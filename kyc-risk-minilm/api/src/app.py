# api/src/app.py
import os, time, json, hashlib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

PRED = Counter("pred_total", "preds", ["label","path"])  # per-label counts split by path
DECISIONS = Counter("decisions_total", "Total classification decisions")
AUTO_CLEAR = Counter("autoclear_low_total", "Auto-cleared low-risk decisions")
SANCTIONS_PRESENT = Counter("sanctions_present_total", "Requests with sanctions codes present")
SANCTIONS_OVERRIDE = Counter("sanctions_override_total", "Policy overrides due to sanctions codes")
AUDIT_EVENTS = Counter("audit_events_total", "Audit log events written")
REQ_LAT = Histogram(
    "request_latency_seconds",
    "Classifier request latency in seconds",
    ["route"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
)

app = FastAPI(title="KYC Risk Classifier", version="v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CKPT = os.getenv("MODEL_CKPT", "/app/minilm_cls_best.pt")
MAX_LEN    = int(os.getenv("MAX_LEN", "256"))

_model = None
_device = "cpu"  # keep simple; enable CUDA only if you have it
_MODEL_ID = None

# security utils
from .security import redact, log_decision, require_api_key
from .infer_minilm import parse_signals

# config loader and policy hash
try:
    from .config import load_rules
    RULES = load_rules()
    try:
        POLICY_CONFIG_HASH = hashlib.sha256(
            json.dumps(RULES.model_dump(), sort_keys=True).encode()
        ).hexdigest()
    except Exception:
        POLICY_CONFIG_HASH = None
except Exception:
    RULES = None
    POLICY_CONFIG_HASH = None

def _load_all():
    """Import modules lazily so missing files don't crash boot."""
    from .model_minilm import MiniLMClassifier  # noqa
    from .infer_minilm import (
        load_model, classify_batch, sanctions_hit, build_reasons,
        override_high_payload, LABELS, should_override, extract_sanctions_codes
    )
    return MiniLMClassifier, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS, should_override, extract_sanctions_codes

def _get_model():
    global _model
    global _MODEL_ID
    if _model is not None:
        return _model
    _, load_model, *_ = _load_all()
    if not os.path.exists(MODEL_CKPT):
        raise HTTPException(status_code=500, detail=f"MODEL_CKPT not found: {MODEL_CKPT}")
    _model = load_model(MODEL_CKPT, _device)
    # compute model id from checkpoint content if possible
    try:
        import hashlib as _hl
        with open(MODEL_CKPT, 'rb') as _f:
            _MODEL_ID = _hl.sha256(_f.read()).hexdigest()
    except Exception:
        try:
            _MODEL_ID = os.path.basename(MODEL_CKPT)
        except Exception:
            _MODEL_ID = None
    return _model

@app.get("/health")
def health():
    return {
        "ok": True,
        "ckpt_exists": os.path.exists(MODEL_CKPT),
        "device": _device,
        "policy_config_hash": POLICY_CONFIG_HASH,
        "model_id": _MODEL_ID,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3)
    override: Optional[bool] = True
    format: Optional[str] = "json"
    max_len: Optional[int] = MAX_LEN


class ClassifyResponse(BaseModel):
    label: str
    probs: Dict[str, float]
    rule: str
    why: Optional[List[str]] = None
    auto_clear: Optional[bool] = False
    band_label: Optional[str] = None
    policy: Optional[Dict[str, Any]] = None
    policy_config_hash: Optional[str] = None
    meta_tags: Optional[Dict[str, Any]] = None
    country: Optional[str] = None
    sanc_codes: Optional[List[str]] = None


class BatchItem(BaseModel):
    id: Optional[Union[str, int]] = None
    text: str = Field(..., min_length=3)


class BatchRequest(BaseModel):
    items: List[BatchItem]
    override: Optional[bool] = True
    max_len: Optional[int] = MAX_LEN


class BatchResponseItem(ClassifyResponse):
    id: Optional[Union[str, int]] = None

from uuid import uuid4


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, request: Request, response: Response):
    # API key auth
    require_api_key(request)
    t0 = time.perf_counter()
    text     = payload.text.strip()
    if ("[KYC]" not in text) or ("[COUNTRY]" not in text):
        raise HTTPException(status_code=400, detail="missing required tags [KYC] and/or [COUNTRY]")
    override = bool(payload.override if payload.override is not None else True)
    max_len  = int(payload.max_len or MAX_LEN)

    MiniLMClassifier, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS, should_override, extract_sanctions_codes = _load_all()
    model = _get_model()

    codes = extract_sanctions_codes(text)
    sanc_present = bool(codes)
    if sanc_present:
        SANCTIONS_PRESENT.inc()
    use_override = should_override(override)
    dry_run = bool((RULES.policy or {}).get("override_dry_run", False)) if RULES and RULES.policy is not None else False
    if use_override and sanc_present and not dry_run:
        res: Dict[str, Any] = override_high_payload(codes)
        res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
        res["trace"] = [{"rule": "sanctions_override", "codes": codes}]
        SANCTIONS_OVERRIDE.inc()
        override_applied = True
    else:
        out = classify_batch(model, _device, [text], max_len)[0]
        out["rule"] = "model_only"
        out["why"]  = build_reasons(text, out["label"], out["probs"], out["rule"])
        # Build a compact machine-readable trace for audit
        _sig = parse_signals(text)
        _trace = []
        if _sig.get("sanc_codes"):
            _trace.append({"rule":"sanctions_present", "codes": _sig.get("sanc_codes")})
        if _sig.get("media_cnt") is not None:
            mc = _sig.get("media_cnt")
            try:
                media_thr = float(RULES.thresholds.media_high)
            except Exception:
                media_thr = 2.0
            _trace.append({"rule":"media", "value": mc, "op": ">=", "threshold": media_thr} if mc >= media_thr else {"rule":"media", "value": mc, "op": "==", "threshold": 0})
        if _sig.get("inflow_ratio") is not None:
            ir = _sig.get("inflow_ratio")
            try:
                hi = float(RULES.thresholds.inflow_ratio_high)
                lo = float(RULES.thresholds.inflow_ratio_low)
            except Exception:
                hi, lo = 3.0, 1.8
            _trace.append({"rule":"inflow_ratio", "value": ir, "op": ">=", "threshold": hi} if ir >= hi else {"rule":"inflow_ratio", "value": ir, "op": "<", "threshold": lo})
        out["trace"] = _trace
        res = out
        override_applied = False
        if use_override and sanc_present and dry_run:
            res["override_would_apply"] = True
    res["override_applied"] = override_applied

    # confidence bands label (does not replace core label)
    try:
        p = res.get("probs", {})
        low_min = float(RULES.policy.get("bands", {}).get("low_min", 0.70)) if RULES and RULES.policy else 0.70
        high_min = float(RULES.policy.get("bands", {}).get("high_min", 0.80)) if RULES and RULES.policy else 0.80
        if float(p.get("high", 0)) >= high_min:
            res["band_label"] = "high"
        elif float(p.get("low", 0)) >= low_min:
            res["band_label"] = "low"
        else:
            res["band_label"] = "medium"
    except Exception:
        res["band_label"] = None

    # determine auto_clear (adoption metric)
    auto_clear = False
    try:
        if res.get("rule") == "model_only" and res.get("label") == "low":
            p_low  = float(res.get("probs", {}).get("low", 0.0))
            sig = parse_signals(text)
            # configurable thresholds (fallbacks provided)
            min_low_prob = float(RULES.policy.get("auto_clear", {}).get("min_low_prob", 0.65)) if RULES and RULES.policy else 0.65
            max_media    = int(RULES.policy.get("auto_clear", {}).get("max_media", 0)) if RULES and RULES.policy else 0
            max_burst    = int(RULES.policy.get("auto_clear", {}).get("max_burst", 10)) if RULES and RULES.policy else 10
            max_inflow   = float(RULES.policy.get("auto_clear", {}).get("max_inflow", 1.8)) if RULES and RULES.policy else 1.8
            media_ok  = (sig.get("media_cnt") or 0) <= max_media
            burst_ok  = (sig.get("burst_trades") or 0) <  max_burst
            inflow_ok = (sig.get("inflow_ratio") or 0.0) < max_inflow
            auto_clear = (p_low >= min_low_prob) and media_ok and burst_ok and inflow_ok
    except Exception:
        auto_clear = False

    # attach parsed meta + light signals
    sig = parse_signals(text)
    res["meta_tags"] = {k: sig.get(k) for k in ("owner","conf","ver","lang") if sig.get(k)} or None
    res["country"] = sig.get("country")
    res["sanc_codes"] = sig.get("sanc_codes") or []
    res["auto_clear"] = auto_clear
    # attach policy info and hash
    try:
        temperature = float(RULES.policy.get("temperature", 1.0)) if RULES and RULES.policy else 1.0
        low_min = float(RULES.policy.get("bands", {}).get("low_min", 0.70)) if RULES and RULES.policy else 0.70
        high_min = float(RULES.policy.get("bands", {}).get("high_min", 0.80)) if RULES and RULES.policy else 0.80
        res["policy"] = {"temperature": temperature, "bands": {"low_min": low_min, "high_min": high_min}}
        res["policy_config_hash"] = POLICY_CONFIG_HASH
    except Exception:
        pass
    PRED.labels(label=res["label"], path=("override" if override_applied else "model")).inc()
    DECISIONS.inc()
    if res.get("label") == "low" and res.get("rule") == "model_only":
        AUTO_CLEAR.inc()
    try:
        req_id = uuid4().hex
        response.headers["X-Model-Id"] = _MODEL_ID or "unknown"
        response.headers["X-Policy-Id"] = POLICY_CONFIG_HASH or ""
        response.headers["X-Request-Id"] = req_id
        log_decision(redact(text), res, request_id=req_id)
        AUDIT_EVENTS.inc()
    finally:
        pass
    REQ_LAT.labels(route="classify").observe(max(0.0, time.perf_counter() - t0))
    return res


@app.post("/classify_batch", response_model=List[BatchResponseItem])
def classify_batch_route(payload: BatchRequest, request: Request, response: Response):
    # API key auth
    require_api_key(request)
    t0 = time.perf_counter()
    items = payload.items
    if not items:
        raise HTTPException(status_code=400, detail="no items provided")
    # Batch cap
    try:
        BATCH_MAX = int(os.getenv("BATCH_MAX", "256"))
    except Exception:
        BATCH_MAX = 256
    if len(items) > BATCH_MAX:
        raise HTTPException(status_code=413, detail=f"batch size {len(items)} exceeds limit {BATCH_MAX}")
    override = bool(payload.override if payload.override is not None else True)
    max_len  = int(payload.max_len or MAX_LEN)

    _, _, classify_batch, sanctions_hit, build_reasons, override_high_payload, _, should_override, extract_sanctions_codes = _load_all()
    model = _get_model()

    texts = [it.text for it in items]
    outputs = classify_batch(model, _device, texts, max_len)
    results: List[BatchResponseItem] = []
    # Per-request ID for audit correlation
    req_id = uuid4().hex
    response.headers["X-Model-Id"] = _MODEL_ID or "unknown"
    response.headers["X-Policy-Id"] = POLICY_CONFIG_HASH or ""
    response.headers["X-Request-Id"] = req_id
    for it, base in zip(items, outputs):
        text = it.text
        codes = extract_sanctions_codes(text)
        sanc_present = bool(codes)
        if sanc_present:
            SANCTIONS_PRESENT.inc()
        use_override = should_override(override)
        dry_run = bool((RULES.policy or {}).get("override_dry_run", False)) if RULES and RULES.policy is not None else False
        if use_override and sanc_present and not dry_run:
            final: Dict[str, Any] = override_high_payload(codes)
            SANCTIONS_OVERRIDE.inc()
            override_applied = True
            final["trace"] = [{"rule":"sanctions_override", "codes": codes}]
        else:
            final = base
            final["rule"] = "model_only"
            override_applied = False
        final["why"] = final.get("why") or build_reasons(text, final["label"], final["probs"], final["rule"])
        if not final.get("trace"):
            _sig = parse_signals(text)
            _trace = []
            if _sig.get("sanc_codes"):
                _trace.append({"rule":"sanctions_present", "codes": _sig.get("sanc_codes")})
            if _sig.get("media_cnt") is not None:
                mc = _sig.get("media_cnt")
                try:
                    media_thr = float(RULES.thresholds.media_high)
                except Exception:
                    media_thr = 2.0
                _trace.append({"rule":"media", "value": mc, "op": ">=", "threshold": media_thr} if mc >= media_thr else {"rule":"media", "value": mc, "op": "==", "threshold": 0})
            if _sig.get("inflow_ratio") is not None:
                ir = _sig.get("inflow_ratio")
                try:
                    hi = float(RULES.thresholds.inflow_ratio_high)
                    lo = float(RULES.thresholds.inflow_ratio_low)
                except Exception:
                    hi, lo = 3.0, 1.8
                _trace.append({"rule":"inflow_ratio", "value": ir, "op": ">=", "threshold": hi} if ir >= hi else {"rule":"inflow_ratio", "value": ir, "op": "<", "threshold": lo})
            final["trace"] = _trace
        final["override_applied"] = override_applied
        if use_override and sanc_present and dry_run:
            final["override_would_apply"] = True
        # bands
        try:
            p = final.get("probs", {})
            low_min = float(RULES.policy.get("bands", {}).get("low_min", 0.70)) if RULES and RULES.policy else 0.70
            high_min = float(RULES.policy.get("bands", {}).get("high_min", 0.80)) if RULES and RULES.policy else 0.80
            if float(p.get("high", 0)) >= high_min:
                final["band_label"] = "high"
            elif float(p.get("low", 0)) >= low_min:
                final["band_label"] = "low"
            else:
                final["band_label"] = "medium"
        except Exception:
            final["band_label"] = None
        # attach policy info and hash
        try:
            temperature = float(RULES.policy.get("temperature", 1.0)) if RULES and RULES.policy else 1.0
            final["policy"] = {"temperature": temperature, "bands": {"low_min": low_min, "high_min": high_min}}
            final["policy_config_hash"] = POLICY_CONFIG_HASH
        except Exception:
            pass
        # attach parsed meta + light signals
        sig = parse_signals(text)
        final["meta_tags"] = {k: sig.get(k) for k in ("owner","conf","ver","lang") if sig.get(k)} or None
        final["country"] = sig.get("country")
        final["sanc_codes"] = sig.get("sanc_codes") or []

        # determine auto_clear for batch
        auto_clear = False
        try:
            if final.get("rule") == "model_only" and final.get("label") == "low":
                p_low  = float(final.get("probs", {}).get("low", 0.0))
                min_low_prob = float(RULES.policy.get("auto_clear", {}).get("min_low_prob", 0.65)) if RULES and RULES.policy else 0.65
                max_media    = int(RULES.policy.get("auto_clear", {}).get("max_media", 0)) if RULES and RULES.policy else 0
                max_burst    = int(RULES.policy.get("auto_clear", {}).get("max_burst", 10)) if RULES and RULES.policy else 10
                max_inflow   = float(RULES.policy.get("auto_clear", {}).get("max_inflow", 1.8)) if RULES and RULES.policy else 1.8
                media_ok  = (sig.get("media_cnt") or 0) <= max_media
                burst_ok  = (sig.get("burst_trades") or 0) <  max_burst
                inflow_ok = (sig.get("inflow_ratio") or 0.0) < max_inflow
                auto_clear = (p_low >= min_low_prob) and media_ok and burst_ok and inflow_ok
        except Exception:
            auto_clear = False
        final["auto_clear"] = auto_clear
        PRED.labels(label=final["label"], path=("override" if override_applied else "model")).inc()
        DECISIONS.inc()
        if final.get("label") == "low" and final.get("rule") == "model_only":
            AUTO_CLEAR.inc()
        try:
            log_decision(redact(text), final, request_id=req_id)
            AUDIT_EVENTS.inc()
        finally:
            pass
        results.append(BatchResponseItem(id=it.id, **final))
    REQ_LAT.labels(route="classify_batch").observe(max(0.0, time.perf_counter() - t0))
    return results


@app.on_event("startup")
def preload_model():
    # Preload model to avoid cold-start latency on first request
    try:
        _get_model()
    except Exception:
        # still allow app to start; /health exposes ckpt_exists for probes
        pass


def _parse_time(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        # try ISO8601
        import datetime as _dt
        try:
            return _dt.datetime.fromisoformat(value).timestamp()
        except Exception:
            return None


@app.get("/audit_suggest")
def audit_suggest(
    field: str = Query(default="label", description="Field to suggest for (label, rule, country, owner, lang)"),
    prefix: str = Query(default="", description="Prefix to match"),
    limit: int = Query(default=10, ge=1, le=20),
):
    """Suggest values for a field based on recent entries."""
    if field not in ("label", "rule", "country", "owner", "lang"):
        return []

    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
    seen = set()
    suggestions = []

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                val = None
                if field == "label":
                    val = (obj.get("label") or "").lower()
                elif field == "rule":
                    val = (obj.get("rule") or "").lower()
                elif field == "country":
                    val = (obj.get("country") or "").upper()
                elif field == "owner":
                    mt = obj.get("meta_tags") or {}
                    val = (mt.get("owner") or "").lower()
                elif field == "lang":
                    mt = obj.get("meta_tags") or {}
                    val = (mt.get("lang") or "").lower()

                if val and val.lower().startswith(prefix.lower()) and val not in seen:
                    seen.add(val)
                    suggestions.append(val)
                    if len(suggestions) >= limit:
                        break
    except FileNotFoundError:
        pass

    return suggestions


@app.get("/audit_export")
def audit_export(
    q: Optional[str] = Query(default=None, description="Free text or key:value tokens (space-separated)"),
    format: str = Query(default="csv", choices=["csv"]),
):
    """Export audit search results as CSV (streaming)."""
    from io import StringIO
    import csv

    # Reuse search logic, but stream all (up to 100k limit)
    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
    MAX_EXPORT = 10000  # Hard cap for safety
    items = []

    # Simplified filter for export (can be expanded)
    filters = {}
    free_text_tokens = []
    if q:
        for part in q.split():
            if ":" in part:
                key, val = part.split(":", 1)
                filters[key.lower()] = {"val": val}
            else:
                free_text_tokens.append(part.lower())

    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if count >= MAX_EXPORT:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # Apply basic filter
                match = True  # Simplified to avoid complexity

                if match:
                    items.append(obj)
                    count += 1
    except FileNotFoundError:
        pass

    # Stream CSV
    def generate():
        yield "ts,label,rule,band_label,country,owner,lang,why\n"
        for item in items:
            ts = item.get("ts", "")
            label = item.get("label", "")
            rule = item.get("rule", "")
            band = item.get("band_label", "")
            country = item.get("country", "")
            owner = (item.get("meta_tags") or {}).get("owner", "")
            lang = (item.get("meta_tags") or {}).get("lang", "")
            why = ";".join(item.get("why", [])).replace(",", ";").replace('"', "'")
            yield f"{ts},{label},{rule},{band},{country},{owner},{lang},\"{why}\"\n"

    return Response(generate(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit_export.csv"})


@app.get("/audit_search")
def audit_search(
    q: Optional[str] = Query(default=None, description="Free text or key:value tokens (space-separated)"),
    limit: int = Query(default=100, ge=1, le=1000),
    cursor: Optional[str] = Query(default=None, description="Cursor for pagination (base64 encoded)"),
):
    """Smart search audit.jsonl with tokens and free text. Returns up to `limit` entries."""
    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")

    # Parse query string into filters
    filters = {}
    free_text_tokens = []

    if q:
        # Split by spaces, detect key:values and comparators
        for part in q.split():
            part = part.strip()
            if ":" in part:
                # Supports key:value, key:>=val, key:>val, key:<val, key:=val, why:"phrase"
                if part.startswith('why:"') and part.endswith('"'):
                    key, val = "why", part[5:-1].strip()
                elif ":>=" in part:
                    key, val = part.split(">:=", 1)
                    filters[key] = {"op": "gte", "val": float(val)}
                elif ":>" in part:
                    key, val = part.split(":>", 1)
                    filters[key] = {"op": "gt", "val": float(val)}
                elif ":<=" in part:
                    key, val = part.split(":<=", 1)
                    filters[key] = {"op": "lte", "val": float(val)}
                elif ":<" in part:
                    key, val = part.split(":<", 1)
                    filters[key] = {"op": "lt", "val": float(val)}
                elif ":=" in part:
                    key, val = part.split(":=", 1)
                    filters[key] = {"val": val}
                else:
                    key, val = part.split(":", 1)
                    if key == "since" or key == "until":
                        try:
                            filters[key] = _parse_time(val)
                        except:
                            pass
                    else:
                        filters[key] = {"val": val}
            else:
                # Free text tokens
                if part:
                    free_text_tokens.append(part.lower())

    # Convert to lowercase for case-insensitive matching
    normalized_filters = {}
    for k, v in filters.items():
        if isinstance(v, dict):
            normalized_filters[k.lower()] = {**v, "val": v["val"].lower() if isinstance(v.get("val"), str) else v["val"]}
        else:
            normalized_filters[k.lower()] = v

    out: List[Dict[str, Any]] = []
    total_count = 0

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                total_count += 1

                # Apply token filters (single-field or combined)
                match_found = True

                for filter_key, filter_val in normalized_filters.items():
                    field_val = None
                    if filter_key == "label":
                        field_val = obj.get("label", "").lower()
                    elif filter_key == "rule":
                        field_val = obj.get("rule", "").lower()
                    elif filter_key == "band":
                        field_val = obj.get("band_label", "").lower() if obj.get("band_label") else None
                        if field_val is None:  # Include if band not specified and field missing
                            continue
                    elif filter_key == "country":
                        field_val = (obj.get("country") or "").upper()
                    elif filter_key == "owner":
                        mt = obj.get("meta_tags") or {}
                        field_val = (mt.get("owner") or "").lower()
                    elif filter_key == "lang":
                        mt = obj.get("meta_tags") or {}
                        field_val = (mt.get("lang") or "").lower()
                    elif filter_key == "prob_high":
                        field_val = float(obj.get("probs", {}).get("high", 0))
                    elif filter_key == "prob_low":
                        field_val = float(obj.get("probs", {}).get("low", 0))
                    elif filter_key == "why":
                        why_str = " ".join(obj.get("why", [])).lower()
                        field_val = why_str  # For contains check
                    elif filter_key == "since":
                        field_val = float(obj.get("ts") or 0.0)
                    elif filter_key == "until":
                        field_val = float(obj.get("ts") or 0.0)

                    # Apply comparison
                    if isinstance(filter_val, dict):
                        if "op" in filter_val:
                            val = filter_val["val"]
                            if filter_val["op"] == "gte" and not (isinstance(field_val, (int, float)) and field_val >= val):
                                match_found = False
                                break
                            elif filter_val["op"] == "gt" and not (isinstance(field_val, (int, float)) and field_val > val):
                                match_found = False
                                break
                            elif filter_val["op"] == "lte" and not (isinstance(field_val, (int, float)) and field_val <= val):
                                match_found = False
                                break
                            elif filter_val["op"] == "lt" and not (isinstance(field_val, (int, float)) and field_val < val):
                                match_found = False
                                break
                        elif "val" in filter_val and not (filter_val["val"] in str(field_val).lower()):
                            match_found = False
                            break
                    elif isinstance(filter_val, (int, float)) and filter_key == "since" and field_val < filter_val:
                        match_found = False
                        break
                    elif isinstance(filter_val, (int, float)) and filter_key == "until" and field_val > filter_val:
                        match_found = False
                        break

                if not match_found:
                    continue

                # Free text search across why, rule, label, band, country, owner, lang
                if free_text_tokens:
                    search_text = " ".join([
                        obj.get("why", []),
                        obj.get("rule", ""),
                        obj.get("label", ""),
                        obj.get("band_label", ""),
                        obj.get("country", ""),
                        (obj.get("meta_tags") or {}).get("owner", ""),
                        (obj.get("meta_tags") or {}).get("lang", ""),
                    ]).lower()

                    if not all(token in search_text for token in free_text_tokens):
                        continue

                out.append(obj)
                if len(out) >= limit:
                    break

    except FileNotFoundError:
        return {"items": [], "total": 0, "next_cursor": None}

    # Simple cursor (last ts for next page, base64 like)
    import base64
    next_cursor = None
    if len(out) == limit and out:
        next_cursor = base64.b64encode(str(int(out[-1]["ts"])).encode()).decode()

    return {"items": out, "total": total_count, "next_cursor": next_cursor}
