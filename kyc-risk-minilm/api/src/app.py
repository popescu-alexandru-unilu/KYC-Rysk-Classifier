# api/src/app.py
import os, time, json, hashlib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

PRED = Counter("pred_total", "preds", ["label"])  # per-label counts
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

# security utils
from .security import redact, log_decision
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
        override_high_payload, LABELS
    )
    return MiniLMClassifier, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS

def _get_model():
    global _model
    if _model is not None:
        return _model
    _, load_model, *_ = _load_all()
    if not os.path.exists(MODEL_CKPT):
        raise HTTPException(status_code=500, detail=f"MODEL_CKPT not found: {MODEL_CKPT}")
    _model = load_model(MODEL_CKPT, _device)
    return _model

@app.get("/health")
def health():
    return {
        "ok": True,
        "ckpt_exists": os.path.exists(MODEL_CKPT),
        "device": _device
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

@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest):
    t0 = time.perf_counter()
    text     = payload.text.strip()
    override = bool(payload.override if payload.override is not None else True)
    max_len  = int(payload.max_len or MAX_LEN)

    MiniLMClassifier, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS = _load_all()
    model = _get_model()

    sanc_present = sanctions_hit(text)
    if sanc_present:
        SANCTIONS_PRESENT.inc()
    if override and sanc_present:
        res: Dict[str, Any] = override_high_payload()
        res["why"] = build_reasons(text, res["label"], res["probs"], res["rule"])
        SANCTIONS_OVERRIDE.inc()
    else:
        out = classify_batch(model, _device, [text], max_len)[0]
        out["rule"] = "model_only"
        out["why"]  = build_reasons(text, out["label"], out["probs"], out["rule"])
        res = out

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
    PRED.labels(label=res["label"]).inc()
    DECISIONS.inc()
    if res.get("label") == "low" and res.get("rule") == "model_only":
        AUTO_CLEAR.inc()
    try:
        log_decision(redact(text), res)
        AUDIT_EVENTS.inc()
    finally:
        pass
    REQ_LAT.labels(route="classify").observe(max(0.0, time.perf_counter() - t0))
    return res


@app.post("/classify_batch", response_model=List[BatchResponseItem])
def classify_batch_route(payload: BatchRequest):
    t0 = time.perf_counter()
    items = payload.items
    if not items:
        raise HTTPException(status_code=400, detail="no items provided")
    override = bool(payload.override if payload.override is not None else True)
    max_len  = int(payload.max_len or MAX_LEN)

    _, _, classify_batch, sanctions_hit, build_reasons, override_high_payload, _ = _load_all()
    model = _get_model()

    texts = [it.text for it in items]
    outputs = classify_batch(model, _device, texts, max_len)
    results: List[BatchResponseItem] = []
    for it, base in zip(items, outputs):
        text = it.text
        sanc_present = sanctions_hit(text)
        if sanc_present:
            SANCTIONS_PRESENT.inc()
        if override and sanc_present:
            final: Dict[str, Any] = override_high_payload()
            SANCTIONS_OVERRIDE.inc()
        else:
            final = base
            final["rule"] = "model_only"
        final["why"] = final.get("why") or build_reasons(text, final["label"], final["probs"], final["rule"])
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
        PRED.labels(label=final["label"]).inc()
        DECISIONS.inc()
        if final.get("label") == "low" and final.get("rule") == "model_only":
            AUTO_CLEAR.inc()
        try:
            log_decision(redact(text), final)
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


@app.get("/audit_search")
def audit_search(
    label: Optional[str] = Query(default=None),
    rule: Optional[str] = Query(default=None),
    band: Optional[str] = Query(default=None),
    country: Optional[str] = Query(default=None),
    owner: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None, description="epoch seconds or ISO"),
    until: Optional[str] = Query(default=None, description="epoch seconds or ISO"),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Search audit.jsonl by common fields. Returns up to `limit` entries."""
    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
    t_since = _parse_time(since)
    t_until = _parse_time(until)
    out: List[Dict[str, Any]] = []
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
                if label and obj.get("label") != label:
                    continue
                if rule and obj.get("rule") != rule:
                    continue
                if band and obj.get("band_label") != band:
                    continue
                if country and (obj.get("country") or "").upper() != country.upper():
                    continue
                if owner:
                    mt = obj.get("meta_tags") or {}
                    if (mt.get("owner") or "").lower() != owner.lower():
                        continue
                if lang:
                    mt = obj.get("meta_tags") or {}
                    if (mt.get("lang") or "").lower() != lang.lower():
                        continue
                ts = float(obj.get("ts") or 0.0)
                if t_since and ts < t_since:
                    continue
                if t_until and ts > t_until:
                    continue
                out.append(obj)
                if len(out) >= limit:
                    break
    except FileNotFoundError:
        return []
    return out
