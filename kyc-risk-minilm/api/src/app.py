# api/src/app.py
import os, time, json, hashlib, base64, tempfile, shutil, csv, zipfile, io
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
import threading

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
    allow_origins=["http://localhost:3000", "http://46.62.218.2:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CKPT = os.getenv("MODEL_CKPT", "/app/models/minilm_cls_best")
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
    from .model_loader import load_model
    loaded = load_model()
    _model = loaded["model"]
    # compute model id from checkpoint content if possible
    try:
        import hashlib as _hl
        if loaded["type"] == "hf":
            ckpt_path = os.getenv("MODEL_CKPT", "")
            _MODEL_ID = _hl.sha256(ckpt_path.encode()).hexdigest()
        else:
            with open(MODEL_CKPT, 'rb') as _f:
                _MODEL_ID = _hl.sha256(_f.read()).hexdigest()
    except Exception:
        try:
            _MODEL_ID = os.path.basename(MODEL_CKPT)
        except Exception:
            _MODEL_ID = None
    return _model

def _validate_ckpt(path_str: str):
    """Lightweight checkpoint validation for health endpoint.
    - HF dir: require config.json(with model_type), pytorch_model.bin, tokenizer files
    - Torch file: require .pt/.pth exists
    Returns a dict with details and ok flag.
    """
    out = {
        "ok": False,
        "type": None,
        "exists": False,
        "reason": None,
        "num_labels": None,
        "id2label": None,
        "missing": [],
    }
    try:
        p = Path(path_str or "")
        if not p.exists():
            out["reason"] = f"not found: {p}"
            return out
        out["exists"] = True
        if p.is_dir():
            out["type"] = "hf"
            # required files
            req = ["config.json", "pytorch_model.bin"]
            opt_tokenizer = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "merges.txt", "vocab.json"]
            for f in req:
                if not (p / f).exists():
                    out["missing"].append(f)
            # need at least one tokenizer artifact
            if not any((p / f).exists() for f in opt_tokenizer):
                out["missing"].append("tokenizer_files")
            # inspect config.json
            try:
                cfg = json.load(open(p / "config.json", encoding="utf-8"))
                out["num_labels"] = cfg.get("num_labels")
                out["id2label"] = cfg.get("id2label")
                if not cfg.get("model_type"):
                    out["missing"].append("config.model_type")
            except Exception as e:
                out["reason"] = f"config.json parse failed: {e}"
            # label mismatch note (non-fatal for health but useful)
            try:
                from .model_minilm import NUM_LABELS as CODE_NUM_LABELS
                if out.get("num_labels") and CODE_NUM_LABELS and int(out["num_labels"]) != int(CODE_NUM_LABELS):
                    out["label_mismatch"] = {"ckpt": out["num_labels"], "code": CODE_NUM_LABELS}
            except Exception:
                pass
            if not out["missing"] and not out.get("reason"):
                out["ok"] = True
            else:
                if not out.get("reason"):
                    out["reason"] = f"missing: {', '.join(out['missing'])}"
            return out
        else:
            # single file
            ext = p.suffix.lower()
            if ext in (".pt", ".pth"):
                out["type"] = "torch"
                out["ok"] = True
                return out
            out["type"] = "unknown"
            out["reason"] = f"unsupported file ext: {ext}"
            return out
    except Exception as e:
        out["reason"] = str(e)
        return out

@app.get("/health")
def health():
    override_enabled = True
    try:
        override_enabled = bool((RULES.policy or {}).get("override_enabled", True))
    except Exception:
        pass
    ck = _validate_ckpt(MODEL_CKPT)
    payload = {
        "ok": bool(ck.get("ok")),
        "ckpt_exists": bool(ck.get("exists")),
        "ckpt_type": ck.get("type"),
        "ckpt_reason": ck.get("reason"),
        "ckpt_num_labels": ck.get("num_labels"),
        "ckpt_id2label": ck.get("id2label"),
        "device": _device,
        "policy_config_hash": POLICY_CONFIG_HASH,
        "model_id": _MODEL_ID,
        "override_enabled": override_enabled,
    }
    # If checkpoint invalid, return 503 to trip healthcheck
    if not payload["ok"]:
        from fastapi import status
        return Response(content=json.dumps(payload), media_type="application/json", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return payload

@app.on_event("startup")
def _maybe_eager_load():
    """Optionally load model at startup to fail fast.
    Set EAGER_LOAD=true to enable. Defaults to true.
    """
    flag = os.getenv("EAGER_LOAD", "true").strip().lower() in ("1","true","yes","on")
    if not flag:
        return
    # Validate checkpoint first; if invalid, raise to stop startup
    ck = _validate_ckpt(MODEL_CKPT)
    if not ck.get("ok"):
        raise RuntimeError(f"Invalid checkpoint for startup: {ck}")
    # Load once
    _ = _get_model()

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

# Batch job store (in-memory, for demo; use Redis or DB for production)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class BatchFileUploadRequest(BaseModel):
    file_data: str = Field(..., description="Base64 encoded file content")
    file_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="MIME type or file extension")
    columns_mapping: Optional[Dict[str, str]] = None  # column_name -> field_name mapping


class BatchJobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed | cancelled
    created_at: float
    filename: str
    total_rows: Optional[int] = None
    processed_rows: Optional[int] = None
    success_count: Optional[int] = None
    failed_count: Optional[int] = None
    eta_seconds: Optional[int] = None
    last_error: Optional[str] = None
    download_url_results: Optional[str] = None
    download_url_errors: Optional[str] = None


class BatchJobResultSummary(BaseModel):
    total: int
    low: int
    medium: int
    high: int
    critical: int
    override_applied: int


def _process_batch_job(job_id: str):
    """Background task to process a batch job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job['status'] not in ('queued',):
            return
        job['status'] = 'running'
        _jobs[job_id] = job

    try:
        # Get model early
        model = _get_model()
        _, _, classify_batch_func, sanctions_hit, build_reasons, override_high_payload, _, should_override, extract_sanctions_codes = _load_all()

        # Parse file data (placeholder for actual parsing; add try/except)
        file_data = base64.b64decode(job['file_data'])  # job needs to store file_data
        ext = job['file_type'].lower()

        rows = []
        if ext == 'csv':
            content = file_data.decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
        elif ext in ('json', 'jsonl'):
            content = file_data.decode('utf-8')
            if ext == 'json':
                rows = json.loads(content)
                if not isinstance(rows, list):
                    rows = [rows]
            else:  # jsonl
                rows = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        else:
            # For XLSX, you'd need openpyxl: from openpyxl import load_workbook; wb = load_workbook(io.BytesIO(data)); rows = list(ws.values)[1:] but to keep simple, raise error
            rows = []  # Placeholder

        if not rows:
            raise ValueError("No rows parsed")

        # Convert to text items (assume 'name', 'country', 'sanctions', etc. columns)
        items = []
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                name = row.get('name', row.get('KYC', ''))
                country = row.get('country', row.get('COUNTRY', ''))
                sanctions = row.get('sanctions', row.get('SANCTIONS', ''))
                media = row.get('media', row.get('MEDIA', '0'))
                text = f'[KYC] Name: {name}\n[COUNTRY] {country}\n[SANCTIONS] {sanctions}\n[MEDIA] {media} mentions'
            else:
                text = str(row)
            items.append({'id': i, 'text': text})

        # Batch classify
        texts = [it['text'] for it in items]
        outputs = classify_batch_func(model, _device, texts, MAX_LEN)

        results = []
        for it, out in zip(items, outputs):
            text = it['text']
            codes = extract_sanctions_codes(text)
            # (Similar logic to classify_batch_route)
            res = out
            res['id'] = it['id']
            res['override_applied'] = False
            res['why'] = build_reasons(text, res['label'], res['probs'], res.get('rule', 'model_only'))
            # Audit would go here if needed
            results.append(res)

        # Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            job['results_path'] = f.name

        with _jobs_lock:
            _jobs[job_id]['status'] = 'done'
            _jobs[job_id]['processed_rows'] = len(results)
            _jobs[job_id]['success_count'] = len(results)
            _jobs[job_id]['failed_count'] = 0
            _jobs[job_id]['total_rows'] = len(results)
            _jobs[job_id]['download_url_results'] = f"/batch/{job_id}/results.csv"

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['last_error'] = str(e)


@app.post("/batch", response_model=BatchJobStatus)
def create_batch_job(payload: BatchFileUploadRequest, background_tasks: BackgroundTasks, request: Request):
    """Start a new batch screening job."""
    require_api_key(request)
    job_id = uuid4().hex
    created_at = time.time()

    # Basic file validation (mock for now)
    file_size = len(payload.file_data) * 3 // 4  # approx base64 decoded size
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="File too large")

    ext = payload.file_type.lower()
    accepted = ['csv', 'xlsx', 'xls', 'json', 'jsonl', 'txt', 'zip']
    if ext not in accepted:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Store job
    job = {
        'job_id': job_id,
        'status': 'queued',
        'created_at': created_at,
        'filename': payload.file_name,
        'file_type': payload.file_type,
        'file_data': payload.file_data,  # Store base64 data
        'columns_mapping': payload.columns_mapping or {},
        'last_error': None,
        'total_rows': None,
        'processed_rows': None,
        'success_count': None,
        'failed_count': None,
        'eta_seconds': None,
        'results_path': None,
        'errors_path': None,
        'download_url_results': None,
        'download_url_errors': None,
    }

    with _jobs_lock:
        _jobs[job_id] = job

    # Start background processing
    background_tasks.add_task(_process_batch_job, job_id)

    return BatchJobStatus(**job)


@app.get("/batch/{job_id}/status", response_model=BatchJobStatus)
def get_batch_status(job_id: str, request: Request):
    """Get batch job status."""
    require_api_key(request)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return BatchJobStatus(**job)


@app.post("/batch/{job_id}/cancel", response_model=dict)
def cancel_batch_job(job_id: str, request: Request):
    """Cancel a batch job."""
    require_api_key(request)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job['status'] = 'cancelled'
    return {'status': 'cancelled'}


@app.get("/batch/{job_id}/results.{format}")
def download_batch_results(job_id: str, format: str, request: Request, include: str = Query("why", description="Comma-separated fields like 'why'"), limit: int = Query(None, ge=1, le=10000), offset: int = Query(0, ge=0)):
    """Download batch results in JSON or CSV.

    For JSON:
    - Supports limit/offset for pagination
    - Returns array of results with total_count
    - Ensures all values are JSON-serializable

    For CSV:
    - Full export (limit ignored for now, but could be paginated if needed)
    - Always returns entire dataset as attachment
    """
    require_api_key(request)
    if format not in ('json', 'csv'):
        raise HTTPException(status_code=400, detail="Format must be json or csv")

    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] != 'done':
        raise HTTPException(status_code=409, detail=f"Job not completed (status: {job['status']})")

    if not job.get('results_path') or not os.path.exists(job['results_path']):
        raise HTTPException(status_code=404, detail="Results file not found")

    # Load results (be careful with memory for large files)
    try:
        with open(job['results_path'], 'r', encoding='utf-8') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Results file is corrupted: {str(e)}")

    # Ensure results is a list
    if not isinstance(results, list):
        raise HTTPException(status_code=500, detail="Results file format invalid")

    total_count = len(results)
    if total_count == 0:
        # Empty results
        if format == 'json':
            empty_response = {"results": [], "total_count": 0, "offset": offset, "limit": limit}
            return Response(
                json.dumps(empty_response, separators=(',', ':')),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename=batch-{job_id}-results.json'}
            )
        else:
            # Empty CSV with headers
            csv_data = "id,label,rule,low,medium,high,why\n".encode('utf-8')
            return Response(
                csv_data,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename=batch-{job_id}-results.csv'}
            )

    # Apply pagination for JSON, but for CSV return full dataset
    if format == 'json':
        # Default limit for UI table view
        if limit is None:
            limit = 1000  # Good for table view, prevents memory issues

        start_idx = offset
        end_idx = start_idx + limit
        paginated_results = results[start_idx:end_idx]

        # Ensure all values are JSON-serializable
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, bool, str)) or obj is None:
                return obj  # native JSON types
            else:
                # Convert problematic types to strings
                return str(obj)

        paginated_results = [sanitize_for_json(r) for r in paginated_results]

        response_data = {
            "results": paginated_results,
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": end_idx < total_count
        }

        json_response = json.dumps(response_data, separators=(',', ':'), ensure_ascii=False)
        return Response(
            json_response.encode('utf-8'),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename=batch-{job_id}-results.json'}
        )

    else:  # format == 'csv'
        # Generate CSV as bytes to avoid encoding issues
        def generate_csv_lines(results_list):
            yield "id,label,rule,low_pct,medium_pct,high_pct,why\n"
            for r in results_list:
                # Ensure safe defaults
                row_id = str(r.get('id', ''))
                label = str(r.get('label', 'low'))
                rule = str(r.get('rule', 'model_only'))

                probs = r.get('probs', {}) if isinstance(r.get('probs'), dict) else {}
                low_pct = int(float(probs.get('low', 0)) * 100)
                med_pct = int(float(probs.get('medium', 0)) * 100)
                high_pct = int(float(probs.get('high', 0)) * 100)

                why_list = r.get('why', []) if isinstance(r.get('why'), list) else []
                why_str = ";".join(str(w) for w in why_list)

                # Escape quotes in fields that might contain semicolons
                for field in [why_str]:
                    if '"' in field or ';' in field or ',' in field:
                        field = f'"{field.replace(chr(34), chr(34)+chr(34))}"'

                yield f"{row_id},{label},{rule},{low_pct}%,{med_pct}%,{high_pct}%,{why_str}\n"

        # Collect all lines and encode to bytes in one go
        csv_content = "".join(generate_csv_lines(results))
        csv_bytes = csv_content.encode('utf-8-sig')  # utf-8-sig for Excel compatibility

        return Response(
            csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename=batch-{job_id}-results.csv'}
        )


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, request: Request, response: Response):
    # API key auth
    require_api_key(request)
    t0 = time.perf_counter()
    text     = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required and cannot be empty")
    text = text.strip()
    if ("[KYC]" not in text) or ("[COUNTRY]" not in text):
        raise HTTPException(status_code=400, detail="missing required tags [KYC] and/or [COUNTRY]")
    override = bool(payload.override if payload.override is not None else True)
    max_len  = int(payload.max_len or MAX_LEN)

    try:
        MiniLMClassifier, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS, should_override, extract_sanctions_codes = _load_all()
        model = _get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    codes = extract_sanctions_codes(text)
    sanc_present = bool(codes)
    if sanc_present:
        SANCTIONS_PRESENT.inc()
    use_override = should_override(override)
    dry_run = bool((RULES.policy or {}).get("override_dry_run", False)) if RULES and RULES.policy is not None else False
    if sanc_present and use_override and not dry_run:
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
        override_applied = use_override and sanc_present and not dry_run
        if override_applied:
            final: Dict[str, Any] = override_high_payload(codes)
            final["why"] = build_reasons(text, final["label"], final["probs"], final["rule"])
            final["trace"] = [{"rule": "sanctions_override", "codes": codes}]
            SANCTIONS_OVERRIDE.inc()
        else:
            final = base
            final["rule"] = "model_only"
            final["why"] = build_reasons(text, final["label"], final["probs"], final["rule"])
            _sig = parse_signals(text)
            _trace = []
            if _sig.get("sanc_codes"):
                _trace.append({"rule": "sanctions_present", "codes": _sig.get("sanc_codes")})
            if _sig.get("media_cnt") is not None:
                mc = _sig.get("media_cnt")
                try:
                    media_thr = float(RULES.thresholds.media_high)
                except Exception:
                    media_thr = 2.0
                _trace.append({"rule": "media", "value": mc, "op": ">=", "threshold": media_thr} if mc >= media_thr else {"rule": "media", "value": mc, "op": "==", "threshold": 0})
            if _sig.get("inflow_ratio") is not None:
                ir = _sig.get("inflow_ratio")
                try:
                    hi = float(RULES.thresholds.inflow_ratio_high)
                    lo = float(RULES.thresholds.inflow_ratio_low)
                except Exception:
                    hi, lo = 3.0, 1.8
                _trace.append({"rule": "inflow_ratio", "value": ir, "op": ">=", "threshold": hi} if ir >= hi else {"rule": "inflow_ratio", "value": ir, "op": "<", "threshold": lo})
            final["trace"] = _trace
            if use_override and sanc_present and dry_run:
                final["override_would_apply"] = True
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
            # try relative durations (parse float + suffix)
            value_lower = value.lower()
            if value_lower.endswith('h'):
                hours = float(value[:-1])
                return _dt.datetime.now().timestamp() - (hours * 3600)
            elif value_lower.endswith('d'):
                days = float(value[:-1])
                return _dt.datetime.now().timestamp() - (days * 86400)
            elif value_lower.endswith('w'):
                weeks = float(value[:-1])
                return _dt.datetime.now().timestamp() - (weeks * 604800)
            elif value_lower.endswith('m'):
                months = float(value[:-1])
                return _dt.datetime.now().timestamp() - (months * 2629746)  # approx 30.44 days
            else:
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

    # Reuse search logic
    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
    MAX_EXPORT = 10000  # Hard cap for safety

    # Parse query string into filters (same as audit_search)
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
                        parsed_val = _parse_time(val)
                        if parsed_val is None:
                            raise HTTPException(status_code=400, detail=f"Invalid {key} format: {val}. Use ISO8601, Unix timestamp, or relative (e.g., 24h, 1d, 7w)")
                        filters[key] = parsed_val
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

    items = []

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

                items.append(obj)
                count += 1
    except FileNotFoundError:
        pass

    if not items:
        # Return empty CSV with headers
        return Response("ts,label,rule,band_label,country,owner,lang,why\n", media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit_export.csv"})

    # Generate CSV
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

    return Response("\n".join(generate()), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=audit_export.csv"})


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
