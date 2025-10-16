import time, json, hashlib, hmac, os, re, yaml
from fastapi import HTTPException
from fastapi import Request
from prometheus_client import Counter

# Metrics
AUDIT = Counter("audited_total", "audit log writes")
REDACT = Counter("redactions_applied", "redaction substitutions", ["type"])


def _load_cfg():
    path = os.getenv("RISK_RULES", "config/risk_rules.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

CFG = _load_cfg()

_PATTERNS = {
    "long_number": re.compile(r"\b\d{6,}\b"),
    "alnum_id":    re.compile(r"\b[A-Z0-9][A-Z0-9\-]{5,}\b", re.I),
    "iban":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", re.I),
    "swift":       re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b", re.I),
    "email":       re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.I),
    "phone":       re.compile(r"(?:(?:\+|00)\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{3,5}"),
}


def require_api_key(request: Request) -> None:
    """Validate API key from Authorization or X-API-Key headers.
    - Accepts: Authorization: Bearer <KEY> or X-API-Key: <KEY>
    - Env: API_KEY must be set (single key)
    """
    expected = os.getenv("API_KEY")
    if not expected:
        # If not configured, treat as disabled
        return
    # Try Authorization: Bearer <key>
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    got = None
    if auth and isinstance(auth, str):
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            got = parts[1]
    # Fallback to X-API-Key
    if not got:
        got = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if not got or got != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def redact(t: str) -> str:
    rules = CFG.get("redaction", {
        "long_number": "<REDACTED_NUM>",
        "alnum_id":    "<REDACTED_ID>",
        "iban":        "<REDACTED_IBAN>",
        "swift":       "<REDACTED_SWIFT>",
        "email":       "<REDACTED_EMAIL>",
        "phone":       "<REDACTED_PHONE>",
    })
    out = t
    for key, pat in _PATTERNS.items():
        repl = rules.get(key)
        if not repl:
            continue
        try:
            cnt = len(list(pat.finditer(out)))
            if cnt:
                REDACT.labels(type=key).inc(cnt)
                out = pat.sub(repl, out)
        except Exception:
            continue
    return out


def log_decision(inp: str, out: dict, *, secret: str | None = None, path: str = "logs/audit.jsonl", request_id: str | None = None) -> None:
    secret = secret or os.getenv("AUDIT_HMAC", "change_me_in_prod")
    blob = {
        "ts": time.time(),
        "text_sha": hashlib.sha256(inp.encode()).hexdigest(),
        "label": out.get("label"),
        "rule": out.get("rule"),
        "probs": out.get("probs"),
        "why": out.get("why"),
        "trace": out.get("trace"),
        "band_label": out.get("band_label"),
        "policy_config_hash": out.get("policy_config_hash"),
        "meta_tags": out.get("meta_tags"),
        "country": out.get("country"),
        "sanc_codes": out.get("sanc_codes"),
    }
    # optional override metadata for richer audit
    if out.get("override_reason"):
        blob["override_reason"] = out.get("override_reason")
    if out.get("sanctions_codes"):
        blob["sanctions_codes"] = out.get("sanctions_codes")
    if request_id:
        blob["request_id"] = request_id
    # optionally persist minimal policy context
    pol = out.get("policy") or {}
    if pol:
        blob["policy"] = {
            "temperature": pol.get("temperature"),
            "bands": pol.get("bands"),
        }
    mac = hmac.new(secret.encode(), json.dumps(blob, sort_keys=True).encode(), "sha256").hexdigest()
    blob["hmac"] = mac
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(blob) + "\n")
    AUDIT.inc()
