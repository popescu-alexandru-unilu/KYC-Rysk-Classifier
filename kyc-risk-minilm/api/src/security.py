import time, json, hashlib, hmac, os, re


def redact(t: str) -> str:
    t = re.sub(r"\b[0-9]{6,}\b", "<REDACTED_NUM>", t)
    t = re.sub(r"\b[A-Z0-9]{6,9}\b", "<REDACTED_ID>", t)
    return t


def log_decision(inp: str, out: dict, *, secret: str | None = None, path: str = "logs/audit.jsonl") -> None:
    secret = secret or os.getenv("AUDIT_HMAC", "change_me_in_prod")
    blob = {
        "ts": time.time(),
        "text_sha": hashlib.sha256(inp.encode()).hexdigest(),
        "label": out.get("label"),
        "rule": out.get("rule"),
        "probs": out.get("probs"),
        "why": out.get("why"),
        "band_label": out.get("band_label"),
        "policy_config_hash": out.get("policy_config_hash"),
        "meta_tags": out.get("meta_tags"),
        "country": out.get("country"),
        "sanc_codes": out.get("sanc_codes"),
    }
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
