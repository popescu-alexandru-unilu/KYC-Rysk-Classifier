import json, hmac, hashlib
from api.src.security import log_decision, redact


def test_hmac_tamper_detection(tmp_path, monkeypatch):
    # Use test secret from conftest or default
    secret = "test_secret"
    monkeypatch.setenv("AUDIT_HMAC", secret)
    out = {
        "label": "low",
        "rule": "model_only",
        "probs": {"low": 1.0, "high": 0.0},
        "why": ["No adverse media mentions found"],
    }
    p = tmp_path / "audit.jsonl"
    log_decision(redact("Name John, email john@example.com, acct 1234567890"), out, path=str(p))

    line = p.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    mac = obj.pop("hmac")
    calc = hmac.new(secret.encode(), json.dumps(obj, sort_keys=True).encode(), "sha256").hexdigest()
    assert mac == calc
    # Tamper and expect mismatch
    obj["label"] = "high"
    calc2 = hmac.new(secret.encode(), json.dumps(obj, sort_keys=True).encode(), "sha256").hexdigest()
    assert mac != calc2


def test_redaction_rules_direct():
    # Direct redact() function test for configurable patterns
    s = "IBAN GB12BARC12345678901234 swift ABCDGB2L email a.b+c@example.com phone +1 202-555-0199 id AA12BB34 and number 1234567"
    red = redact(s)
    # Placeholders should replace sensitive patterns
    assert "<REDACTED_IBAN>" in red
    assert "<REDACTED_SWIFT>" in red
    assert "<REDACTED_EMAIL>" in red
    assert "<REDACTED_PHONE>" in red
    assert "<REDACTED_ID>" in red or "<REDACTED_NUM>" in red

