import json, os
from fastapi.testclient import TestClient


def get_client():
    from api.src.app import app
    return TestClient(app)


def write_audit_lines(path):
    rows = [
        {
            "ts": 1700000000.0,
            "text_sha": "a"*64,
            "label": "high",
            "rule": "sanctions_override",
            "probs": {"low": 0.0, "medium": 0.0, "high": 1.0},
            "why": ["Sanctions/PEP code(s) present: us-bis-el"],
            "band_label": "high",
            "country": "DE",
            "meta_tags": {"owner": "TeamA", "lang": "en"},
            "hmac": "0"*64,
        },
        {
            "ts": 1700000001.0,
            "text_sha": "b"*64,
            "label": "low",
            "rule": "model_only",
            "probs": {"low": 0.9, "medium": 0.1, "high": 0.0},
            "why": ["No adverse media mentions found"],
            # band_label missing intentionally
            "country": "de",
            "meta_tags": {"owner": "teama", "lang": "EN"},
            "hmac": "0"*64,
        },
        {
            "ts": 1700000002.0,
            "text_sha": "c"*64,
            "label": "medium",
            "rule": "model_only",
            "probs": {"low": 0.1, "medium": 0.8, "high": 0.1},
            "why": ["Adverse media mentions: 2 count exceeds threshold 2, indicating significant reputational risk"],
            "band_label": "medium",
            "country": "FR",
            "meta_tags": {},
            "hmac": "0"*64,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_audit_search_filters(tmp_path, monkeypatch):
    # Point search to temp audit file
    audit_path = tmp_path / "audit.jsonl"
    write_audit_lines(audit_path)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(audit_path))

    c = get_client()

    # Case-insensitive label filter
    r = c.get("/audit_search", params={"label": "HIGH"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1 and items[0]["label"] == "high"

    # Band filter includes entries with missing band_label
    r = c.get("/audit_search", params={"band": "HIGH"})
    assert r.status_code == 200
    items = r.json()
    # Should at least include the high band row and the one missing band
    labels = {it.get("label") for it in items}
    assert "high" in labels
    assert any("band_label" not in it or it.get("band_label") is None for it in items)

    # Country case-insensitive
    r = c.get("/audit_search", params={"country": "de"})
    assert r.status_code == 200
    items = r.json()
    assert all((it.get("country") or "").upper() == "DE" for it in items)

    # Owner/lang case-insensitive via meta_tags
    r = c.get("/audit_search", params={"owner": "teama", "lang": "en"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1
    mt = items[0].get("meta_tags") or {}
    assert (mt.get("owner") or "").lower() == "teama"
    assert (mt.get("lang") or "").lower() == "en"

