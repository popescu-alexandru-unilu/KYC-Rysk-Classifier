import os
from fastapi.testclient import TestClient


def get_client():
    from api.src.app import app
    return TestClient(app)


def test_classify_batch_basic_and_ids(monkeypatch):
    c = get_client()
    payload = {
        "items": [
            {"id": "a", "text": "[KYC] Name: A [COUNTRY] DE [SANCTIONS] list=none [MEDIA] 0 mentions"},
            {"id": "b", "text": "[KYC] Name: B [COUNTRY] HK [SANCTIONS] list=US-BIS-EL [MEDIA] 0 mentions"},
        ],
        "override": True,
    }
    r = c.post("/classify_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2
    ids = {it.get("id") for it in data}
    assert ids == {"a", "b"}
    # One item should be override applied
    ov = {it.get("id"): it.get("override_applied") for it in data}
    assert ov["a"] in (False, None)
    assert ov["b"] is True
    # Headers should include request id
    assert r.headers.get("X-Request-Id")


def test_classify_batch_cap(monkeypatch):
    c = get_client()
    # Force cap to 2
    monkeypatch.setenv("BATCH_MAX", "2")
    payload = {
        "items": [
            {"text": "[KYC] Name: A [COUNTRY] DE [SANCTIONS] list=none"},
            {"text": "[KYC] Name: B [COUNTRY] DE [SANCTIONS] list=none"},
            {"text": "[KYC] Name: C [COUNTRY] DE [SANCTIONS] list=none"},
        ]
    }
    r = c.post("/classify_batch", json=payload)
    assert r.status_code == 413
    assert "exceeds limit" in r.text

