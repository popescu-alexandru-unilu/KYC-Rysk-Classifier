import json, time, os
from fastapi.testclient import TestClient


def get_client():
    # Import inside to ensure env vars are set by conftest
    from api.src.app import app
    return TestClient(app)


def test_health():
    c = get_client()
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("ckpt_exists") in (True, False)


def test_classify_and_audit(tmp_path):
    c = get_client()
    # warm-up request (don’t measure)
    c.post("/classify", json={"text": "[KYC] Name: John [COUNTRY] DE [SANCTIONS] list=none"})

    # measure and validate
    t0 = time.perf_counter()
    r = c.post("/classify", json={"text": "[KYC] Name: Jane [COUNTRY] DE [SANCTIONS] list=none"})
    dt = time.perf_counter() - t0
    assert r.status_code == 200
    data = r.json()
    assert set(data["probs"].keys()) >= {"low", "high"}
    assert data["label"] in ("low", "medium", "high")
    assert data.get("rule") in ("model_only", "sanctions_override")

    # latency smoke check (should be below 0.5s on CPU)
    assert dt < 0.5

    # audit log line appended with hmac
    with open("logs/audit.jsonl", encoding="utf-8") as f:
        last = None
        for line in f:
            if line.strip():
                last = json.loads(line)
    assert last is not None and "hmac" in last and len(last["hmac"]) == 64


def test_policy_override_sanctions():
    c = get_client()
    r = c.post("/classify", json={
        "text": "[KYC] Name: ACME [COUNTRY] HK [SANCTIONS] list=US-BIS-EL",
        "override": True
    })
    assert r.status_code == 200
    data = r.json()
    assert data["rule"] == "sanctions_override"
    assert data["label"] == "high"


def test_metrics_counters_increase():
    c = get_client()
    # Capture metrics before
    m0 = c.get("/metrics").text
    # trigger classify with sanctions present to bump counters
    c.post("/classify", json={"text": "[SANCTIONS] list=US-BIS-EL"})
    m1 = c.get("/metrics").text
    # Counters should appear and increase
    assert "pred_total" in m1
    assert "decisions_total" in m1 and metric_value(m1, "decisions_total") >= metric_value(m0, "decisions_total")
    assert "sanctions_present_total" in m1
    assert metric_value(m1, "sanctions_present_total") >= metric_value(m0, "sanctions_present_total")


def metric_value(metrics_text: str, metric: str) -> float:
    v = 0.0
    for line in metrics_text.splitlines():
        if line.startswith(metric + " "):
            try:
                v = float(line.split()[1])
            except Exception:
                pass
    return v

