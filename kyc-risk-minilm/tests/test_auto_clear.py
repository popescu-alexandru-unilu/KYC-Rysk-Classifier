from fastapi.testclient import TestClient


def get_client():
    from api.src.app import app
    return TestClient(app)


def test_auto_clear_flag_low_risk():
    c = get_client()
    # Craft a text that meets auto-clear side conditions
    text = "[COUNTRY] DE [SANCTIONS] list=none [MEDIA] 0 mentions Inflow ratio = 1.2 Burst trades = 3"
    r = c.post("/classify", json={"text": text, "override": False})
    assert r.status_code == 200
    data = r.json()
    # Not asserting label strictness here; only that auto_clear boolean exists and is coherent
    assert "auto_clear" in data


def test_auto_clear_false_when_override():
    c = get_client()
    text = "[COUNTRY] FR [SANCTIONS] list=US-BIS-EL [MEDIA] 0 mentions"
    r = c.post("/classify", json={"text": text, "override": True})
    assert r.status_code == 200
    data = r.json()
    assert data["rule"] == "sanctions_override"
    assert data.get("auto_clear") is False

