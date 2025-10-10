from fastapi.testclient import TestClient


def get_client():
    from api.src.app import app
    return TestClient(app)


def test_none_plus_media_burst_is_not_low():
    c = get_client()
    text = "[SANCTIONS] list=none [MEDIA] 5 mentions Burst trades = 60 [COUNTRY] DE"
    r = c.post("/classify", json={"text": text, "override": False})
    assert r.status_code == 200
    data = r.json()
    # expect medium or high based on thresholds (ours: high due to burst>=50)
    assert data["label"] in ("medium", "high")

