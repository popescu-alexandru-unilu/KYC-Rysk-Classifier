import types
from fastapi.testclient import TestClient


def get_client(monkeypatch, *, sanc_present: bool, probs: dict, label: str, bands=(0.70, 0.80)):
    # import app module
    from api.src import app as appmod

    # fake loaders to avoid real model load
    def fake_load_all():
        # return tuple like (_MiniLMCls, load_model, classify_batch, sanctions_hit, build_reasons, override_high_payload, LABELS)
        def fake_load_model(ckpt, device):
            return object()

        def fake_classify_batch(model, device, texts, max_len):
            # return one result per text
            return [{"probs": probs.copy(), "label": label} for _ in texts]

        def fake_sanctions_hit(text):
            return sanc_present

        def fake_build_reasons(text, label, probs, rule):
            return ["stub reason"]

        # reuse real override_high_payload
        from api.src.infer_minilm import override_high_payload, LABELS
        class _MiniLM: pass
        return (_MiniLM, fake_load_model, fake_classify_batch, fake_sanctions_hit, fake_build_reasons, override_high_payload, LABELS)

    monkeypatch.setattr(appmod, "_load_all", fake_load_all)
    monkeypatch.setattr(appmod, "_get_model", lambda: object())

    # set policy bands
    policy_obj = types.SimpleNamespace(policy={"bands": {"low_min": bands[0], "high_min": bands[1]}, "temperature": 1.0})
    monkeypatch.setattr(appmod, "RULES", policy_obj, raising=False)

    return TestClient(appmod.app)


def test_band_high_without_override(monkeypatch):
    c = get_client(
        monkeypatch,
        sanc_present=True,
        probs={"low": 0.05, "medium": 0.10, "high": 0.85},
        label="high",
        bands=(0.70, 0.80),
    )
    r = c.post("/classify", json={"text": "[SANCTIONS] list=US-BIS-EL", "override": False})
    assert r.status_code == 200
    data = r.json()
    assert data["band_label"] == "high"
    assert data["rule"] == "model_only"


def test_override_forces_high(monkeypatch):
    c = get_client(
        monkeypatch,
        sanc_present=True,
        probs={"low": 0.90, "medium": 0.09, "high": 0.01},
        label="low",
        bands=(0.70, 0.80),
    )
    r = c.post("/classify", json={"text": "[SANCTIONS] list=US-SAM", "override": True})
    assert r.status_code == 200
    data = r.json()
    assert data["rule"] == "sanctions_override"
    assert data["label"] == "high"

