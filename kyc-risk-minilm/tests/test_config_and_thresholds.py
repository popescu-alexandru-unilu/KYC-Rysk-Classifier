import importlib


def IM():
    return importlib.import_module("src.infer_minilm")


def test_media_threshold_override(monkeypatch):
    mod = IM()
    # Lower media_high to 1 and assert reason triggers for 2 mentions
    thr = mod.THR
    class T:
        media_high = 1
        inflow_ratio_high = getattr(thr, "inflow_ratio_high", 3.0)
        inflow_ratio_low = getattr(thr, "inflow_ratio_low", 1.8)
        burst_high = getattr(thr, "burst_high", 50)
        burst_low = getattr(thr, "burst_low", 10)
    monkeypatch.setattr(mod, "THR", T, raising=False)

    text = "[KYC] Name: X [COUNTRY] DE [SANCTIONS] list=none [MEDIA] 2 mentions"
    reasons = mod.build_reasons(text, "low", {"low": 0.9, "medium": 0.05, "high": 0.05}, "model_only")
    assert any("Adverse media mentions" in r for r in reasons)

