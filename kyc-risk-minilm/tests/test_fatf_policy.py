import importlib


def IM():
    return importlib.import_module("src.infer_minilm")


def test_country_bucket_list_mode(monkeypatch):
    mod = IM()
    fatf = {
        "mode": "list",
        "high_risk": ["IRAN", "DPRK"],
        "monitor": ["TURKEY", "PAKISTAN"],
    }
    cfg = dict(mod.CFG)
    cfg["fatf"] = fatf
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)

    assert mod.country_bucket("IRAN") == "high_risk"
    assert mod.country_bucket("DPRK") == "high_risk"
    assert mod.country_bucket("TURKEY") == "monitor"
    assert mod.country_bucket("FR") is None


def test_apply_fatf_bump_list_mode(monkeypatch):
    mod = IM()
    fatf = {
        "mode": "list",
        "high_risk": ["IRAN"],
        "monitor": ["TURKEY"],
    }
    cfg = dict(mod.CFG)
    cfg["fatf"] = fatf
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)

    # High-risk bumps
    assert mod.apply_fatf_bump("low", "IRAN") == "medium"
    assert mod.apply_fatf_bump("medium", "IRAN") == "high"
    assert mod.apply_fatf_bump("high", "IRAN") == "high"
    # Monitor does not bump
    assert mod.apply_fatf_bump("low", "TURKEY") == "low"
    # None stays
    assert mod.apply_fatf_bump("low", "FR") == "low"


def test_country_bucket_score_mode_and_bump(monkeypatch):
    mod = IM()
    fatf = {
        "mode": "score",
        "scores": {"IRAN": 95, "FR": 10},
        "bump_threshold": 80,
    }
    cfg = dict(mod.CFG)
    cfg["fatf"] = fatf
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)

    assert mod.country_bucket("IRAN") == "high_score"
    assert mod.country_bucket("FR") is None
    assert mod.apply_fatf_bump("low", "IRAN") == "medium"
    assert mod.apply_fatf_bump("low", "FR") == "low"

