import importlib


def IM():
    return importlib.import_module("src.infer_minilm")


def _mk_text(country: str) -> str:
    return f"[KYC] Name: X\n[COUNTRY] {country}\n[SANCTIONS] list=none\n[MEDIA] 0 mentions"


def test_build_reasons_fatf_list_highrisk(monkeypatch):
    mod = IM()
    cfg = dict(mod.CFG)
    cfg["fatf"] = {"mode": "list", "high_risk": ["IRAN"], "monitor": ["TURKEY"]}
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)
    text = _mk_text("IRAN")
    reasons = mod.build_reasons(text, "low", {"low": 0.9, "medium": 0.05, "high": 0.05}, "model_only")
    assert any("FATF high-risk jurisdiction (policy bump applied)" in r for r in reasons)


def test_build_reasons_fatf_list_monitor(monkeypatch):
    mod = IM()
    cfg = dict(mod.CFG)
    cfg["fatf"] = {"mode": "list", "high_risk": ["IRAN"], "monitor": ["TURKEY"]}
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)
    text = _mk_text("TURKEY")
    reasons = mod.build_reasons(text, "low", {"low": 0.9, "medium": 0.05, "high": 0.05}, "model_only")
    assert any("FATF on monitoring list" in r for r in reasons)


def test_build_reasons_fatf_score_high(monkeypatch):
    mod = IM()
    cfg = dict(mod.CFG)
    cfg["fatf"] = {"mode": "score", "scores": {"IRAN": 95, "FR": 10}, "bump_threshold": 80}
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)
    text = _mk_text("IRAN")
    reasons = mod.build_reasons(text, "low", {"low": 0.9, "medium": 0.05, "high": 0.05}, "model_only")
    assert any("FATF score ≥ 80 (policy bump applied)" in r for r in reasons)


def test_build_reasons_no_flags(monkeypatch):
    mod = IM()
    cfg = dict(mod.CFG)
    cfg["fatf"] = {"mode": "list", "high_risk": ["IRAN"], "monitor": ["TURKEY"]}
    monkeypatch.setattr(mod, "CFG", cfg, raising=False)
    text = _mk_text("FR")
    reasons = mod.build_reasons(text, "low", {"low": 0.9, "medium": 0.05, "high": 0.05}, "model_only")
    assert any("no additional FATF flags" in r for r in reasons)

