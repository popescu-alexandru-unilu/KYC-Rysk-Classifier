import pytest


def _import_mod():
    import importlib
    mod = importlib.import_module("src.infer_minilm")
    return mod


def test_none_no_codes():
    IM = _import_mod()
    t = "[SANCTIONS] list=none"
    assert IM.extract_sanctions_codes(t) == []
    assert IM.sanctions_hit(t) is False


def test_us_sdn_parsed_and_hits():
    IM = _import_mod()
    t = "[SANCTIONS] list=US-SDN"
    codes = IM.extract_sanctions_codes(t)
    assert "us-sdn" in codes
    assert IM.sanctions_hit(t) is True


def test_none_then_us_sdn():
    IM = _import_mod()
    t = "[SANCTIONS] list=none; US-SDN"
    codes = IM.extract_sanctions_codes(t)
    assert "us-sdn" in codes
    assert IM.sanctions_hit(t) is True


def test_stop_at_next_tag():
    IM = _import_mod()
    t = "[SANCTIONS] list=eu-blr [MEDIA] 0"
    codes = IM.extract_sanctions_codes(t)
    assert codes == ["eu-blr"]
    assert IM.sanctions_hit(t) is True


def test_whitelist_blocks_unknown_by_default_empty_then_allows(monkeypatch):
    IM = _import_mod()
    t = "[SANCTIONS] list=unknown | jp-meti"
    # With a non-empty whitelist that does not include jp-meti, it should be filtered out
    monkeypatch.setattr(IM, "_valid_codes", {"us-sdn"}, raising=False)
    assert IM.extract_sanctions_codes(t) == []
    assert IM.sanctions_hit(t) is False
    # If whitelist includes jp-meti, it should pass
    monkeypatch.setattr(IM, "_valid_codes", {"us-sdn", "jp-meti"}, raising=False)
    assert IM.extract_sanctions_codes(t) == ["jp-meti"]
    assert IM.sanctions_hit(t) is True


def test_space_separated_alias_behavior(monkeypatch):
    IM = _import_mod()
    t = "[SANCTIONS] list=US SDN"
    # With empty whitelist (default behavior), tokens remain and sanctions_hit is True
    monkeypatch.setattr(IM, "_valid_codes", set(), raising=False)
    assert IM.extract_sanctions_codes(t) == ["us", "sdn"]
    assert IM.sanctions_hit(t) is True
    # With a non-empty whitelist lacking these tokens, they are filtered out
    monkeypatch.setattr(IM, "_valid_codes", {"us-sdn"}, raising=False)
    assert IM.extract_sanctions_codes(t) == []
    assert IM.sanctions_hit(t) is False

