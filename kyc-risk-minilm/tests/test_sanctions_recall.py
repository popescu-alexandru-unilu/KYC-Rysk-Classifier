from src.infer_minilm import sanctions_hit


def ok(t):
    return sanctions_hit(t)


def nok(t):
    return not sanctions_hit(t)


def test_semicolon_combo():
    assert ok("[SANCTIONS] list= none ; US-BIS-EL")


def test_spaces_code():
    assert ok("[SANCTIONS] list= US SAM")


def test_notes_sdn():
    assert ok("[SANCTIONS] list=none; remarks: SDN")


def test_none_only():
    assert nok("[SANCTIONS] list= none ")

