from src.config import load_config


def test_loads_target_and_demo_default():
    c = load_config()
    assert c["client"]["target"] == [2, 7]
    assert c["demo_mode"] is True
    assert c["holds_s"]["H2"] == 300


def test_full_timeline_uses_full_holds():
    c = load_config(full_timeline=True)
    assert c["demo_mode"] is False
    assert c["holds_s"]["H2"] == 5736
