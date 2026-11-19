import numpy as np

from src.config import load_config
from src.geometry import module_ids
from src.modules.health_monitor import (
    HealthMonitor, PowerState, array_total, module_powers)

cfg = load_config()
TARGET = tuple(cfg["client"]["target"])


def run(cfg, strike=True, seed=None, t_end=1000):
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    state, mon, events = PowerState(), HealthMonitor(cfg), []
    for t in range(0, t_end + 1):
        p = module_powers(float(t), cfg, state, rng, strike=strike)
        for e in mon.step(float(t), p):
            events.append(e)
            if e.kind == "failed":
                state.open_switch.add(e.module)
    return events


def first_event(events, kind):
    return next(e for e in events if e.kind == kind)


def replaced_ratio(cfg):
    rng = np.random.default_rng(1)
    state = PowerState()
    state.replaced.add(TARGET)
    p = module_powers(100.0, cfg, state, rng, strike=False)
    others = [v for k, v in p.items() if k != TARGET]
    return p[TARGET] / np.median(others)


def test_service_request_about_661s():
    assert 655 <= first_event(run(cfg, strike=True), "service_request").t <= 665


def test_failed_targets_struck_module():
    assert first_event(run(cfg), "failed").module == TARGET


def test_aging_only_no_request():
    assert not any(e.kind == "service_request" for e in run(cfg, strike=False))


def test_new_module_ratio_passes():
    assert 1.10 < replaced_ratio(cfg) < 1.18


def test_power_restored():
    rng = np.random.default_rng(3)
    state = PowerState()
    before = array_total(module_powers(10.0, cfg, state, rng, strike=True))
    state.struck.add(TARGET)
    state.open_switch.add(TARGET)
    faulted = array_total(module_powers(700.0, cfg, state, rng))
    assert faulted < 0.99 * before
    state.struck.discard(TARGET)
    state.open_switch.discard(TARGET)
    state.replaced.add(TARGET)
    after = array_total(module_powers(900.0, cfg, state, rng))
    assert after >= 0.99 * before


def test_same_seed_same_result():
    a = first_event(run(cfg, seed=42), "service_request").t
    b = first_event(run(cfg, seed=42), "service_request").t
    assert a == b


def test_checkout_pass_and_fail():
    rng = np.random.default_rng(5)
    state, mon = PowerState(), HealthMonitor(cfg)
    state.replaced.add(TARGET)
    mon.start_checkout(TARGET, 0.0, 10.0)
    ev = []
    for t in range(0, 12):
        ev += mon.step(float(t), module_powers(float(t), cfg, state, rng, strike=False))
    assert [e.kind for e in ev if e.kind.startswith("checkout")] == ["checkout_pass"]

    state2, mon2, ev2 = PowerState(), HealthMonitor(cfg), []
    state2.open_switch.add(TARGET)
    mon2.start_checkout(TARGET, 0.0, 10.0)
    for t in range(0, 12):
        ev2 += mon2.step(float(t), module_powers(float(t), cfg, state2, rng, strike=False))
    assert any(e.kind == "checkout_fail" for e in ev2)


def test_sixty_ids():
    assert len(module_ids(cfg)) == 60
