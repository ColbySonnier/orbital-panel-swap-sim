import numpy as np

from src.config import load_config
from src.geometry import hill_dcm, waypoints_A
from src.modules.rel_guidance import (StraightLeg, cw_propagate,
                                      cw_relative_accel, pd_accel,
                                      radial_hop_dv)

cfg = load_config()
MU = cfg["orbit"]["mu_km3s2"]
RE = cfg["orbit"]["r_earth_km"]
n = np.sqrt(MU / (RE + cfg["orbit"]["alt_km"]) ** 3)
norm = np.linalg.norm


def test_n_value():
    assert abs(n - 0.0010948) < 1e-7


def test_far_hop_dv():
    assert abs(radial_hop_dv(1800, 0.0010948) - 0.4926) < 1e-3


def test_hop_lands_on_h1():
    dv = radial_hop_dv(1800, n)  # magnitude; +y drift needs radially INWARD push
    x = cw_propagate([0, -2000, 0, -dv, 0, 0], n, np.pi / n)
    assert abs(x[1] - (-200)) < 1.0 and abs(x[0]) < 1.0


def test_cw_accel():
    a = cw_relative_accel([1, 2, 3], [0.1, 0.2, 0.3], n)
    assert np.allclose(a, [3 * n**2 + 2 * n * 0.2, -2 * n * 0.1, -n**2 * 3])


def test_pd_accel():
    a = pd_accel(np.array([1.0, 0, 0]), np.array([0, 2.0, 0]), 0.02, 0.9)
    assert np.allclose(a, [0.02**2, 2 * 0.9 * 0.02 * 2, 0])


def _leg():
    return StraightLeg(np.array([0, 0, 0.0]), np.array([30, 0, 0.0]), 0.05, 0.001)


def test_leg_respects_speed():
    leg = _leg()
    ts = np.linspace(0, leg.duration, 2000)
    assert max(norm(leg.at(t)[1]) for t in ts) <= 0.05 + 1e-9


def test_leg_arrives():
    leg = _leg()
    assert np.allclose(leg.at(leg.duration)[0], [30, 0, 0], atol=1e-6)
    assert np.allclose(leg.at(leg.duration + 100)[0], [30, 0, 0], atol=1e-6)
    assert np.allclose(leg.at(-5)[0], [0, 0, 0])


def test_short_leg_triangular():
    leg = StraightLeg(np.zeros(3), np.array([0.0, 0.0, 0.5]), 0.05, 0.001)
    ts = np.linspace(0, leg.duration, 500)
    assert max(norm(leg.at(t)[1]) for t in ts) < 0.05
    assert np.allclose(leg.at(leg.duration)[0], [0, 0, 0.5], atol=1e-9)


def test_legs_outside_keepout():
    tilt = np.radians(8)
    c, s = np.cos(tilt), np.sin(tilt)
    dcm_HN = np.eye(3)
    dcm_AH = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    w = waypoints_A(cfg, dcm_AH @ dcm_HN, dcm_HN)
    pts = []
    for a, b in (("H2", "W3"), ("W3", "H3")):
        leg = StraightLeg(w[a], w[b], cfg["speeds"]["leg_mps"],
                          cfg["speeds"]["accel_mps2"])
        pts += [leg.at(t)[0] for t in np.linspace(0, leg.duration, 500)]
    assert min(norm(p) for p in pts) >= cfg["client"]["keepout_m"]
