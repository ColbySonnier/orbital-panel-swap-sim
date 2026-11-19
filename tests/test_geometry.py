import numpy as np
import pytest

from src.config import load_config
from src.geometry import (array_dcm, client_mass_props, hill_dcm,
                          module_center, module_ids, waypoints_A)

cfg = load_config()
R = np.array([6928e3, 0.0, 0.0])
V = np.array([0.0, 7.58e3, 0.0])
S = np.array([0.6, -0.3, 0.7])
S = S / np.linalg.norm(S)


def frames():
    return array_dcm(S, R, V), hill_dcm(R, V)


def test_target_module_center():
    assert np.allclose(module_center(2, 7, cfg), [5, -5, 0])


def test_sixty_modules():
    ids = module_ids(cfg)
    assert len(ids) == 60 and (4, 4) not in ids


def test_dcms_orthonormal():
    for d in (hill_dcm(R, V), array_dcm(S, R, V)):
        assert np.allclose(d @ d.T, np.eye(3))
        assert np.isclose(np.linalg.det(d), 1.0)


def test_array_axes():
    A, H = frames()
    assert np.allclose(A @ S, [0, 0, 1])
    yH = H[1]
    yA = A @ yH  # Hill y seen in A: no x component, positive y component
    assert abs(yA[0]) < 1e-12 and yA[1] > 0


def test_berth_and_h3():
    A, H = frames()
    w = waypoints_A(cfg, A, H)
    assert np.allclose(w["BERTH"], [5, -5, -2.6])
    assert np.allclose(w["H3"], [5, -5, -20])
    assert np.allclose(w["TARGET"], [5, -5, 0])


def test_w3_is_h2_minus_normal():
    A, H = frames()
    w = waypoints_A(cfg, A, H)
    assert np.allclose(w["W3"] - w["H2"], [0, 0, -20])


def test_h2_includes_com_offset():
    A, H = frames()
    w = waypoints_A(cfg, A, H)
    _, com, _ = client_mass_props(cfg, set())
    h2 = A @ H.T @ np.array(cfg["waypoints"]["H2"]) + com
    assert np.allclose(w["H2"], h2)


def test_client_mass():
    m, com, I = client_mass_props(cfg, set())
    assert abs(m - 6120) < 1
    assert np.allclose(com[:2], 0) and com[2] < 0
    assert np.allclose(I, I.T) and np.all(np.linalg.eigvalsh(I) > 0)


def test_missing_module_reduces_mass_and_shifts_com():
    m0, c0, _ = client_mass_props(cfg, set())
    m1, c1, _ = client_mass_props(cfg, {(2, 7)})
    assert np.isclose(m0 - m1, 12)
    assert not np.allclose(c0, c1)
