"""Scenario integration tests (Task 7).

The module fixture flies the whole demo-mode mission in Basilisk without
Vizard (a few minutes). The `slow` tests run it again with the Vizard replay.
"""
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.geometry import module_center
from src.modules.arm_motion import SwapSequence, berth_pose_A
from src.plots import make_plots
from src.scenario_servicing import REPO, run
from src.viz_setup import R_MO, module_dcm_NO

cfg = load_config()


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    return run(load_config(), viz=False, outdir=tmp_path_factory.mktemp("noviz"))


@pytest.fixture(scope="module")
def viz_log(tmp_path_factory):
    """Replay goes to a temp folder (never overwrites outputs/)."""
    return run(load_config(), viz=True, outdir=tmp_path_factory.mktemp("viz"))


# --- fast frame checks ---------------------------------------------------------

def test_seated_module_coplanar_with_array():
    """A seated module_target (S frame at berth) maps onto the array plane:
    its OBJ frame equals A and its center is the slot center."""
    c_S, dcm_SM = SwapSequence(cfg).at(0.0).bodies["module_target"]
    pos_A, dcm_AS = berth_pose_A(cfg)
    dcm_AO = dcm_AS @ dcm_SM @ R_MO
    assert np.allclose(dcm_AO, np.eye(3), atol=1e-9)
    assert np.allclose(dcm_AS @ c_S + pos_A, module_center(*cfg["client"]["target"], cfg))


def test_module_dcm_NO_maps_obj_normal_to_x_M():
    dcm_NS = np.eye(3)
    dcm_SM = np.eye(3)
    dcm_NO = module_dcm_NO(dcm_NS, dcm_SM)
    assert np.allclose(dcm_NO @ [0, 0, 1.0], [1.0, 0, 0])  # OBJ +z -> x_M


# --- full mission (no viz) --------------------------------------------------------

def test_mission_completes(log):
    assert log.phase[-1] == 9


def test_phase_order(log):
    seen = []
    for p in log.phase:
        if not seen or seen[-1] != p:
            seen.append(int(p))
    assert seen == list(range(10))


def test_power_restored(log):
    pre = log.t < cfg["power"]["strike_t_s"]
    assert log.array_power[-1] >= 0.99 * log.array_power[pre].mean()


def test_dv_budget(log):
    assert log.dv_used[-1] <= cfg["budget"]["dv_mps"]


def test_no_timeouts(log):
    assert not any(s == "timeout" for s in log.sub)


def test_plots_written(log, tmp_path):
    paths = make_plots(log, tmp_path)
    assert [p.name for p in paths] == ["hill_path.png", "range_speed.png",
                                       "module_ratio.png", "delta_v.png"]
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)


# --- with the Vizard replay ----------------------------------------------------------

@pytest.mark.slow
def test_viz_file_small(viz_log):
    f = viz_log.replay
    assert f.name == Path(cfg["viz"]["save_file"]).name and REPO / "outputs" not in f.parents
    assert 0 < f.stat().st_size < 300e6


@pytest.mark.slow
def test_ambient_applied(viz_log):
    assert viz_log.viz_settings["ambient"] == cfg["viz"]["ambient"]


def test_corridor(log):
    """Inside the 15 m keep-out sphere the servicer stays in the approach cone."""
    c = cfg["client"]
    tgt = module_center(*c["target"], cfg)
    inside = [p for p in log.rel_A[:, :3] if np.linalg.norm(p) < c["keepout_m"]]
    assert inside
    for p in inside:
        d = p - tgt
        along = -d[2]
        assert 0.0 <= along <= c["cone_len_m"]
        assert np.linalg.norm(d[:2]) <= along * np.tan(np.radians(c["cone_half_deg"])) + 1e-6


def test_rigid_link_while_inhibited(log):
    """While thrusters are inhibited rel_A stays fixed (servicer locked to the client)."""
    i6 = log.phase == 6
    assert np.ptp(log.rel_A[i6, :3], axis=0).max() < 0.01
    assert np.abs(log.rel_A[i6, 3:]).max() < 1e-3


def test_mass_handoff(log):
    m0 = cfg["servicer"]["mass_kg"]
    assert log.mass.max() == m0 + cfg["client"]["module_mass_kg"]
    assert log.mass[0] == m0 and log.mass[-1] == m0
