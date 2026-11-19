"""Scenario integration tests (task G3: rail gantry).

The module fixture flies the whole demo-mode mission in Basilisk without
Vizard (seconds). The `slow` tests run it again with the Vizard replay.

Review Focus 4: every pose check below is made on the INERTIAL states that
BodyWriter wrote into the replay messages (read back from the messages), and
the expected slot pose is rebuilt here from the client's own state message:
r_slot = r_client + dcm_NA @ (slot_A - com_A), dcm = dcm_NA (B = A).
"""
from pathlib import Path

import numpy as np
import pytest
from Basilisk.utilities import RigidBodyKinematics as rbk

from src.config import load_config
from src.geometry import client_mass_props, module_center
from src.plots import make_plots
from src.scenario_servicing import PHASES, REPO, run

cfg = load_config()
_, COM, _ = client_mass_props(cfg, set())
SLOT_A = module_center(*cfg["client"]["target"], cfg)
G = cfg["gantry"]
BAY_W_PARKED_A = np.array([G["park_x"]["B"],
                           G["carriage_park_y"] + G["waste_side"] * G["bay_pitch"] / 2,
                           G["bay_z0"]])


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    return run(load_config(), viz=False, outdir=tmp_path_factory.mktemp("noviz"))


@pytest.fixture(scope="module")
def viz_log(tmp_path_factory):
    """Replay goes to a temp folder (never overwrites outputs/)."""
    return run(load_config(), viz=True, outdir=tmp_path_factory.mktemp("viz"))


# --- helpers ---------------------------------------------------------------------

def client_frame(tr, i):
    """(r_client, dcm_NA) at trace row i from the client's state message."""
    return np.array(tr["r_c"][i]), rbk.MRP2C(np.array(tr["sigma_c"][i])).T


def body(tr, name, i):
    """(r_N, dcm_NB) written for a body at trace row i."""
    b = tr[name]
    return np.array(b["r"][i]), rbk.MRP2C(np.array(b["sigma"][i])).T


def to_A(tr, i, r_N):
    r_c, dcm_NA = client_frame(tr, i)
    return dcm_NA.T @ (r_N - r_c) + COM


def slot_N(tr, i):
    r_c, dcm_NA = client_frame(tr, i)
    return r_c + dcm_NA @ (SLOT_A - COM), dcm_NA


# --- mission flow -------------------------------------------------------------------

def test_phase_order(log):
    seen = []
    for p in log.phase:
        if not seen or seen[-1] != p:
            seen.append(int(p))
    assert seen == list(range(len(PHASES))) == list(range(6))


def test_phase_starts_follow_events(log):
    ev = {k: t for t, k, _ in reversed(log.events)}  # first time of each kind
    ps = log.phase_start
    assert ps[1] == pytest.approx(ev["service_request"])
    assert ps[2] - ps[1] == pytest.approx(cfg["holds_s"]["go"], abs=1.0)
    assert ps[3] < ev["checkout_pass"] <= ps[4]
    for k in ("capture", "release_client", "insert_contact", "latched", "release_tool"):
        assert ps[2] <= ev[k] < ps[3], k


def test_gantry_holds_still_during_checkout(log):
    i3 = log.phase == 3
    assert i3.sum() > 5
    for a in (log.bridge_x, log.carriage_y, log.fork_z):
        assert np.ptp(a[i3], axis=0).max() == 0.0


def test_sequence_clock_contiguous(log):
    """Phases 2 and 4 run contiguous sequence time; 0, 1 hold 0, 3 holds still, 5 the end."""
    s, ph = log.seq_t, log.phase
    assert np.all(s[ph <= 1] == 0.0)
    assert np.all(np.diff(s) >= 0)
    d2 = np.diff(s[ph == 2])
    assert np.allclose(d2, 1.0)
    assert np.ptp(s[ph == 3]) == 0.0
    i4 = np.flatnonzero(ph == 4)
    assert s[i4[0]] - s[i4[0] - 1] == pytest.approx(1.0)  # resumes where it paused
    assert s[ph == 5].min() == pytest.approx(log.seq_duration)


def test_red_before_green(log):
    ov = list(log.overlay)
    assert "red" in ov and "green" in ov
    i_red, i_green = ov.index("red"), ov.index("green")
    assert i_red < i_green
    assert "red" not in ov[i_green:]
    assert log.t[i_red] == pytest.approx(log.phase_start[1])


def test_power_restored(log):
    pre = log.t < cfg["power"]["strike_t_s"]
    assert log.array_power[-1] >= 0.99 * log.array_power[pre].mean()


def test_done_tail(log):
    assert log.t[-1] - log.phase_start[5] >= cfg["run"]["home_tail_s"] - 1.0


# --- inertial replay poses (Review Focus 4) -------------------------------------

def test_spare_seated_in_inertial_frame_at_end(log):
    tr = log.trace
    i = len(tr["t"]) - 1
    r_slot, dcm_NA = slot_N(tr, i)
    r, dcm = body(tr, "spare_0", i)
    assert np.linalg.norm(r - r_slot) < 1e-3
    assert np.allclose(dcm, dcm_NA, atol=1e-9)


def test_spare_seated_from_release_on(log):
    tr = log.trace
    t_rel = next(t for t, k, _ in log.events if k == "release_tool")
    rows = [i for i, t in enumerate(tr["t"]) if t >= t_rel]
    assert len(rows) > 100
    for i in rows:
        assert np.linalg.norm(body(tr, "spare_0", i)[0] - slot_N(tr, i)[0]) < 1e-3


def test_module_target_seated_at_start_and_in_bay_W_at_end(log):
    tr = log.trace
    assert tr["t"][0] == 0.0
    r_slot, dcm_NA = slot_N(tr, 0)
    r, dcm = body(tr, "module_target", 0)
    assert np.linalg.norm(r - r_slot) < 1e-3
    assert np.allclose(dcm, dcm_NA, atol=1e-9)
    i = len(tr["t"]) - 1
    assert np.linalg.norm(to_A(tr, i, body(tr, "module_target", i)[0]) - BAY_W_PARKED_A) < 1e-3


def test_gantry_bodies_ride_the_client(log):
    """Bridge B in the replay = client pose o its A-frame pose (parked at the end)."""
    tr = log.trace
    i = len(tr["t"]) - 1
    p_A = to_A(tr, i, body(tr, "bridge_B", i)[0])
    assert np.allclose(p_A, [G["park_x"]["B"], 0.0, G["bridge_z"]], atol=1e-3)
    assert np.allclose(body(tr, "bridge_B", i)[1], client_frame(tr, i)[1], atol=1e-9)


def test_overlay_on_slot_while_visible(log):
    tr = log.trace
    park = cfg["viz"]["park_m"]
    shown = {"red": 0, "green": 0}
    for i in range(len(tr["t"])):
        r_slot, dcm_NA = slot_N(tr, i)
        r_c = client_frame(tr, i)[0]
        for color in shown:
            r, dcm = body(tr, f"overlay_{color}", i)
            if np.linalg.norm(r - r_c) < 100.0:
                shown[color] += 1
                assert np.linalg.norm(r - r_slot) < 1e-3
                assert np.allclose(dcm, dcm_NA, atol=1e-9)
            else:  # parked 1 km along -z_A from the slot
                assert np.allclose(dcm_NA.T @ (r - r_slot), [0, 0, -park], atol=1e-3)
    assert shown["red"] > 60 and shown["green"] > 60


def test_overlay_visibility_matches_log(log):
    tr = log.trace
    for i, t in enumerate(tr["t"]):
        k = int(np.searchsorted(log.t, t))
        ov = log.overlay[k]
        r_c = client_frame(tr, i)[0]
        for color in ("red", "green"):
            near = np.linalg.norm(body(tr, f"overlay_{color}", i)[0] - r_c) < 100.0
            assert near == (ov == color), (t, color, ov)


def test_hud_text(log):
    hud = log.trace["hud"]
    assert hud and all("Phase" in h and "kW" in h and "%" in h for h in hud)
    assert any("Gantry: Drop in" in h for h in hud)
    assert any("Array power restored" in h for h in hud)


def test_plots_written(log, tmp_path):
    paths = make_plots(log, tmp_path)
    assert [p.name for p in paths] == ["module_ratio.png", "array_power.png",
                                       "gantry_motion.png", "timeline.png"]
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
    assert 0.5 <= cfg["viz"]["ambient"] <= 0.6


@pytest.mark.slow
def test_viz_cameras(viz_log):
    assert viz_log.viz_settings["cameras"] == ["Wide orbit view", "Gantry view", "Swap close-up"]


@pytest.mark.slow
def test_viz_run_spare_seated(viz_log):
    tr = viz_log.trace
    i = len(tr["t"]) - 1
    assert np.linalg.norm(body(tr, "spare_0", i)[0] - slot_N(tr, i)[0]) < 1e-3


# --- model mapping (no extra rotation) ---------------------------------------------

def test_every_body_has_a_model_and_A_axes():
    """Every replay body maps to a built model, and every gantry/panel body keeps
    A axes (R_A = I) through the whole sequence, so models need no rotation."""
    from models.build_models import MODEL_FILES
    from src.modules.gantry_motion import GantrySequence
    from src.viz_setup import model_key
    seq = GantrySequence(cfg)
    for t in np.linspace(0.0, seq.duration, 41):
        for name, (_, R) in seq.at(t).bodies.items():
            assert np.array_equal(R, np.eye(3)), (t, name)
    for name in [*seq.at(0.0).bodies, "overlay_red", "overlay_green"]:
        assert Path(MODEL_FILES[model_key(name)]).exists(), name
    assert model_key("bridge_A") == model_key("bridge_B") == "bridge"
    assert model_key("fork_W") == "fork_W_head" and model_key("spare_3") == "module"


# --- failed checkout ---------------------------------------------------------------

def test_failed_checkout_ends_cleanly(tmp_path, monkeypatch, capsys):
    """Force checkout_fail with an unreachable band: the run stops at the failed
    checkout (not t_max), beats()/summary()/plots and main() all cope."""
    import src.scenario_servicing as sc
    bad = load_config()
    bad["power"]["checkout_band"] = [1.5, 2.0]
    monkeypatch.setattr(sc, "load_config", lambda full_timeline=False: bad)
    sc.main(["--no-viz", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "CHECKOUT FAILED" in out and "video beats" in out
    log = run(bad, viz=False, outdir=tmp_path)
    assert log.outcome == "checkout_fail"
    assert log.phase[-1] == 3 and 4 not in log.phase_start
    assert log.t[-1] < bad["run"]["t_max_s"] - 1000
    assert [n for n, _, _ in sc.beats(log)][-1] == "Checkout hold, then green"
