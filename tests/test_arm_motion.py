import numpy as np
import pytest

from src.config import load_config
from src.geometry import module_center
from src.modules.arm_motion import (IKError, SwapSequence, a_to_s, berth_pose_A,
                                    bus_half, fk, grapple_d, ik, module_corners_S,
                                    module_half, shoulder_S,
                                    obb_overlap, rack_module_pose,
                                    rack_slot_pose, s_to_a, target_fixture_S)

cfg = load_config()
norm = np.linalg.norm
deg = np.radians
# Mission numbers come from config; literals below are requirement thresholds.
SHOULDER = shoulder_S(cfg)
BUS_HALF = bus_half(cfg)
MODULE_HALF = module_half(cfg)
GD = grapple_d(cfg)  # tool point to module center along the tool axis
RACK = cfg["arm"]["rack"]
APPROACH = cfg["arm"]["approach_m"]
TIP_CONTACT = cfg["arm"]["tip_contact_mps"]
REACH = sum(cfg["arm"]["links_m"])
ARRAY_CLEAR = -0.10  # Review Focus 3: carried corners stay at z_A <= -0.10
Q0 = deg([20, -30, 60, -20, 15, 10])
Q_SEED = deg([0, -60, 120, -60, 0, 0])
MODULES = ["module_target", "spare_0", "spare_1", "spare_2", "spare_3"]


@pytest.fixture(scope="module")
def seq():
    return SwapSequence(cfg, target_fixture_S(cfg))


@pytest.fixture(scope="module")
def samples(seq):
    ts = np.arange(0.0, seq.duration + 1e-9, 1.0)
    return ts, [seq.at(t) for t in ts]


def z_A(p_S):
    return s_to_a(p_S, cfg)[..., 2]


def contact_points(seq):
    return [target_fixture_S(cfg)[0], rack_slot_pose(3, cfg)[0],
            rack_slot_pose(4, cfg)[0]]


# --- kinematics -------------------------------------------------------------

def test_shoulder_on_pedestal_ahead_of_front_face():
    assert SHOULDER[0] > BUS_HALF[0] + 0.06 and SHOULDER[1] == SHOULDER[2] == 0


def test_zero_pose_reach():
    frames = fk(np.zeros(6), cfg)
    assert len(frames) == 8  # base, 6 links, tool
    assert np.allclose(frames[0][0], SHOULDER)
    assert abs(norm(frames[-1][0] - SHOULDER) - REACH) < 1e-6
    assert np.allclose(frames[-1][1], np.eye(3))


def test_fk_dcms_orthonormal():
    for _, R in fk(Q0, cfg):
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)


def test_ik_roundtrip():
    p, R = fk(Q0, cfg)[-1]
    q = ik(p, R, Q_SEED, cfg)
    p2, R2 = fk(q, cfg)[-1]
    assert np.allclose(p2, p, atol=1e-4) and np.allclose(R2, R, atol=1e-4)


def test_obb_overlap_basic():
    h = np.array([0.5, 0.5, 0.5])
    assert obb_overlap(np.zeros(3), np.eye(3), h, [0.9, 0, 0], np.eye(3), h)
    assert not obb_overlap(np.zeros(3), np.eye(3), h, [1.1, 0, 0], np.eye(3), h)


def test_obb_overlap_rotated_near_touch():
    # unit cube vs a cube turned 45 deg about z: its edge reaches 0.5*sqrt(2)
    h = np.array([0.5, 0.5, 0.5])
    R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                  [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]])
    gap = 0.5 + 0.5 * np.sqrt(2)
    assert obb_overlap(np.zeros(3), np.eye(3), h, [gap - 1e-3, 0, 0], R, h)
    assert not obb_overlap(np.zeros(3), np.eye(3), h, [gap + 1e-3, 0, 0], R, h)


def test_ik_unreachable_raises():
    with pytest.raises(IKError):
        ik(SHOULDER + [REACH + 0.5, 0, 0], np.eye(3), Q_SEED, cfg)


# --- frames and fixtures ----------------------------------------------------

def test_berth_pose_and_frame_helpers():
    pos, dcm_AS = berth_pose_A(cfg)
    assert np.allclose(pos, [5, -5, -2.6])
    assert np.allclose(dcm_AS @ [1, 0, 0], [0, 0, 1])   # S x = +z_A
    assert np.allclose(dcm_AS @ [0, 1, 0], [0, 1, 0])   # S y = +y_A
    assert np.allclose(dcm_AS @ [0, 0, 1], [-1, 0, 0])  # S z = -x_A
    p = np.array([0.3, -0.2, 1.1])
    assert np.allclose(a_to_s(s_to_a(p, cfg), cfg), p)


def test_target_fixture():
    p, R = target_fixture_S(cfg)
    tgt = module_center(*cfg["client"]["target"], cfg)
    assert np.allclose(s_to_a(p, cfg), tgt + [0, 0, -GD])
    assert np.allclose(R, np.eye(3))  # tool axis +x_S = +z_A


def test_rack_slot_geometry():
    for k in range(6):
        c, Rm = rack_module_pose(k, cfg)
        side = 1 if k < 4 else -1  # spares on top (+z), stow on the bottom
        n = k if k < 4 else k - 4
        assert np.allclose(c, [RACK["x_m"], 0, side * (RACK["z0_m"] + RACK["dz_m"] * n)])
        corners = module_corners_S(c, Rm, cfg)
        assert corners[:, 0].max() < SHOULDER[0]  # front edge behind the shoulder
        assert np.allclose(np.abs(corners[:, 1]).max(), 1.0)  # parallel to x-y
        assert not obb_overlap(c, Rm, MODULE_HALF, np.zeros(3), np.eye(3), BUS_HALF)
        p, R = rack_slot_pose(k, cfg)
        assert np.isclose(p[2], c[2] + side * GD)  # fixture faces outward
        assert np.allclose(R[:, 0], [0, 0, -side])  # tool points inward


def test_rack_poses_follow_config():
    import copy
    c2 = copy.deepcopy(cfg)
    c2["arm"]["rack"].update(x_m=-0.5, z0_m=0.6, dz_m=0.1)
    c2["client"]["grapple_gap_m"] = 0.07
    c, _ = rack_module_pose(2, c2)
    assert np.allclose(c, [-0.5, 0, 0.8])
    assert np.isclose(rack_slot_pose(2, c2)[0][2], 0.8 + 0.025 + 0.07)


def test_all_targets_reachable():
    for pose in [target_fixture_S(cfg)] + [rack_slot_pose(k, cfg) for k in range(6)]:
        q = ik(*pose, Q_SEED, cfg)
        assert np.all(np.abs(q) <= deg(170) + 1e-12)
        assert np.allclose(fk(q, cfg)[-1][0], pose[0], atol=1e-4)


# --- swap sequence ----------------------------------------------------------

def test_duration_is_conops(seq):
    real = sum(cfg["arm"]["step_min"]) * 60.0
    assert seq.duration == pytest.approx(real + sum(v[1] - v[0] for v in seq.stretched.values()))
    assert seq.duration >= real
    assert set(seq.stretched) <= {"6.7", "6.12"}


def test_joint_limits_and_rate(samples):
    _, frames = samples
    q = np.array([f.q for f in frames])
    assert np.all(np.abs(q) <= deg(170) + 1e-12)
    assert np.abs(np.diff(q, axis=0)).max() <= deg(cfg["arm"]["rate_dps"]) + 1e-9


def test_tip_speed_near_contact(seq, samples):
    _, frames = samples
    tips = np.array([fk(f.q, cfg)[-1][0] for f in frames])
    v = norm(np.diff(tips, axis=0), axis=1)
    pts = contact_points(seq)
    near = np.array([min(norm(p - c) for c in pts) <= APPROACH for p in tips])
    near = near[:-1] | near[1:]
    assert near.sum() > 50
    assert v[near].max() <= TIP_CONTACT + 1e-9
    flagged = np.array([f.contact for f in frames])
    assert flagged.any() and v[flagged[:-1] | flagged[1:]].max() <= TIP_CONTACT + 1e-9


def test_tip_speed_free(samples):
    _, frames = samples
    tips = np.array([fk(f.q, cfg)[-1][0] for f in frames])
    assert norm(np.diff(tips, axis=0), axis=1).max() <= cfg["arm"]["tip_free_mps"] + 1e-9


def test_module_clears_array(samples):
    # Review Focus 3: every carried module corner stays 10 cm behind the array
    _, frames = samples
    seat_c = target_fixture_S(cfg)[0] + [GD, 0, 0]
    n = 0
    for f in frames:
        if f.held is None or f.step in ("6.5", "6.9"):
            continue
        if norm(f.bodies[f.held][0] - seat_c) < 1e-6:
            # grappled but still seated in its slot: not being carried
            assert f.step in ("6.3", "6.4", "6.10", "6.11")
            continue
        n += 1
        c, R = f.bodies[f.held]
        assert z_A(module_corners_S(c, R, cfg)).max() <= ARRAY_CLEAR
    assert n > 1000


def test_carried_module_clear_of_bus_and_rack(samples):
    _, frames = samples
    for f in frames:
        if f.held is None:
            continue
        c, R = f.bodies[f.held]
        assert not obb_overlap(c, R, MODULE_HALF, np.zeros(3), np.eye(3), BUS_HALF), f.step
        for name in MODULES:
            if name != f.held:
                c2, R2 = f.bodies[name]
                assert not obb_overlap(c, R, MODULE_HALF, c2, R2, MODULE_HALF), (f.step, name)


def test_arm_never_passes_through_array(samples):
    _, frames = samples
    zmax = max(z_A(np.array([p for p, _ in fk(f.q, cfg)])).max() for f in frames)
    assert zmax <= -GD + 1e-6  # tool tip at most reaches the grapple point


def test_step_order(seq):
    assert [e[1] for e in seq.events] == ["capture", "release_client",
                                          "insert_contact", "latched", "release_tool"]
    ts = [e[0] for e in seq.events]
    assert ts == sorted(ts)


def test_event_steps(seq):
    steps = {name: seq.at(t).step for t, name in seq.events}
    assert steps["capture"] == "6.3"
    assert steps["insert_contact"] == "6.9"
    t_ins = dict((n, t) for t, n in seq.events)["insert_contact"]
    tip = fk(seq.at(t_ins).q, cfg)[-1][0]
    assert abs(norm(tip - target_fixture_S(cfg)[0]) - APPROACH) < 5e-3


def test_bodies_start_and_end(seq):
    seat_c = target_fixture_S(cfg)[0] + [GD, 0, 0]
    b0, b1 = seq.at(0.0).bodies, seq.at(seq.duration).bodies
    assert set(b0) == set(MODULES)
    assert np.allclose(b0["module_target"][0], seat_c)
    assert np.allclose(b1["spare_3"][0], seat_c, atol=1e-6)
    assert np.allclose(b1["spare_3"][1], np.eye(3), atol=1e-6)
    assert np.allclose(b1["module_target"][0], rack_module_pose(4, cfg)[0], atol=1e-6)
    assert np.allclose(b1["module_target"][1], rack_module_pose(4, cfg)[1], atol=1e-6)
    for k in range(3):
        assert np.allclose(b1[f"spare_{k}"][0], rack_module_pose(k, cfg)[0])
    assert seq.at(0.0).held is None and seq.at(seq.duration).held is None
    assert np.allclose(seq.at(seq.duration).q, seq.at(0.0).q)  # stowed


def test_held_body_follows_tool(samples):
    _, frames = samples
    held = [f for f in frames if f.held is not None]
    assert {f.step for f in held} >= {"6.5", "6.6", "6.8", "6.9"}
    for f in held:
        p, R = fk(f.q, cfg)[-1]
        c, Rm = f.bodies[f.held]
        assert np.allclose(c, p + GD * R[:, 0]) and np.allclose(Rm, R)

def test_bodies_never_snap(samples):
    # grab and release hand over without a jump; carried motion stays smooth
    _, frames = samples
    for a, b in zip(frames[:-1], frames[1:]):
        for name in MODULES:
            (c0, R0), (c1, R1) = a.bodies[name], b.bodies[name]
            assert norm(c1 - c0) <= 0.03, (a.step, name)
            assert np.degrees(np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1))) <= 2.0


LINK_R = 0.06  # arm link radius for clearance checks


def _link_segments(q):
    f = fk(q, cfg)
    ends = [f[0][0], f[3][0], f[4][0], f[7][0]]
    return list(zip(ends[:-1], ends[1:]))


def _seg_box_dist(a, b, c, R, h, n=60):
    """Min distance from segment a-b to the box (center c, axes R, half h)."""
    pts = np.linspace(a, b, n)
    loc = (pts - c) @ R
    return norm(np.maximum(np.abs(loc) - h, 0.0), axis=1).min()


def _grapple(c, R):
    return c - GD * R[:, 0]


def test_arm_links_clear_of_bus_and_panels(samples):
    # Every link capsule clears the bus and every panel (racked, seated or
    # held). Only the wrist link's tool end may touch the panel whose
    # fixture it is at (within 0.12 m of that grapple point).
    _, frames = samples
    worst = np.inf
    for f in frames:
        segs = _link_segments(f.q)
        tool = segs[-1][1]
        for a, b in segs:
            d = _seg_box_dist(a, b, np.zeros(3), np.eye(3), BUS_HALF)
            assert d > LINK_R, (f.step, "bus")
            worst = min(worst, d)
        for name in MODULES:
            c, R = f.bodies[name]
            for i, (a, b) in enumerate(segs):
                if i == 2 and norm(tool - _grapple(c, R)) < 0.12:
                    b = b - 0.15 * (b - a) / norm(b - a)  # tool end is in contact
                assert _seg_box_dist(a, b, c, R, MODULE_HALF) > LINK_R, (f.step, name, i)
    assert worst > LINK_R
