"""Gantry config and motion sequence (task G1)."""
import numpy as np
import pytest

from models.build_models import module_visual_m, truss_members
from src.config import load_config
from src.geometry import module_center, module_ids
from src.modules.gantry_motion import (GantrySequence, aabb_dist, body_half,
                                       fork_geometry, gantry_layout, seg_aabb_dist)

cfg = load_config()
G = cfg["gantry"]
TARGET = tuple(cfg["client"]["target"])
SLOT = module_center(*TARGET, cfg)
LAY = gantry_layout(cfg)
GOFF = LAY["grapple_offset"]           # module center minus fork tip, along +z
PANEL_HALF = body_half(cfg)["spare_0"]
PANELS = ["module_target"] + [f"spare_{k}" for k in range(G["n_fresh"])]
BODIES = ["bridge_A", "bridge_B", "carriage", "fork_W", "fork_F"] + PANELS
EVENTS = ["capture", "release_client", "insert_contact", "latched", "release_tool"]
TOL = 1e-9


@pytest.fixture(scope="module")
def seq():
    return GantrySequence(cfg, bridge="B")


def samples(seq, dt):
    ts = np.arange(0.0, seq.duration, dt)
    return np.append(ts, seq.duration)


@pytest.fixture(scope="module")
def frames(seq):
    ts = samples(seq, 0.25)
    return ts, [seq.at(t) for t in ts]


# --- client geometry for clearance checks (from models.build_models.build_client) ---

def bus_obstacles(cfg):
    """[('box', c, h) | ('cyl', (x, y), R, z0, z1) | ('cap', p0, p1, r)] in A."""
    c = cfg["client"]
    a, zb = c["bus_size_m"], -c["bus_offset_m"]
    top, zd = zb + a / 2, -1.0
    out = [("box", np.array([0, 0, zb]), np.array([a / 2 + 0.1, a / 2 + 0.1, a / 2 + 0.05]))]
    for sg in (-1, 1):  # radiator wings along +-x and their hinge tube / housing
        x0 = sg * (a / 2 + 0.25)
        out.append(("cap", np.array([sg * a / 2, 0, zb]), np.array([x0, 0, zb]), 0.08))
        out.append(("box", np.array([x0, 0, zb]), np.array([0.15, 0.15, 0.15])))
        for k in range(2):
            xa = x0 + sg * (0.15 + k * 2.9)
            out.append(("box", np.array([xa + sg * 1.4, 0, zb]), np.array([1.42, 0.035, 1.345])))
    out.append(("cyl", (0.0, 0.0), 1.56, top, zd))  # adapter cone + flanges, bounded
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(("cap", np.array([sx * 2, sy * 2, zd]), np.array([sx * 1.7, sy * 1.7, top]), 0.04))
    return out


def obstacle_dist(ob, c, h):
    if ob[0] == "box":
        return aabb_dist(ob[1], ob[2], c, h)
    if ob[0] == "cap":
        return seg_aabb_dist(ob[1], ob[2], c, h) - ob[3]
    (x, y), R, z0, z1 = ob[1:]
    q = np.clip([x, y], c[:2] - h[:2], c[:2] + h[:2])
    dxy = max(0.0, np.hypot(x - q[0], y - q[1]) - R)
    zc, zh = (z0 + z1) / 2, abs(z1 - z0) / 2
    dz = max(0.0, abs(zc - c[2]) - zh - h[2])
    return float(np.hypot(dxy, dz))


TRUSS = truss_members(cfg)
T_P0 = np.array([m[1] for m in TRUSS])
T_P1 = np.array([m[2] for m in TRUSS])
T_R = np.array([m[3] for m in TRUSS])
T_LO = np.minimum(T_P0, T_P1) - T_R[:, None]
T_HI = np.maximum(T_P0, T_P1) + T_R[:, None]


def truss_clearance(c, h, near=0.3):
    """Min surface distance from an A-aligned box to the truss capsules."""
    gap = np.maximum(0.0, np.maximum(T_LO - (c + h), (c - h) - T_HI)).max(axis=1)
    idx = np.nonzero(gap < near)[0]
    if len(idx) == 0:
        return near
    return min(seg_aabb_dist(T_P0[i], T_P1[i], c, h) - T_R[i] for i in idx)


SEATED = [module_center(*rc, cfg) for rc in module_ids(cfg) if rc != TARGET]


# --- tests ----------------------------------------------------------------------

def test_layout_values():
    assert LAY["rail_y"] == 8.6 and LAY["rail_z"] == -2.0 and LAY["bridge_z"] == -2.0
    assert LAY["park_x"] == {"A": -8.6, "B": 8.6}
    assert LAY["bay_pitch"] == 2.1 and LAY["bay_z0"] == -1.25 and LAY["stack_dz"] == 0.17
    assert np.allclose(LAY["grapple_point"], SLOT + [0, 0, -0.075])
    assert np.isclose(LAY["carriage_z"], (G["bridge_z"] + G["magazine_top_z"]) / 2)


def test_events_in_order(seq):
    assert [e[1] for e in seq.events] == EVENTS
    ts = [e[0] for e in seq.events]
    assert all(0 < a < b for a, b in zip(ts, ts[1:])) and ts[-1] < seq.duration


def test_steps_and_duration(seq):
    assert [s[1] for s in seq.steps] == ["roll_out", "align_waste", "grab", "pull", "shift",
                                         "drop_in", "latch", "retract", "roll_home"]
    assert seq.steps[0][0] == 0.0
    assert all(a[0] < b[0] for a, b in zip(seq.steps, seq.steps[1:]))
    assert 600 <= seq.duration <= 1200  # spec: on the order of 10-20 min
    assert seq.at(seq.duration).step == "done"


def test_event_semantics(seq):
    ev = dict((n, t) for t, n in seq.events)
    g = LAY["grapple_point"][2]
    f = seq.at(ev["capture"])
    assert np.isclose(f.fork_z["W"], g, atol=1e-9)
    assert seq.at(ev["capture"] - 1.0).fork_z["W"] < g - 1e-4
    p = seq.at(ev["release_client"]).bodies["module_target"][0]
    assert np.isclose(SLOT[2] - p[2], cfg["client"]["module_thick_m"] + G["release_clear_m"], atol=1e-6)
    p = seq.at(ev["insert_contact"]).bodies["spare_0"][0]
    assert np.isclose(SLOT[2] - p[2], G["contact_zone_m"], atol=1e-6)
    assert seq.at(ev["latched"]).step in ("latch", "retract")
    st = dict((n, t) for t, n in seq.steps)
    assert np.isclose(ev["latched"], st["latch"] + G["latch_s"])
    assert np.isclose(ev["release_tool"], st["retract"])
    assert seq.at(ev["release_tool"] + 1.0).fork_z["F"] < g - 1e-6


def test_speed_limits(frames):
    ts, fr = frames
    g, bay = LAY["grapple_point"][2], LAY["bay_z0"] - GOFF
    zone = G["contact_zone_m"]
    vmax = {"travel": 0.0, "fork": 0.0, "contact": 0.0}
    for (t0, a), (t1, b) in zip(zip(ts, fr), zip(ts[1:], fr[1:])):
        dt = t1 - t0
        for k in ("A", "B"):
            vmax["travel"] = max(vmax["travel"], abs(b.bridge_x[k] - a.bridge_x[k]) / dt)
        vmax["travel"] = max(vmax["travel"], abs(b.carriage_y - a.carriage_y) / dt)
        for k in ("W", "F"):
            v = abs(b.fork_z[k] - a.fork_z[k]) / dt
            vmax["fork"] = max(vmax["fork"], v)
            near = lambda z: min(abs(z - g), abs(z - bay)) <= zone + 1e-12
            if near(a.fork_z[k]) and near(b.fork_z[k]):
                vmax["contact"] = max(vmax["contact"], v)
    assert vmax["travel"] <= G["travel_mps"] + TOL
    assert vmax["fork"] <= G["fork_mps"] + TOL
    assert vmax["contact"] <= G["contact_mps"] + TOL


def test_spare_seated_at_end(seq):
    p, R = seq.at(seq.duration).bodies["spare_0"]
    assert np.allclose(p, SLOT, atol=1e-6) and np.allclose(R, np.eye(3))


def test_dead_panel_in_waste_bay(seq):
    f = seq.at(seq.duration)
    p, R = f.bodies["module_target"]
    wy = f.carriage_y + G["waste_side"] * G["bay_pitch"] / 2
    assert np.allclose(p, [f.bridge_x["B"], wy, G["bay_z0"]], atol=1e-6)
    assert np.allclose(R, np.eye(3))


def test_no_panel_collisions(seq):
    """Review Focus 1: panels vs truss, other panels, bridge and carriage boxes."""
    half = body_half(cfg)
    worst = {"truss": np.inf, "panel": np.inf, "frame": np.inf}
    for t in samples(seq, 0.5):
        f = seq.at(t)
        pos = {n: f.bodies[n][0] for n in BODIES}
        for i, n in enumerate(PANELS):
            c = pos[n]
            worst["truss"] = min(worst["truss"], truss_clearance(c, PANEL_HALF))
            others = [pos[m] for m in PANELS[i + 1:]] + [s for s in SEATED if np.linalg.norm(s - c) < 3]
            for o in others:
                worst["panel"] = min(worst["panel"], aabb_dist(c, PANEL_HALF, o, PANEL_HALF))
            for b in ("bridge_A", "bridge_B", "carriage"):
                worst["frame"] = min(worst["frame"], aabb_dist(c, PANEL_HALF, pos[b], half[b]))
    print("min clearance (m):", worst)
    assert worst["truss"] > 0.002
    assert worst["panel"] > 0.01
    assert worst["frame"] > 0.01


def held(seq, f, t):
    """{fork: panel it holds at t}."""
    ev = dict((n, tt) for tt, n in seq.events)
    out = {}
    if t >= ev["capture"]:
        out["W"] = "module_target"
    if t <= ev["release_tool"]:
        out["F"] = "spare_0"
    return out


@pytest.fixture(scope="module")
def fork_clearance(seq):
    """Min clearances of each fork's boxes (head, crossbar, masts/arms) over 0.25 s samples."""
    obs = bus_obstacles(cfg)
    w = {k: {"panel": np.inf, "spare_1_3": np.inf, "module_target": np.inf, "seated": np.inf,
             "truss": np.inf, "other_fork": np.inf, "bus": np.inf} for k in ("W", "F")}
    for t in samples(seq, 0.25):
        f = seq.at(t)
        h = held(seq, f, t)
        boxes = {k: [(f.bodies[f"fork_{k}"][0] + o, hh)
                     for o, hh in fork_geometry(cfg, k, f.fork_z[k])] for k in ("W", "F")}
        for k, other in (("W", "F"), ("F", "W")):
            for c, hh in boxes[k]:
                for n in PANELS:
                    if h.get(k) == n:
                        continue
                    d = aabb_dist(c, hh, f.bodies[n][0], PANEL_HALF)
                    w[k]["panel"] = min(w[k]["panel"], d)
                    key = "module_target" if n == "module_target" else ("spare_1_3" if n != "spare_0" else None)
                    if key:
                        w[k][key] = min(w[k][key], d)
                for sc in SEATED:
                    if abs(sc[0] - c[0]) < 3 + hh[0] and abs(sc[1] - c[1]) < 3 + hh[1]:
                        w[k]["seated"] = min(w[k]["seated"], aabb_dist(c, hh, sc, PANEL_HALF))
                w[k]["truss"] = min(w[k]["truss"], truss_clearance(c, hh))
                w[k]["bus"] = min(w[k]["bus"], min(obstacle_dist(o, c, hh) for o in obs))
                for c2, h2 in boxes[other]:
                    w[k]["other_fork"] = min(w[k]["other_fork"], aabb_dist(c, hh, c2, h2))
    print("fork min clearance (m):", w)
    return w


def test_fork_W_clear(fork_clearance):
    w = fork_clearance["W"]
    assert w["panel"] > 0.01 and w["seated"] > 0.01 and w["truss"] > 0.01
    assert w["other_fork"] > 0.1 and w["bus"] >= 0.1


def test_fork_F_clear(fork_clearance):
    """Fork F vs spares 1-3 (the stack does not index: it would drop spare_1 onto the returning crossbar), the truss at the slot, bay W / module_target,
    fork W and the bus."""
    w = fork_clearance["F"]
    assert w["spare_1_3"] >= 0.01 and w["module_target"] > 0.1 and w["panel"] >= 0.01
    assert w["seated"] > 0.01 and w["truss"] >= 0.01
    assert w["other_fork"] > 0.1 and w["bus"] >= 0.1


def test_fork_F_outside_footprint_beside_stack():
    """Fork F arms sit outside the panel footprint in x and inside the carriage."""
    arms = fork_geometry(cfg, "F", LAY["bay_grip_z"])[2:]
    assert len(arms) == 2
    for c, h in arms:
        assert abs(c[0]) - h[0] > PANEL_HALF[0] + 0.01
        assert abs(c[0]) + h[0] <= G["carriage_size"][0] / 2


def test_crossbar_fits_between_spares():
    """At the bay grip, fork F's head + crossbar sit between spare_0's knob and spare_1."""
    tip = LAY["bay_grip_z"]
    zmin = min(tip + o[2] - h[2] for o, h in fork_geometry(cfg, "F"))
    spare1_front = LAY["bay_z0"] - G["stack_dz"] + PANEL_HALF[2]
    assert zmin - spare1_front >= 0.01


def test_stack_fixtures_clear_carriage():
    """Top spare's grapple knob stays clear of the carriage plate."""
    n = G["n_fresh"] - 1
    knob = LAY["bay_z0"] - n * G["stack_dz"] - GOFF
    assert knob - G["magazine_top_z"] >= 0.01


def test_bridge_clears_bus(seq):
    """Review Focus 3: gantry bodies stay >= 0.1 m from the bus, wings and standoff."""
    half = body_half(cfg)
    obs = bus_obstacles(cfg)
    worst = np.inf
    for t in samples(seq, 1.0):
        f = seq.at(t)
        for n in ["bridge_A", "bridge_B", "carriage"] + PANELS:
            c = f.bodies[n][0]
            worst = min(worst, min(obstacle_dist(o, c, half[n]) for o in obs))
    print("min bus clearance (m):", worst)
    assert worst >= 0.1


def test_mechanism_connected(seq):
    """Review Focus 2: carriage on its bridge, forks on the carriage, held panel on its fork."""
    ev = dict((n, t) for t, n in seq.events)
    lim = G["rail_y"] - G["carriage_size"][1] / 2
    side = {"W": G["waste_side"], "F": -G["waste_side"]}
    for t in samples(seq, 1.0):
        f = seq.at(t)
        b = f.bodies
        assert f.bridge_x["A"] == G["park_x"]["A"]
        assert G["x_limits"]["B"][0] - TOL <= f.bridge_x["B"] <= G["x_limits"]["B"][1] + TOL
        assert np.allclose(b["bridge_B"][0], [f.bridge_x["B"], 0, G["bridge_z"]])
        assert np.allclose(b["bridge_A"][0], [G["park_x"]["A"], 0, G["bridge_z"]])
        assert np.allclose(b["carriage"][0], [f.bridge_x["B"], f.carriage_y, LAY["carriage_z"]])
        assert abs(f.carriage_y) <= lim + TOL
        for k in ("W", "F"):
            tip = b[f"fork_{k}"][0]
            y = f.carriage_y + side[k] * G["bay_pitch"] / 2
            assert np.allclose(tip, [f.bridge_x["B"], y, f.fork_z[k]])
            assert LAY["bay_z0"] - GOFF - TOL <= tip[2] <= LAY["grapple_point"][2] + TOL
        for n in BODIES:
            assert np.allclose(b[n][1], np.eye(3))
        if t >= ev["capture"]:
            assert np.allclose(b["module_target"][0], b["fork_W"][0] + [0, 0, GOFF], atol=1e-9)
        else:
            assert np.allclose(b["module_target"][0], SLOT)
        if t <= ev["release_tool"]:
            assert np.allclose(b["spare_0"][0], b["fork_F"][0] + [0, 0, GOFF], atol=1e-9)
        else:
            assert np.allclose(b["spare_0"][0], SLOT, atol=1e-9)
        for k in range(1, G["n_fresh"]):  # the rest of the stack rides in bay F
            p = b[f"spare_{k}"][0]
            assert np.isclose(p[0], f.bridge_x["B"]) and np.isclose(p[1], b["fork_F"][0][1])
            assert np.isclose(p[2], LAY["bay_z0"] - k * G["stack_dz"])


def test_no_teleports(seq):
    """Review Focus 5: no body jumps more than travel_mps * 1 s between 1 s samples."""
    ts = samples(seq, 0.1)
    prev = seq.at(ts[0]).bodies
    worst = 0.0
    for t0, t in zip(ts, ts[1:]):
        cur = seq.at(t).bodies
        for n in BODIES:
            worst = max(worst, np.linalg.norm(cur[n][0] - prev[n][0]) / (t - t0))
        prev = cur
    assert worst <= G["travel_mps"] + 1e-6


def test_ends_parked(seq):
    f0, f1 = seq.at(0.0), seq.at(seq.duration)
    for f in (f0, f1):
        assert f.bridge_x["B"] == G["park_x"]["B"] and f.bridge_x["A"] == G["park_x"]["A"]
        assert f.carriage_y == G["carriage_park_y"]
    assert seq.at(-5.0).bridge_x["B"] == G["park_x"]["B"]
    assert seq.at(seq.duration + 100).bridge_x["B"] == G["park_x"]["B"]


def test_contact_flag(seq):
    ev = dict((n, t) for t, n in seq.events)
    assert seq.at(ev["capture"]).contact
    assert not seq.at(1.0).contact
    assert not seq.at(seq.duration).contact


def test_panel_visual_size_matches_model():
    assert np.allclose(PANEL_HALF[:2], module_visual_m(cfg) / 2)
