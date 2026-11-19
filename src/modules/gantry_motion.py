"""On-board rail gantry: layout and scripted panel-swap motion (Fidelity Level 1).

Spec: docs/superpowers/specs/2026-11-19-rail-gantry-design.md. Everything is in
the array frame A (origin at the array center, +z toward the Sun, back is -z).

Body frames in GantryFrame.bodies, all as (pos_A, dcm) with dcm = I (A axes):
  bridge_A, bridge_B   bridge beam center (x_bridge, 0, bridge_z)
  carriage             carriage center (x_bridge, carriage_y, carriage_z);
                       carriage_z is the midpoint of bridge_z and magazine_top_z
  fork_W, fork_F       fork TIP (the grapple point it engages); the shaft runs
                       along -z from the tip up to the carriage. Fork k sits at
                       y = carriage_y + side_k * bay_pitch / 2 with
                       side_W = waste_side and side_F = -waste_side
  module_target,       module center; panels translate only
  spare_0..spare_3

A held panel's center is its fork tip + grapple_offset along +z
(grapple_offset = module_thick_m / 2 + client.grapple_gap_m).

Every axis moves on a trapezoid-like 1-D profile: accel_mps2 ramps, a cruise
limit, and contact_mps within contact_zone_m of either end of a fork stroke
(the array grapple / seat at one end, the magazine grip / seat at the other).
Only one axis among bridge and carriage moves at a time.
"""
from dataclasses import dataclass, field

import numpy as np

from src.geometry import module_center

MODULE_GAP_M = 0.015   # == models.build_models.MODULE_GAP_M (a test checks the panel size)
STEPS = ["roll_out", "align_waste", "grab", "pull", "shift", "drop_in", "latch", "retract", "roll_home"]
EVENTS = ["capture", "release_client", "insert_contact", "latched", "release_tool"]
_I3 = np.eye(3)


# --- geometry helpers -------------------------------------------------------------

def aabb_dist(c1, h1, c2, h2):
    """Surface distance between two A-aligned boxes (center, half extents); 0 if they touch."""
    gap = np.maximum(0.0, np.abs(np.asarray(c1, float) - c2) - np.asarray(h1) - h2)
    return float(np.linalg.norm(gap))


def _point_aabb(p, c, h):
    return float(np.linalg.norm(np.maximum(0.0, np.abs(p - c) - h)))


def seg_aabb_dist(p0, p1, c, h, iters=80):
    """Distance from segment p0-p1 to an A-aligned box (convex, golden-section on t)."""
    p0, p1, c, h = (np.asarray(v, float) for v in (p0, p1, c, h))
    d = p1 - p0
    f = lambda t: _point_aabb(p0 + t * d, c, h)
    if not d.any():
        return f(0.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = 0.0, 1.0
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
    return min(f(0.0), f(1.0), f1, f2)


# --- layout -----------------------------------------------------------------------

def gantry_layout(cfg):
    """Static gantry geometry in A from cfg['gantry'] and the client target."""
    G, c = cfg["gantry"], cfg["client"]
    goff = c["module_thick_m"] / 2 + c["grapple_gap_m"]
    slot = module_center(*c["target"], cfg)
    return {
        "rail_y": G["rail_y"], "rail_z": G["rail_z"], "bridge_z": G["bridge_z"],
        "park_x": dict(G["park_x"]), "x_limits": {k: list(v) for k, v in G["x_limits"].items()},
        "bay_pitch": G["bay_pitch"], "bay_z0": G["bay_z0"], "stack_dz": G["stack_dz"],
        "n_fresh": G["n_fresh"], "carriage_z": G["carriage_z"],
        "magazine_top_z": G["magazine_top_z"], "waste_side": G["waste_side"],
        "carriage_y_limit": G["rail_y"] - G["carriage_size"][1] / 2,
        "grapple_offset": goff,                       # module center - fork tip, along +z
        "slot_center": slot,
        "grapple_point": slot + np.array([0.0, 0.0, -goff]),
        "bay_grip_z": G["bay_z0"] - goff,             # fork tip z on a bottom-bay panel
    }


def body_half(cfg):
    """Half extents (A axes) of each body's collision box, centered on its frame origin.

    Fork entries cover only the tip head; the shaft is a G2 visual detail."""
    G, c = cfg["gantry"], cfg["client"]
    bx, bz = G["bridge_size"]
    p = (c["slot_m"] - 2 * MODULE_GAP_M) / 2
    panel = np.array([p, p, c["module_thick_m"] / 2])
    out = {f"bridge_{k}": np.array([bx / 2, G["rail_y"], bz / 2]) for k in ("A", "B")}
    out["carriage"] = np.array(G["carriage_size"], float) / 2
    out["fork_W"] = out["fork_F"] = np.array([0.1, 0.1, 0.03])
    out["module_target"] = panel
    for k in range(G["n_fresh"]):
        out[f"spare_{k}"] = panel
    return out


# --- 1-D motion profile --------------------------------------------------------------

class Move1D:
    """Rest-to-rest move s0 -> s1: accel limit a, cruise vmax, and vslow within `zone`
    of the start and/or end. Time-optimal under those limits (forward/backward pass on
    a fine grid, exact constant-accel timing per cell); continuous velocity."""

    def __init__(self, s0, s1, vmax, a, vslow=None, zone=0.0,
                 slow_start=False, slow_end=False, ds=2e-4):
        self.s0, self.s1 = float(s0), float(s1)
        L = abs(self.s1 - self.s0)
        self.sign = 1.0 if self.s1 >= self.s0 else -1.0
        self.L = L
        if L == 0.0:
            self.u, self.t, self.duration = np.array([0.0]), np.array([0.0]), 0.0
            return
        n = max(int(np.ceil(L / ds)), 4)
        u = np.linspace(0.0, L, n + 1)
        du = L / n
        lim = np.full(n + 1, float(vmax))
        z = zone + 2 * du  # pad so a whole grid cell touching the zone is slow
        if vslow is not None:
            if slow_start:
                lim[u <= z] = min(vslow, vmax)
            if slow_end:
                lim[u >= L - z] = min(vslow, vmax)
        v = np.empty(n + 1)
        v[0] = 0.0
        k = 2 * a * du
        for i in range(1, n + 1):
            v[i] = min(lim[i], np.sqrt(v[i - 1] ** 2 + k))
        v[n] = 0.0
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + k))
        dt = 2 * du / (v[:-1] + v[1:])
        self.u, self.t = u, np.concatenate([[0.0], np.cumsum(dt)])
        self.duration = float(self.t[-1])

    def pos(self, tau):
        if tau <= 0.0 or self.L == 0.0:
            return self.s0
        if tau >= self.duration:
            return self.s1
        return self.s0 + self.sign * float(np.interp(tau, self.t, self.u))

    def time_at(self, d):
        """Time when the distance travelled reaches d."""
        return float(np.interp(d, self.u, self.t))


class _Track:
    """A scalar driven by moves placed at absolute start times; holds between them."""

    def __init__(self, s0):
        self.s0, self.moves = float(s0), []

    def add(self, t0, move):
        self.moves.append((float(t0), move))

    @property
    def end(self):
        return self.moves[-1][1].s1 if self.moves else self.s0

    def __call__(self, t):
        val = self.s0
        for t0, m in self.moves:
            if t < t0:
                break
            val = m.pos(t - t0)
        return val


# --- sequence ---------------------------------------------------------------------

@dataclass
class GantryFrame:
    bridge_x: dict
    carriage_y: float
    fork_z: dict
    step: str
    bodies: dict = field(default_factory=dict)
    contact: bool = False


class GantrySequence:
    """The 9-step swap (spec 'Swap sequence') for the client target with one bridge.

    .duration, .events [(t, name)], .steps [(t_start, name)], .at(t) -> GantryFrame.
    Before t = 0 the gantry is parked (step 'idle'); from .duration on it is
    parked again with the swap done (step 'done')."""

    def __init__(self, cfg, bridge="B"):
        G = cfg["gantry"]
        self.cfg, self.bridge = cfg, bridge
        self.other = "A" if bridge == "B" else "B"
        lay = self.lay = gantry_layout(cfg)
        self.ws = G["waste_side"]
        self.half_pitch = G["bay_pitch"] / 2
        slot = lay["slot_center"]
        g, grip = lay["grapple_point"][2], lay["bay_grip_z"]
        a, settle, latch = G["accel_mps2"], G["settle_s"], G["latch_s"]

        x0, xs = G["park_x"][bridge], slot[0]
        lo, hi = G["x_limits"][bridge]
        if not lo - 1e-9 <= xs <= hi + 1e-9:
            raise ValueError(f"target x {xs} outside bridge {bridge} limits {G['x_limits'][bridge]}")
        y0 = G["carriage_park_y"]
        y_w = slot[1] - self.ws * self.half_pitch   # bay W over the slot
        y_f = slot[1] + self.ws * self.half_pitch   # bay F over the slot
        for y in (y0, y_w, y_f):
            if abs(y) > lay["carriage_y_limit"] + 1e-9:
                raise ValueError(f"carriage y {y} outside rails (limit {lay['carriage_y_limit']}); "
                                 "flip gantry.waste_side")

        def travel(s0, s1):
            return Move1D(s0, s1, G["travel_mps"], a)

        def stroke(s0, s1):
            return Move1D(s0, s1, G["fork_mps"], a, G["contact_mps"], G["contact_zone_m"], True, True)

        self.bx, self.cy = _Track(x0), _Track(y0)
        self.fz = {"W": _Track(grip), "F": _Track(grip)}
        self.stack = _Track(0.0)                    # index offset of spares 1..n-1 (+z)
        steps, events = [], []
        t = 0.0

        steps.append((t, "roll_out"))
        m = travel(x0, xs); self.bx.add(t, m); t += m.duration + settle

        steps.append((t, "align_waste"))
        m = travel(y0, y_w); self.cy.add(t, m); t += m.duration + settle

        steps.append((t, "grab"))
        m = stroke(grip, g); self.fz["W"].add(t, m)
        events.append((t + m.duration, "capture"))
        t += m.duration + latch                     # grapple close + slot latches open

        steps.append((t, "pull"))
        m = stroke(g, grip); self.fz["W"].add(t, m)
        events.append((t + m.time_at(cfg["client"]["module_thick_m"] + 0.02), "release_client"))
        t += m.duration + settle

        steps.append((t, "shift"))
        m = travel(y_w, y_f); self.cy.add(t, m); t += m.duration + settle

        steps.append((t, "drop_in"))
        m = stroke(grip, g); self.fz["F"].add(t, m)
        events.append((t + m.time_at(m.L - G["contact_zone_m"]), "insert_contact"))
        t += m.duration

        steps.append((t, "latch"))
        t += latch
        events.append((t, "latched"))
        t += settle                                 # confirm latch before letting go

        steps.append((t, "retract"))
        events.append((t, "release_tool"))
        self.t_release_tool = t
        m = stroke(g, grip); self.fz["F"].add(t, m)
        dur = m.duration
        if G.get("index_stack", True) and G["n_fresh"] > 1:
            mi = Move1D(0.0, G["stack_dz"], G["contact_mps"], a)
            self.stack.add(t, mi)
            dur = max(dur, mi.duration)
        t += dur + settle

        steps.append((t, "roll_home"))
        m = travel(y_f, y0); self.cy.add(t, m); t += m.duration + settle
        m = travel(xs, x0); self.bx.add(t, m); t += m.duration

        self.duration = t
        self.steps, self.events = steps, events
        self.t_capture = events[0][0]

    def _step(self, t):
        if t < 0.0:
            return "idle"
        if t >= self.duration:
            return "done"
        name = self.steps[0][1]
        for t0, n in self.steps:
            if t >= t0:
                name = n
        return name

    def at(self, t):
        G, lay = self.cfg["gantry"], self.lay
        x, cy = self.bx(t), self.cy(t)
        fz = {k: tr(t) for k, tr in self.fz.items()}
        bz, goff = lay["bridge_z"], lay["grapple_offset"]
        side = {"W": self.ws, "F": -self.ws}
        tip = {k: np.array([x, cy + side[k] * self.half_pitch, fz[k]]) for k in ("W", "F")}
        up = np.array([0.0, 0.0, goff])
        slot = lay["slot_center"]

        b = {f"bridge_{self.bridge}": np.array([x, 0.0, bz]),
             f"bridge_{self.other}": np.array([G["park_x"][self.other], 0.0, bz]),
             "carriage": np.array([x, cy, lay["carriage_z"]]),
             "fork_W": tip["W"], "fork_F": tip["F"]}
        b["module_target"] = tip["W"] + up if t >= self.t_capture else slot.copy()
        b["spare_0"] = tip["F"] + up if t <= self.t_release_tool else slot.copy()
        off = self.stack(t)
        for k in range(1, G["n_fresh"]):
            b[f"spare_{k}"] = np.array([x, tip["F"][1], G["bay_z0"] - k * G["stack_dz"] + off])

        g = lay["grapple_point"]
        contact = any(np.allclose(tip[k][:2], g[:2]) and abs(fz[k] - g[2]) <= G["contact_zone_m"] + 1e-12
                      for k in ("W", "F"))
        return GantryFrame(bridge_x={self.bridge: x, self.other: G["park_x"][self.other]},
                           carriage_y=cy, fork_z=fz, step=self._step(t),
                           bodies={n: (p, _I3.copy()) for n, p in b.items()}, contact=contact)
