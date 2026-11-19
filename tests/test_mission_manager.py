"""Mission manager tests against a point-mass CW plant (tests only).

Plant: Hill-frame CW equations, RK4 at 1 s, plus the commanded force
rotated N -> H. Frames: circular orbit from config (550 km, inc 97.6,
RAAN 90, arg lat 0 at t = 0); sun fixed in N, 8 deg off the orbit normal
(beta ~ 82 deg); dcm_AN constant (array_dcm at t = 0); dcm_HN rotates at n.
While thrusters are inhibited the plant locks rel_A (rigid arm link).
A HealthMonitor + PowerState run alongside (strike at 600 s) with the
glue the controller contract describes.
"""
import numpy as np
import pytest

from src.config import load_config
from src.geometry import array_dcm, client_mass_props, hill_dcm, waypoints_A
from src.modules.health_monitor import HealthMonitor, PowerState, module_powers
from src.modules.mission_manager import (PHASES, Command, MissionManager,
                                         cw_target_velocity)
from src.modules.rel_guidance import cw_propagate, radial_hop_dv

cfg = load_config()
norm = np.linalg.norm
o = cfg["orbit"]
R = o["r_earth_km"] + o["alt_km"]
N_MEAN = np.sqrt(o["mu_km3s2"] / R**3)
TARGET_ID = tuple(cfg["client"]["target"])
T_MAX = 40000.0


def orbit_rv(t):
    i, raan = np.radians(o["inc_deg"]), np.radians(o["raan_deg"])
    u = np.radians(o["arg_lat_deg"]) + N_MEAN * t
    P = np.array([np.cos(raan), np.sin(raan), 0.0])
    Q = np.array([-np.cos(i) * np.sin(raan), np.cos(i) * np.cos(raan), np.sin(i)])
    r = R * (np.cos(u) * P + np.sin(u) * Q)
    v = R * N_MEAN * (-np.sin(u) * P + np.cos(u) * Q)
    return r, v


def cw_deriv(x, a):
    n = N_MEAN
    return np.array([x[3], x[4], x[5],
                     3 * n**2 * x[0] + 2 * n * x[4] + a[0],
                     -2 * n * x[3] + a[1],
                     -n**2 * x[2] + a[2]])


def rk4(x, a, dt):
    k1 = cw_deriv(x, a)
    k2 = cw_deriv(x + dt / 2 * k1, a)
    k3 = cw_deriv(x + dt / 2 * k2, a)
    k4 = cw_deriv(x + dt * k3, a)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


W_H = np.array([0.0, 0.0, N_MEAN])  # Hill angular velocity, in H


def run_mission():
    r0, v0 = orbit_rv(0.0)
    h = np.cross(r0, v0)
    h /= norm(h)
    b = np.radians(8.0)
    sun = np.cos(b) * h + np.sin(b) * r0 / norm(r0)
    dcm_AN = array_dcm(sun, r0, v0)
    com = client_mass_props(cfg, set())[1]
    mass = cfg["servicer"]["mass_kg"]

    mm = MissionManager(cfg)
    rng = np.random.default_rng(cfg["seed"])
    pstate, mon = PowerState(), HealthMonitor(cfg)
    x = np.array([*cfg["waypoints"]["H0"], 0.0, 0.0, 0.0], float)
    lock = None
    installed = False
    log = {k: [] for k in ("t", "rel_H", "rel_A", "cmd", "events")}
    t, home_t = 0.0, None
    while t <= T_MAX:
        dcm_HN = hill_dcm(*orbit_rv(t))
        dcm_AH = dcm_AN @ dcm_HN.T
        if lock is not None:  # rigid link: co-move with the array
            x[:3] = dcm_AH.T @ (lock - com)
            x[3:] = -np.cross(W_H, x[:3])
        rel_A = np.concatenate([dcm_AH @ x[:3] + com,
                                dcm_AH @ (x[3:] + np.cross(W_H, x[:3]))])
        events = mon.step(t, module_powers(t, cfg, pstate, rng, strike=True))
        for e in events:
            if e.kind == "failed":
                pstate.open_switch.add(e.module)
        cmd = mm.update(t, x.copy(), rel_A, events, dcm_HN, dcm_AN)
        if cmd.module_installed and not installed:
            installed = True
            pstate.open_switch.discard(TARGET_ID)
            pstate.replaced.add(TARGET_ID)
            mon.reset(TARGET_ID)
            mon.start_checkout(TARGET_ID, t, cfg["holds_s"]["checkout"])
        for k, v in (("t", t), ("rel_H", x.copy()), ("rel_A", rel_A),
                     ("cmd", cmd), ("events", events)):
            log[k].append(v)
        if cmd.phase == 9 and home_t is None:
            home_t = t
        if home_t is not None and t - home_t >= 600:
            break
        lock = rel_A[:3].copy() if cmd.thrusters_inhibited else None
        if lock is None:
            x = rk4(x, dcm_HN @ cmd.force_N / mass, 1.0)
        t += 1.0
    log["t"] = np.array(log["t"])
    log["rel_H"] = np.array(log["rel_H"])
    log["rel_A"] = np.array(log["rel_A"])
    log["phase"] = np.array([c.phase for c in log["cmd"]])
    log["mm"] = mm
    log["dcm_AN"] = dcm_AN
    log["wp"] = waypoints_A(cfg, dcm_AN, hill_dcm(r0, v0))
    return log


@pytest.fixture(scope="module")
def sim():
    return run_mission()


def first_t(log, pred):
    return next(t for t, c in zip(log["t"], log["cmd"]) if pred(c))


# --- unit checks (fast) ------------------------------------------------------

def test_phase_names():
    assert PHASES == ["Monitoring", "Go / no-go", "Far approach", "Inspection",
                      "Move to back side", "Final approach", "Module exchange",
                      "Checkout", "Departure", "Home"]


def test_targeting_reduces_to_radial_hop():
    T = np.pi / N_MEAN
    v = cw_target_velocity([0, -2000, 0], [0, -200, 0], T, N_MEAN)
    assert np.allclose(v, [-radial_hop_dv(1800, N_MEAN), 0, 0], atol=1e-9)


def test_targeting_hits_target_off_axis():
    r0, p1, tau = np.array([5.0, -40, 2]), np.array([0, -200.0, 0]), 1500.0
    v = cw_target_velocity(r0, p1, tau, N_MEAN)
    assert norm(cw_propagate([*r0, *v], N_MEAN, tau)[:3] - p1) < 1e-6


def test_monitoring_holds_without_request():
    mm = MissionManager(cfg)
    r0, v0 = orbit_rv(0.0)
    dcm = hill_dcm(r0, v0)
    x = np.array([*cfg["waypoints"]["H0"], 0, 0, 0], float)
    c = mm.update(0.0, x, np.zeros(6), [], dcm, dcm)
    assert isinstance(c, Command)
    assert c.phase == 0 and c.overlay == "none" and norm(c.force_N) < 1e-9


class _Ev:
    def __init__(self, t, kind, module):
        self.t, self.kind, self.module = t, kind, module


def _mm_in_checkout():
    mm = MissionManager(cfg)
    mm._enter(7, 0.0)
    mm._installed = True
    return mm


def _step(mm, t, events=()):
    r0, v0 = orbit_rv(t)
    dcm = hill_dcm(r0, v0)
    rel_A = np.concatenate([mm.BERTH, np.zeros(3)])
    return mm.update(t, np.zeros(6), rel_A, list(events), dcm, dcm)


def test_tolerances_in_config():
    for k in ("H0", "H1", "H2", "H3", "berth"):
        assert set(cfg["tolerances"][k]) == {"pos_m", "vel_mps"}


def test_phase_timeouts_flattened():
    to = cfg["phase_timeout_s"]
    assert to["monitoring"] is None and to["home"] is None
    assert all(isinstance(to[k], (int, float)) for k in to if k not in ("monitoring", "home"))


def test_checkout_pass_wrong_module_ignored():
    mm = _mm_in_checkout()
    _step(mm, 1.0, [_Ev(1.0, "checkout_pass", (1, 1))])
    c = _step(mm, 2.0)
    assert c.phase == 7 and c.overlay != "green"
    _step(mm, 3.0, [_Ev(3.0, "checkout_pass", TARGET_ID)])
    assert _step(mm, 4.0).phase == 8


def test_checkout_fail_label():
    mm = _mm_in_checkout()
    c = _step(mm, 1.0, [_Ev(1.0, "checkout_fail", TARGET_ID)])
    assert c.phase == 7 and c.sub == "checkout_fail"


def test_timeout_label_keeps_flying():
    mm = _mm_in_checkout()
    t = cfg["phase_timeout_s"]["checkout"] + 1.0
    c = _step(mm, t)
    assert c.phase == 7 and c.sub == "timeout"


# --- full mission ------------------------------------------------------------

def test_phase_order(sim):
    seen = []
    for p in sim["phase"]:
        if not seen or seen[-1] != p:
            seen.append(int(p))
    assert seen == list(range(10))


def test_red_then_green(sim):
    t_red = first_t(sim, lambda c: c.overlay == "red")
    t_green = first_t(sim, lambda c: c.overlay == "green")
    assert t_red < t_green


def test_berth_accuracy(sim):
    i = int(np.argmax(sim["phase"] == 6))
    p, v = sim["rel_A"][i, :3], sim["rel_A"][i, 3:]
    tol = cfg["tolerances"]["berth"]
    assert norm(p - sim["wp"]["BERTH"]) < tol["pos_m"] and norm(v) < tol["vel_mps"]


def test_ends_at_h0(sim):
    tol = cfg["tolerances"]["H0"]
    assert norm(sim["rel_H"][-1, :3] - np.array(cfg["waypoints"]["H0"])) < tol["pos_m"]
    assert norm(sim["rel_H"][-1, 3:]) < tol["vel_mps"]


def test_no_timeouts(sim):
    assert not any(c.sub == "timeout" for c in sim["cmd"])


def test_dv_budget(sim):
    assert sim["mm"].dv_used <= 6.0


def test_force_within_limit(sim):
    fmax = cfg["servicer"]["max_thrust_n"]
    assert max(norm(c.force_N) for c in sim["cmd"]) <= fmax + 1e-9


def in_cone(p, wp):
    c = cfg["client"]
    d = p - wp["TARGET"]
    along = -d[2]  # cone axis is -z_A
    lateral = norm(d[:2])
    return (0.0 <= along <= c["cone_len_m"]
            and lateral <= along * np.tan(np.radians(c["cone_half_deg"])))


def test_corridor(sim):
    keep = cfg["client"]["keepout_m"]
    inside = [p for p in sim["rel_A"][:, :3] if norm(p) < keep]
    assert inside, "never entered the keep-out sphere"
    assert all(in_cone(p, sim["wp"]) for p in inside)


def test_inhibit_while_linked(sim):
    ev = dict((name, t) for t, name in sim["mm"].swap.events)
    windows = [(ev["capture"], ev["release_client"]),
               (ev["insert_contact"], ev["release_tool"])]

    def linked(c):
        return c.arm_t is not None and any(a <= c.arm_t <= b for a, b in windows)

    linked_cmds = [c for c in sim["cmd"] if linked(c)]
    assert linked_cmds
    assert all(c.thrusters_inhibited and norm(c.force_N) == 0.0 for c in linked_cmds)


def test_module_installed_from_phase_7(sim):
    """Phase 6 ends at "spare latched, arm stowed": installed from phase 7 on."""
    assert all(c.module_installed == (c.phase >= 7) for c in sim["cmd"])


def test_green_only_after_checkout_hold(sim):
    t7 = sim["t"][int(np.argmax(sim["phase"] == 7))]
    t_green = first_t(sim, lambda c: c.overlay == "green")
    assert t_green >= t7 + cfg["holds_s"]["checkout"] - 1.0
    assert not any(c.overlay == "green" for c in sim["cmd"] if c.phase <= 6)


def test_speed_limits(sim):
    """ConOps closing-speed limits inside 200 m, 20 % margin.

    Hop / Hill-hold phases (orbital segments) use LVLH range and speed
    (|rel_H|); array-frame phases (4-8 forced-motion legs and berth) use
    range and speed relative to the array center (|rel_A|). The < 10 m
    zone means within 10 m of the array plane while over the array face.
    """
    half = cfg["client"]["slot_m"] * cfg["client"]["grid"] / 2
    slow = cfg["speeds"]["slow_zone_m"]
    bad = []
    for t, xh, xa, c in zip(sim["t"], sim["rel_H"], sim["rel_A"], sim["cmd"]):
        if c.frame == "A":
            rng, spd, p = norm(xa[:3]), norm(xa[3:]), xa[:3]
            if abs(p[2]) < slow and max(abs(p[0]), abs(p[1])) <= half:
                lim = 0.02
            elif rng < 30:
                lim = 0.05
            elif rng < 200:
                lim = 0.10
            else:
                continue
        else:
            rng, spd = norm(xh[:3]), norm(xh[3:])
            if rng < 30:
                lim = 0.05
            elif rng < 200:
                lim = 0.10
            else:
                continue
        if spd > 1.2 * lim:
            bad.append((t, c.phase, round(rng, 2), round(spd, 4), lim))
    assert not bad, bad[:10]
