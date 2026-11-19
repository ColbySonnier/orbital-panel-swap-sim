"""Mission manager: the phase state machine for the servicer.

Flies the servicer from H0 (2 km behind the client) to a berth behind the
array, runs the arm swap, waits for checkout, and flies home to H0.

Frames (see geometry.py): H = Hill (rotating, origin at the client CoM),
A = array frame (inertial, origin at the array center), N = inertial.

Inputs each step: rel_H = servicer state relative to the client CoM in H
(position, velocity in the rotating Hill frame); rel_A = servicer state
relative to the array center in A (velocity is the inertial relative
velocity written in A).

Control:
  Far range (phases 0-3, departure hops, home) works in H.
    Hops: finite burns at max_thrust_n. The first burn is the CW two-impulse
    solution for a half-orbit transfer; from rest on the V-bar it equals the
    radial hop (radial_hop_dv, inward for +y, outward for -y). One midpoint
    correction re-solves the targeting from the current state; a braking
    burn centred on the arrival time closes the hop, then PD holds the point.
    Holds: a = pd(err) - cw_relative_accel(r_H, v_H)  (CW feedforward).
  Close range (phases 4-8 array legs, berth) works in A.
    a = leg acc + pd(err) - g_A, where g_A is the relative-gravity term:
    the CW relative acceleration with the rotating-frame (Coriolis and
    centrifugal) parts removed, i.e. the tidal term (2n^2 x, -n^2 y, -n^2 z)
    in H, rotated into A. A is inertial, so the full CW expression would
    add a fictitious ~3e-5 m/s^2 bias near berth, which the PD turns into
    a ~6-8 cm offset (more than the 5 cm berth tolerance).
  Force clip: by vector norm to max_thrust_n (keeps the direction).

Glue contract (caller): on a "failed" event open the module switch; when
Command.module_installed first becomes True do the power-state swap,
monitor.reset(module) and monitor.start_checkout(...). The manager leaves
Checkout only after a "checkout_pass" event (it remembers one that arrives
while the arm is still stowing).
"""
from dataclasses import dataclass, field

import numpy as np

from src.geometry import waypoints_A
from src.modules.arm_motion import SwapSequence
from src.modules.rel_guidance import (StraightLeg, cw_propagate,
                                      cw_relative_accel, pd_accel)

PHASES = ["Monitoring", "Go / no-go", "Far approach", "Inspection",
          "Move to back side", "Final approach", "Module exchange",
          "Checkout", "Departure", "Home"]

_EPS_DV = 1e-6       # m/s, a burn is done below this
_BRAKE_DONE = 1e-3   # m/s, braking ends below this Hill speed


@dataclass
class Command:
    phase: int
    phase_name: str
    force_N: np.ndarray
    thrusters_inhibited: bool
    arm_t: float | None
    overlay: str                 # "none" | "red" | "green"
    module_installed: bool = False
    frame: str = "H"             # frame the controller flies in: "H" or "A"
    sub: str = ""                # sub-step label (e.g. "coast", "leg 2/4")


def mean_motion(cfg):
    o = cfg["orbit"]
    return float(np.sqrt(o["mu_km3s2"] / (o["r_earth_km"] + o["alt_km"]) ** 3))


def _stm(n, tau):
    """6x6 CW state transition matrix."""
    return np.column_stack([cw_propagate(e, n, tau) for e in np.eye(6)])


def cw_target_velocity(r0, p1, tau, n):
    """Velocity at r0 that coasts to p1 after tau seconds (CW two-impulse).

    In-plane is always solvable; the out-of-plane channel is singular at
    half and full orbits, where vz = 0 is returned (the hold PD cleans up).
    """
    r0, p1 = np.asarray(r0, float), np.asarray(p1, float)
    phi = _stm(n, tau)
    rr, rv = phi[:3, :3], phi[:3, 3:]
    rhs = p1 - rr @ r0
    v = np.zeros(3)
    v[:2] = np.linalg.solve(rv[:2, :2], rhs[:2])  # in/out-of-plane decouple
    s = np.sin(n * tau)
    if abs(s) > 0.1:
        v[2] = rhs[2] / rv[2, 2]
    return v


def tidal_accel_H(r_H, v_H, n):
    """Relative gravity (inertial relative accel) in H: CW minus the
    rotating-frame Coriolis and centrifugal terms."""
    r_H, v_H = np.asarray(r_H, float), np.asarray(v_H, float)
    w = np.array([0.0, 0.0, n])
    fict = -2 * np.cross(w, v_H) - np.cross(w, np.cross(w, r_H))
    return cw_relative_accel(r_H, v_H, n) - fict


class _Burn:
    """Finite burn of a fixed Hill-frame dv vector at amax."""

    def __init__(self, dv, amax):
        self.rem = np.asarray(dv, float).copy()
        self.amax = amax
        self.last_a = np.zeros(3)

    def account(self, dt):
        self.rem -= self.last_a * dt
        self.last_a = np.zeros(3)

    @property
    def done(self):
        return float(np.linalg.norm(self.rem)) < _EPS_DV

    def accel(self, dt_nom):
        m = float(np.linalg.norm(self.rem))
        if m < _EPS_DV:
            return np.zeros(3)
        self.last_a = self.rem / m * min(self.amax, m / max(dt_nom, 1e-3))
        return self.last_a


class _Hop:
    """Half-orbit CW transfer to p1 with finite burns (Hill frame)."""

    def __init__(self, t, r, v, p1, n, amax):
        self.p1, self.n, self.amax = np.asarray(p1, float), n, amax
        T = np.pi / n
        dv = cw_target_velocity(r, p1, T, n) - v
        t_b = float(np.linalg.norm(dv)) / amax
        self.t_arr = t + t_b / 2 + T
        self.t_mid = t + t_b / 2 + T / 2
        self.burn = _Burn(dv, amax)
        self.state = "burn"
        self.t_brake = None

    def step(self, t, r, v, dt_prev, dt_nom):
        """Hill accel command, or None once holding (caller runs the PD)."""
        if self.burn is not None:
            self.burn.account(dt_prev)
        if self.state in ("burn", "correct"):
            if not self.burn.done:
                return self.burn.accel(dt_nom)
            self.state = "coast" if self.state == "burn" else "coast2"
            self.burn = None
        if self.state == "coast":
            if t < self.t_mid:
                return np.zeros(3)
            tau = self.t_arr - t
            v_req = cw_target_velocity(r, self.p1, tau, self.n)
            v_arr = cw_propagate([*r, *v_req], self.n, tau)[3:]
            self.t_brake = self.t_arr - float(np.linalg.norm(v_arr)) / self.amax / 2
            self.burn = _Burn(v_req - v, self.amax)
            self.state = "correct"
            return self.burn.accel(dt_nom)
        if self.state == "coast2":
            if t < self.t_brake:
                return np.zeros(3)
            self.state = "brake"
        if self.state == "brake":
            s = float(np.linalg.norm(v))
            if s < _BRAKE_DONE:
                self.state = "hold"
                return None
            return -v / s * min(self.amax, s / max(dt_nom, 1e-3))
        return None


class _Legs:
    """Back-to-back StraightLegs starting at t0 (A frame)."""

    def __init__(self, t0, points, speeds, a_max):
        self.t0 = t0
        self.legs = [StraightLeg(p0, p1, v, a_max)
                     for p0, p1, v in zip(points[:-1], points[1:], speeds)]
        self.end = points[-1]
        self.duration = sum(l.duration for l in self.legs)

    def at(self, t):
        if self.done(t):
            # StraightLeg.at clamps but keeps acc = -a at the end; hold cleanly.
            return self.end.copy(), np.zeros(3), np.zeros(3), len(self.legs) - 1
        tl = t - self.t0
        for i, leg in enumerate(self.legs):
            if tl < leg.duration or i == len(self.legs) - 1:
                p, v, a = leg.at(tl)
                return p, v, a, i
            tl -= leg.duration

    def done(self, t):
        return t - self.t0 >= self.duration


class MissionManager:
    """Phase state machine. update() once per sim step; returns a Command.

    geo (optional dict) may supply precomputed pieces: "swap" (SwapSequence).
    """

    def __init__(self, cfg, geo=None):
        geo = geo or {}
        self.cfg = cfg
        self.n = mean_motion(cfg)
        self.mass = float(cfg["servicer"]["mass_kg"])
        self.fmax = float(cfg["servicer"]["max_thrust_n"])
        self.amax = self.fmax / self.mass
        w = cfg["waypoints"]
        self.H = {k: np.array(w[k], float) for k in ("H0", "H1", "H2")}
        eye = np.eye(3)
        wp = waypoints_A(cfg, eye, eye)  # TARGET, H3, BERTH don't depend on dcms
        self.TARGET, self.H3, self.BERTH = wp["TARGET"], wp["H3"], wp["BERTH"]
        z = np.array([0.0, 0.0, 1.0])
        self.P10 = self.TARGET - cfg["speeds"]["slow_zone_m"] * z  # slow-zone gate
        sp = cfg["speeds"]
        self.v_leg, self.v_fin, self.a_leg = sp["leg_mps"], sp["final_mps"], sp["accel_mps2"]
        self.wn, self.zeta = cfg["control"]["wn"], cfg["control"]["zeta"]
        self.holds = cfg["holds_s"]
        self.swap = geo.get("swap") or SwapSequence(cfg)
        ev = {name: float(t) for t, name in self.swap.events}
        self._inhibit = [(ev["capture"], ev["release_client"]),
                         (ev["insert_contact"], ev["release_tool"])]
        self._t_latched = ev["latched"]

        self.phase = 0
        self.dv_used = 0.0
        self._t_prev = None
        self._dt_nom = 1.0
        self._last_F = np.zeros(3)
        self._t_phase = 0.0
        self._svc = False
        self._pass = False
        self._installed = False
        self._hop = None
        self._legs = None
        self._hold_t0 = None
        self._sub = ""
        self._dep = None  # departure sub-step

    # --- helpers ------------------------------------------------------------

    def _enter(self, phase, t):
        self.phase, self._t_phase = phase, t
        self._hop = self._legs = self._hold_t0 = None
        self._sub = ""

    def _clip(self, F):
        m = float(np.linalg.norm(F))
        return F * (self.fmax / m) if m > self.fmax else F

    def _hill_hold(self, p_ref, r, v):
        a = pd_accel(p_ref - r, -v, self.wn, self.zeta) - cw_relative_accel(r, v, self.n)
        return a

    def _a_track(self, ref, rel_A, rel_H, dcm_AH):
        p, v, acc = ref
        g_A = dcm_AH @ tidal_accel_H(rel_H[:3], rel_H[3:], self.n)
        return acc + pd_accel(p - rel_A[:3], v - rel_A[3:], self.wn, self.zeta) - g_A

    def _hop_step(self, t, rel_H, p1):
        """Hill accel for an active hop (PD hold after the brake)."""
        r, v = rel_H[:3], rel_H[3:]
        a = self._hop.step(t, r, v, self._dt_prev, self._dt_nom)
        self._sub = self._hop.state
        if a is None:
            if self._hold_t0 is None:
                self._hold_t0 = t
            return self._hill_hold(p1, r, v)
        return a

    @staticmethod
    def _settled(err, vel, tol_p, tol_v):
        return np.linalg.norm(err) < tol_p and np.linalg.norm(vel) < tol_v

    def _predict_dcm_HN(self, dcm_HN, dt):
        """Hill axes after dt on the circular orbit (rotation about z_H)."""
        c, s = np.cos(self.n * dt), np.sin(self.n * dt)
        x, y, z = dcm_HN
        return np.vstack([c * x + s * y, -s * x + c * y, z])

    # --- main ---------------------------------------------------------------

    def update(self, t, rel_H, rel_A, events, dcm_HN, dcm_AN):
        t = float(t)
        rel_H = np.asarray(rel_H, float)
        rel_A = np.asarray(rel_A, float)
        dcm_HN, dcm_AN = np.asarray(dcm_HN, float), np.asarray(dcm_AN, float)
        dcm_AH = dcm_AN @ dcm_HN.T
        self._dt_prev = 0.0 if self._t_prev is None else t - self._t_prev
        if self._dt_prev > 0:
            self.dv_used += float(np.linalg.norm(self._last_F)) / self.mass * self._dt_prev
            self._dt_nom = self._dt_prev
        self._t_prev = t
        for e in events:
            if e.kind == "service_request":
                self._svc = True
            elif e.kind == "checkout_pass":
                self._pass = True

        a_H = a_A = None
        inhibited, arm_t = False, None
        r, v = rel_H[:3], rel_H[3:]
        for _ in range(4):  # a phase may hand over within the same step
            ph = self.phase
            if ph == 0:
                if self._svc:
                    self._enter(1, t)
                    continue
                a_H = self._hill_hold(self.H["H0"], r, v)
            elif ph == 1:
                if t - self._t_phase >= self.holds["go"]:
                    self._enter(2, t)
                    self._hop = _Hop(t, r, v, self.H["H1"], self.n, self.amax)
                    continue
                a_H = self._hill_hold(self.H["H0"], r, v)
            elif ph in (2, 3):
                p1 = self.H["H1"] if ph == 2 else self.H["H2"]
                hold = self.holds["H1"] if ph == 2 else self.holds["H2"]
                tol = (5.0, 0.02) if ph == 2 else (0.5, 0.01)
                a_H = self._hop_step(t, rel_H, p1)
                if (self._hold_t0 is not None and t - self._hold_t0 >= hold
                        and self._settled(p1 - r, v, *tol)):
                    if ph == 2:
                        self._enter(3, t)
                        self._hop = _Hop(t, r, v, self.H["H2"], self.n, self.amax)
                    else:
                        self._enter(4, t)
                    continue
            elif ph == 4:
                if self._legs is None:
                    p0 = rel_A[:3].copy()
                    w3 = p0 - self.cfg["waypoints"]["W3_offset_m"] * np.array([0, 0, 1.0])
                    self._legs = _Legs(t, [p0, w3, self.H3], [self.v_leg] * 2, self.a_leg)
                a_A = self._follow(t, rel_A, rel_H, dcm_AH, "leg")
                if self._hold_done(t, self.holds["H3"], self.H3, rel_A, 0.2, 0.005):
                    self._enter(5, t)
                    continue
            elif ph == 5:
                if self._legs is None:
                    self._legs = _Legs(t, [self.H3, self.P10, self.BERTH],
                                       [self.v_leg, self.v_fin], self.a_leg)
                a_A = self._follow(t, rel_A, rel_H, dcm_AH, "leg")
                if self._hold_done(t, self.holds["berth"], self.BERTH, rel_A, 0.05, 0.005):
                    self._enter(6, t)
                    continue
            elif ph == 6:
                arm_t = t - self._t_phase
                if arm_t >= self.swap.duration:
                    self._enter(7, t)
                    arm_t = None
                    continue
                inhibited = any(a <= arm_t <= b for a, b in self._inhibit)
                if arm_t >= self._t_latched:
                    self._installed = True
                self._sub = self.swap.at(arm_t).step
                a_A = None if inhibited else self._a_track(
                    (self.BERTH, np.zeros(3), np.zeros(3)), rel_A, rel_H, dcm_AH)
            elif ph == 7:
                if self._pass and t > self._t_phase:
                    self._enter(8, t)
                    continue
                self._sub = "awaiting checkout"
                a_A = self._a_track((self.BERTH, np.zeros(3), np.zeros(3)),
                                    rel_A, rel_H, dcm_AH)
            elif ph == 8:
                res = self._departure(t, rel_H, rel_A, dcm_HN, dcm_AN, dcm_AH)
                if res is None:
                    self._enter(9, t)
                    continue
                a_H, a_A = res
            else:  # 9 Home
                a_H = self._hill_hold(self.H["H0"], r, v)
            break

        if inhibited:
            F_N = np.zeros(3)
        elif a_A is not None:
            F_N = self._clip(dcm_AN.T @ (self.mass * a_A))
        else:
            F_N = self._clip(dcm_HN.T @ (self.mass * a_H))
        self._last_F = F_N
        overlay = "green" if self._pass else ("red" if self._svc else "none")
        return Command(phase=self.phase, phase_name=PHASES[self.phase], force_N=F_N,
                       thrusters_inhibited=inhibited, arm_t=arm_t, overlay=overlay,
                       module_installed=self._installed,
                       frame="A" if a_A is not None or inhibited else "H",
                       sub=self._sub)

    def _follow(self, t, rel_A, rel_H, dcm_AH, label):
        p, v, acc, i = self._legs.at(t)
        self._sub = f"{label} {i + 1}/{len(self._legs.legs)}"
        if self._legs.done(t):
            if self._hold_t0 is None:
                self._hold_t0 = t
            self._sub = "hold"
        return self._a_track((p, v, acc), rel_A, rel_H, dcm_AH)

    def _hold_done(self, t, hold, p_ref, rel_A, tol_p, tol_v):
        return (self._hold_t0 is not None and t - self._hold_t0 >= hold
                and self._settled(p_ref - rel_A[:3], rel_A[3:], tol_p, tol_v))

    def _departure(self, t, rel_H, rel_A, dcm_HN, dcm_AN, dcm_AH):
        """Returns (a_H, a_A) for this step, or None when home."""
        r, v = rel_H[:3], rel_H[3:]
        if self._dep is None:
            self._dep = "out"
            self._legs = _Legs(t, [self.BERTH, self.P10, self.H3],
                               [self.v_fin, self.v_leg], self.a_leg)
        if self._dep == "out":
            a = self._follow(t, rel_A, rel_H, dcm_AH, "back out")
            if self._legs.done(t) and self._settled(self.H3 - rel_A[:3], rel_A[3:], 0.2, 0.005):
                self._dep = "around"
                self._legs = self._around_legs(t, dcm_HN, dcm_AN)
                self._hold_t0 = None
            return None, a
        if self._dep == "around":
            a = self._follow(t, rel_A, rel_H, dcm_AH, "around")
            if self._legs.done(t) and self._settled(self._legs.end - rel_A[:3], rel_A[3:], 0.5, 0.01):
                self._dep = "hop H1"
                self._hop, self._hold_t0 = _Hop(t, r, v, self.H["H1"], self.n, self.amax), None
            else:
                return None, a
        if self._dep == "hop H1":
            a = self._hop_step(t, rel_H, self.H["H1"])
            self._sub = "hop to H1: " + self._sub
            if self._hold_t0 is not None and self._settled(self.H["H1"] - r, v, 5.0, 0.02):
                self._dep = "hop H0"
                self._hop, self._hold_t0 = _Hop(t, r, v, self.H["H0"], self.n, self.amax), None
            else:
                return a, None
        if self._dep == "hop H0":
            a = self._hop_step(t, rel_H, self.H["H0"])
            self._sub = "hop to H0: " + self._sub
            if self._hold_t0 is not None and self._settled(self.H["H0"] - r, v, 5.0, 0.02):
                return None
            return a, None

    def _around_legs(self, t, dcm_HN, dcm_AN):
        """H3 -> W3' -> H2' where H2' is the A-frame spot of Hill H2 at the
        predicted leg end (the Hill frame turns ~1 rad during the legs)."""
        zA = np.array([0.0, 0.0, 1.0])
        off = self.cfg["waypoints"]["W3_offset_m"]
        dt = 0.0
        for _ in range(4):
            h2 = waypoints_A(self.cfg, dcm_AN, self._predict_dcm_HN(dcm_HN, dt))["H2"]
            legs = _Legs(t, [self.H3, h2 - off * zA, h2], [self.v_leg] * 2, self.a_leg)
            dt = legs.duration
        return legs
