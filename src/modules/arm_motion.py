"""Arm kinematics and the module swap sequence (ConOps steps 6.1-6.12).

Frames:
  S  servicer body: origin at the servicer CoM, +x out of the front (arm)
     face. The bus is a 1.2 (x) x 1.0 x 1.0 m box, front face at x = +0.6.
  A  array frame (see geometry.py). At berth the servicer CoM is at
     BERTH = target + (0, 0, -berth_offset) and S x = +z_A, S y = +y_A,
     S z = -x_A.
  L  link frames from fk(); the tool frame's +x is the approach axis.
  M  module body frame: +x_M is the module normal pointing from the back
     (fixture) face to the front (sun) face; y_M, z_M lie in the 2 x 2 m
     plane. A held module's M frame is aligned with the tool frame and its
     center sits GRAPPLE_D ahead of the tool point along +x_tool.
dcm_XY maps Y-frame vectors into X (v_X = dcm_XY @ v_Y), so the columns of
dcm_SL are the L axes written in S.

Arm: shoulder at the front-face center (0.6, 0, 0) (controller ruling; the
task brief's 0.5 predates the 1.2 m body length). Joint axes at q = 0
(arm straight along +x_S): z, y at the shoulder; y at the elbow (1.1 m out);
y, z, x at the wrist (another 1.1 m), then 0.3 m to the tool point.

Rack: spare slots 0-3 on the +y side, stow slots 4-5 on the -y side. Panels
lie parallel to the S x-z plane, 0.08 m apart outward from |y| = 0.55 m,
centered at x = -0.4 so the 2 m panels end flush with the front face.
Fixtures face outward, so the tool points inward (-y on +y slots). Racked
panels are rolled RACK_ROLL about their normal (same square footprint).

Motion: each step is built from joint-space moves and Cartesian moves. A
Cartesian move is solved by IK on a dense grid (<= 5 mm apart) and
interpolated linearly in joint space between grid points; joint-space moves
interpolate linearly between two IK solutions. Every move follows a
trapezoidal time law (15 % ramp up, 15 % ramp down) on a time-warped path. Its minimum time is set
by the joint-rate limit and the tool-speed limit (free or contact); moves
are then slowed to fill the real ConOps step duration (up to SLOW_MAX x),
with any remainder held at the end of the step. A step is stretched only if
its moves cannot fit its ConOps duration (recorded in SwapSequence.stretched).
"""
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from src.geometry import module_center

FRONT_X = 0.6
BUS_HALF = np.array([0.6, 0.5, 0.5])
GRAPPLE_GAP = 0.05          # grapple point behind the module back face
RACK_X, RACK_Y0, RACK_DY = -0.4, 0.55, 0.08
RACK_STANDOFF = 0.25        # pull-out distance from a rack slot
APPROACH = 0.10             # tool standoff before capture / after release
PULL = 0.50                 # extract / insert stroke at the array
SPARE_SLOT, STOW_SLOT = 3, 4
STEPS = ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6",
         "6.7", "6.8", "6.9", "6.10", "6.11", "6.12"]
Q_STOW = np.radians([0.0, -80.0, 160.0, -80.0, 0.0, 0.0])
# Via-point for the 6.7 cross from the stow slot to the spare slot: upper arm
# straight up, elbow folded, so the tool sweeps a small radius (2 cm/s cap).
Q_CROSS = np.radians([0.0, -90.0, 150.0, 100.0, 0.0, 0.0])
LATCH_S = 10.0              # latch / grapple actuation holds
SLOW_MAX = 2.5              # moves run at no less than 1/SLOW_MAX of the limit
MARGIN = 0.97               # fraction of each limit the profiles may use
RAMP = 0.15
RACK_ROLL = np.radians(-90.0)  # panel roll in the rack, +y side (sign picked so joint 6 stays in limits)
TILT = np.radians(30.0)  # panel tilt while the carry crosses the wrist singularity


class IKError(RuntimeError):
    """IK did not converge to the requested tool pose."""


# --- small rotation helpers -------------------------------------------------

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _log(R):
    """Rotation vector of R."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    if np.pi - th < 1e-6:
        w = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(w))
        w = R[:, i] + np.eye(3)[i]
        return th * w / np.linalg.norm(w)
    return th / (2 * np.sin(th)) * v


def _exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# --- frames -----------------------------------------------------------------

def berth_pose_A(cfg):
    """(pos_A, dcm_AS) of the servicer at berth."""
    tgt = module_center(*cfg["client"]["target"], cfg)
    pos = tgt + np.array([0.0, 0.0, -cfg["waypoints"]["berth_offset_m"]])
    dcm_AS = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    return pos, dcm_AS


def s_to_a(p_S, cfg):
    """Points (..., 3) in S -> A at berth."""
    pos, dcm_AS = berth_pose_A(cfg)
    return np.asarray(p_S, float) @ dcm_AS.T + pos


def a_to_s(p_A, cfg):
    """Points (..., 3) in A -> S at berth."""
    pos, dcm_AS = berth_pose_A(cfg)
    return (np.asarray(p_A, float) - pos) @ dcm_AS


def grapple_d(cfg):
    """Tool point to module center distance along the tool axis."""
    return cfg["client"]["module_thick_m"] / 2 + GRAPPLE_GAP


def target_fixture_S(cfg):
    """Tool pose (pos_S, dcm_S) that grapples the target module in its slot."""
    tgt = module_center(*cfg["client"]["target"], cfg)
    p_A = tgt + np.array([0.0, 0.0, -grapple_d(cfg)])
    _, dcm_AS = berth_pose_A(cfg)
    # Seated module: x_M = +z_A, y_M = +y_A, z_M = -x_A, i.e. aligned with S.
    dcm_AM = dcm_AS
    return a_to_s(p_A, cfg), dcm_AS.T @ dcm_AM


def module_half(cfg):
    s = cfg["client"]["slot_m"]
    return np.array([cfg["client"]["module_thick_m"] / 2, s / 2, s / 2])


def module_corners_S(c, R, half=None):
    """8 corners (8, 3) of a module with center c and dcm_SM R."""
    h = np.array([0.025, 1.0, 1.0]) if half is None else half
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return np.asarray(c) + (signs * h) @ np.asarray(R).T


def obb_overlap(c1, R1, h1, c2, R2, h2, eps=1e-9):
    """Separating-axis test for two oriented boxes (columns of R = box axes)."""
    R1, R2 = np.asarray(R1), np.asarray(R2)
    d = np.asarray(c2, float) - np.asarray(c1, float)
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            a = np.cross(R1[:, i], R2[:, j])
            if np.linalg.norm(a) > 1e-9:
                axes.append(a / np.linalg.norm(a))
    for a in axes:
        r1 = np.sum(np.abs(R1.T @ a) * h1)
        r2 = np.sum(np.abs(R2.T @ a) * h2)
        if abs(d @ a) > r1 + r2 + eps:
            return False
    return True


def _rack_y(k):
    if k < 4:
        return RACK_Y0 + RACK_DY * k
    return -(RACK_Y0 + RACK_DY * (k - 4))


def rack_module_pose(k, cfg):
    """(center_S, dcm_SM) of the module parked in rack slot k (0-5)."""
    y = _rack_y(k)
    psi = -np.pi / 2 if y > 0 else np.pi / 2  # x_M points at the bus
    # Rolled RACK_ROLL about x_M (mirrored on the -y side). The square panel
    # has the same footprint at any multiple of 90 deg; this roll lets the
    # arm carry the panel square-aligned through its wrist flip (see
    # SwapSequence._rack_transfer_legs) with joint 6 inside its limits.
    return np.array([RACK_X, y, 0.0]), _rz(psi) @ _rx(np.sign(y) * RACK_ROLL)


def rack_slot_pose(k, cfg):
    """Tool pose (pos_S, dcm_S) that grapples the module in rack slot k."""
    c, R = rack_module_pose(k, cfg)
    return c - grapple_d(cfg) * R[:, 0], R


# --- kinematics -------------------------------------------------------------

def fk(q, cfg):
    """[(pos_S, dcm_SL)] for base, links 1-6 and the tool (8 entries)."""
    L1, L2, L3 = cfg["arm"]["links_m"]
    sh = np.array([FRONT_X, 0.0, 0.0])
    R1 = _rz(q[0])
    R2 = R1 @ _ry(q[1])
    el = sh + R2[:, 0] * L1
    R3 = R2 @ _ry(q[2])
    wr = el + R3[:, 0] * L2
    R4 = R3 @ _ry(q[3])
    R5 = R4 @ _rz(q[4])
    R6 = R5 @ _rx(q[5])
    tool = wr + R6[:, 0] * L3
    return [(sh, np.eye(3)), (sh, R1), (sh, R2), (el, R3), (wr, R4),
            (wr, R5), (wr, R6), (tool, R6)]


def _jacobian(q, cfg):
    f = fk(q, cfg)
    pt = f[-1][0]
    # joint i axis in S and a point on it
    axes = [(f[0][1][:, 2], f[0][0]), (f[1][1][:, 1], f[1][0]),
            (f[2][1][:, 1], f[3][0]), (f[3][1][:, 1], f[4][0]),
            (f[4][1][:, 2], f[4][0]), (f[5][1][:, 0], f[4][0])]
    J = np.zeros((6, 6))
    for i, (a, p) in enumerate(axes):
        J[:3, i] = np.cross(a, pt - p)
        J[3:, i] = a
    return J, f[-1]


def _dls(pos, R, q, cfg, iters=200, tol=1e-7):
    """Damped least squares from q toward tool pose (pos, R)."""
    lim = np.radians(cfg["arm"]["limit_deg"])
    q = np.clip(np.array(q, float), -lim, lim)
    for _ in range(iters):
        J, (p, Rc) = _jacobian(q, cfg)
        e = np.concatenate([pos - p, _log(R @ Rc.T)])
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol:
            return q, True
        n = np.linalg.norm(e)
        if n > 0.2:
            e = e * 0.2 / n
        lam2 = 1e-4 + 1e-2 * min(n, 1.0) ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
        q = np.clip(q + dq, -lim, lim)
    return q, False


def ik(pos_S, dcm_S, q_seed, cfg, n_seeds=40):
    """Joint angles placing the tool at (pos_S, dcm_S). Damped least squares
    from q_seed, then from deterministic random seeds; returns the converged
    solution nearest q_seed. Raises IKError if none converges."""
    pos = np.asarray(pos_S, float)
    R = np.asarray(dcm_S, float)
    q_seed = np.asarray(q_seed, float)
    q, ok = _dls(pos, R, q_seed, cfg)
    if ok:
        return q
    lim = np.radians(cfg["arm"]["limit_deg"])
    rng = np.random.default_rng(1)
    best = None
    for _ in range(n_seeds):
        q, ok = _dls(pos, R, rng.uniform(-lim, lim, 6), cfg)
        if ok and (best is None or np.linalg.norm(q - q_seed) < np.linalg.norm(best - q_seed)):
            best = q
    if best is None:
        raise IKError(f"no IK solution for tool at {pos}")
    return best


# --- time law ---------------------------------------------------------------

_VPK = 1.0 / (1.0 - RAMP)  # peak ds/dtau of the trapezoid


def _prof(tau):
    tau = min(max(tau, 0.0), 1.0)
    a = _VPK / RAMP
    if tau < RAMP:
        return 0.5 * a * tau * tau
    if tau > 1 - RAMP:
        return 1 - 0.5 * a * (1 - tau) ** 2
    return 0.5 * _VPK * RAMP + _VPK * (tau - RAMP)


def _prof_inv(s):
    a = _VPK / RAMP
    s0 = 0.5 * _VPK * RAMP
    if s < s0:
        return np.sqrt(2 * s / a)
    if s > 1 - s0:
        return 1 - np.sqrt(2 * (1 - s) / a)
    return RAMP + (s - s0) / _VPK


class _Move:
    """Joint path through grid points q[i], time-warped to the local limits.

    Each grid segment gets a path-speed cap from the joint-rate and tool-
    speed limits; the caps are smoothed (window min, then window mean, which
    never exceeds the cap) and integrated into a normalized time grid tau[i].
    q is piecewise linear in tau. A trapezoid on tau (peak 1/(1-RAMP)) then
    gives the motion; t_min is the shortest duration that respects every cap.
    """

    def __init__(self, q, cfg, v_tip, win=8):
        self.q = np.asarray(q, float)
        dq = np.abs(np.diff(self.q, axis=0)).max(axis=1)
        tips = np.array([fk(qi, cfg)[-1][0] for qi in self.q])
        dp = np.linalg.norm(np.diff(tips, axis=0), axis=1)
        rate = np.radians(cfg["arm"]["rate_dps"]) * MARGIN
        # seconds per segment at the cap (both limits, per unit of "1 s")
        dt = np.maximum(np.maximum(dq / rate, dp / (v_tip * MARGIN)), 1e-6)
        # smooth the speed cap 1/dt: running min, then running mean
        n = len(dt)
        cap = 1.0 / dt
        lo = np.array([cap[max(0, i - win):i + win + 1].min() for i in range(n)])
        cap = np.array([lo[max(0, i - win):i + win + 1].mean() for i in range(n)])
        seg_t = 1.0 / cap
        self.tau = np.concatenate([[0.0], np.cumsum(seg_t)])
        self.t_min = _VPK * self.tau[-1]
        self.tau /= self.tau[-1]

    def q_at(self, s):
        return np.array([np.interp(s, self.tau, self.q[:, j]) for j in range(6)])


def _mirror(q):
    """Joint angles of the arm's mirror image across the S x-z plane."""
    return np.asarray(q) * np.array([-1, 1, 1, 1, -1, -1])


def _joint_move(q0, q1, cfg, n=80, via=None):
    """Joint-space straight line q0 -> q1, optionally through a via-point
    (no stop at the via)."""
    s = np.linspace(0, 1, n)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if via is None:
        q = q0 + np.outer(s, q1 - q0)
    else:
        q = np.vstack([q0 + np.outer(s, via - q0), (via + np.outer(s, q1 - via))[1:]])
    return _Move(q, cfg, cfg["arm"]["tip_free_mps"])


def _cart_move(legs, q_seed, cfg, v_tip, step=0.005, max_jump_deg=3.0):
    """legs: list of callables u -> (tool_pos, tool_R), u in [0, 1]. Each leg
    is gridded by its swept length (position + 0.3 m/rad of rotation) and
    solved by IK point to point from q_seed. Raises IKError if the solution
    jumps branch anywhere, including away from q_seed at the first point."""
    poses = []
    for leg in legs:
        u = np.linspace(0, 1, 401)
        pr = [leg(x) for x in u]
        dl = [np.linalg.norm(b[0] - a[0]) + 0.3 * np.linalg.norm(_log(b[1] @ a[1].T))
              for a, b in zip(pr[:-1], pr[1:])]
        L = sum(dl)
        n = max(4, int(np.ceil(L / step)))
        pts = [leg(x) for x in np.linspace(0, 1, n + 1)]
        if poses:
            pts = pts[1:]
        poses += pts
    qs, q = [], np.asarray(q_seed, float)
    for p, R in poses:
        qn, ok = _dls(p, R, q, cfg)
        if not ok:
            qn = ik(p, R, q, cfg)
        if np.abs(qn - q).max() > np.radians(max_jump_deg):  # q = seed at first
            raise IKError(f"IK branch jump near tool {p}")
        q = qn
        qs.append(q)
    return _Move(np.array(qs), cfg, v_tip)


# --- swap sequence ----------------------------------------------------------

@dataclass
class ArmFrame:
    """Arm state at one instant. bodies: {name: (center_S, dcm_SM)} for every
    module; contact: tool point within APPROACH (0.10 m) of a grapple point
    the sequence uses (target slot, spare slot 3, stow slot 4)."""
    q: np.ndarray
    step: str
    held: str | None
    bodies: dict
    contact: bool


@dataclass
class _Seg:
    t0: float
    t1: float
    step: str
    move: _Move | None
    q: np.ndarray       # joint angles for holds
    held: str | None
    parked: dict


class SwapSequence:
    """Scripted swap at the real ConOps step durations.

    at(t) -> ArmFrame; events -> [(t, name)]; stretched -> {step: (conops_s,
    actual_s)} for any step whose moves could not fit its ConOps duration.
    """

    def __init__(self, cfg, target_fixture_S=None):
        self.cfg = cfg
        if target_fixture_S is None:
            target_fixture_S = globals()["target_fixture_S"](cfg)
        if isinstance(target_fixture_S, tuple):
            fix_p, fix_R = (np.asarray(x, float) for x in target_fixture_S)
        else:
            fix_p, fix_R = np.asarray(target_fixture_S, float), np.eye(3)
        self._d = grapple_d(cfg)
        self._step_s = dict(zip(STEPS, [60.0 * m for m in cfg["arm"]["step_min"]]))
        self.events, self.stretched, self._segs = [], {}, []
        self._t, self._q, self._held = 0.0, Q_STOW.copy(), None
        self._parked = {"module_target": (fix_p + self._d * fix_R[:, 0], fix_R.copy())}
        for k in range(4):
            self._parked[f"spare_{k}"] = rack_module_pose(k, cfg)
        spare, stow = rack_slot_pose(SPARE_SLOT, cfg), rack_slot_pose(STOW_SLOT, cfg)
        self.contact_points = [fix_p, spare[0], stow[0]]
        self._build(fix_p, fix_R, spare, stow)
        self.duration = self._t
        self._t0 = [s.t0 for s in self._segs]

    # --- path helpers -------------------------------------------------------

    def _tool(self, c, psi, theta=0.0, roll=0.0):
        """Tool pose holding a module centered at c: yaw psi about z_S, then
        pitch theta about the tool y axis, then roll about the tool axis."""
        R = _rz(psi) @ _ry(theta) @ _rx(roll)
        return np.asarray(c, float) - self._d * R[:, 0], R

    def _rack_transfer_legs(self, slot):
        """Module-center legs from a rack slot (seated) out to PULL behind
        the target slot. Built for the +y side and mirrored for -y.
        Returns (extract_leg, transfer_legs) as tool-pose callables.

        Route (+y side; yaw psi of x_M about z_S, yo = slot |y| + standoff):
          extract  slide out along +y to y = yo                 psi = -90
          fwd      slide forward along +x to center x = 1.4     psi = -90
          pivot    yaw to 0 about the panel's bus-side edge, held at
                   (0.4, yo): the panel stays outboard of the bus and the
                   rack, and its far edge never passes x = 2.4
          tilt     in place at (0.4, yo + 1): pitch the panel TILT about
                   its horizontal in-plane axis
          a        slide forward along +x to (1.3, yo + 1), tilted. The wrist
                   flips here (joints 2+3+4 go from ~180 to ~0 deg); the tilt
                   keeps it off the singularity where the joint 4 and 6 axes
                   align, which a level carry would have to cross
          b1       to (1.5, yo + 0.5), tilted
          spin     in place: roll the panel from RACK_ROLL to 0. Centered at
                   x = 1.5 the panel clears the bus, the rack and the array
                   at any roll (half-diagonal 1.41 m)
          b2       diagonal to PULL behind the slot, untilting
        The panel keeps RACK_ROLL (square-aligned) through the wrist flip;
        joint 6 absorbs the flip's 180 deg of roll inside its limits.
        """
        sgn = 1.0 if _rack_y(slot) > 0 else -1.0
        ys = abs(_rack_y(slot))
        yo = ys + RACK_STANDOFF
        seat_c = self._seat_c
        end = np.array([seat_c[0] - PULL, sgn * seat_c[1]])
        piv = np.array([0.4, yo])

        def mk(cfun, psifun, thfun=lambda u: 0.0, roll=0.0):
            rfun = roll if callable(roll) else (lambda u: roll)

            def leg(u):
                c = cfun(u)
                return self._tool([c[0], sgn * c[1], 0.0], sgn * psifun(u), thfun(u),
                                  sgn * rfun(u))
            return leg

        def lin(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            return lambda u: a + u * (b - a)

        def smooth(u):
            return u * u * (3 - 2 * u)

        q, r = -np.pi / 2, RACK_ROLL
        extract = mk(lin([RACK_X, ys], [RACK_X, yo]), lambda u: q, roll=r)
        fwd = mk(lin([RACK_X, yo], [1.4, yo]), lambda u: q, roll=r)
        pivot = mk(lambda u: piv + np.array([-np.sin(q * (1 - u)), np.cos(q * (1 - u))]),
                   lambda u: q * (1 - u), roll=r)
        top = np.array([0.4, yo + 1.0])
        tilt = mk(lambda u: top, lambda u: 0.0, lambda u: TILT * smooth(u), roll=r)
        a = mk(lin(top, [1.3, yo + 1.0]), lambda u: 0.0, lambda u: TILT, roll=r)
        mid = np.array([1.5, yo + 0.5])
        b1 = mk(lin([1.3, yo + 1.0], mid), lambda u: 0.0, lambda u: TILT, roll=r)
        spin = mk(lambda u: mid, lambda u: 0.0, lambda u: TILT,
                  roll=lambda u: r * (1 - smooth(u)))
        b2 = mk(lin(mid, end), lambda u: 0.0, lambda u: TILT * (1 - smooth(u)))
        return extract, [fwd, pivot, tilt, a, b1, spin, b2]

    @staticmethod
    def _rev(leg):
        return lambda u: leg(1.0 - u)

    # --- timeline -----------------------------------------------------------

    def _seg(self, dur, step, move=None):
        q = self._q.copy() if move is None else None
        self._segs.append(_Seg(self._t, self._t + dur, step, move, q,
                               self._held, dict(self._parked)))
        self._t += dur
        if move is not None:
            self._q = move.q[-1].copy()

    def _run(self, step, items):
        """items: ('move', _Move[, (s, event)]), ('hold', s), ('event', name),
        ('grab', body), ('release',), ('fill',)."""
        D = self._step_s[step]
        moves = [it[1] for it in items if it[0] == "move"]
        holds = sum(it[1] for it in items if it[0] == "hold")
        tmin = sum(m.t_min for m in moves)
        k = 1.0
        if tmin > 0:
            if holds + tmin > D:
                self.stretched[step] = (D, holds + tmin)
                D = holds + tmin
            k = min(SLOW_MAX, (D - holds) / tmin)
        fill = D - holds - k * tmin
        if not any(it[0] == "fill" for it in items):
            items = list(items) + [("fill",)]
        for it in items:
            kind = it[0]
            if kind == "move":
                T = k * it[1].t_min
                if len(it) > 2:
                    s_ev, name = it[2]  # fraction of the move's grid points
                    m = it[1]
                    tau = np.interp(s_ev * (len(m.q) - 1), np.arange(len(m.q)), m.tau)
                    self.events.append((self._t + T * _prof_inv(tau), name))
                self._seg(T, step, it[1])
            elif kind == "hold":
                self._seg(it[1], step)
            elif kind == "fill":
                if fill > 1e-9:
                    self._seg(fill, step)
            elif kind == "event":
                self.events.append((self._t, it[1]))
            elif kind == "grab":
                self._held = it[1]
                self._parked.pop(it[1])
            elif kind == "release":
                p, R = fk(self._q, self.cfg)[-1]
                self._parked[self._held] = (p + self._d * R[:, 0], R)
                self._held = None

    def _build(self, fix_p, fix_R, spare, stow):
        cfg = self.cfg
        x = fix_R[:, 0]
        self._seat_c = fix_p + self._d * x
        free, cont = cfg["arm"]["tip_free_mps"], cfg["arm"]["tip_contact_mps"]
        pre = (fix_p - APPROACH * x, fix_R)
        out = (fix_p - PULL * x, fix_R)

        def line(a, b):
            return lambda u: (a[0] + u * (b[0] - a[0]), a[1])

        def cart(legs, v, seed=None):
            return _cart_move(legs, self._q if seed is None else seed, cfg, v)

        def joint_to(pose, seed=None):
            q1 = ik(*pose, self._q if seed is None else seed, cfg)
            return _joint_move(self._q, q1, cfg)

        self._run("6.1", [("fill",)])
        self._run("6.2", [("move", joint_to(pre))])
        self._run("6.3", [("move", cart([line(pre, (fix_p, fix_R))], cont)),
                          ("grab", "module_target"), ("event", "capture")])
        self._run("6.4", [("fill",)])
        self._run("6.5", [("move", cart([line((fix_p, fix_R), out)], cont)),
                          ("event", "release_client")])
        # 6.6: swing to the stow slot, insert, latch, let go.
        ext, xfer = self._rack_transfer_legs(STOW_SLOT)
        m_xfer = cart([self._rev(l) for l in reversed(xfer)], free)
        m_ins = cart([self._rev(ext)], cont, m_xfer.q[-1])
        self._run("6.6", [("move", m_xfer), ("move", m_ins), ("hold", LATCH_S),
                          ("release",)])
        # 6.7: back off the stowed module, cross to the spare, grapple it.
        st_p, st_R = stow
        back4 = (st_p - APPROACH * st_R[:, 0], st_R)
        m_back = cart([line(stow, back4)], cont)
        q_back = m_back.q[-1]
        sp_p, sp_R = spare
        pre3 = (sp_p - APPROACH * sp_R[:, 0], sp_R)
        # Same wrist branch as at the stow slot (mirror image), so the spare
        # carry mirrors the damaged-module carry.
        q_pre3 = ik(*pre3, _mirror(q_back), cfg)
        m_cross = _joint_move(q_back, q_pre3, cfg, via=Q_CROSS)
        m_app = cart([line(pre3, spare)], cont, q_pre3)
        self._run("6.7", [("hold", LATCH_S), ("move", m_back), ("move", m_cross),
                          ("move", m_app), ("hold", LATCH_S),
                          ("grab", f"spare_{SPARE_SLOT}")])
        # 6.8: pull the spare out of the rack and carry it behind the slot.
        ext, xfer = self._rack_transfer_legs(SPARE_SLOT)
        m_ext = cart([ext], cont)
        m_xfer = cart(xfer, free, m_ext.q[-1])
        self._run("6.8", [("move", m_ext), ("move", m_xfer)])
        m_ins = cart([line(out, (fix_p, fix_R))], cont)
        s_contact = (PULL - APPROACH) / PULL
        self._run("6.9", [("move", m_ins, (s_contact, "insert_contact"))])
        self._run("6.10", [("fill",), ("event", "latched")])
        m_back = cart([line((fix_p, fix_R), pre)], cont)
        self._run("6.11", [("hold", LATCH_S), ("release",), ("event", "release_tool"),
                           ("move", m_back)])
        self._run("6.12", [("move", _joint_move(self._q, Q_STOW, cfg))])

    # --- query --------------------------------------------------------------

    def at(self, t):
        t = min(max(float(t), 0.0), self.duration)
        i = max(0, bisect_right(self._t0, t) - 1)
        sg = self._segs[i]
        if sg.move is None:
            q = sg.q.copy()
        else:
            tau = (t - sg.t0) / (sg.t1 - sg.t0) if sg.t1 > sg.t0 else 1.0
            q = sg.move.q_at(_prof(tau))
        p, R = fk(q, self.cfg)[-1]
        bodies = dict(sg.parked)
        if sg.held is not None:
            bodies[sg.held] = (p + self._d * R[:, 0], R)
        contact = min(np.linalg.norm(p - c) for c in self.contact_points) <= APPROACH + 1e-9
        return ArmFrame(q=q, step=sg.step, held=sg.held, bodies=bodies, contact=contact)
