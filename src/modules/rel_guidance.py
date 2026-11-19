"""Relative guidance: Clohessy-Wiltshire hops and forced-motion legs.

Hill frame: x radial out, y along-track, z orbit normal.
CW: xdd = 3n^2 x + 2n ydot, ydd = -2n xdot, zdd = -n^2 z.
"""
import numpy as np


def radial_hop_dv(dy_m, n):
    """Magnitude of the radial impulse that drifts a servicer at rest by dy_m
    along-track after half an orbit (t = pi/n): |dy| = 4 dv / n -> dv = dy n / 4.

    SIGN: from the CW STM, y(pi/n) = y0 - 4*xdot0/n, so a radially OUTWARD
    (+x) push drifts toward -y. To go +y (e.g. H0 -> H1) push radially INWARD
    (xdot0 = -dv). The magnitude returned here is always positive.
    """
    return abs(dy_m) * n / 4.0


def cw_propagate(x0, n, t):
    """Closed-form CW state transition. x0 = [x,y,z,vx,vy,vz] in Hill."""
    x, y, z, vx, vy, vz = np.asarray(x0, float)
    c, s = np.cos(n * t), np.sin(n * t)
    return np.array([
        (4 - 3 * c) * x + s / n * vx + 2 / n * (1 - c) * vy,
        6 * (s - n * t) * x + y + 2 / n * (c - 1) * vx + (4 * s - 3 * n * t) / n * vy,
        c * z + s / n * vz,
        3 * n * s * x + c * vx + 2 * s * vy,
        6 * n * (c - 1) * x - 2 * s * vx + (4 * c - 3) * vy,
        -n * s * z + c * vz,
    ])


def cw_relative_accel(r_H, v_H, n):
    """CW differential acceleration (feedforward) in Hill."""
    r_H = np.asarray(r_H, float)
    v_H = np.asarray(v_H, float)
    return np.array([3 * n**2 * r_H[0] + 2 * n * v_H[1],
                     -2 * n * v_H[0],
                     -n**2 * r_H[2]])


def pd_accel(pos_err, vel_err, wn, zeta):
    """Commanded accel: wn^2 * pos_err + 2 zeta wn * vel_err (errors = desired - actual)."""
    return wn**2 * np.asarray(pos_err, float) + 2 * zeta * wn * np.asarray(vel_err, float)


class StraightLeg:
    """Straight-line trapezoidal-speed profile from p0 to p1 (rest to rest)."""

    def __init__(self, p0, p1, v_max, a_max):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        d = self.p1 - self.p0
        self.L = float(np.linalg.norm(d))
        self.u = d / self.L if self.L > 0 else np.zeros(3)
        self.a = float(a_max)
        if self.L >= v_max**2 / self.a:
            self.v = float(v_max)
            self.t_acc = self.v / self.a
            self.t_cruise = (self.L - self.v**2 / self.a) / self.v
        else:  # triangular
            self.v = float(np.sqrt(self.L * self.a))
            self.t_acc = self.v / self.a
            self.t_cruise = 0.0
        self.duration = 2 * self.t_acc + self.t_cruise

    def at(self, t):
        """(pos, vel, acc) at time t, clamped to [0, duration]."""
        t = min(max(float(t), 0.0), self.duration)
        ta, tc, a, v = self.t_acc, self.t_cruise, self.a, self.v
        if t < ta:
            s, sd, sdd = 0.5 * a * t * t, a * t, a
        elif t < ta + tc:
            s, sd, sdd = 0.5 * a * ta**2 + v * (t - ta), v, 0.0
        else:
            td = self.duration - t
            s, sd, sdd = self.L - 0.5 * a * td * td, a * td, -a
        if self.L == 0:
            s = sd = sdd = 0.0
        return self.p0 + s * self.u, sd * self.u, sdd * self.u
