"""Geometry and frames for the rail-gantry demo.

Frames:
  N  inertial
  H  Hill: x radial out, y in-track (orthogonal to x), z orbit normal (r x v)
  A  array: origin at array center, z_A = sun unit vector,
     y_A = Hill y projected perpendicular to z_A, x_A = y_A x z_A.
     The client body frame B is defined aligned with A.
dcm_XN maps N-frame vectors into X (v_X = dcm_XN @ v_N).
"""
import numpy as np


def module_center(r, c, cfg):
    """Module (r, c) center in A, meters."""
    s = cfg["client"]["slot_m"]
    off = s * (cfg["client"]["grid"] + 1) / 2.0
    return np.array([s * c - off, s * r - off, 0.0])


def module_ids(cfg):
    """All (r, c) module ids, 1-based (1..grid), bus slots excluded."""
    n = cfg["client"]["grid"]
    bus = {tuple(b) for b in cfg["client"]["bus_slots"]}
    return [(r, c) for r in range(1, n + 1) for c in range(1, n + 1) if (r, c) not in bus]


def hill_dcm(r_N, v_N):
    r_N = np.asarray(r_N, float)
    v_N = np.asarray(v_N, float)
    x = r_N / np.linalg.norm(r_N)
    z = np.cross(r_N, v_N)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def array_dcm(sun_hat_N, r_N, v_N):
    z = np.asarray(sun_hat_N, float)
    z = z / np.linalg.norm(z)
    yh = hill_dcm(r_N, v_N)[1]
    y = yh - np.dot(yh, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z])


def _plate_inertia(m, a, b, c):
    """Solid box a x b x c (about its center, axes x,y,z)."""
    return np.diag([m * (b * b + c * c), m * (a * a + c * c),
                    m * (a * a + b * b)]) / 12.0


def _shift(I, m, d):
    """Parallel-axis term for offset d."""
    return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def client_mass_props(cfg, missing):
    """(mass, CoM, inertia about CoM) in A axes; origin at array center."""
    c = cfg["client"]
    missing = {tuple(m) for m in missing}
    bodies = []  # (mass, center, inertia about own center)
    a = c["bus_size_m"]
    bodies.append((c["bus_mass_kg"], np.array([0, 0, -c["bus_offset_m"]]),
                   _plate_inertia(c["bus_mass_kg"], a, a, a)))
    side = c["slot_m"] * c["grid"]
    bodies.append((c["truss_mass_kg"], np.zeros(3),
                   _plate_inertia(c["truss_mass_kg"], side, side, 0.0)))
    mm = c["module_mass_kg"]
    Im = _plate_inertia(mm, c["slot_m"], c["slot_m"], c["module_thick_m"])
    for rc in module_ids(cfg):
        if rc not in missing:
            bodies.append((mm, module_center(*rc, cfg), Im))
    mass = sum(b[0] for b in bodies)
    com = sum(b[0] * b[1] for b in bodies) / mass
    I = sum(_shift(b[2], b[0], b[1] - com) for b in bodies)
    return mass, com, I
