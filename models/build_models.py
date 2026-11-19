"""Build the textured OBJ+MTL models for the Vizard replay.

    python -m models.build_models          # build models/obj/ + previews

Every model is written in the frame its Vizard body uses (Task 7 attaches
each OBJ with vizSupport.createCustomModel):

  client          A axes, origin at the array center. Array plane z = 0,
                  cells on +z (sun). 59 modules (M(2,7) is its own body),
                  truss behind the array, 4 m gold bus at z = -4.
  module          A axes when seated, origin at the module center. Cells on
                  +z, grapple fixture on the -z (back) face, its tip at the
                  tool point (z = -(t/2 + grapple_gap)).
  servicer        S axes (+x = arm face), origin at the servicer CoM.
  arm_link_1..6   each in its own link frame from arm_motion.fk().
  tool            tool frame (origin at the tool point, +x = approach axis).
  overlay_*       module frame, a 2.02 m plate at z = -0.04 (behind the back).
  keepout_sphere  A axes, origin at the array center: the client.keepout_m
                  sphere, amber, semi-transparent (MTL d, see MAT_ALPHA).

All geometry is procedural: real CC0 textures (ambientCG) on beveled,
lathed and trussed parts. Textures are copied or derived next to the OBJs
and referenced by relative path in each MTL.
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import load_config
from src.geometry import module_center, module_ids
from src.modules.arm_motion import fk

HERE = Path(__file__).resolve().parent
OBJ_DIR = HERE / "obj"
DL = HERE / "downloads"

NAMES = ["client", "module", "servicer", *(f"arm_link_{i}" for i in range(1, 7)),
         "tool", "overlay_red", "overlay_green", "keepout_sphere"]
MODEL_FILES = {n: str(OBJ_DIR / f"{n}.obj") for n in NAMES}

# --- materials ----------------------------------------------------------------
# name: (diffuse texture, normal map or None, Ks, Ns, Ke, meters per texture repeat)
MATS = {
    "solar_cells": ("solar_cells.jpg", "solar_normal.jpg", 0.5, 120, 0.0, None),
    "gold_mli":    ("gold_mli.jpg", "foil_normal.jpg", 0.6, 60, 0.0, 0.7),
    "white_mli":   ("white_mli.jpg", "foil_normal.jpg", 0.2, 20, 0.0, 0.5),
    "kapton":      ("kapton.jpg", "foil_normal.jpg", 0.4, 40, 0.0, 0.3),
    "aluminum":    ("aluminum.jpg", "metal_normal.jpg", 0.7, 90, 0.0, 0.4),
    "dark_metal":  ("dark_metal.jpg", "metal_normal.jpg", 0.4, 60, 0.0, 0.3),
    "carbon":      ("carbon.jpg", "carbon_normal.jpg", 0.5, 80, 0.0, 0.12),
    "radiator":    ("radiator.jpg", None, 0.3, 30, 0.0, 1.0),
    "optics":      ("optics.png", None, 0.9, 200, 0.0, 0.1),
    "overlay_red":   ("overlay_red.png", None, 0.0, 1, 0.8, 1.0),
    "overlay_green": ("overlay_green.png", None, 0.0, 1, 0.8, 1.0),
    "keepout":       ("keepout.png", None, 0.0, 1, 0.25, 4.0),
}
MAT_ALPHA = {"keepout": 0.15}  # MTL d; Vizard's OBJ loader renders d < 1 transparent
CELLS_PER_TILE = 8          # SolarPanel001 has an 8 x 8 cell grid per tile
CELL_M = 0.15               # target cell pitch on a module
MODULE_GAP_M = 0.015        # visual gap between a module edge and its slot edge


# --- textures -------------------------------------------------------------------

def _src(folder, stem, kind):
    p = DL / folder / f"{stem}_1K-JPG_{kind}.jpg"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m models.fetch_assets` first")
    return p


def _gray(p):
    return np.asarray(Image.open(p).convert("L"), float) / 255.0


def _rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), float) / 255.0


def _save(a, path, quality=90):
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    im.save(path, quality=quality) if path.suffix == ".jpg" else im.save(path)


def prepare_textures(out):
    """Copy the CC0 maps used and derive the tinted / generated ones into out."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(_src("solarpanel001", "SolarPanel001", "Color"), out / "solar_cells.jpg")
    shutil.copy(_src("solarpanel001", "SolarPanel001", "NormalGL"), out / "solar_normal.jpg")
    shutil.copy(_src("foil002", "Foil002", "NormalGL"), out / "foil_normal.jpg")
    shutil.copy(_src("metal051a", "Metal051A", "NormalGL"), out / "metal_normal.jpg")
    shutil.copy(_src("fabric004", "Fabric004", "NormalGL"), out / "carbon_normal.jpg")
    shutil.copy(_src("fabric004", "Fabric004", "Color"), out / "carbon.jpg")

    crinkle = _gray(_src("foil002", "Foil002", "Displacement"))[..., None]
    gold = np.array([0.80, 0.58, 0.20]) * (0.45 + 0.85 * crinkle)
    gold += np.clip(crinkle - 0.75, 0, 1) * np.array([1.0, 0.9, 0.6]) * 1.2  # glints
    _save(gold, out / "gold_mli.jpg")
    _save(np.array([0.72, 0.40, 0.08]) * (0.6 + 0.6 * crinkle), out / "kapton.jpg")

    # white beta-cloth / MLI blanket: soft crinkle plus quilting stitches
    white = np.array([0.93, 0.93, 0.91]) * (0.84 + 0.2 * crinkle)
    n = white.shape[0]
    for k in range(0, n, n // 4):
        for d in range(0, n, 12):  # dashed stitch lines on a 4 x 4 grid per tile
            white[k:k + 3, d:d + 7] *= 0.78
            white[d:d + 7, k:k + 3] *= 0.78
    _save(white, out / "white_mli.jpg")

    metal = _rgb(_src("metal051a", "Metal051A", "Color"))
    _save(metal * np.array([0.78, 0.79, 0.81]), out / "aluminum.jpg")
    _save(metal * np.array([0.17, 0.17, 0.19]), out / "dark_metal.jpg")

    # radiator: white silver-Teflon panel with embedded heat-pipe lines
    rng = np.random.default_rng(1)
    rad = 0.9 + 0.03 * rng.standard_normal((512, 512))
    rad = np.asarray(Image.fromarray((rad * 255).clip(0, 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(2)), float) / 255
    for x in range(16, 512, 64):
        rad[:, x:x + 5] *= 0.82
    rad[:6, :] *= 0.6; rad[-6:, :] *= 0.6; rad[:, :6] *= 0.6; rad[:, -6:] *= 0.6
    _save(np.repeat(rad[..., None], 3, 2) * np.array([0.98, 0.99, 1.0]), out / "radiator.jpg")

    g = np.linspace(0, 1, 128)[:, None, None]
    _save(np.array([0.02, 0.04, 0.10]) + 0.25 * g * np.array([0.3, 0.5, 1.0]) * (1 - g),
          out / "optics.png")
    for name, col in (("overlay_red", (255, 32, 24)), ("overlay_green", (40, 255, 70))):
        Image.new("RGB", (64, 64), col).save(out / f"{name}.png")
    Image.new("RGB", (64, 64), (255, 158, 38)).save(out / "keepout.png")


# --- mesh parts -----------------------------------------------------------------

class Part:
    """Unshared-vertex triangle mesh with per-vertex normals and UVs."""

    def __init__(self, V, N, UV, F):
        self.V, self.N, self.UV, self.F = (np.asarray(V, float), np.asarray(N, float),
                                           np.asarray(UV, float), np.asarray(F, int))

    def drop_facing(self, n, tol=0.99):
        """Remove flat triangles whose normal points along n (hidden faces)."""
        keep = [f for f in self.F if self.N[f].mean(0) @ np.asarray(n, float) < tol]
        return Part(self.V, self.N, self.UV, np.array(keep, int).reshape(-1, 3))

    def xf(self, R=np.eye(3), t=(0, 0, 0)):
        R = np.asarray(R, float)
        return Part(self.V @ R.T + np.asarray(t, float), self.N @ R.T, self.UV, self.F)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


AXIS = {"z": np.eye(3), "x": rot_y(np.pi / 2), "y": rot_x(-np.pi / 2),
        "-z": rot_x(np.pi), "-x": rot_y(-np.pi / 2), "-y": rot_x(np.pi / 2)}


def lathe(profile, tex_m=0.4, seg=24, axis="z", at=(0, 0, 0)):
    """Revolve (s, r) points about the local axis (s along it).

    Trace the profile so the outside is on the right (normal = (ds, -dr)):
    e.g. a closed cylinder is (s0,0) (s0,r) (s1,r) (s1,0). Flat along the
    profile (crisp bevels), smooth around it."""
    V, N, UV, F = [], [], [], []
    th = np.linspace(0, 2 * np.pi, seg + 1)
    c, s_ = np.cos(th), np.sin(th)
    arc = 0.0
    rmax = max(r for _, r in profile)
    for (s0, r0), (s1, r1) in zip(profile[:-1], profile[1:]):
        ds, dr = s1 - s0, r1 - r0
        L = np.hypot(ds, dr)
        if L < 1e-12:
            continue
        nr, ns = ds / L, -dr / L
        base = len(V)
        for (s, r, a) in ((s0, r0, arc), (s1, r1, arc + L)):
            for j in range(seg + 1):
                V.append((r * c[j], r * s_[j], s))
                N.append((nr * c[j], nr * s_[j], ns))
                UV.append((j / seg * max(1.0, round(2 * np.pi * rmax / tex_m)), a / tex_m))
        for j in range(seg):
            a, b = base + j, base + j + 1
            cc, d = b + seg + 1, a + seg + 1
            if r0 > 1e-9:
                F.append((a, b, cc))
            if r1 > 1e-9:
                F.append((a, cc, d))
        arc += L
    return Part(V, N, UV, F).xf(AXIS[axis], at)


def tube(p0, p1, r, tex_m=0.3, seg=12, caps=True):
    """Closed cylinder from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    z = d / L
    x = np.cross(z, [0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    prof = [(0, 0), (0, r), (L, r), (L, 0)] if caps else [(0, r), (L, r)]
    return lathe(prof, tex_m, seg).xf(R, p0)


def hull(points, tex_m=0.4):
    """Convex hull (exact, small point sets) with flat normals and box-projected UVs.

    Each supporting plane becomes one fan-triangulated polygon, so coplanar
    faces (a box side) are clean quads rather than joggled slivers."""
    P = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    scale = np.ptp(P, axis=0).max()
    eps = 1e-7 * max(scale, 1e-3)
    planes = []
    n_pts = len(P)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            for k in range(j + 1, n_pts):
                n = np.cross(P[j] - P[i], P[k] - P[i])
                ln = np.linalg.norm(n)
                if ln < 1e-12 * scale * scale:
                    continue
                n /= ln
                d = P @ n - n @ P[i]
                if d.max() <= eps:
                    pass
                elif d.min() >= -eps:
                    n, d = -n, -d
                else:
                    continue
                if not any(abs(n @ q - 1) < 1e-9 and abs(n @ P[i] - e) < eps for q, e in planes):
                    planes.append((n, n @ P[i]))
    V, N = [], []
    for n, e in planes:
        on = P[np.abs(P @ n - e) < eps]
        c = on.mean(0)
        u = on[0] - c
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        order = np.argsort(np.arctan2((on - c) @ w, (on - c) @ u))
        ring = on[order]
        for a in range(1, len(ring) - 1):
            V += [ring[0], ring[a], ring[a + 1]]
            N += [n] * 3
    V, N = np.array(V), np.array(N)
    k = np.abs(N).argmax(1)
    UV = np.where(k[:, None] == 0, V[:, [1, 2]], np.where(k[:, None] == 1, V[:, [0, 2]], V[:, [0, 1]]))
    return Part(V, N, UV / tex_m, np.arange(len(V)).reshape(-1, 3))


def cbox(size, center=(0, 0, 0), bevel=0.0, tex_m=0.4):
    """Box with chamfered edges (bevel in meters)."""
    h = np.asarray(size, float) / 2
    b = min(bevel, *(h * 0.9)) if bevel else 0.0
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                sg = np.array([sx, sy, sz])
                for i in range(3):
                    p = h - b
                    p[i] = h[i]
                    pts.append(sg * p)
    return hull(pts, tex_m).xf(t=center)


def quad(c, u, v, uv_rep=(1, 1)):
    """Single-sided rectangle centered c with half-edge vectors u, v (normal u x v)."""
    c, u, v = (np.asarray(a, float) for a in (c, u, v))
    V = [c - u - v, c + u - v, c + u + v, c - u + v]
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    UV = [(0, 0), (uv_rep[0], 0), (uv_rep[0], uv_rep[1]), (0, uv_rep[1])]
    return Part(V, [n] * 4, UV, [(0, 1, 2), (0, 2, 3)])


def bell(r_throat, r_exit, length, axis, at):
    """Hollow thruster nozzle bell pointing along axis from at."""
    k = np.linspace(0, 1, 7)
    outer = [(length * t, r_throat * 1.4 + (r_exit - r_throat * 1.4) * t ** 1.5) for t in k]
    inner = [(length * t, r_throat + (r_exit * 0.9 - r_throat) * t ** 1.5) for t in k[::-1]]
    return lathe([(0, 0)] + outer + inner + [(0, 0)], 0.2, 20, axis, at)


def housing(half, r, axis, at=(0, 0, 0), flange=0.012, fw=0.016, bevel=0.005):
    """Joint housing: cylinder with beveled end flanges."""
    R = r + flange
    h, b = half, bevel
    prof = [(-h, 0), (-h, R - b), (-h + b, R), (-h + fw, R), (-h + fw, r), (h - fw, r),
            (h - fw, R), (h - b, R), (h, R - b), (h, 0)]
    return lathe(prof, 0.3, 28, axis, at)


class Model:
    def __init__(self):
        self.chunks = []    # (group, material, Part)

    def add(self, group, mat, *parts):
        for p in parts:
            self.chunks.append((group, mat, p))
        return self

    def merge(self, other, group, R=np.eye(3), t=(0, 0, 0)):
        for _, m, p in other.chunks:
            self.chunks.append((group, m, p.xf(R, t)))


# --- writer -----------------------------------------------------------------------

def write_obj(model, name, out):
    """name.obj + name.mtl in out. Each group's vertices are written inside it."""
    groups = {}
    for g, m, p in model.chunks:
        groups.setdefault(g, []).append((m, p))
    used = sorted({m for _, m, _ in model.chunks})
    lines = [f"# {name}: built by models/build_models.py", f"mtllib {name}.mtl"]
    ov = ot = on = 1
    for g, items in groups.items():
        lines.append(f"g {g}")
        for m, p in items:
            if not len(p.F):
                continue
            # dedupe positions, UVs and normals within the chunk (shorter file)
            V, iv = np.unique(np.round(p.V, 5), axis=0, return_inverse=True)
            T, it = np.unique(np.round(p.UV, 4), axis=0, return_inverse=True)
            N, in_ = np.unique(np.round(p.N, 3), axis=0, return_inverse=True)
            iv, it, in_ = iv.ravel() + ov, it.ravel() + ot, in_.ravel() + on
            lines.append(f"usemtl {m}")
            lines += [f"v {a:.5g} {b:.5g} {c:.5g}" for a, b, c in V]
            lines += [f"vt {a:.4g} {b:.4g}" for a, b in T]
            lines += [f"vn {a:.3g} {b:.3g} {c:.3g}" for a, b, c in N]
            lines += ["f " + " ".join(f"{iv[i]}/{it[i]}/{in_[i]}" for i in tri) for tri in p.F]
            ov, ot, on = ov + len(V), ot + len(T), on + len(N)
    (out / f"{name}.obj").write_text("\n".join(lines) + "\n")
    mtl = []
    for m in used:
        tex, nrm, ks, ns, ke, _ = MATS[m]
        mtl += [f"newmtl {m}", "Ka 0.25 0.25 0.25", "Kd 1.0 1.0 1.0",
                f"Ks {ks} {ks} {ks}", f"Ns {ns}", f"Ke {ke} {ke} {ke}" if ke else "Ke 0 0 0",
                f"d {MAT_ALPHA.get(m, 1.0)}", "illum 2", f"map_Kd {tex}"]
        if nrm:
            mtl.append(f"map_Bump {nrm}")
        mtl.append("")
    (out / f"{name}.mtl").write_text("\n".join(mtl))


# --- module ---------------------------------------------------------------------

def tex(mat):
    return MATS[mat][5]


def build_module(cfg):
    """One solar module in its own frame (A axes when seated)."""
    s = module_visual_m(cfg)                            # 1.97 m in a 2 m slot
    t = cfg["client"]["module_thick_m"]
    h, ht = s / 2, t / 2
    fw = 0.04                                           # frame width
    m = Model()
    # panel: aluminum frame, carbon substrate, cell face
    m.add("panel", "aluminum",
          cbox((s, fw, t), (0, h - fw / 2, 0), 0.006, tex("aluminum")),
          cbox((s, fw, t), (0, -h + fw / 2, 0), 0.006, tex("aluminum")),
          cbox((fw, s - 2 * fw, t), (h - fw / 2, 0, 0), 0.006, tex("aluminum")),
          cbox((fw, s - 2 * fw, t), (-h + fw / 2, 0, 0), 0.006, tex("aluminum")))
    sub = cbox((s - 2 * fw + 0.01, s - 2 * fw + 0.01, t * 0.64), (0, 0, -t * 0.08), 0.0, tex("carbon"))
    m.add("panel", "carbon", sub.drop_facing((0, 0, 1)))   # top is hidden under the cells
    w = h - fw
    rep = round(2 * w / CELL_M) / CELLS_PER_TILE
    m.add("panel", "solar_cells", quad((0, 0, ht * 0.5), (w, 0, 0), (0, w, 0), (rep, rep)))
    zb = -t * 0.08 - t * 0.32                            # substrate back face
    # back-face hardware: stiffener ribs, harness, corner latches
    for y in (-0.62, 0.62):
        m.add("hardware", "aluminum", cbox((2 * w - 0.1, 0.03, 0.014), (0, y, zb - 0.007), 0.003))
    for y in (-0.3, 0.3):
        m.add("hardware", "kapton", cbox((1.5, 0.025, 0.004), (0, y, zb - 0.002), 0.0012))
    m.add("hardware", "kapton", cbox((0.025, 0.6, 0.004), (0.2, 0, zb - 0.002), 0.0012))
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.add("hardware", "dark_metal",
                  cbox((0.09, 0.09, 0.016), (sx * (w - 0.08), sy * (w - 0.08), zb - 0.008), 0.004))
    # grapple fixture: base plate, shaft and knob, three cam ramps
    tip = -(ht + cfg["client"]["grapple_gap_m"])
    m.add("fixture", "aluminum", cbox((0.18, 0.18, 0.008), (0, 0, -ht - 0.004), 0.003))
    z0 = -ht - 0.008
    m.add("fixture", "aluminum", lathe([(tip, 0), (tip, 0.02), (tip + 0.003, 0.026),
                                        (tip + 0.008, 0.026), (tip + 0.008, 0.016),
                                        (z0, 0.016), (z0, 0)], 0.1, 20))
    for k in range(3):
        a = 2 * np.pi * k / 3
        pts = [(r * np.cos(a) + d * -np.sin(a), r * np.sin(a) + d * np.cos(a), z)
               for r, zt in ((0.03, z0 - 0.03), (0.085, z0 - 0.006))
               for d in (-0.01, 0.01) for z in (z0, zt)]
        m.add("fixture", "dark_metal", hull(pts, 0.1))
    # data/power connector beside the fixture
    m.add("connector", "dark_metal", cbox((0.09, 0.06, 0.028), (0.2, 0, -ht - 0.014), 0.004))
    for k in (-1, 0, 1):
        m.add("connector", "gold_mli", tube((0.2 + 0.025 * k, 0, -ht - 0.028),
                                            (0.2 + 0.025 * k, 0, -ht - 0.036), 0.006, 0.05, 10))
    return m


def build_overlay(color):
    m = Model()
    m.add("overlay", f"overlay_{color}", cbox((2.02, 2.02, 0.01), (0, 0, -0.04), 0.0, 1.0))
    return m


def build_keepout(cfg, n=48):
    """Keep-out sphere about the array center (single-sided, outward normals)."""
    R = cfg["client"]["keepout_m"]
    th = np.linspace(0, np.pi, n + 1)
    m = Model()
    m.add("keepout", "keepout", lathe(list(zip(-R * np.cos(th), R * np.sin(th))), 4.0, 2 * n))
    return m


# --- client ---------------------------------------------------------------------

# Slot-boundary truss: thin enough that a module (MODULE_GAP_M in from each
# slot edge) pulled straight back or carried in its row never touches it.
BOUNDARY_TUBE_R = 0.012
TRUSS_TOP_Z, TRUSS_KEEL_Z = -0.065, -1.0


def module_visual_m(cfg):
    """Visual edge length of a module: slot minus a gap on each side."""
    return cfg["client"]["slot_m"] - 2 * MODULE_GAP_M


def truss_members(cfg):
    """[(kind, p0, p1, r)] of the client truss in A ("tube" or "node" capsules).

    Top grid on every slot boundary just behind the modules (z > -0.10 keeps
    the carry corridor clear), latch brackets at the grid nodes, and deep
    Warren keels in the planes y = const every 4 m."""
    c = cfg["client"]
    s, n = c["slot_m"], c["grid"]
    half = s * n / 2
    r = BOUNDARY_TUBE_R
    zt, zd = TRUSS_TOP_Z, TRUSS_KEEL_Z
    zl = -c["module_thick_m"] / 2 - 0.012               # latch brackets
    lines = np.arange(-half, half + 1e-9, s)
    out = []
    for v in lines:
        out += [("tube", (-half, v, zt), (half, v, zt), r), ("tube", (v, -half, zt), (v, half, zt), r)]
        for u in lines:
            out += [("node", (u, v, zt), (u, v, zt), r * np.sqrt(2)),
                    ("node", (u, v, zl), (u, v, zl), r * np.sqrt(2))]
    for y in lines[::2]:
        out.append(("tube", (-half, y, zd), (half, y, zd), r))
        for i, x in enumerate(lines):
            out += [("tube", (x, y, zt), (x, y, zd), r), ("node", (x, y, zd), (x, y, zd), r * np.sqrt(2))]
            if i < n:
                a, b = (x, x + s) if i % 2 == 0 else (x + s, x)
                out.append(("tube", (a, y, zt), (b, y, zd), r))
    return [(k, np.array(p0, float), np.array(p1, float), rr) for k, p0, p1, rr in out]


def build_client(cfg):
    c = cfg["client"]
    s, n = c["slot_m"], c["grid"]
    half = s * n / 2
    m = Model()
    module = build_module(cfg)
    target = tuple(c["target"])
    bus_slots = {tuple(b) for b in c["bus_slots"]}
    for rc in module_ids(cfg):
        if rc == target:
            continue
        m.merge(module, f"module_{rc[0]}_{rc[1]}", t=module_center(*rc, cfg))

    for kind, p0, p1, r in truss_members(cfg):
        if kind == "node":
            hw = r / np.sqrt(2)  # square node whose half-diagonal is r
            mat = "dark_metal" if p0[2] > -0.05 else "aluminum"
            m.add("truss_nodes", mat, cbox((2 * hw, 2 * hw, 0.024 if p0[2] > -0.05 else 0.05), p0, 0.003, 0.2))
        else:
            m.add("truss", "carbon", tube(p0, p1, r, tex("carbon"), 10))
    zd = TRUSS_KEEL_Z

    # bus mount face in the center 2 x 2 slots
    rows = [r for r, _ in bus_slots]
    cols = [cc for _, cc in bus_slots]
    lo = module_center(min(rows), min(cols), cfg) - s / 2
    hi = module_center(max(rows), max(cols), cfg) + s / 2
    mid, ext = (lo + hi) / 2, hi - lo - 0.02
    m.add("mount", "white_mli", cbox((ext[0], ext[1], 0.04), (mid[0], mid[1], -0.005), 0.02, tex("white_mli")))
    m.add("mount", "aluminum", lathe([(0.015, 0), (0.015, 0.55), (0.06, 0.55), (0.06, 0.45),
                                      (0.03, 0.45), (0.03, 0)], 0.4, 40, at=(mid[0], mid[1], 0)))
    m.add("mount", "dark_metal", lathe([(0.015, 0.0), (0.015, 0.25), (0.15, 0.18), (0.22, 0.0)],
                                       0.3, 28, at=(mid[0], mid[1], 0)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = (mid[0] + sx * 1.6, mid[1] + sy * 1.6, 0.015)
            m.add("mount", "aluminum", cbox((0.14, 0.14, 0.06), (p[0], p[1], 0.045), 0.01, 0.2))
            m.add("mount", "optics", quad((p[0], p[1], 0.0755), (0.05, 0, 0), (0, 0.05, 0)))

    # standoff: adapter cone from the keel bottom to the bus top, plus struts
    a = c["bus_size_m"]
    zb = -c["bus_offset_m"]
    top = zb + a / 2
    m.add("standoff", "white_mli", lathe([(top, 0), (top, 1.5), (top + 0.1, 1.5), (zd, 1.0), (zd, 0)],
                                         tex("white_mli"), 40))
    m.add("standoff", "aluminum", lathe([(zd - 0.08, 1.0), (zd - 0.08, 1.09), (zd - 0.02, 1.09),
                                         (zd - 0.02, 1.0)], 0.3, 40),
          lathe([(top + 0.08, 1.47), (top + 0.08, 1.56), (top + 0.14, 1.56), (top + 0.14, 1.47)], 0.3, 40))
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.add("standoff", "carbon", tube((sx * 2, sy * 2, zd), (sx * 1.7, sy * 1.7, top), 0.04, 0.12, 12))

    # bus: gold MLI cube with bolted radiator panels, radiator wings, antennas
    m.add("bus", "gold_mli", cbox((a, a, a), (0, 0, zb), 0.08, tex("gold_mli")))
    for ax in (0, 1):
        for sg in (-1, 1):
            n_ = np.zeros(3); n_[ax] = sg
            u = np.zeros(3); u[1 - ax] = 1.6
            v = np.array([0, 0, 1.6])
            ctr = np.array([0, 0, zb]) + n_ * (a / 2 + 0.02)
            size = np.abs(u) * 2 + np.abs(v) * 2 + np.abs(n_) * 0.03
            m.add("radiators", "radiator", cbox(size, ctr, 0.01, 3.2))
            strip = np.abs(u) * 2.1 + np.abs(n_) * 0.05 + [0, 0, 0.06]
            m.add("bus_hardware", "aluminum", cbox(strip, ctr + v * 1.04, 0.01), cbox(strip, ctr - v * 1.04, 0.01))
    for sg in (-1, 1):  # deployable radiator wings along +-x, edge-on to the sun
        x0 = sg * (a / 2 + 0.25)
        m.add("bus_hardware", "aluminum", tube((sg * a / 2, 0, zb), (x0, 0, zb), 0.08, 0.3, 16))
        m.add("bus_hardware", "dark_metal", housing(0.15, 0.1, "y", (x0, 0, zb)))
        for k in range(2):
            xa = x0 + sg * (0.15 + k * 2.9)
            m.add("radiators", "radiator", cbox((2.8, 0.05, 2.6), (xa + sg * 1.4, 0, zb), 0.01, 2.8))
            m.add("bus_hardware", "aluminum", cbox((2.84, 0.07, 0.05), (xa + sg * 1.4, 0, zb + 1.32), 0.01),
                  cbox((2.84, 0.07, 0.05), (xa + sg * 1.4, 0, zb - 1.32), 0.01))
    # high-gain antenna on the anti-sun face, star trackers, thrusters
    zf = zb - a / 2
    m.add("bus_hardware", "aluminum", tube((1.0, -1.0, zf), (1.0, -1.0, zf - 0.8), 0.06, 0.3, 14))
    rr = np.linspace(0, 0.9, 9)
    dish = [(0.32 * (r / 0.9) ** 2, r) for r in rr] + [(0.32 * (r / 0.9) ** 2 + 0.02, r) for r in rr[::-1]]
    m.add("bus_hardware", "white_mli", lathe(dish, 0.4, 36, "-z", (1.0, -1.0, zf - 0.8)))
    m.add("bus_hardware", "dark_metal", tube((1.0, -1.0, zf - 0.8), (1.0, -1.0, zf - 1.15), 0.03, 0.2, 10))
    for p in ((-1.3, 1.3), (-1.3, -1.3)):
        m.add("bus_hardware", "dark_metal", cbox((0.25, 0.25, 0.2), (p[0], p[1], zf - 0.1), 0.02))
        m.add("bus_hardware", "aluminum", lathe([(0, 0.08), (0.18, 0.13), (0.18, 0.12), (0.01, 0.07)],
                                                0.2, 20, "-z", (p[0], p[1], zf - 0.2)))
        m.add("bus_hardware", "optics", quad((p[0], p[1], zf - 0.2 - 0.005), (0.07, 0, 0), (0, -0.07, 0)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = np.array([sx * (a / 2 - 0.3), sy * (a / 2 - 0.3), zf])
            m.add("bus_hardware", "dark_metal", cbox((0.3, 0.3, 0.12), corner - [0, 0, 0.06], 0.02))
            m.add("bus_hardware", "dark_metal", bell(0.025, 0.08, 0.2, "-z", corner - [0, 0, 0.12]))
    return m


# --- servicer -------------------------------------------------------------------

def build_servicer(cfg):
    sx_, sy_, sz_ = cfg["servicer"]["size_m"]
    hx, hy, hz = sx_ / 2, sy_ / 2, sz_ / 2
    m = Model()
    m.add("bus", "gold_mli", cbox((sx_, sy_, sz_), (0, 0, 0), 0.03, tex("gold_mli")))
    # body solar panels (+-y) and an aft radiator
    for sg in (-1, 1):
        m.add("bus_panels", "aluminum", cbox((sx_ - 0.16, 0.012, sz_ - 0.26), (0, sg * (hy + 0.006), 0), 0.004))
        rep = round((sx_ - 0.2) / CELL_M) / CELLS_PER_TILE, round((sz_ - 0.3) / CELL_M) / CELLS_PER_TILE
        m.add("bus_panels", "solar_cells", quad((0, sg * (hy + 0.0125), 0), (-sg * (hx - 0.1), 0, 0),
                                                (0, 0, hz - 0.15), rep))
    m.add("bus_panels", "radiator", cbox((0.012, sy_ - 0.2, sz_ - 0.3), (-hx - 0.006, 0, 0.0), 0.004, 0.8))

    # shoulder pedestal: front face -> link 1 housing
    sh = np.array(cfg["arm"]["shoulder_m"], float)
    x1 = sh[0] - (JOINT_R + 0.01)                        # embeds in the link-1 housing core
    L = x1 - hx
    m.add("pedestal", "aluminum", lathe([(0, 0), (0, 0.17), (0.02, 0.17), (0.02, 0.13), (0.07, 0.095),
                                         (L - 0.025, 0.095), (L - 0.025, 0.11), (L, 0.11), (L, 0)],
                                        0.3, 32, "x", (hx, 0, 0)))
    m.add("pedestal", "dark_metal", lathe([(0.09, 0.0965), (0.13, 0.0965)], 0.2, 32, "x", (hx, 0, 0)))
    for k in range(6):  # gussets
        a = 2 * np.pi * k / 6 + np.pi / 6
        d = np.array([0, np.cos(a), np.sin(a)])
        e = np.array([0, -np.sin(a), np.cos(a)]) * 0.006
        pts = [p + q for p in (np.array([hx + 0.02, 0, 0]) + d * 0.165, np.array([hx + 0.02, 0, 0]) + d * 0.09,
                               np.array([hx + 0.11, 0, 0]) + d * 0.09) for q in (e, -e)]
        m.add("pedestal", "aluminum", hull(pts, 0.2))

    # front sensors: stereo camera, lidar, docking lights
    m.add("sensors", "dark_metal", cbox((0.1, 0.1, 0.08), (hx + 0.05, 0.3, 0.3), 0.01))
    m.add("sensors", "aluminum", lathe([(0, 0.035), (0.04, 0.035), (0.06, 0.045), (0.06, 0.04),
                                        (0.02, 0.028), (0.02, 0)], 0.1, 20, "x", (hx + 0.1, 0.3, 0.3)))
    m.add("sensors", "optics", lathe([(0.022, 0.028), (0.022, 0)], 0.1, 20, "x", (hx + 0.1, 0.3, 0.3)))
    m.add("sensors", "aluminum", lathe([(0, 0), (0, 0.07), (0.1, 0.07), (0.1, 0)], 0.2, 28, "x", (hx, -0.3, 0.3)))
    m.add("sensors", "optics", lathe([(0.03, 0.0715), (0.07, 0.0715)], 0.1, 28, "x", (hx, -0.3, 0.3)))
    m.add("sensors", "dark_metal", lathe([(0.1005, 0.06), (0.1005, 0.0)], 0.1, 28, "x", (hx, -0.3, 0.3)))
    for y in (-0.3, 0.3):
        m.add("sensors", "aluminum", cbox((0.03, 0.09, 0.05), (hx + 0.015, y, -0.32), 0.006))
        m.add("sensors", "optics", quad((hx + 0.0305, y, -0.32), (0, 0.035, 0), (0, 0, 0.018)))

    # RCS pods at the eight bus corners (front pods fire +x, aft -x) and the main engine
    for fx in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                p = np.array([fx * (hx - 0.06), y * (hy - 0.06), z * (hz - 0.06)])
                m.add("thrusters", "dark_metal", cbox((0.1, 0.1, 0.1), p + [fx * 0.05, 0, 0], 0.012))
                m.add("thrusters", "dark_metal", bell(0.006, 0.02, 0.045, "x" if fx > 0 else "-x",
                                                      p + [fx * 0.1, 0, 0]))
                m.add("thrusters", "dark_metal", bell(0.006, 0.02, 0.045, "y" if y > 0 else "-y",
                                                      p + [fx * 0.05, y * 0.05, 0]))
    m.add("thrusters", "dark_metal", bell(0.03, 0.11, 0.22, "-x", (-hx - 0.012, 0, 0)))
    m.add("thrusters", "aluminum", lathe([(0, 0.0), (0, 0.06), (0.04, 0.045), (0.04, 0)], 0.2, 24, "-x", (-hx - 0.012 + 0.04, 0, 0)))

    # spares/stow rack: deck beams on the bus face, corner rails, slot latches
    r = cfg["arm"]["rack"]
    s = cfg["client"]["slot_m"]
    x0, x1r = r["x_m"] - s / 2, r["x_m"] + s / 2
    for sg, name, nslots in ((1, "rack_top", 4), (-1, "rack_bottom", 2)):
        zf = sg * hz
        zs = [sg * (r["z0_m"] + r["dz_m"] * k) for k in range(nslots)]
        zo = zs[-1] + sg * 0.06
        zd = zf + sg * 0.012
        for y in (-0.42, 0.42):
            m.add(name, "aluminum", cbox((s + 0.04, 0.04, 0.024), (r["x_m"], y, zd), 0.004))
        for x in (x0, x1r):
            m.add(name, "aluminum", cbox((0.04, s + 0.08, 0.024), (x, 0, zd), 0.004))
            for y in (-1, 1):
                yp = y * (s / 2 + 0.02)
                m.add(name, "aluminum", cbox((0.04, 0.04, abs(zo - zf)), (x, yp, (zo + zf) / 2), 0.006))
                m.add(name, "dark_metal", cbox((0.05, 0.05, 0.02), (x, yp, zo), 0.005))
                for z in zs:
                    m.add(name, "dark_metal", cbox((0.05, 0.02, 0.016), (x, yp - y * 0.025, z), 0.003))
    return m


# --- arm ------------------------------------------------------------------------

BOOM_R, CORE_R, JOINT_R = 0.055, 0.035, 0.068


def link_spans(cfg):
    """Joint-to-joint offset of links 1-6 along each link's +x (from fk at q = 0)."""
    f = fk(np.zeros(6), cfg)
    return [float((f[i + 1][0] - f[i][0]) @ f[i][1][:, 0]) for i in range(1, 7)]


def _boom(m, L):
    """Canadarm-style boom from x = 0 to L: core stubs, collars, white MLI tube."""
    c0, c1 = 0.075, L - 0.075
    m.add("span", "aluminum", lathe([(0, 0), (0, CORE_R), (c0, CORE_R), (c0, 0)], 0.2, 20, "x"),
          lathe([(c1, 0), (c1, CORE_R), (L, CORE_R), (L, 0)], 0.2, 20, "x"))
    m.add("span", "aluminum", lathe([(c0, 0), (c0, 0.062), (c0 + 0.008, 0.066), (c0 + 0.03, 0.066),
                                     (c0 + 0.03, 0.058), (c0 + 0.05, 0.058), (c0 + 0.05, 0)], 0.2, 24, "x"),
          lathe([(c1 - 0.05, 0), (c1 - 0.05, 0.058), (c1 - 0.03, 0.058), (c1 - 0.03, 0.066),
                 (c1 - 0.008, 0.066), (c1, 0.062), (c1, 0)], 0.2, 24, "x"))
    m.add("span", "white_mli", lathe([(c0 + 0.05, BOOM_R), (c1 - 0.05, BOOM_R)], tex("white_mli"), 24, "x"))
    for xb in np.linspace(c0 + 0.05, c1 - 0.05, 4)[1:-1]:
        m.add("bands", "dark_metal", lathe([(xb - 0.01, 0), (xb - 0.01, BOOM_R + 0.003),
                                            (xb + 0.01, BOOM_R + 0.003), (xb + 0.01, 0)], 0.1, 24, "x"))


def _clevis(m, x, r=JOINT_R):
    """Fork plates either side of the next pitch joint at (x, 0, 0)."""
    for sg in (-1, 1):
        y0 = sg * 0.064
        m.add("clevis", "aluminum", lathe([(0, 0), (0, r), (0.014, r), (0.014, 0)], 0.2, 24,
                                          "y" if sg > 0 else "-y", (x, y0, 0)))
        m.add("clevis", "aluminum", cbox((0.09, 0.014, 0.09), (x - 0.06, y0 + sg * 0.007, 0), 0.004))
        m.add("clevis", "dark_metal", lathe([(0.014, 0), (0.014, 0.03), (0.02, 0.03), (0.02, 0)], 0.1, 16,
                                            "y" if sg > 0 else "-y", (x, y0, 0)))


def _camera(m, at):
    at = np.asarray(at, float)
    m.add("camera", "white_mli", cbox((0.09, 0.06, 0.06), at, 0.008, 0.2))
    m.add("camera", "dark_metal", lathe([(0, 0.022), (0.03, 0.024), (0.03, 0.0)], 0.1, 16, "x", at + [0.045, 0, 0]))
    m.add("camera", "optics", lathe([(0.0305, 0.018), (0.0305, 0)], 0.1, 16, "x", at + [0.045, 0, 0]))
    m.add("camera", "aluminum", cbox((0.03, 0.03, 0.03), at - [0, 0, 0.04], 0.004))


def build_link(i, cfg):
    """arm_link_i in its fk link frame."""
    L = link_spans(cfg)[i - 1]
    m = Model()
    if i == 1:    # shoulder yaw (z)
        m.add("housing", "aluminum", housing(0.1, JOINT_R + 0.01, "z"))
        m.add("housing", "dark_metal", lathe([(-0.02, JOINT_R + 0.012), (0.02, JOINT_R + 0.012)], 0.2, 28, "z"))
    elif i in (2, 3):  # shoulder / elbow pitch (y) + boom
        m.add("housing", "aluminum", housing(0.056, JOINT_R, "y"))
        m.add("housing", "dark_metal", lathe([(-0.012, JOINT_R + 0.002), (0.012, JOINT_R + 0.002)], 0.2, 28, "y"))
        _boom(m, L)
        _clevis(m, L)
        if i == 3:
            _camera(m, (0.24, 0, BOOM_R + 0.045))
    elif i == 4:  # wrist pitch (y)
        m.add("housing", "aluminum", housing(0.056, JOINT_R - 0.004, "y"))
    elif i == 5:  # wrist yaw (z)
        m.add("housing", "aluminum", housing(0.07, 0.05, "z"))
        m.add("housing", "dark_metal", lathe([(-0.012, 0.0515), (0.012, 0.0515)], 0.2, 24, "z"))
    elif i == 6:  # wrist roll (x): roll collar and the mating flange the tool bolts to
        x1 = L - EE_LEN
        m.add("span", "aluminum", lathe([(0, 0), (0, CORE_R), (0.07, CORE_R), (0.07, 0)], 0.2, 20, "x"))
        m.add("span", "dark_metal", lathe([(0.07, 0), (0.07, 0.07), (0.075, 0.075), (x1 - 0.012, 0.075),
                                           (x1 - 0.012, EE_R + 0.004), (x1, EE_R + 0.004), (x1, 0)],
                                          0.2, 28, "x"))
    return m


EE_LEN, EE_R, EE_RIN, EE_DEPTH = 0.19, 0.085, 0.07, 0.11


def build_tool(cfg):
    """End effector in the tool frame: shell from x = -EE_LEN to the open mouth
    at the tool point (x = 0), snare inside, camera, lights and drive ribs on
    the shell. Every part touches the shell; its back mates link 6's flange."""
    m = Model()
    x0 = -EE_LEN
    m.add("effector", "white_mli", lathe([(x0, 0), (x0, EE_R - 0.006), (x0 + 0.006, EE_R), (-0.02, EE_R)],
                                         0.3, 28, "x"))
    m.add("effector", "aluminum", lathe([(-0.02, EE_R), (-0.02, EE_R + 0.006), (-0.004, EE_R + 0.006),
                                         (0, EE_R), (0, EE_RIN), (-EE_DEPTH, EE_RIN), (-EE_DEPTH, 0)],
                                        0.2, 28, "x"))
    for k in range(3):  # snare drive ribs bonded to the shell
        a = 2 * np.pi * k / 3 + np.pi / 6
        d = np.array([0, np.cos(a), np.sin(a)])
        m.add("effector", "dark_metal", cbox((0.14, 0.02, 0.02), (0, 0, 0), 0.004).xf(
            rot_x(a), np.array([-0.1, 0, 0]) + d * (EE_R + 0.006)))
    # snare: hub on the cavity floor, three wires anchored in the cavity wall
    m.add("snare", "dark_metal", lathe([(-EE_DEPTH, 0), (-EE_DEPTH, 0.022), (-0.08, 0.014), (-0.08, 0)],
                                       0.1, 16, "x"))
    pts = [np.array([-0.035, EE_RIN * np.cos(a), EE_RIN * np.sin(a)]) for a in np.radians([90, 210, 330])]
    for k in range(3):
        m.add("snare", "dark_metal", tube(pts[k], pts[(k + 1) % 3], 0.004, 0.05, 8))
    # camera on top of the shell (box sits on the shell surface)
    at = np.array([-0.12, 0.0, EE_R + 0.028])
    m.add("camera", "white_mli", cbox((0.09, 0.06, 0.06), at, 0.008, 0.2))
    m.add("camera", "dark_metal", lathe([(0, 0.022), (0.03, 0.024), (0.03, 0.0)], 0.1, 16, "x", at + [0.045, 0, 0]))
    m.add("camera", "optics", lathe([(0.0305, 0.018), (0.0305, 0)], 0.1, 16, "x", at + [0.045, 0, 0]))
    # lights on the shell sides, facing +x
    for y in (-1, 1):
        c = np.array([-0.04, y * (EE_R + 0.008), 0.0])
        m.add("lights", "aluminum", cbox((0.04, 0.02, 0.03), c, 0.004))
        m.add("lights", "optics", quad(c + [0.0205, 0, 0], (0, 0.008, 0), (0, 0, 0.012)))
    return m


# --- build ------------------------------------------------------------------------

def builders(cfg):
    out = {"client": build_client(cfg), "module": build_module(cfg), "servicer": build_servicer(cfg),
           "tool": build_tool(cfg), "overlay_red": build_overlay("red"),
           "overlay_green": build_overlay("green"), "keepout_sphere": build_keepout(cfg)}
    for i in range(1, 7):
        out[f"arm_link_{i}"] = build_link(i, cfg)
    return out


def build_all(cfg=None, out=OBJ_DIR, previews=True):
    cfg = cfg or load_config()
    out = Path(out)
    prepare_textures(out)
    models = builders(cfg)
    for name, model in models.items():
        write_obj(model, name, out)
    if previews:
        from models.preview import render_all
        render_all(models, out)
    return {n: str(out / f"{n}.obj") for n in models}


def main():
    files = build_all()
    for n, p in files.items():
        p = Path(p)
        nf = sum(1 for l in p.open() if l.startswith("f "))
        print(f"{n:14s} {nf:7d} faces {p.stat().st_size / 1e6:6.2f} MB")
    print(f"previews: {OBJ_DIR / 'previews'}")


if __name__ == "__main__":
    main()
