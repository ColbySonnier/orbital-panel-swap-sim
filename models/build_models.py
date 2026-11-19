"""Build the textured OBJ+MTL models for the Vizard replay.

    python -m models.build_models          # build models/obj/ + previews

Every model is built in the Basilisk body frame of the Vizard body that draws
it (listed below), then write_obj converts it to the axes Vizard 2.4 draws
OBJ vertices in (VIZARD_FROM_BODY; src/viz_setup.py attaches each OBJ with
vizSupport.createCustomModel, rotation 0, scale 1):

  Vizard 2.4 puts an OBJ vertex o in Unity's left-handed space as-is: the
  Dummiesman loader's (-1, 1, 1) root scale is overwritten by
  ModelDirectoryPanelMethods.FinalizeCustomModel (localScale = scale), and
  Unity axes are u = (b_y, b_z, -b_x) of the Basilisk body frame
  (OrbitVectorMath.TransformFromBSKCStoUnity). So a vertex is drawn at body
  b = (-o_z, o_x, o_y): a mirror. write_obj writes o = (b_y, b_z, -b_x)
  (normals too) and reverses each triangle (Unity's front faces wind the
  other way in that improper frame). Evidence: .superpowers/sdd/
  2026-11-19-rail-gantry/vizfix-report.md.

  client          A axes, origin at the array center. Array plane z = 0,
                  cells on +z (sun). 59 modules (M(2,7) is its own body),
                  truss behind the array, 4 m gold bus at z = -4.
  module          A axes when seated, origin at the module center. Cells on
                  +z, grapple fixture on the -z (back) face, its tip at the
                  fork grapple point (z = -(t/2 + grapple_gap)).
  overlay_*       module frame: a glowing outline frame around the slot with an
                  open centre (viz.overlay_frame): back bars + sun-side rim.
  bridge          gantry bridge (A axes), origin on the rail line at the beam
                  center (x, 0, bridge_z); carbon box truss, drive trucks on
                  the client's rails (which are part of `client`).
  carriage        carriage + magazine, origin at the plate center
                  (x, carriage_y, carriage_z); saddles the bridge beam.
  fork_W_head,    fork heads, origin at the fork tip (grapple point); match
  fork_F_head     gantry_motion.fork_geometry (F: head, crossbar, arm fittings).
  fork_*_stageN   telescoping stages, origin at the stage's +z (head) end,
                  running along -z (gantry_motion.stage_layout).

All geometry is procedural: real CC0 textures (ambientCG) on beveled,
lathed and trussed parts. Textures are copied or derived next to the OBJs
and referenced by relative path in each MTL.
"""
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import load_config
from src.geometry import module_center, module_ids
from src.modules.gantry_motion import stage_layout

HERE = Path(__file__).resolve().parent
OBJ_DIR = HERE / "obj"
DL = HERE / "downloads"

def _stage_names():
    cfg = load_config()
    return [n for k in ("W", "F") for n in stage_layout(cfg, k)["names"]]


GANTRY_NAMES = ["bridge", "carriage", "fork_W_head", "fork_F_head", *_stage_names()]
NAMES = ["client", "module", "overlay_red", "overlay_green", *GANTRY_NAMES]
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
}
MAT_ALPHA = {}  # name: MTL d; Vizard's OBJ loader renders d < 1 transparent
CELLS_PER_TILE = 8          # SolarPanel001 has an 8 x 8 cell grid per tile
CELL_M = 0.15               # target cell pitch on a module


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
    key = tuple(np.round(np.asarray(size, float), 9)) + (round(float(bevel), 9), float(tex_m))
    return _cbox_at_origin(key).xf(t=center)


@lru_cache(maxsize=None)
def _cbox_at_origin(key):
    """The hull is O(n^3) in its points, so each box shape is built once (UVs are taken
    at the origin, so a translated copy is identical to building it in place)."""
    *size, bevel, tex_m = key
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
    return hull(pts, tex_m)


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

# Body frame -> Vizard 2.4 OBJ file axes (see the module docstring): o = M @ b,
# M = OrbitVectorMath.TransformFromBSKCStoUnity. det(M) = -1, so write_obj also
# reverses the triangle winding.
VIZARD_FROM_BODY = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def vizard_draws_at(o):
    """Body-frame point(s) Vizard 2.4 draws OBJ file vertex(es) o at (inverse of M)."""
    return np.asarray(o, float) @ VIZARD_FROM_BODY


def write_obj(model, name, out):
    """name.obj + name.mtl in out. Each group's vertices are written inside it,
    converted from the body frame to Vizard's OBJ axes (VIZARD_FROM_BODY)."""
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
            p = Part(p.V @ VIZARD_FROM_BODY.T, p.N @ VIZARD_FROM_BODY.T, p.UV, p.F[:, ::-1])
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


def _frame_bars(inner, outer, z, end):
    """Four bars of a square outline frame: (size, center). `end` is each bar's
    half-length along its edge."""
    w, c, (z0, z1) = outer - inner, (outer + inner) / 2, z
    zc, dz = (z0 + z1) / 2, z1 - z0
    out = []
    for sg in (-1, 1):
        out.append(((2 * end, w, dz), (0.0, sg * c, zc)))   # edges at y = +-c
        out.append(((w, 2 * end, dz), (sg * c, 0.0, zc)))   # edges at x = +-c
    return out


def build_overlay(color, cfg):
    """Slot overlay: a glowing outline frame (viz.overlay_frame), open centre.
    Back bars behind the truss grid, a sun-side rim around the cell face."""
    o = cfg["viz"]["overlay_frame"]
    b, f = o["back"], o["front"]
    m = Model()
    for size, ctr in _frame_bars(b["inner"], b["outer"], b["z"], b["end"]):
        m.add("overlay_back", f"overlay_{color}", cbox(size, ctr, 0.0, 1.0))
    # front rim: x-edge bars run the full outer length, y-edge bars fill between
    w, c = f["outer"] - f["inner"], (f["outer"] + f["inner"]) / 2
    zc, dz = sum(f["z"]) / 2, f["z"][1] - f["z"][0]
    for sg in (-1, 1):
        m.add("overlay_front", f"overlay_{color}",
              cbox((2 * f["outer"], w, dz), (0.0, sg * c, zc), 0.0, 1.0),
              cbox((w, 2 * f["inner"], dz), (sg * c, 0.0, zc), 0.0, 1.0))
    return m


# --- client ---------------------------------------------------------------------

# Slot-boundary truss: thin enough that a module (client.module_gap_m in from each
# slot edge) pulled straight back or carried in its row never touches it.
BOUNDARY_TUBE_R = 0.012
TRUSS_TOP_Z, TRUSS_KEEL_Z = -0.065, -1.0


def module_visual_m(cfg):
    """Visual edge length of a module: slot minus a gap on each side."""
    return cfg["client"]["slot_m"] - 2 * cfg["client"]["module_gap_m"]


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
    add_rails(m, cfg)
    return m


# --- gantry (task G2) ---------------------------------------------------------------
# Frames are the GantrySequence body frames (src/modules/gantry_motion.py), A axes:
#   bridge     origin on the rail line at the beam center (x, 0, bridge_z)
#   carriage   origin at the plate center (x, carriage_y, carriage_z)
#   fork_*_head  origin at the fork tip (grapple point); hardware runs along -z
#   fork_*_stageN  origin at the stage's +z (head-side) end; the stage runs along -z

def rail_dims(cfg):
    """(width y, height z, half length x) of the rails, from gantry config: the rail runs
    rail_end_margin_m past the farthest bridge park position at each end."""
    G = cfg["gantry"]
    w, h = (float(v) for v in G["rail_section"])
    return w, h, max(abs(v) for v in G["park_x"].values()) + G["rail_end_margin_m"]


POST_DY = 0.2          # support posts this far outboard of the rail center line
BEAM_END = 0.18        # bridge beam ends this far inboard of the rail center line
GUIDE_GAP = 0.012      # bay guide inner face to panel edge
CHANNEL = 0.06         # carriage channel wall to beam side face (rollers run in it)


def bx(lo, hi, bevel=0.003, tex_m=0.3):
    """Beveled box from its min / max corners."""
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return cbox(hi - lo, (lo + hi) / 2, bevel, tex_m)


def extrusion(m, group, lo, hi, slot_axes=(), bevel=0.004):
    """Aluminium T-slot extrusion: beveled bar plus dark slot inlays (1 mm proud)
    centered on the faces normal to slot_axes, running its long axis."""
    lo, hi = np.minimum(lo, hi).astype(float), np.maximum(lo, hi).astype(float)
    m.add(group, "aluminum", bx(lo, hi, bevel, tex("aluminum")))
    L = int(np.argmax(hi - lo))
    for a in slot_axes:
        o = 3 - L - a                                   # the third axis
        c, w = (lo[o] + hi[o]) / 2, 0.22 * (hi[o] - lo[o])
        for face, sg in ((hi[a], 1), (lo[a], -1)):
            s0, s1 = lo.copy(), hi.copy()
            s0[L], s1[L] = lo[L] + 2 * bevel, hi[L] - 2 * bevel
            s0[o], s1[o] = c - w, c + w
            s0[a], s1[a] = face - sg * 0.002, face + sg * 0.001
            m.add(group, "dark_metal", bx(s0, s1, 0.0005, 0.2))


def cable_chain(m, group, path, wdir, link=0.05, w=0.07, h=0.03):
    """Drag chain along a dense polyline `path`: overlapping links of two dark side
    plates and an aluminium crossbar; wdir is the chain's width direction."""
    path = np.asarray(path, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    at = lambda d: np.array([np.interp(d, s, path[:, k]) for k in range(3)])
    wdir = np.asarray(wdir, float)
    for i in range(int(s[-1] // link)):
        a, b = at(i * link), at(min((i + 1) * link + 0.004, s[-1]))
        t = (b - a) / np.linalg.norm(b - a)
        R = np.column_stack([t, wdir, np.cross(t, wdir)])
        L = np.linalg.norm(b - a)
        for sg in (-1, 1):
            m.add(group, "dark_metal", cbox((L, 0.008, h), (0, sg * (w / 2 - 0.004), 0), 0.002, 0.1)
                  .xf(R, (a + b) / 2))
        m.add(group, "aluminum", cbox((0.012, w - 0.016, 0.006), (0, 0, h / 2 - 0.003), 0.001, 0.1)
              .xf(R, (a + b) / 2))


def add_rails(m, cfg):
    """Static gantry rails (A frame) at y = +-rail_y, z = rail_z: aluminium extrusion with
    hardened running strips, end stops, and a carbon side truss up to the keel nodes."""
    RAIL_W, RAIL_H, RAIL_X = rail_dims(cfg)
    G, c = cfg["gantry"], cfg["client"]
    ry, rz = G["rail_y"], G["rail_z"]
    hw, hh = RAIL_W / 2, RAIL_H / 2
    half = c["slot_m"] * c["grid"] / 2
    keel_x = np.arange(-half, half + 1e-9, c["slot_m"])
    py = ry + POST_DY
    ends = [-RAIL_X + 0.2, RAIL_X - 0.2]
    for sy, name in ((1, "rail_p"), (-1, "rail_n")):
        Y = lambda a: sy * a
        y0 = Y(ry)
        m.add(name, "aluminum", bx((-RAIL_X, y0 - hw, rz - hh + 0.01), (RAIL_X, y0 + hw, rz + hh - 0.01),
                                   0.006, tex("aluminum")))
        for zz in (rz + hh - 0.01, rz - hh + 0.01):  # running strips: wheel faces at rz +- RAIL_H / 2
            z1 = rz + hh if zz > rz else rz - hh
            m.add(name, "dark_metal", bx((-RAIL_X + 0.01, y0 - 0.04, zz), (RAIL_X - 0.01, y0 + 0.04, z1), 0.001, 0.3))
        for dz in (-0.05, 0.05):    # T-slots on the outer face (supports bolt between them)
            m.add(name, "dark_metal", bx((-RAIL_X + 0.02, Y(ry + hw - 0.002), rz + dz - 0.004),
                                         (RAIL_X - 0.02, Y(ry + hw + 0.002), rz + dz + 0.004), 0.0005, 0.2))
        for sx in (-1, 1):          # end stops with a buffer facing the bridge truck bumpers
            x1 = sx * RAIL_X
            m.add("rail_stops", "dark_metal", bx((x1 - sx * 0.08, y0 - 0.07, rz - 0.22), (x1, y0 + 0.07, rz + 0.11), 0.01))
            m.add("rail_stops", "dark_metal",
                  lathe([(0, 0), (0, 0.03), (0.05, 0.03), (0.06, 0.02), (0.06, 0)], 0.1, 16,
                        "-x" if sx > 0 else "x", (x1 - sx * 0.08, y0, rz - 0.18)))
        # supports: bracket on the outer face -> carbon post at |y| = py -> keel node outrigger
        for x in list(keel_x) + ends:
            m.add("rail_supports", "aluminum", bx((x - 0.06, Y(ry + hw), rz - 0.04), (x + 0.06, Y(py + 0.03), rz + 0.04),
                                                 0.005, 0.2))
            top = -1.0 if x in keel_x else rz + 0.1
            m.add("rail_supports", "carbon", tube((x, Y(py), rz - 0.04), (x, Y(py), top), 0.03, tex("carbon"), 12))
            if x in keel_x:
                m.add("rail_supports", "aluminum", cbox((0.08, 0.08, 0.08), (x, Y(py), -1.0), 0.008, 0.2))
                m.add("rail_supports", "carbon", tube((x, Y(half), -1.0), (x, Y(py), -1.0), 0.025, tex("carbon"), 12))
        xs = list(keel_x)
        m.add("rail_supports", "carbon", tube((xs[0], Y(py), -1.0), (xs[-1], Y(py), -1.0), 0.02, tex("carbon"), 10))
        for i, (xa, xb) in enumerate(zip(xs, xs[1:])):   # Warren diagonals in the post plane
            a, b = (xa, xb) if i % 2 == 0 else (xb, xa)
            m.add("rail_supports", "carbon", tube((a, Y(py), -1.0), (b, Y(py), rz + 0.04), 0.018, tex("carbon"), 10))
        for xe, xk in ((ends[0], xs[0]), (ends[1], xs[-1])):
            m.add("rail_supports", "carbon", tube((xk, Y(py), -1.0), (xe, Y(py), rz + 0.08), 0.018, tex("carbon"), 10))


def build_bridge(cfg):
    """Bridge: carbon box truss spanning y between the rails, with side guide rails for the
    carriage rollers, a top slide rail, an internal harness, and a drive truck at each end
    whose wheels run on the top, bottom and inner faces of the rail."""
    RAIL_W, RAIL_H, RAIL_X = rail_dims(cfg)
    G = cfg["gantry"]
    ry = G["rail_y"]
    hx, hz = np.array(G["bridge_size"], float) / 2
    Y = ry - BEAM_END
    rc = 0.025
    cx, cz = hx - rc, hz - rc
    m = Model()
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add("beam", "carbon", tube((sx * cx, -Y, sz * cz), (sx * cx, Y, sz * cz), rc, tex("carbon"), 14))
    nb = int(round(2 * Y / 0.6))
    ys = np.linspace(-Y, Y, nb + 1)
    rb = 0.011
    for i, y in enumerate(ys):
        for sx in (-1, 1):
            for sz in (-1, 1):
                m.add("beam", "aluminum", cbox((0.05, 0.05, 0.05), (sx * cx, y, sz * cz), 0.006, 0.2))
        m.add("beam", "carbon", tube((-cx, y, cz), (cx, y, cz), rb, 0.12, 8), tube((-cx, y, -cz), (cx, y, -cz), rb, 0.12, 8),
              tube((cx, y, -cz), (cx, y, cz), rb, 0.12, 8), tube((-cx, y, -cz), (-cx, y, cz), rb, 0.12, 8))
        if i < nb:
            ya, yb = (ys[i], ys[i + 1]) if i % 2 == 0 else (ys[i + 1], ys[i])
            m.add("beam", "carbon", tube((-cx, ya, cz), (cx, yb, cz), 0.009, 0.12, 8),
                  tube((-cx, ya, -cz), (cx, yb, -cz), 0.009, 0.12, 8),
                  tube((cx, ya, -cz), (cx, yb, cz), 0.009, 0.12, 8), tube((-cx, yb, -cz), (-cx, ya, cz), 0.009, 0.12, 8))
    for sx in (-1, 1):  # side guide rails: the carriage rollers run on the steel strips at |x| = hx
        x0, x1 = sorted((sx * (hx - 0.02), sx * (hx - 0.004)))
        m.add("beam", "aluminum", bx((x0, -Y, hz - 0.1), (x1, Y, hz - 0.04), 0.002, tex("aluminum")))
        x0, x1 = sorted((sx * (hx - 0.004), sx * hx))
        m.add("beam", "dark_metal", bx((x0, -Y + 0.01, hz - 0.09), (x1, Y - 0.01, hz - 0.05), 0.0005, 0.3))
    m.add("beam", "aluminum", bx((-0.04, -Y, hz - 0.025), (0.04, Y, hz), 0.003, tex("aluminum")))   # top slide rail
    m.add("beam", "dark_metal", bx((0.04, -Y + 0.01, hz - 0.02), (0.06, Y - 0.01, hz), 0.002, 0.3))  # drive rack
    m.add("beam", "kapton", tube((0.05, -Y, -cz + rb + 0.02), (0.05, Y, -cz + rb + 0.02), 0.02, tex("kapton"), 10))

    for sy in (1, -1):   # drive trucks (|y| = a below), wrapping the rail from the inboard side
        Yy = lambda a: sy * a
        rin, rout = ry - RAIL_W / 2, ry + RAIL_W / 2
        e0, e1 = Y, rin - 0.06
        m.add("truck", "aluminum", bx((-0.3, Yy(e0), -0.2), (0.3, Yy(e1), 0.2), 0.012, tex("aluminum")))
        for sz in (1, -1):
            z0, z1 = sorted((sz * 0.16, sz * 0.2))
            m.add("truck", "aluminum", bx((-0.3, Yy(e1), z0), (0.3, Yy(rout + 0.04), z1), 0.008, tex("aluminum")))
            z0, z1 = sorted((sz * 0.075, sz * 0.16))
            m.add("truck", "aluminum", bx((-0.3, Yy(rout + 0.01), z0), (0.3, Yy(rout + 0.04), z1), 0.006, tex("aluminum")))
            zc = sz * (RAIL_H / 2 + 0.045)
            for x in (-0.17, 0.17):   # load wheels on the top / bottom running strips
                prof = [(-0.03, 0), (-0.03, 0.036), (-0.026, 0.045), (0.026, 0.045), (0.03, 0.036), (0.03, 0)]
                m.add("wheels", "dark_metal", lathe(prof, 0.1, 24, "y", (x, Yy(ry), zc)))
                m.add("wheels", "aluminum", lathe([(-0.031, 0), (-0.031, 0.022), (0.031, 0.022), (0.031, 0)],
                                                  0.1, 16, "y", (x, Yy(ry), zc)))
                m.add("wheels", "aluminum", tube((x, Yy(e1), zc), (x, Yy(rout + 0.01), zc), 0.012, 0.1, 10))
            m.add("truck", "dark_metal", lathe([(0, 0), (0, 0.025), (0.05, 0.025), (0.06, 0.016), (0.06, 0)],
                                               0.1, 14, "x" if sz > 0 else "-x", (0.3 * sz, Yy(ry), -0.18)))
        for x in (-0.08, 0.08):        # guide wheels on the rail's inner face
            ac = rin - 0.03
            m.add("wheels", "dark_metal", lathe([(-0.02, 0), (-0.02, 0.03), (0.02, 0.03), (0.02, 0)], 0.1, 20,
                                                "z", (x, Yy(ac), 0.0)))
            m.add("wheels", "aluminum", tube((x, Yy(ac), -0.16), (x, Yy(ac), 0.16), 0.008, 0.1, 8))
        # drive: gearbox on the +x end of the top wheel line, motor running inboard, encoder cap
        m.add("drive", "dark_metal", bx((0.3, Yy(e1 + 0.01), 0.08), (0.4, Yy(rout + 0.04), 0.2), 0.01, 0.2))
        zc = RAIL_H / 2 + 0.045
        m.add("drive", "aluminum", tube((0.35, Yy(e1 + 0.01), zc), (0.35, Yy(e0 - 0.08), zc), 0.045, 0.2, 20))
        for k in range(3):
            a = e0 - 0.01 - k * 0.025
            m.add("drive", "aluminum", tube((0.35, Yy(a), zc), (0.35, Yy(a - 0.008), zc), 0.05, 0.1, 20))
        m.add("drive", "dark_metal", tube((0.35, Yy(e0 - 0.08), zc), (0.35, Yy(e0 - 0.11), zc), 0.035, 0.1, 16))
        m.add("drive", "kapton", tube((0.35, Yy(e0 - 0.095), zc + 0.03), (0.2, Yy(e0), 0.17), 0.008, 0.1, 6))
    return m


def build_carriage(cfg):
    """Carriage + magazine in its body frame: a split plate of T-slot extrusions and carbon
    skins saddling the bridge beam (rollers on the beam's guide rails, slide decks on its top),
    the bay W lift mount on a deck over the beam, the bay F arm mounts on the plate edges,
    side frames with bay guides up past the front of the stack, drives and a cable chain."""
    G, c = cfg["gantry"], cfg["client"]
    hx, hy, hz = np.array(G["carriage_size"], float) / 2
    hp = G["bay_pitch"] / 2
    yW, yF = G["waste_side"] * hp, -G["waste_side"] * hp
    bhx = G["bridge_size"][0] / 2
    wall, ext = bhx + CHANNEL, 0.08
    cz = G["carriage_z"]
    ztop = G["bay_z0"] + c["module_thick_m"] / 2 + 0.075 - cz     # frame top, past the stack front
    ph = module_visual_m(cfg) / 2
    stW, stF = stage_layout(cfg, "W"), stage_layout(cfg, "F")
    m = Model()

    # plate: two halves either side of the beam channel
    for sx in (-1, 1):
        S = lambda a, b: sorted((sx * a, sx * b))
        x0, x1 = S(hx - ext, hx)
        extrusion(m, "plate", (x0, -hy, -hz), (x1, hy, hz), (0,))
        x0, x1 = S(wall, wall + ext)
        extrusion(m, "plate", (x0, -hy, -hz), (x1, hy, hz), (0,))
        x0, x1 = S(wall + ext, hx - ext)
        for sy in (-1, 1):
            y0, y1 = sorted((sy * (hy - ext), sy * hy))
            extrusion(m, "plate", (x0, y0, -hz), (x1, y1, hz), (1,))
        cuts = [-hy + ext, -hp - 0.03, -hp + 0.03, -0.03, 0.03, hp - 0.03, hp + 0.03, hy - ext]
        for ya, yb in zip(cuts[1::2][:-1], cuts[2::2]):
            extrusion(m, "plate", (x0, ya, -hz), (x1, yb, hz), (1,))
        for ya, yb in zip(cuts[0::2], cuts[1::2]):
            m.add("plate", "carbon", bx((x0, ya, hz - 0.02), (x1, yb, hz), 0.002, tex("carbon")),
                  bx((x0, ya, -hz), (x1, yb, -hz + 0.012), 0.002, tex("carbon")))
    # slide decks bridging the channel; their undersides ride the beam's top slide rail
    s1 = stW["sections"][0] / 2 + 0.002
    w = wall + ext
    d = (hz, hz + 0.03)
    m.add("deck", "aluminum", bx((-w, yW - 0.3, d[0]), (-s1[0], yW + 0.3, d[1]), 0.004, tex("aluminum")),
          bx((s1[0], yW - 0.3, d[0]), (w, yW + 0.3, d[1]), 0.004, tex("aluminum")),
          bx((-s1[0], yW + s1[1], d[0]), (s1[0], yW + 0.3, d[1]), 0.002, tex("aluminum")),
          bx((-s1[0], yW - 0.3, d[0]), (s1[0], yW - s1[1], d[1]), 0.002, tex("aluminum")))
    ties = [0.0, -(hy - 0.04), hy - 0.04]
    for y in ties:
        m.add("deck", "aluminum", bx((-w, y - 0.04, d[0]), (w, y + 0.04, d[1]), 0.004, tex("aluminum")))
    pads = [(x, y, hz) for y in ties for x in (-0.03, 0.03)] + \
           [(x, yW + dy, hz) for x in (-0.03, 0.03) for dy in (-0.2, 0.2)]

    # bay W lift mount: collar round stage 1 on the deck, lift motor and gearbox beside it
    o = stW["sections"][0][0] / 2 + 0.045
    z0, z1 = d[1], d[1] + 0.09
    for a, b in (((-o, -o), (o, -s1[1])), ((-o, s1[1]), (o, o)), ((-o, -s1[1]), (-s1[0], s1[1])),
                 ((s1[0], -s1[1]), (o, s1[1]))):
        m.add("mount_W", "aluminum", bx((a[0], yW + a[1], z0), (b[0], yW + b[1], z1), 0.004, tex("aluminum")))
    m.add("mount_W", "dark_metal", bx((o - 0.01, yW - 0.045, z0), (o + 0.13, yW + 0.045, z0 + 0.06), 0.006, 0.2))
    m.add("mount_W", "aluminum", tube((o + 0.075, yW, z0 + 0.06), (o + 0.075, yW, z0 + 0.2), 0.042, 0.2, 20))
    m.add("mount_W", "dark_metal", tube((o + 0.075, yW, z0 + 0.2), (o + 0.075, yW, z0 + 0.23), 0.03, 0.1, 16))
    # bay F arm mounts: U-blocks round each stage-1 arm on the plate edge, lift drives behind
    fx, fy = stF["sections"][0] / 2
    for xa in stF["x"]:
        sx = np.sign(xa)
        S = lambda a, b: sorted((xa + sx * a, xa + sx * b))
        for sy in (-1, 1):
            y0, y1 = sorted((yF + sy * (fy + 0.005), yF + sy * (fy + 0.105)))
            x0, x1 = sorted((sx * (ph + 0.015), sx * hx))
            m.add("mount_F", "aluminum", bx((x0, y0, hz), (x1, y1, hz + 0.1), 0.006, tex("aluminum")))
        x0, x1 = S(fx + 0.004, hx - abs(xa))
        m.add("mount_F", "aluminum", bx((x0, yF - fy - 0.105, hz), (x1, yF + fy + 0.105, hz + 0.1), 0.003, tex("aluminum")))
        xm = sx * 0.7
        m.add("drives", "dark_metal", bx((xm - 0.08, yF - 0.07, -hz - 0.06), (xm + 0.08, yF + 0.07, -hz), 0.006, 0.2))
        m.add("drives", "aluminum", tube((xm, yF, -hz - 0.06), (xm, yF, -hz - 0.24), 0.05, 0.2, 20))
        m.add("drives", "dark_metal", tube((xm, yF, -hz - 0.24), (xm, yF, -hz - 0.27), 0.035, 0.1, 16))
        x0, x1 = sorted((xm, xa))
        m.add("drives", "dark_metal", bx((x0, yF - 0.03, -hz - 0.04), (x1, yF + 0.03, -hz), 0.004, 0.2))

    # magazine side frames: posts, top beams, bracing (clear of both bay footprints)
    posts_y = [-(hy - 0.04), 0.0, hy - 0.04]
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (hx - ext), sx * hx))
        for y in posts_y:
            extrusion(m, "frame", (x0, y - 0.04, hz), (x1, y + 0.04, ztop - 0.06), (0, 1))
        y0, y1 = sorted((G["waste_side"] * 0.04, G["waste_side"] * (hy - ext)))   # bay W side rail
        extrusion(m, "frame", (x0, y0, 0.37), (x1, y1, 0.43), (0,))
    for y in posts_y:
        extrusion(m, "frame", (-hx, y - 0.04, ztop - 0.06), (hx, y + 0.04, ztop), (1,))
        for sg in (-1, 1):
            m.add("frame", "carbon", tube((-sg * (hx - ext), y, hz + 0.02), (sg * (hx - ext), y, ztop - 0.07),
                                          0.015, tex("carbon"), 10))
    # bay guides: L-angles at the 8 bay corners, GUIDE_GAP off the panel edges, tied to the top beams
    g, t, leg = ph + GUIDE_GAP, 0.006, 0.05
    for yc in (yW, yF):
        for sx in (-1, 1):
            for sy in (-1, 1):
                X = lambda a: sx * a
                Yc = lambda a: yc + sy * a
                m.add("guides", "aluminum",
                      bx((X(g), Yc(g - leg), hz), (X(g + t), Yc(g + t), ztop - 0.06), 0.001, tex("aluminum")),
                      bx((X(g - leg), Yc(g), hz), (X(g + t), Yc(g + t), ztop - 0.06), 0.001, tex("aluminum")))
                yb = min(posts_y, key=lambda p: abs(p - Yc(g)))
                ya = yb + 0.04 * np.sign(Yc(g) - yb)
                m.add("frame", "aluminum", bx((X(g - leg), ya, ztop - 0.06), (X(g + t), Yc(g + t), ztop - 0.02),
                                              0.002, 0.2))
    # rollers on the beam's guide rails, carriage y drive (friction wheel + motor), avionics
    zr = G["bridge_z"] + G["bridge_size"][1] / 2 - 0.07 - cz    # guide strip center, carriage frame
    rollers = []
    for y in (-1.7, -0.6, 0.6, 1.7):
        for sx in (-1, 1):
            xc = sx * (bhx + 0.03)
            m.add("rollers", "dark_metal", lathe([(-0.015, 0), (-0.015, 0.03), (0.015, 0.03), (0.015, 0)],
                                                 0.1, 20, "z", (xc, y, zr)))
            m.add("rollers", "aluminum", tube((xc, y, zr - 0.035), (xc, y, zr + 0.045), 0.008, 0.1, 8))
            for za, zb in ((zr + 0.025, zr + 0.045), (zr - 0.035, zr - 0.02)):
                x0, x1 = sorted((sx * (bhx + 0.01), sx * wall))
                m.add("rollers", "aluminum", bx((x0, y - 0.045, za), (x1, y + 0.045, zb), 0.002, 0.2))
            rollers.append((sx * bhx, y, zr))
    yd = 0.3
    xd = bhx + 0.03
    m.add("drives", "dark_metal", lathe([(-0.016, 0), (-0.016, 0.03), (0.016, 0.03), (0.016, 0)], 0.1, 20, "z",
                                        (xd, yd, zr)))
    rollers.append((bhx, yd, zr))
    m.add("drives", "aluminum", tube((xd, yd, -hz - 0.07), (xd, yd, zr), 0.01, 0.1, 8))
    m.add("drives", "dark_metal", bx((xd - 0.02, yd - 0.08, -hz - 0.07), (wall + 0.27, yd + 0.08, -hz), 0.006, 0.2))
    m.add("drives", "aluminum", tube((wall + 0.19, yd, -hz - 0.07), (wall + 0.19, yd, -hz - 0.26), 0.045, 0.2, 20))
    m.add("drives", "dark_metal", tube((wall + 0.19, yd, -hz - 0.26), (wall + 0.19, yd, -hz - 0.29), 0.03, 0.1, 16))
    m.add("drives", "white_mli", bx((-0.95, -0.45, -hz - 0.16), (-0.45, 0.45, -hz), 0.015, tex("white_mli")))
    m.add("drives", "radiator", bx((-0.93, -0.43, -hz - 0.17), (-0.47, 0.43, -hz - 0.16), 0.002, 0.5))
    # cable chain: service loop from the avionics box, hanging under the plate
    x0 = -0.7
    zu, zl, r = -hz - 0.015, -hz - 0.015 - 0.18, 0.09
    ya, yb, yc = -0.44, -1.75, -0.9
    up = np.column_stack([np.full(30, x0), np.linspace(ya, yb, 30), np.full(30, zu)])
    th = np.linspace(0, np.pi, 16)
    bend = np.column_stack([np.full(16, x0), yb - r * np.sin(th), (zu + zl) / 2 + r * np.cos(th)])
    lo = np.column_stack([np.full(20, x0), np.linspace(yb, yc, 20), np.full(20, zl)])
    cable_chain(m, "cable_chain", np.vstack([up, bend[1:], lo[1:]]), (1, 0, 0))
    m.add("cable_chain", "aluminum", bx((x0 - 0.05, yc - 0.03, zl - 0.02), (x0 + 0.05, yc + 0.01, -hz), 0.004, 0.2))
    # bay camera on the center top beam, looking at the array
    m.add("camera", "white_mli", bx((-0.05, -0.035, ztop), (0.05, 0.035, ztop + 0.05), 0.006, 0.2))
    m.add("camera", "dark_metal", lathe([(0, 0.022), (0.02, 0.024), (0.02, 0.0)], 0.1, 16, "z", (0, 0, ztop + 0.05)))
    m.add("camera", "optics", lathe([(0.0205, 0.018), (0.0205, 0)], 0.1, 16, "z", (0, 0, ztop + 0.05)))
    m.anchors = {"roller": rollers, "pad": pads, "mount_W": [(x, yW, hz) for x in stW["x"]],
                 "mount_F": [(x, yF, hz) for x in stF["x"]], "bay_y": {"W": yW, "F": yF}}
    return m


def _grapple_face(m, group, r_in=0.03, r_out=0.055, jaw_r=0.075, z=0.0):
    """Grapple receptacle on a head's +z face (at z): aluminium ring and three cam jaws
    that close on the module fixture knob."""
    m.add(group, "aluminum", lathe([(z + 0.0005, r_out), (z + 0.0005, r_in)], 0.1, 28))
    m.add(group, "dark_metal", lathe([(z + 0.0003, r_in), (z + 0.0003, 0.0)], 0.1, 20))
    for k in range(3):
        a = 2 * np.pi * k / 3 + np.pi / 2
        m.add(group, "dark_metal", cbox((0.032, 0.02, 0.008), (0, 0, 0), 0.002, 0.1)
              .xf(rot_z(a), (jaw_r * np.cos(a), jaw_r * np.sin(a), z - 0.002)))


def build_fork_W_head(cfg):
    """Fork W grapple head in the fork frame (tip at the origin, head behind it in -z)."""
    G = cfg["gantry"]
    h = np.array(G["fork_W_head"]["size"], float) / 2
    oz = G["fork_W_head"]["offset"][2]
    z0, z1 = oz - h[2], oz + h[2]
    m = Model()
    m.add("head", "aluminum", bx((-h[0], -h[1], z0), (h[0], h[1], z1 - 0.014), 0.008, tex("aluminum")))
    m.add("head", "dark_metal", bx((-h[0] + 0.005, -h[1] + 0.005, z1 - 0.014), (h[0] - 0.005, h[1] - 0.005, z1),
                                   0.003, 0.2))
    _grapple_face(m, "head", z=z1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.add("head", "aluminum", lathe([(z1 - 0.002, 0), (z1 - 0.002, 0.008), (z1 + 0.002, 0.007), (z1 + 0.002, 0)],
                                            0.05, 8, at=(sx * 0.072, sy * 0.072, 0)))
    m.add("head", "dark_metal", bx((h[0], -0.03, z0 + 0.01), (h[0] + 0.004, 0.03, z0 + 0.035), 0.001, 0.1))
    m.add("head", "optics", quad((-h[0] - 0.0004, 0, oz), (0, 0, 0.01), (0, 0.022, 0)))
    s = stage_layout(cfg, "W")["sections"][-1] / 2      # socket collar round the last stage
    for a, b in (((-s[0] - 0.014, -s[1] - 0.014), (s[0] + 0.014, -s[1] - 0.001)),
                 ((-s[0] - 0.014, s[1] + 0.001), (s[0] + 0.014, s[1] + 0.014)),
                 ((-s[0] - 0.014, -s[1]), (-s[0] - 0.001, s[1])), ((s[0] + 0.001, -s[1]), (s[0] + 0.014, s[1]))):
        m.add("head", "dark_metal", bx((a[0], a[1], z0 - 0.004), (b[0], b[1], z0 + 0.002), 0.001, 0.1))
    return m


def build_fork_F_head(cfg):
    """Fork F head in the fork frame: thin grapple head on a carbon / aluminium crossbar
    that carries the panel from behind, with end fittings where the arms socket in."""
    G = cfg["gantry"]
    hd, cb = G["fork_F_head"], G["fork_F_crossbar"]
    hh, ho = np.array(hd["size"], float) / 2, np.array(hd["offset"], float)
    ch, co = np.array(cb["size"], float) / 2, np.array(cb["offset"], float)
    arm = G["fork_F_arm"]
    ax, ahx = arm["offset"][0], arm["size"][0] / 2
    z0, z1 = co[2] - ch[2], co[2] + ch[2]
    xin = ax - ahx                                  # inner face of the arms
    m = Model()
    m.add("head", "aluminum", bx(ho - hh, ho + hh, 0.004, tex("aluminum")))
    _grapple_face(m, "head", z=ho[2] + hh[2])
    m.add("crossbar", "aluminum", bx((-0.12, -ch[1], z0), (0.12, ch[1], z1), 0.005, tex("aluminum")))
    m.add("crossbar", "carbon", bx((-xin, -ch[1] + 0.015, z0 + 0.005), (xin, ch[1] - 0.015, z1 - 0.005), 0.002,
                                   tex("carbon")))
    for sy in (-1, 1):
        y0, y1 = sorted((sy * (ch[1] - 0.015), sy * ch[1]))
        m.add("crossbar", "aluminum", bx((-xin, y0, z0), (xin, y1, z1), 0.003, tex("aluminum")))
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (xin - 0.004), sx * (ax + ahx + 0.005)))
        m.add("fittings", "aluminum", bx((x0, -ch[1] - 0.0045, z0 - 0.035), (x1, ch[1] + 0.0045, z1), 0.005, tex("aluminum")))
        for y in (-0.06, 0.06):
            m.add("fittings", "dark_metal", lathe([(0, 0), (0, 0.008), (0.004, 0.007), (0.004, 0)], 0.05, 8, "-z",
                                                  (sx * ax, y, z0 - 0.035)))
    return m


def build_stage(cfg, which, i):
    """Telescoping stage i of fork `which` in its stage frame (origin at the +z end).
    W: carbon square tube with aluminium wear collars. F: two aluminium T-slot arms
    with dark collars, at the fork_F_arm x offsets."""
    st = stage_layout(cfg, which)
    sx_, sy_ = st["sections"][i - 1]
    L, e, cl = st["length"], 0.003, 0.03
    y = st["y"]
    m = Model()
    for x in st["x"]:
        lo, hi = np.array([x - sx_ / 2, y - sy_ / 2]), np.array([x + sx_ / 2, y + sy_ / 2])
        core, collar = ("carbon", "aluminum") if which == "W" else ("aluminum", "dark_metal")
        m.add("tube", core, bx((*(lo + e), -L + cl), (*(hi - e), -cl), 0.003, tex(core)))
        for za, zb in ((-cl, 0.0), (-L, -L + cl)):
            m.add("collars", collar, bx((*lo, za), (*hi, zb), 0.002, 0.1))
        if which == "F":
            for sg in (-1, 1):
                yy = sorted((y + sg * (sy_ / 2 - e - 0.002), y + sg * (sy_ / 2 - e + 0.001)))
                m.add("tube", "dark_metal", bx((x - 0.3 * sx_, yy[0], -L + cl + 0.01), (x + 0.3 * sx_, yy[1], -cl - 0.01),
                                               0.0005, 0.1))
    return m


def gantry_anchors(cfg):
    """Contact points of the gantry models in their own frames (for connectedness tests)."""
    RAIL_W, RAIL_H, RAIL_X = rail_dims(cfg)
    G = cfg["gantry"]
    ry = G["rail_y"]
    rin = ry - RAIL_W / 2
    br = {"wheel_top": [(x, sy * ry, RAIL_H / 2) for sy in (1, -1) for x in (-0.17, 0.17)],
          "wheel_bottom": [(x, sy * ry, -RAIL_H / 2) for sy in (1, -1) for x in (-0.17, 0.17)],
          "wheel_side": [(x, sy * rin, 0.0) for sy in (1, -1) for x in (-0.08, 0.08)]}
    return {"bridge": br, "carriage": build_carriage(cfg).anchors}


# --- build ------------------------------------------------------------------------

def builders(cfg):
    out = {"client": build_client(cfg), "module": build_module(cfg),
           "overlay_red": build_overlay("red", cfg), "overlay_green": build_overlay("green", cfg)}
    out.update({"bridge": build_bridge(cfg), "carriage": build_carriage(cfg),
                "fork_W_head": build_fork_W_head(cfg), "fork_F_head": build_fork_F_head(cfg)})
    for k in ("W", "F"):
        for i, n in enumerate(stage_layout(cfg, k)["names"], 1):
            out[n] = build_stage(cfg, k, i)
    return out


def build_all(cfg=None, out=OBJ_DIR, previews=True):
    cfg = cfg or load_config()
    out = Path(out)
    prepare_textures(out)
    models = builders(cfg)
    for name, model in models.items():
        write_obj(model, name, out)
    if previews:
        from models.preview import render_all, render_assembled
        render_all(models, out)
        render_assembled(cfg, models, out)
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
