"""Task 6: textured OBJ models for the Vizard replay."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import load_config

cfg = load_config()
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
G = cfg["gantry"]
N_STAGES = {k: len(G[f"fork_{k}_stages"]) for k in ("W", "F")}
GANTRY_KEYS = {"bridge", "carriage", "fork_W_head", "fork_F_head",
               *(f"fork_{k}_stage{i}" for k in ("W", "F") for i in range(1, N_STAGES[k] + 1))}
KEYS = {"client", "module", "overlay_red", "overlay_green", *GANTRY_KEYS}


@pytest.fixture(scope="module", autouse=True)
def built():
    from models.build_models import build_all
    build_all(previews=False)


def files():
    from models.build_models import MODEL_FILES
    return MODEL_FILES


def read_obj_raw(name):
    """{group: (n, 3) vertex array} exactly as written in the OBJ file (Vizard axes);
    a vertex belongs to the group open when written."""
    groups, cur = {}, "default"
    for line in Path(files()[name]).read_text().splitlines():
        if line.startswith("g "):
            cur = line[2:].strip()
            groups.setdefault(cur, [])
        elif line.startswith("v "):
            groups.setdefault(cur, []).append([float(t) for t in line.split()[1:4]])
    return {g: np.array(v).reshape(-1, 3) for g, v in groups.items() if v}


def read_obj(name):
    """read_obj_raw converted to the body frame: where Vizard 2.4 draws each vertex."""
    from models.build_models import vizard_draws_at
    return {g: vizard_draws_at(v) for g, v in read_obj_raw(name).items()}


def all_verts(name):
    return np.vstack(list(read_obj(name).values()))


def extents(name, group=None):
    v = read_obj(name)[group] if group else all_verts(name)
    return v.max(0) - v.min(0)


def count_module_groups(name):
    return sum(1 for g in read_obj(name) if g.startswith("module_"))


def test_model_keys():
    assert set(files()) == KEYS


def test_all_models_exist():
    assert all(Path(p).exists() and Path(p).with_suffix(".mtl").exists()
               for p in files().values())


def test_textures_referenced():
    for p in files().values():
        mtl = Path(p).with_suffix(".mtl").read_text()
        assert "map_Kd" in mtl
        # every material has a texture, and every texture file is next to the OBJ
        assert mtl.count("newmtl") == mtl.count("map_Kd")
        for line in mtl.splitlines():
            if line.split()[:1] in (["map_Kd"], ["map_Bump"]):
                assert (Path(p).parent / line.split()[-1]).exists(), line


def test_uvs_and_normals_present():
    for p in files().values():
        txt = Path(p).read_text()
        assert "\nvt " in txt and "\nvn " in txt
        f = next(l for l in txt.splitlines() if l.startswith("f "))
        assert all(len(c.split("/")) == 3 and c.split("/")[1] for c in f.split()[1:])


def test_files_small():
    for p in files().values():
        assert Path(p).stat().st_size < 20e6


def test_module_dims():
    # Expected size is the config-driven visual size: slot minus a 1.5 cm gap per
    # side = 1.97 m (0.03 under the 2.0 m slot, so the brief's +-0.02 on 2.0 no
    # longer applies).
    from models.build_models import module_visual_m
    vis = module_visual_m(cfg)
    assert vis == pytest.approx(cfg["client"]["slot_m"] - 0.03)
    assert np.allclose(extents("module", "panel"), [vis, vis, 0.05], atol=0.002)
    assert np.allclose(extents("module")[:2], [vis, vis], atol=0.002)


def test_module_cells_on_front_and_fixture_on_back():
    g = read_obj("module")
    assert abs(g["panel"][:, 2].max() - 0.025) < 0.005
    tip = g["fixture"][:, 2].min()
    expect = -(cfg["client"]["module_thick_m"] / 2 + cfg["client"]["grapple_gap_m"])
    assert abs(tip - expect) < 0.005  # fixture tip at the fork grapple point
    assert np.abs(g["fixture"][:, :2]).max() < 0.15  # centered on the back


def read_faces(name):
    """(n_faces, k, 3) body-frame vertex arrays of every face (all faces in these OBJs
    are polygons)."""
    from models.build_models import vizard_draws_at
    V, F = [], []
    for line in Path(files()[name]).read_text().splitlines():
        if line.startswith("v "):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    V = vizard_draws_at(np.array(V))
    return [V[f] for f in F]


def test_overlays_are_open_frames():
    """I1: each overlay is a glowing outline frame around the slot: no face covers
    the inner ~1.8 m square (the seated panel shows through from both sides), and
    it has a back-side frame (behind the truss grid) and a sun-side rim."""
    o = cfg["viz"]["overlay_frame"]
    hole = 0.9  # half of the 1.8 m square that must stay open
    for n in ("overlay_red", "overlay_green"):
        faces = read_faces(n)
        assert len(faces) > 50
        for f in faces:
            lo, hi = f[:, :2].min(0), f[:, :2].max(0)
            overlap = np.all(hi > -hole) and np.all(lo < hole)
            assert not overlap, (n, lo, hi)
        v = all_verts(n)
        assert v[:, 2].min() == pytest.approx(o["back"]["z"][0], abs=1e-3)
        assert v[:, 2].max() == pytest.approx(o["front"]["z"][1], abs=1e-3)
        assert np.abs(v[:, :2]).max() == pytest.approx(max(o["back"]["outer"], o["front"]["outer"]), abs=1e-3)
        # the rim sits outside the panel footprint, which moves along z through it
        assert np.abs(v[:, :2]).max(1).min() >= module_visual_half() - 1e-9


def module_visual_half():
    from models.build_models import module_visual_m
    return module_visual_m(cfg) / 2


def overlay_boxes_A(name="overlay_red"):
    """Overlay chunk AABBs placed on the target slot (A frame)."""
    from models.build_models import builders
    from src.geometry import module_center
    slot = module_center(*cfg["client"]["target"], cfg)
    return [(c + slot, h) for _, c, h in chunk_boxes(builders(cfg)[name])]


def test_overlay_frame_clears_truss():
    from models.build_models import truss_members
    from src.modules.gantry_motion import seg_aabb_dist
    worst = np.inf
    for c, h in overlay_boxes_A():
        for _, p0, p1, r in truss_members(cfg):  # long grid tubes span the array: no xy cull
            worst = min(worst, seg_aabb_dist(p0, p1, c, h) - r)
    print("overlay-truss min clearance (m):", worst)
    assert worst >= 0.001


def test_overlay_frame_clears_swap():
    """No fork head, fork stage or carried panel (actual model parts) touches the
    overlay frame at any time of the swap (1 s samples)."""
    m = models()
    ov = overlay_boxes_A()
    C = np.array([c for c, _ in ov])
    H = np.array([h for _, h in ov])
    parts = {"fork_W": chunk_boxes(m["fork_W_head"]), "fork_F": chunk_boxes(m["fork_F_head"]),
             "module_target": chunk_boxes(m["module"]), "spare_0": chunk_boxes(m["module"])}
    for k in ("W", "F"):
        for n in stage_layout(cfg, k)["names"]:
            parts[n] = chunk_boxes(m[n])
    worst = np.inf
    for t in np.arange(0.0, SEQ.duration, 1.0):
        f = SEQ.at(t)
        for n, boxes in parts.items():
            p = f.bodies[n][0]
            if n in ("module_target", "spare_0") and np.allclose(p, LAY["slot_center"]):
                continue  # a seated panel sits inside the frame by design (checked below)
            worst = min(worst, min_box_dist(boxes, p, C, H))
    seated = min_box_dist(parts["spare_0"], LAY["slot_center"], C, H)
    print("overlay-swap min clearance (m):", worst, "seated panel:", seated)
    assert worst >= 0.001 and seated >= 0.001


def test_client_has_59_modules():  # M(2,7) is its own body
    assert count_module_groups("client") == 59
    assert "module_2_7" not in read_obj("client")


def test_client_layout():
    from src.geometry import module_center
    g = read_obj("client")
    for (r, c) in [(1, 1), (8, 8), (2, 6), (3, 7)]:
        v = g[f"module_{r}_{c}"]
        mid = (v.max(0) + v.min(0)) / 2
        assert np.allclose(mid[:2], module_center(r, c, cfg)[:2], atol=0.01)
    bus = g["bus"]
    assert np.allclose((bus.max(0) + bus.min(0)) / 2, [0, 0, -cfg["client"]["bus_offset_m"]], atol=0.05)
    assert np.allclose(bus.max(0) - bus.min(0), [cfg["client"]["bus_size_m"]] * 3, atol=0.05)


def test_assets_licensed():
    man = yaml.safe_load((MODELS / "assets_manifest.yaml").read_text())
    doc = (MODELS / "ASSETS.md").read_text()
    assert man["assets"]
    for a in man["assets"]:
        assert a["name"] in doc and a["license"] in doc
        assert {"name", "url", "license", "size_mb", "dest"} <= set(a)


def test_fetch_skips_existing(tmp_path):
    from models.fetch_assets import fetch, load_manifest
    for a in load_manifest():
        (tmp_path / a["dest"]).mkdir()
        (tmp_path / a["dest"] / a["check"]).write_bytes(b"x")
        assert fetch(a, root=tmp_path) is False  # no network touched


def _connected(model, tol=1e-3):
    """True if the chunks' bounding boxes (grown by tol) form one touching cluster."""
    boxes = [(p.V.min(0) - tol, p.V.max(0) + tol) for _, _, p in model.chunks if len(p.F)]
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j, (lo, hi) in enumerate(boxes):
            if j not in seen and np.all(lo <= boxes[i][1]) and np.all(boxes[i][0] <= hi):
                seen.add(j)
                todo.append(j)
    return len(seen) == len(boxes)


def _point_obb_dist(P, c, R, h):
    """Distance from points (n, 3) to the box (center c, axes = columns of R, half h)."""
    q = (P - c) @ R
    d = np.abs(q) - h
    return np.linalg.norm(np.maximum(d, 0), axis=1)


@pytest.mark.slow
def test_truss_clears_carried_modules():
    """Gantry steps pull and drop_in: the module_target and spare_0 boxes
    (visual size, A axes) clear every truss capsule near the target slot."""
    from models.build_models import module_visual_m, truss_members
    from src.geometry import module_center
    from src.modules.gantry_motion import GantrySequence
    seq = GantrySequence(cfg)
    h = np.array([*[module_visual_m(cfg) / 2] * 2, cfg["client"]["module_thick_m"] / 2])
    tgt = module_center(*cfg["client"]["target"], cfg)
    caps = []
    for _, p0, p1, r in truss_members(cfg):
        n = max(2, int(np.linalg.norm(p1 - p0) / 0.003) + 1)
        pts = p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)
        near = np.linalg.norm(pts[:, :2] - tgt[:2], axis=1) < 4.0
        if near.any():
            caps.append((pts[near], r))
    assert len(caps) > 20
    pts = np.vstack([c[0] for c in caps])
    rad = np.concatenate([np.full(len(c[0]), c[1]) for c in caps])
    checked = {"pull": 0, "drop_in": 0}
    for t in np.arange(0.0, seq.duration, 0.5):
        f = seq.at(t)
        if f.step not in checked:
            continue
        body = "module_target" if f.step == "pull" else "spare_0"
        c_A, R_A = f.bodies[body]
        if np.allclose(c_A, tgt):  # seated (the seated panel sits in its slot by design)
            continue
        gap = _point_obb_dist(pts, c_A, R_A, h) - rad
        assert gap.min() > 0.0005, (f.step, t, body, gap.min())
        checked[f.step] += 1
    assert min(checked.values()) > 50

# --- gantry models (task G2) --------------------------------------------------------

from src.modules.gantry_motion import (GantrySequence, aabb_dist, body_half,  # noqa: E402
                                       fork_geometry, gantry_layout, stage_layout)

LAY = gantry_layout(cfg)
BRIDGE_HALF = body_half(cfg)["bridge_A"]
SEQ = GantrySequence(cfg, bridge="B")
TS = np.linspace(0.0, SEQ.duration, 61)


def models():
    from models.build_models import builders
    if not hasattr(models, "m"):
        models.m = builders(cfg)
    return models.m


def anchors():
    from models.build_models import gantry_anchors
    return gantry_anchors(cfg)


def chunk_boxes(model, groups=None, skip=()):
    """[(group, center, half)] AABB per chunk (all gantry bodies use dcm = I)."""
    out = []
    for g, _, p in model.chunks:
        if not len(p.F) or (groups and g not in groups) or g in skip:
            continue
        lo, hi = p.V.min(0), p.V.max(0)
        out.append((g, (lo + hi) / 2, (hi - lo) / 2))
    return out


def group_box(model, group):
    V = np.vstack([p.V for g, _, p in model.chunks if g == group and len(p.F)])
    return V.min(0), V.max(0)


def min_box_dist(boxes, offset, C, H):
    """Min surface distance between boxes [(g, c, h)] shifted by offset and boxes (C, H)."""
    c = np.array([b[1] for b in boxes]) + offset
    h = np.array([b[2] for b in boxes])
    gap = np.maximum(0.0, np.abs(c[:, None] - C[None]) - h[:, None] - H[None])
    return float(np.sqrt((gap ** 2).sum(-1)).min())


def test_gantry_models_connected():
    """Every gantry model is one touching assembly (F stages: each arm on its own)."""
    m = models()
    for n in ("bridge", "carriage", "fork_W_head", "fork_F_head",
              *(f"fork_W_stage{i}" for i in range(1, N_STAGES["W"] + 1))):
        assert _connected(m[n], tol=2e-3), n
    from models.build_models import Model
    for i in range(1, N_STAGES["F"] + 1):
        for sx in (-1, 1):
            half = Model()
            half.chunks = [c for c in m[f"fork_F_stage{i}"].chunks if np.sign(c[2].V[:, 0].mean()) == sx]
            assert half.chunks and _connected(half, tol=2e-3), (i, sx)


def test_rails_in_client():
    g = read_obj("client")
    for name, sy in (("rail_p", 1), ("rail_n", -1)):
        v = g[name]
        mid = (v.max(0) + v.min(0)) / 2
        assert abs(mid[1] - sy * LAY["rail_y"]) < 0.01 and abs(mid[2] - LAY["rail_z"]) < 0.01
        # rail runs rail_end_margin_m past the farthest park position (trucks + end stops)
        xe = max(abs(x) for x in G["park_x"].values()) + G["rail_end_margin_m"]
        assert np.isclose(v[:, 0].min(), -xe, atol=1e-3) and np.isclose(v[:, 0].max(), xe, atol=1e-3)
        w, hgt = G["rail_section"]
        assert np.isclose(v[:, 2].max() - v[:, 2].min(), hgt, atol=1e-3)
        assert np.isclose(np.abs(v[:, 1]).min(), LAY["rail_y"] - w / 2, atol=1e-3)  # inner (guide wheel) face
    assert "rail_supports" in g


def test_rails_supported_by_truss():
    """Rails + supports + truss form one connected structure (nothing floats)."""
    from models.build_models import Model
    c = models()["client"]
    sub = Model()
    sub.chunks = [ch for ch in c.chunks if ch[0] in ("rail_p", "rail_n", "rail_supports", "truss", "truss_nodes")]
    assert _connected(sub, tol=2e-3)


def test_bridge_spans_rails():
    RAIL_H = G["rail_section"][1]
    b = models()["bridge"]
    lo, hi = group_box(b, "beam")
    assert np.allclose(hi[[0, 2]], BRIDGE_HALF[[0, 2]], atol=0.005)   # beam = G1 bridge section
    assert np.allclose(lo[[0, 2]], -BRIDGE_HALF[[0, 2]], atol=0.005)
    top = np.array(anchors()["bridge"]["wheel_top"])
    assert len(top) >= 4
    for sy in (1, -1):
        w = top[np.sign(top[:, 1]) == sy]
        assert len(w) >= 2
        assert np.allclose(w[:, 1], sy * LAY["rail_y"], atol=0.02)   # trucks on the rails (brief: 2 cm)
        assert np.allclose(w[:, 2], RAIL_H / 2, atol=1e-6)
    v = all_verts("bridge")
    assert v[:, 1].max() >= LAY["rail_y"] and v[:, 1].min() <= -LAY["rail_y"]


def test_trucks_ride_on_rails():
    """Review Focus 2: both bridges' wheel contacts lie on the client rail surfaces at all times."""
    g = read_obj("client")
    rails = {1: g["rail_p"], -1: g["rail_n"]}
    a = anchors()["bridge"]
    for t in TS:
        f = SEQ.at(t)
        for n in ("bridge_A", "bridge_B"):
            p = f.bodies[n][0]
            for key, face in (("wheel_top", 1), ("wheel_bottom", -1)):
                for q in np.array(a[key]) + p:
                    r = rails[int(np.sign(q[1]))]
                    assert r[:, 0].min() <= q[0] <= r[:, 0].max()
                    assert r[:, 1].min() <= q[1] <= r[:, 1].max()
                    zf = r[:, 2].max() if face > 0 else r[:, 2].min()
                    assert abs(q[2] - zf) < 1e-3, (t, n, key, q, zf)
            for q in np.array(a["wheel_side"]) + p:           # guide wheels on the inner face
                r = rails[int(np.sign(q[1]))]
                inner = np.abs(r[:, 1]).min()
                assert abs(abs(q[1]) - inner) < 1e-3
                assert r[:, 2].min() <= q[2] <= r[:, 2].max()


def test_carriage_rides_bridge():
    """Review Focus 2: carriage plate sits on the bridge beam; rollers touch the beam sides."""
    m = models()
    blo, bhi = group_box(m["bridge"], "beam")
    plo, phi = group_box(m["carriage"], "plate")
    a = anchors()["carriage"]
    assert abs(phi[2] - (G["magazine_top_z"] - LAY["carriage_z"])) < 1e-3   # +z face = magazine_top_z
    assert abs(plo[2] - (G["bridge_z"] - LAY["carriage_z"])) < 1e-3          # back face = bridge_z
    for t in TS:
        f = SEQ.at(t)
        bp, cp = f.bodies["bridge_B"][0], f.bodies["carriage"][0]
        d = aabb_dist((plo + phi) / 2 + cp, (phi - plo) / 2, (blo + bhi) / 2 + bp, (bhi - blo) / 2)
        assert d < 1e-3, (t, d)
        for q in np.array(a["roller"]) + cp - bp:                # in bridge-local coordinates
            assert abs(abs(q[0]) - bhi[0]) < 1e-3
            assert blo[1] <= q[1] <= bhi[1] and blo[2] <= q[2] <= bhi[2]
        for q in np.array(a["pad"]) + cp - bp:
            assert abs(q[2] - bhi[2]) < 1e-3 and abs(q[0]) <= bhi[0] and blo[1] <= q[1] <= bhi[1]


def test_carriage_parts_clear_beam():
    """No carriage part cuts into the bridge beam (pads and rollers may touch it)."""
    m = models()
    blo, bhi = group_box(m["bridge"], "beam")
    dz = np.array([0.0, 0.0, LAY["carriage_z"] - G["bridge_z"]])   # carriage -> bridge frame
    for g, c, h in chunk_boxes(m["carriage"]):
        pen = np.minimum(c + dz + h, bhi) - np.maximum(c + dz - h, blo)
        assert pen.min() <= 1e-3, (g, c, h)


def test_magazine_bays_match_config():
    m = models()
    a = anchors()["carriage"]
    hp = G["bay_pitch"] / 2
    assert np.isclose(a["bay_y"]["W"], G["waste_side"] * hp) and np.isclose(a["bay_y"]["F"], -G["waste_side"] * hp)
    assert np.isclose(a["bay_y"]["F"] - a["bay_y"]["W"], -G["waste_side"] * G["bay_pitch"])
    from models.build_models import module_visual_m
    ph = module_visual_m(cfg) / 2
    guides = chunk_boxes(m["carriage"], groups={"guides"})
    assert len(guides) >= 8
    for bay in ("W", "F"):
        c = np.array([0, a["bay_y"][bay], 0])
        near = [aabb_dist(gc, gh, c, [ph, ph, 1.0]) for _, gc, gh in guides
                if abs(gc[1] - c[1]) < ph + 0.1]
        assert len(near) >= 4 and min(near) >= 0.005 and max(near) <= 0.03, bay
    # side frames reach past the front of the bottom stack panel: the stack sits in the carriage
    fz = max(c[2] + h[2] for _, c, h in chunk_boxes(m["carriage"], groups={"frame", "guides"}))
    front = G["bay_z0"] + cfg["client"]["module_thick_m"] / 2 - LAY["carriage_z"]
    assert fz >= front + 0.05


def test_stages_meet_carriage_and_head():
    """Review Focus 2: stage 1 on the carriage mount, neighbours overlap, last stage in its head."""
    m = models()
    a = anchors()["carriage"]
    mounts = {"W": np.array(a["mount_W"]), "F": np.array(a["mount_F"])}
    for k in ("W", "F"):
        st = stage_layout(cfg, k)
        hlo, hhi = (np.min([p.V.min(0) for _, _, p in m[f"fork_{k}_head"].chunks], 0),
                    np.max([p.V.max(0) for _, _, p in m[f"fork_{k}_head"].chunks], 0))
        boxes = []
        for n in st["names"]:
            v = np.vstack([p.V for _, _, p in m[n].chunks])
            lo, hi = v.min(0), v.max(0)
            assert abs((hi - lo)[2] - st["length"]) < 2e-3 and abs(hi[2]) < 1e-3, n
            boxes.append((lo, hi))
        for t in TS:
            f = SEQ.at(t)
            cp = f.bodies["carriage"][0]
            tip = f.bodies[f"fork_{k}"][0]
            iv = [(lo + f.bodies[n][0], hi + f.bodies[n][0]) for n, (lo, hi) in zip(st["names"], boxes)]
            tops = [np.array([x, iv[0][0][1] * 0 + tip[1], iv[0][0][2]]) for x in tip[0] + np.array(st["x"])]
            for q in tops:                                       # stage 1 top on a carriage mount
                assert np.min(np.linalg.norm(mounts[k] + cp - q, axis=1)) < 1e-3, (t, k, q)
            for (lo0, hi0), (lo1, hi1) in zip(iv, iv[1:]):
                assert min(hi0[2], hi1[2]) - max(lo0[2], lo1[2]) >= G["stage_overlap_m"] - 2e-3
                assert np.all(lo1[:2] >= lo0[:2] - 1e-9) and np.all(hi1[:2] <= hi0[:2] + 1e-9)  # nested
            z_end = iv[-1][1][2]
            assert hlo[2] + tip[2] <= z_end <= hhi[2] + tip[2], (t, k)   # last stage ends in the head
            assert min(lo[2] for lo, _ in iv) >= G["magazine_top_z"] - 1e-3


def test_fork_heads_match_geometry():
    """Head models fill fork_geometry's head / crossbar boxes and stay inside the fork envelope (5 mm)."""
    m = models()
    tol = 0.005
    for k in ("W", "F"):
        V = np.vstack([p.V for _, _, p in m[f"fork_{k}_head"].chunks])
        lo, hi = V.min(0), V.max(0)
        heads = fork_geometry(cfg, k)
        for c, h in heads:
            assert np.all(lo <= c - h + tol) and np.all(hi >= c + h - tol), (k, c, h)
        env = fork_geometry(cfg, k, LAY["bay_grip_z"])
        inside = np.zeros(len(V), bool)
        for c, h in env:
            inside |= np.all(np.abs(V - c) <= h + tol, axis=1)
        assert inside.all(), (k, V[~inside][:5])


def test_carriage_clears_moving_panels_and_forks():
    """Magazine frame and guides clear every panel and fork box over the sequence."""
    m = models()
    frame = chunk_boxes(m["carriage"], groups={"frame", "guides"})
    from src.modules.gantry_motion import stage_boxes
    ph = body_half(cfg)["spare_0"]
    worst = np.inf
    for t in TS:
        f = SEQ.at(t)
        cp = f.bodies["carriage"][0]
        boxes = [(f.bodies[n][0], ph) for n in ["module_target"] + [f"spare_{i}" for i in range(G["n_fresh"])]]
        for k in ("W", "F"):
            tip = f.bodies[f"fork_{k}"][0]
            boxes += [(tip + o, h) for o, h in fork_geometry(cfg, k)]
            for n, bx in stage_boxes(cfg, k).items():
                boxes += [(f.bodies[n][0] + o, h) for o, h in bx]
        for _, c, h in frame:
            for bc, bh in boxes:
                worst = min(worst, aabb_dist(c + cp, h, bc, bh))
    print("carriage frame min clearance (m):", worst)
    assert worst >= 0.005


def test_gantry_models_clear_client():
    """Bridge and carriage models clear the bus (>= 0.1 m), the truss and the rail supports."""
    from tests.test_gantry_motion import bus_obstacles, obstacle_dist, truss_clearance
    m = models()
    obs = bus_obstacles(cfg)
    sup = chunk_boxes(m["client"], groups={"rail_supports"})
    sup_c, sup_h = np.array([b[1] for b in sup]), np.array([b[2] for b in sup])
    chunks = {n: chunk_boxes(m[n]) for n in ("bridge", "carriage")}
    parts = {}
    for name in ("bridge", "carriage"):    # one AABB per group (conservative, and fast)
        parts[name] = []
        for grp in {g for g, _, _ in m[name].chunks}:
            lo, hi = group_box(m[name], grp)
            parts[name].append((grp, (lo + hi) / 2, (hi - lo) / 2))
    worst = {"bus": np.inf, "truss": np.inf, "supports": np.inf}
    for t in TS[::3]:
        f = SEQ.at(t)
        poses = [("bridge", f.bodies["bridge_A"][0]), ("bridge", f.bodies["bridge_B"][0]),
                 ("carriage", f.bodies["carriage"][0])]
        for name, p in poses:
            for _, c, h in parts[name]:
                worst["bus"] = min(worst["bus"], min(obstacle_dist(o, c + p, h) for o in obs))
                worst["truss"] = min(worst["truss"], truss_clearance(c + p, h))
        for name, p in poses:                    # supports vs every chunk (the trucks pass close by)
            worst["supports"] = min(worst["supports"], min_box_dist(chunks[name], p, sup_c, sup_h))
    print("gantry model clearance (m):", worst)
    assert worst["bus"] >= 0.1 and worst["truss"] >= 0.01 and worst["supports"] >= 0.01


def test_anchors_on_geometry():
    """The exported anchors are real contact points of the built meshes, not just numbers."""
    m = models()
    a = anchors()

    def line_contact(model, groups, q, axis, tol=1e-3):
        """A tread line along `axis` passes through q: vertices match q off-axis and straddle it."""
        V = np.vstack([p.V for g, _, p in model.chunks if g in groups])
        off = [i for i in range(3) if i != axis]
        on = V[np.all(np.abs(V[:, off] - q[off]) < tol, axis=1), axis]
        return len(on) > 0 and on.min() < q[axis] < on.max()

    for key, ax in (("wheel_top", 1), ("wheel_bottom", 1), ("wheel_side", 2)):
        for q in a["bridge"][key]:
            assert line_contact(m["bridge"], {"wheels"}, np.array(q), ax), (key, q)
    for q in a["carriage"]["roller"]:
        assert line_contact(m["carriage"], {"rollers", "drives"}, np.array(q), 2), q
    decks = chunk_boxes(m["carriage"], groups={"deck"})
    for q in np.array(a["carriage"]["pad"]):   # on the underside of a deck plate
        assert any(np.all(np.abs(q[:2] - c[:2]) <= h[:2]) and abs(c[2] - h[2] - q[2]) < 1e-3 for _, c, h in decks), q
    for k in ("W", "F"):                        # each mount surrounds its stage-1 top
        mounts = chunk_boxes(m["carriage"], groups={f"mount_{k}"})
        for q in np.array(a["carriage"][f"mount_{k}"]):
            d = [aabb_dist(c, h, q, np.zeros(3)) for _, c, h in mounts]
            assert sum(x < 0.15 for x in d) >= 3, (k, q)


def test_fork_heads_fill_geometry():
    """Fill, not just containment: each fork_geometry box (head, crossbar, F arm ends) is
    occupied by head-model parts. The parts' AABBs clipped to the box span it within 5 mm on
    every axis, and points every 1 cm along the box's long axis lie inside a part (no gaps)."""
    m = models()
    tol = 0.005
    for k in ("W", "F"):
        parts = np.array([(p.V.min(0), p.V.max(0)) for _, _, p in m[f"fork_{k}_head"].chunks if len(p.F)])
        boxes = list(fork_geometry(cfg, k))
        zmin = parts[:, 0, 2].min()
        for c, h in fork_geometry(cfg, k, LAY["bay_grip_z"])[len(boxes):]:   # arm / mast ends in the head
            top = c[2] + h[2]                                                # head-side end of the member
            if top - zmin > tol:
                lo = max(c[2] - h[2], zmin)
                boxes.append((np.array([c[0], c[1], (lo + top) / 2]), np.array([h[0], h[1], (top - lo) / 2])))
        if k == "F":
            assert len(boxes) == 4                                           # head, crossbar, 2 arm ends
        for c, h in boxes:
            blo, bhi = c - h, c + h
            lo = np.maximum(parts[:, 0], blo)
            hi = np.minimum(parts[:, 1], bhi)
            hit = np.all(hi - lo > 1e-4, axis=1)
            assert hit.any(), (k, c)
            assert np.allclose(lo[hit].min(0), blo, atol=tol) and np.allclose(hi[hit].max(0), bhi, atol=tol), \
                (k, c, h, lo[hit].min(0), hi[hit].max(0))
            ax = int(np.argmax(h))
            n = int(2 * h[ax] / 0.01) + 1
            pts = np.repeat(c[None], n, 0)
            pts[:, ax] = np.linspace(blo[ax] + tol, bhi[ax] - tol, n)
            inside = np.any(np.all((pts[:, None] >= parts[None, :, 0] - 1e-6) &
                                   (pts[:, None] <= parts[None, :, 1] + 1e-6), axis=2), axis=1)
            assert inside.all(), (k, c, pts[~inside][:3])


# --- Vizard frame convention ---------------------------------------------------------
# Evidence (vizfix-report.md in .superpowers/sdd/2026-11-19-rail-gantry/): decompiled IL of
# Vizard 2.4's Assembly-CSharp.dll. Dummiesman.OBJLoader.Load gives the OBJ root a
# (-1, 1, 1) localScale, but ModelDirectoryPanelMethods.FinalizeCustomModel then sets
# localScale = (scale[1], scale[2], scale[0]) = (1, 1, 1) and localPosition =
# (off[1], off[2], -off[0]); OrbitVectorMath.TransformFromBSKCStoUnity is
# u = (b_y, b_z, -b_x). So an OBJ vertex o is drawn at body b = (-o_z, o_x, o_y). Writing the
# client OBJ in body axes (the old convention) put the M(2,7) hole at A (0, 5, -5) and drew
# the parked bridge A along z_A, matching the user's screenshots to ~10 px.

def test_vizard_frame_is_the_decompiled_transform():
    from models.build_models import VIZARD_FROM_BODY, vizard_draws_at
    b = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]])
    to_unity = np.array([[x[1], x[2], -x[0]] for x in b])  # TransformFromBSKCStoUnity
    assert np.allclose(b @ VIZARD_FROM_BODY.T, to_unity)
    assert np.allclose(vizard_draws_at(to_unity), b)
    assert np.linalg.det(VIZARD_FROM_BODY) == pytest.approx(-1.0)  # a mirror: winding flips


def _drawn_A(name, origin_B, com):
    """Where Vizard draws each group of `name`: file vertex -> body (draw map) -> A."""
    from models.build_models import vizard_draws_at
    return {g: vizard_draws_at(v) + origin_B + com for g, v in read_obj_raw(name).items()}


def test_client_gap_is_where_module_target_is_drawn():
    """The client OBJ's missing-module slot, as Vizard draws it (createCustomModel
    offset -com, then the draw map), is exactly where the module_target body and its
    module.obj are drawn: slot M(2,7) at A (5, -5, 0), cells toward +z_A (sun)."""
    from src.geometry import client_mass_props, module_center
    from src.modules.gantry_motion import GantrySequence
    _, com, _ = client_mass_props(cfg, set())
    s, n = cfg["client"]["slot_m"], cfg["client"]["grid"]
    client = _drawn_A("client", -com, com)          # offset = -com_A in viz_setup
    centers = np.array([(v.max(0) + v.min(0)) / 2 for g, v in client.items() if g.startswith("module_")])
    assert np.allclose(centers[:, 2], 0.0, atol=0.05)  # array drawn in the z_A = 0 plane (fixture skews the box)
    lattice = {(int(round((x + s * (n + 1) / 2) / s)), int(round((y + s * (n + 1) / 2) / s)))
               for x, y, _ in centers}                 # (col, row)
    bus = {(c, r) for r, c in cfg["client"]["bus_slots"]}
    gaps = {(c, r) for r in range(1, n + 1) for c in range(1, n + 1)} - lattice - bus
    assert len(gaps) == 1
    (c, r), = gaps
    gap_A = module_center(r, c, cfg)
    p_target, R = GantrySequence(cfg).at(0.0).bodies["module_target"]
    assert np.allclose(gap_A, p_target) and np.array_equal(R, np.eye(3))
    assert (r, c) == tuple(cfg["client"]["target"])
    # module.obj drawn on module_target: centered in the gap, cells (panel top) on +z_A
    mod = _drawn_A("module", p_target - com, com)
    v = np.vstack(list(mod.values()))
    assert np.allclose((v.max(0) + v.min(0))[:2] / 2, gap_A[:2], atol=0.01)
    assert abs(mod["panel"][:, 2].max() - 0.025) < 0.005
    assert mod["fixture"][:, 2].min() < -0.05      # grapple fixture behind the array
    # the bus is drawn behind the array (-z_A), not beside it
    bus_c = (client["bus"].max(0) + client["bus"].min(0)) / 2
    assert np.allclose(bus_c, [0, 0, -cfg["client"]["bus_offset_m"]], atol=0.05)


def test_faces_front_facing_in_unity():
    """Unity's front face is the one numeric cross(b - a, c - a) points to, in the file's
    (left-handed Unity) axes. Every triangle's written normal must agree with it."""
    for name in ("client", "module", "bridge", "carriage"):
        V, N, bad, tot = [], [], 0, 0
        for line in Path(files()[name]).read_text().splitlines():
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("vn "):
                N.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [[int(k) - 1 for k in t.split("/")] for t in line.split()[1:]]
                a, b, c = (np.array(V[i[0]]) for i in idx[:3])
                cr = np.cross(b - a, c - a)
                if np.linalg.norm(cr) < 1e-10:
                    continue
                nrm = np.mean([N[i[2]] for i in idx], axis=0)
                tot += 1
                bad += cr @ nrm < 0
        assert bad <= 0.01 * tot, (name, bad, tot)


# --- stale OBJ detection -----------------------------------------------------------

def test_stale_obj_detected_and_viz_setup_refuses(tmp_path, monkeypatch):
    """OBJs are git-ignored: ones built before the frame fix (no stamp) or missing
    must be flagged, and setup_viz must refuse with a 'build_models' message."""
    from models import build_models as bm
    part = bm.Part(np.eye(3), np.tile([0.0, 0.0, 1.0], (3, 1)), np.zeros((3, 2)), [[0, 1, 2]])
    fresh = bm.Model().add("g", "aluminum", part)
    bm.write_obj(fresh, "x", tmp_path)
    path = tmp_path / "x.obj"
    assert bm.stale_models({"x": str(path)}) == []
    path.write_text("\n".join(l for l in path.read_text().splitlines()
                              if not l.startswith("# vizard-frame")))
    assert bm.stale_models({"x": str(path)}) == ["x"]
    assert bm.stale_models({"y": str(tmp_path / "nope.obj")}) == ["y"]
    import src.viz_setup as vs
    monkeypatch.setattr(vs, "stale_models", lambda: ["client"])
    with pytest.raises(RuntimeError, match="build_models"):
        vs.setup_viz(None, "viz", None, {}, {})


def test_shipped_objs_match_frame_convention():
    from models.build_models import stale_models
    assert stale_models() == []
