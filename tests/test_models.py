"""Task 6: textured OBJ models for the Vizard replay."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import load_config

cfg = load_config()
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
KEYS = {"client", "module", "servicer", "tool", "overlay_red", "overlay_green",
        *(f"arm_link_{i}" for i in range(1, 7))}


@pytest.fixture(scope="module", autouse=True)
def built():
    from models.build_models import build_all
    build_all(previews=False)


def files():
    from models.build_models import MODEL_FILES
    return MODEL_FILES


def read_obj(name):
    """{group: (n, 3) vertex array}; a vertex belongs to the group open when written."""
    groups, cur = {}, "default"
    for line in Path(files()[name]).read_text().splitlines():
        if line.startswith("g "):
            cur = line[2:].strip()
            groups.setdefault(cur, [])
        elif line.startswith("v "):
            groups.setdefault(cur, []).append([float(t) for t in line.split()[1:4]])
    return {g: np.array(v).reshape(-1, 3) for g, v in groups.items() if v}


def all_verts(name):
    return np.vstack(list(read_obj(name).values()))


def extents(name, group=None):
    v = read_obj(name)[group] if group else all_verts(name)
    return v.max(0) - v.min(0)


def count_module_groups(name):
    return sum(1 for g in read_obj(name) if g.startswith("module_"))


def link_lengths():
    """Joint-to-joint offset of each link along its +x axis (fk joint layout)."""
    from models.build_models import link_spans
    return link_spans(cfg)


def extent_along_axis(name):
    """Length of the link's joint-to-joint span member along its +x axis."""
    g = read_obj(name).get("span")
    return 0.0 if g is None else g[:, 0].max() - g[:, 0].min()


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
    assert abs(tip - expect) < 0.005  # fixture tip at the tool point
    assert np.abs(g["fixture"][:, :2]).max() < 0.15  # centered on the back


def test_overlays():
    for n in ("overlay_red", "overlay_green"):
        e = extents(n)
        assert np.allclose(e, [2.02, 2.02, 0.01], atol=0.005)
        assert abs(all_verts(n)[:, 2].mean() + 0.04) < 0.005


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


def test_servicer_bus_and_pedestal():
    g = read_obj("servicer")
    bus = g["bus"]
    assert np.allclose(bus.max(0) - bus.min(0), cfg["servicer"]["size_m"], atol=0.02)
    assert np.allclose((bus.max(0) + bus.min(0)) / 2, 0, atol=0.01)
    ped = g["pedestal"]
    assert abs(ped[:, 0].min() - cfg["servicer"]["size_m"][0] / 2) < 0.02
    assert ped[:, 0].max() <= cfg["arm"]["shoulder_m"][0] + 0.01


def test_servicer_rack_matches_config():
    r = cfg["arm"]["rack"]
    g = read_obj("servicer")
    for side in (1, -1):
        rails = g["rack_top" if side > 0 else "rack_bottom"]
        z = side * rails[:, 2]
        n = 4 if side > 0 else 2
        outer = r["z0_m"] + r["dz_m"] * (n - 1)
        assert z.min() >= cfg["servicer"]["size_m"][2] / 2 - 0.01
        assert z.max() >= outer  # rails reach past the outermost slot
        # rails frame the 2 m panel footprint centered at x_m
        assert abs((rails[:, 0].max() + rails[:, 0].min()) / 2 - r["x_m"]) < 0.05


def test_arm_links_meet():  # Review Focus 2
    from models.build_models import EE_LEN
    L = link_lengths()
    assert L == pytest.approx([0.0, *cfg["arm"]["links_m"][:2], 0.0, 0.0, cfg["arm"]["links_m"][2]])
    for i, Li in enumerate(L):
        name = f"arm_link_{i + 1}"
        if i == 5:
            # link 6 runs wrist -> mating flange; the tool's effector shell runs
            # flange -> tool point. Together they span L6 joint to tool point.
            eff = read_obj("tool")["effector"][:, 0]
            assert abs(eff.max()) < 0.002 and abs(eff.min() + EE_LEN) < 0.002
            assert abs(extent_along_axis(name) + (eff.max() - eff.min()) - Li) < 0.02
            assert abs(read_obj(name)["span"][:, 0].max() - (Li - EE_LEN)) < 0.002  # parts meet
            continue
        assert abs(extent_along_axis(name) - Li) < 0.02, name
        if Li > 0:  # span starts at the link's own joint
            assert abs(read_obj(name)["span"][:, 0].min()) < 0.02
        else:       # joint housing centered on the joint
            v = all_verts(name)
            assert np.all(np.abs((v.max(0) + v.min(0)) / 2) < 0.02), name


def test_tool_at_tool_point():
    v = all_verts("tool")
    assert v[:, 0].max() < 0.01  # nothing past the tool point along +x
    assert v[:, 0].min() > -cfg["arm"]["links_m"][2]


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


def test_tool_and_links_are_one_assembly():
    from models.build_models import build_link, build_tool
    assert _connected(build_tool(cfg))
    for i in range(1, 7):
        assert _connected(build_link(i, cfg)), i


def _point_obb_dist(P, c, R, h):
    """Distance from points (n, 3) to the box (center c, axes = columns of R, half h)."""
    q = (P - c) @ R
    d = np.abs(q) - h
    return np.linalg.norm(np.maximum(d, 0), axis=1)


@pytest.mark.slow
def test_truss_clears_carried_modules():
    """Steps 6.5 (pull) and 6.9 (insert): the module_target and spare_3 boxes
    (visual size) clear every truss capsule near the target slot."""
    from models.build_models import module_visual_m, truss_members
    from src.geometry import module_center
    from src.modules.arm_motion import SwapSequence, berth_pose_A
    seq = SwapSequence(cfg)
    pos, dcm_AS = berth_pose_A(cfg)
    h = np.array([cfg["client"]["module_thick_m"] / 2, *[module_visual_m(cfg) / 2] * 2])
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
    ts = np.arange(0.0, seq.duration, 0.5)
    checked = {"6.5": 0, "6.9": 0}
    for t in ts:
        f = seq.at(t)
        if f.step not in checked:
            continue
        for body in ("module_target", "spare_3"):
            c_S, R_SM = f.bodies[body]
            c_A = dcm_AS @ c_S + pos
            R_AM = dcm_AS @ R_SM
            gap = _point_obb_dist(pts, c_A, R_AM, h) - rad
            assert gap.min() > 0.0005, (f.step, t, body, gap.min())
        checked[f.step] += 1
    assert min(checked.values()) > 50
