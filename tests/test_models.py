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
    assert np.allclose(extents("module", "panel"), [2.0, 2.0, 0.05], atol=0.02)
    assert np.allclose(extents("module")[:2], [2.0, 2.0], atol=0.02)


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
    L = link_lengths()
    assert L == pytest.approx([0.0, *cfg["arm"]["links_m"][:2], 0.0, 0.0, cfg["arm"]["links_m"][2]])
    for i, Li in enumerate(L):
        name = f"arm_link_{i + 1}"
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
