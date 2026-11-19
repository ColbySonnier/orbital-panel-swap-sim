"""Quick previews of the built models: a tiny numpy z-buffer rasterizer.

Orthographic, back faces culled (so a wrong winding shows up as holes),
textures sampled per pixel, Lambert + ambient + a soft specular.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# (elev, azim) views per model; default one view
VIEWS = {"client": [(28, -55), (-30, 125)], "servicer": [(22, 35), (-18, -140)],
         "module": [(35, -60), (-40, 120)]}
LIGHT = np.array([0.45, -0.35, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
BG = np.array([5, 7, 13]) / 255


def _textures(out, mats):
    from models.build_models import MATS
    return {m: np.asarray(Image.open(out / MATS[m][0]).convert("RGB").resize((512, 512)), float) / 255
            for m in mats}


def render(model, out, name, elev, azim, path, size=900, final=420):
    tex = _textures(Path(out), {m for _, m, _ in model.chunks})
    e, a = np.radians(elev), np.radians(azim)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(eye, right)
    allv = np.vstack([p.V for _, _, p in model.chunks if len(p.F)])
    sc = np.column_stack([allv @ right, allv @ up])
    lo, hi = sc.min(0), sc.max(0)
    k = 0.9 * size / (hi - lo).max()
    ctr = (lo + hi) / 2
    img = np.tile(BG, (size, size, 1))
    zb = np.full((size, size), -np.inf)
    h = LIGHT + eye
    h /= np.linalg.norm(h)
    for _, m, p in model.chunks:
        if not len(p.F):
            continue
        T = p.V[p.F]
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        vis = fn @ eye > 0
        X = (T @ right - ctr[0]) * k + size / 2
        Y = size / 2 - (T @ up - ctr[1]) * k
        Z = T @ eye
        UV, NN = p.UV[p.F], p.N[p.F]
        timg = tex[m]
        th, tw = timg.shape[:2]
        for i in np.nonzero(vis)[0]:
            x, y = X[i], Y[i]
            x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), size - 1))
            y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), size - 1))
            if x1 < x0 or y1 < y0:
                continue
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(d) < 1e-12:
                continue
            w0 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / d
            w1 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / d
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            z = w0 * Z[i, 0] + w1 * Z[i, 1] + w2 * Z[i, 2]
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            draw = inside & (z > sub)
            if not draw.any():
                continue
            ws = [w[draw] for w in (w0, w1, w2)]
            uv = sum(w[:, None] * UV[i, j] for j, w in enumerate(ws)) % 1.0
            col = timg[((1 - uv[:, 1]) * (th - 1)).astype(int), (uv[:, 0] * (tw - 1)).astype(int)]
            n = sum(w[:, None] * NN[i, j] for j, w in enumerate(ws))
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            lam = np.clip(n @ LIGHT, 0, 1)
            spec = np.clip(n @ h, 0, 1) ** 30 * 0.35
            sub[draw] = z[draw]
            block = img[y0:y1 + 1, x0:x1 + 1]
            block[draw] = np.clip(col * (0.28 + 0.72 * lam)[:, None] + spec[:, None], 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((final, final), Image.LANCZOS)
    ImageDraw.Draw(im).text((8, 6), f"{name}  ({sum(len(p.F) for _, _, p in model.chunks)} tris)",
                            fill=(230, 230, 230))
    im.save(path)
    return path


def render_all(models, out):
    out = Path(out)
    pdir = out / "previews"
    pdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in models.items():
        for k, (el, az) in enumerate(VIEWS.get(name, [(25, -50)])):
            p = pdir / f"{name}{'' if k == 0 else f'_view{k + 1}'}.png"
            paths.append(render(model, out, name, el, az, p))
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = ims[0].width
    cols = 5
    rows = -(-len(ims) // cols)
    sheet = Image.new("RGB", (cols * w, rows * w), tuple(int(c) for c in BG * 255))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * w, (i // cols) * w))
    sheet.save(pdir / "contact_sheet.png")
    return pdir / "contact_sheet.png"
