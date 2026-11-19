"""Backup-slide plots from a RunLog (matplotlib, light background).

Colours: reference categorical slots 1-3 (blue, orange, aqua), validated
colourblind-safe as a set; neutrals for context; thresholds are dashed and
directly labelled, so nothing is told by colour alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.scenario_servicing import PHASES, STEP_LABELS  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GREEN = "#d03a2f", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SHADE = "#f0efeb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 20, "lines.linewidth": 2.0, "legend.frameon": False,
})


def _minutes(log):
    return log.t / 60.0


def _spans(t, labels):
    """[(t0, t1, label)] contiguous runs of equal labels."""
    labels = list(labels)
    out, a = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[a]:
            out.append((t[a], t[min(i, len(t) - 1)], labels[a]))
            a = i
    return out


def _shade(ax, spans, label=True, min_frac=0.025, fmt=str):
    total = spans[-1][1] - spans[0][0]
    for k, (a, b, p) in enumerate(spans):
        if k % 2:
            ax.axvspan(a, b, color=SHADE, zorder=0, lw=0)
        if label and b - a > min_frac * total:
            ax.text((a + b) / 2, 1.0, fmt(p), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color=INK2)


def _phase_key(fig):
    key = "   ".join(f"{i} {n}" for i, n in enumerate(PHASES))
    fig.text(0.01, 0.005, "Phases: " + key, fontsize=8, color=INK2, ha="left", va="bottom")


def _save(fig, path, key=True):
    if key:
        _phase_key(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def module_ratio(log, path):
    t = _minutes(log)
    ids = [tuple(m) for m in log.module_ids]
    tgt = tuple(log.cfg["client"]["target"])
    i = ids.index(tgt)
    R = log.module_ratios
    others = np.delete(R, i, axis=1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _shade(ax, _spans(t, log.phase))
    ax.fill_between(t, np.nanmin(others, 1), np.nanmax(others, 1), color="#c9c8c2",
                    lw=0, label=f"Other {others.shape[1]} modules (min to max)")
    ax.plot(t, R[:, i], color=BLUE, label="Target M({},{})".format(*tgt))
    p = log.cfg["power"]
    for v, name in ((p["degraded"], "degraded"), (p["service"], "service"), (p["failed"], "failed")):
        ax.axhline(v, color=INK2, ls="--", lw=1)
        ax.text(0.005, v, f"{name} {v:.2f}", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", color=INK2, fontsize=9)
    lo, hi = p["checkout_band"]
    ax.axhspan(lo, hi, color=AQUA, alpha=0.15, lw=0, label=f"Checkout pass band {lo:.2f}-{hi:.2f}")
    ax.set_title("Module power ratio (module power / array median)")
    ax.set_xlabel("Mission time (min)")
    ax.set_ylabel("Power ratio (-)")
    ax.set_ylim(-0.05, 1.35)
    ax.legend(loc="lower right")
    _save(fig, path)


def array_power(log, path):
    t = _minutes(log)
    kw = log.array_power / 1000.0
    pre = log.pre_fault_power / 1000.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _shade(ax, _spans(t, log.phase))
    ax.plot(t, kw, color=BLUE, label="Array power")
    ax.axhline(pre, color=INK2, ls="--", lw=1)
    ax.annotate(f"pre-fault {pre:.1f} kW", (0.3, pre), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 14), textcoords="offset points", va="bottom", ha="center",
                color=INK2, fontsize=9)
    ax.annotate(f"{kw[-1]:.1f} kW ({100 * kw[-1] / pre:.1f}%)", (t[-1], kw[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=INK, fontweight="bold")
    lo = min(kw.min(), pre)
    ax.set_ylim(lo - 0.01 * pre, max(kw.max(), pre) + 0.015 * pre)
    ax.set_title("Array power through fault, swap and checkout")
    ax.set_xlabel("Mission time (min)")
    ax.set_ylabel("Power (kW)")
    _save(fig, path)


def gantry_motion(log, path):
    t = _minutes(log)
    act = (log.phase >= 2) & (log.phase <= 4)
    i0 = max(int(np.flatnonzero(act)[0]) - 30, 0)
    i1 = min(int(np.flatnonzero(act)[-1]) + 30, len(t) - 1)
    sl = slice(i0, i1 + 1)
    steps = [s for s in log.step[sl]]
    abbrev = {v: v.split(",")[0].replace("Align waste bay", "Align") for v in set(steps)}
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    series = [(log.bridge_x[sl], "Bridge B x (m)", [("Bridge B", BLUE)]),
              (log.carriage_y[sl], "Carriage y (m)", [("Carriage", BLUE)]),
              (log.fork_z[sl], "Fork tip z (m)", [("Fork W (waste)", ORANGE), ("Fork F (fresh)", BLUE)])]
    G = log.cfg["gantry"]
    for k, (ax, (y, lab, names)) in enumerate(zip(axs, series)):
        _shade(ax, _spans(t[sl], steps), label=k == 0, min_frac=0.03, fmt=lambda s: abbrev[s])
        y = np.atleast_2d(y.T).T if y.ndim == 1 else y
        for j, (nm, col) in enumerate(names):
            ax.plot(t[sl], y[:, j], color=col, label=nm)
        ax.set_ylabel(lab)
        if len(names) > 1:
            ax.legend(loc="center right")
    axs[0].set_title("Gantry motion in the array frame (step bands)")
    gz, bz = log.fork_z.max(), log.fork_z.min()
    axs[2].axhline(gz, color=INK2, ls="--", lw=1)
    axs[2].text(t[i0], gz, " grapple point at the slot (array side)", va="top", color=INK2, fontsize=9)
    axs[2].text(t[i0], bz, " magazine bay grip", va="bottom", color=INK2, fontsize=9)
    axs[0].axhline(G["park_x"]["B"], color=INK2, ls=":", lw=1)
    axs[2].set_xlabel("Mission time (min)")
    fig.text(0.01, 0.005, "Bands: gantry step. Bridge and carriage <= 5 cm/s, forks <= 2 cm/s "
             "(1 cm/s within 10 cm of contact).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline(log, path):
    t_end = log.t[-1] / 60.0
    ps = log.phase_start
    fig, ax = plt.subplots(figsize=(11, 4.8))
    cols = ["#c9c8c2", "#9fb8d8", BLUE, AQUA, ORANGE, "#c9c8c2"]
    for k in sorted(ps):
        a = ps[k] / 60.0
        b = ps.get(k + 1, log.t[-1]) / 60.0
        ax.barh(1, b - a, left=a, height=0.5, color=cols[k], edgecolor=SURF, lw=1.5)
        ax.text((a + b) / 2, 1, str(k), ha="center", va="center", color=SURF if k in (2, 4) else INK,
                fontweight="bold")
    seen, marks = set(), []
    for t, kind, _ in log.events:
        if kind in seen or kind == "degraded":
            continue
        seen.add(kind)
        marks.append((t / 60.0, kind))
    marks.append((log.cfg["power"]["strike_t_s"] / 60.0, "strike"))
    marks.sort()
    for j, (tm, kind) in enumerate(marks):
        y = 0.58 - 0.12 * (j % 5)
        color = RED if kind in ("strike", "failed", "service_request") else (
            GREEN if kind == "checkout_pass" else INK2)
        ax.plot([tm, tm], [y, 0.75], color=color, lw=1)
        ax.plot(tm, 0.75, "v", color=color, ms=6)
        ax.text(tm, y - 0.02, f"{kind.replace('_', ' ')}\n{tm:.1f} min", ha="center", va="top",
                fontsize=8, color=color)
    ax.set_ylim(-0.15, 1.4)
    ax.set_xlim(0, t_end)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_title("Mission timeline: phases and events")
    ax.set_xlabel("Mission time (min)")
    _save(fig, path)


def make_plots(log, outdir):
    """log.cfg is required. Write module_ratio, array_power, gantry_motion and
    timeline PNGs; return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fn in (("module_ratio", module_ratio), ("array_power", array_power),
                     ("gantry_motion", gantry_motion), ("timeline", timeline)):
        p = outdir / f"{name}.png"
        fn(log, p)
        out.append(p)
    return out
