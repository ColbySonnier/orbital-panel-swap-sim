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

from src.modules.mission_manager import PHASES  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def _phase_spans(log):
    """[(t0_h, t1_h, phase)] contiguous phase intervals."""
    t, ph = log.t / 3600.0, np.asarray(log.phase)
    cut = np.flatnonzero(np.diff(ph)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(t) - 1]])
    return [(t[a], t[b], int(ph[a])) for a, b in zip(starts, ends)]


def _shade_phases(ax, log, label=True):
    for k, (a, b, p) in enumerate(_phase_spans(log)):
        if k % 2:
            ax.axvspan(a, b, color=SHADE, zorder=0, lw=0)
        if label and b - a > 0.02 * (log.t[-1] / 3600):
            ax.text((a + b) / 2, 1.0, str(p), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color=INK2)


def _phase_key(fig):
    key = "   ".join(f"{i} {n}" for i, n in enumerate(PHASES))
    fig.text(0.01, 0.005, "Phases: " + key, fontsize=8, color=INK2, ha="left", va="bottom")


def hill_path(log, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x, y = log.rel_H[:, 0], log.rel_H[:, 1]
    out = np.asarray(log.phase) >= 8
    ax.plot(y[~out], x[~out], color=BLUE, label="Approach (phases 0-7)")
    ax.plot(y[out], x[out], color=ORANGE, label="Departure and home (8-9)")
    w = log.cfg["waypoints"] if log.cfg else {"H0": [0, -2000, 0], "H1": [0, -200, 0], "H2": [0, -30, 0]}
    for k in ("H0", "H1", "H2"):
        ax.plot(w[k][1], w[k][0], "o", ms=8, color=INK, mec=SURF, mew=2, zorder=5)
        ax.annotate(k, (w[k][1], w[k][0]), xytext=(0, 10), textcoords="offset points",
                    ha="center", color=INK, fontweight="bold")
    ax.plot(0, 0, "s", ms=9, color=INK2, mec=SURF, mew=2, zorder=5)
    ax.annotate("Client", (0, 0), xytext=(8, -14), textcoords="offset points", color=INK2)
    ax.set_title("Servicer path relative to the client (Hill frame, orbit plane)")
    ax.set_xlabel("Along-track y (m)   [+ = ahead of the client]")
    ax.set_ylabel("Radial x (m)   [+ = up]")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def range_speed(log, path):
    t = log.t / 3600.0
    rng = np.linalg.norm(log.rel_H[:, :3], axis=1)
    # speed: Hill speed in Hill-frame phases, array-frame speed close in
    spd = np.where((np.asarray(log.phase) >= 4) & (np.asarray(log.phase) <= 7),
                   np.linalg.norm(log.rel_A[:, 3:], axis=1),
                   np.linalg.norm(log.rel_H[:, 3:], axis=1))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.semilogy(t, np.maximum(rng, 0.1), color=BLUE)
    a1.set_title("Range to the client and relative speed")
    a1.set_ylabel("Range (m, log scale)")
    a2.semilogy(t, np.maximum(100 * spd, 1e-3), color=ORANGE)
    a2.set_ylabel("Relative speed (cm/s, log scale)")
    a2.set_xlabel("Mission time (h)")
    for ax in (a1, a2):
        _shade_phases(ax, log, label=ax is a1)
    _phase_key(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def module_ratio(log, path):
    t = log.t / 3600.0
    ids = [tuple(m) for m in log.module_ids]
    tgt = tuple(log.cfg["client"]["target"]) if log.cfg else (2, 7)
    i = ids.index(tgt)
    R = log.module_ratios
    others = np.delete(R, i, axis=1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(t, np.nanmin(others, 1), np.nanmax(others, 1), color="#c9c8c2",
                    lw=0, label="Other 59 modules (min to max)")
    ax.plot(t, R[:, i], color=BLUE, label="Target M({},{})".format(*tgt))
    p = log.cfg["power"] if log.cfg else {"degraded": 0.85, "service": 0.7, "failed": 0.5,
                                           "checkout_band": [0.98, 1.2]}
    for v, name in ((p["degraded"], "degraded"), (p["service"], "service"), (p["failed"], "failed")):
        ax.axhline(v, color=INK2, ls="--", lw=1)
        ax.text(0.005, v, f"{name} {v:.2f}", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", color=INK2, fontsize=9)
    lo, hi = p["checkout_band"]
    ax.axhspan(lo, hi, color=AQUA, alpha=0.15, lw=0, label=f"Checkout pass band {lo:.2f}-{hi:.2f}")
    ax.set_title("Module power ratio (module power / array median)")
    ax.set_xlabel("Mission time (h)")
    ax.set_ylabel("Power ratio (-)")
    ax.set_ylim(-0.05, 1.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def delta_v(log, path):
    t = log.t / 3600.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _shade_phases(ax, log)
    ax.plot(t, log.dv_used, color=BLUE, label="Cumulative delta-v")
    ax.axhline(6.0, color=INK2, ls="--", lw=1)
    ax.text(t[0], 6.0, " budget 6 m/s", va="bottom", color=INK2, fontsize=9)
    ax.annotate(f"{log.dv_used[-1]:.2f} m/s", (t[-1], log.dv_used[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=INK, fontweight="bold")
    ax.set_title("Servicer delta-v used")
    ax.set_xlabel("Mission time (h)")
    ax.set_ylabel("Delta-v (m/s)")
    ax.set_ylim(0, 6.5)
    _phase_key(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(log, outdir):
    """Write hill_path, range_speed, module_ratio and delta_v PNGs; return paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fn in (("hill_path", hill_path), ("range_speed", range_speed),
                     ("module_ratio", module_ratio), ("delta_v", delta_v)):
        p = outdir / f"{name}.png"
        fn(log, p)
        out.append(p)
    return out
