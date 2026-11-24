"""Figures of the article, drawn only from computed outputs.

  fig2_risk_coverage.png      Fig. 2  risk-coverage curves (mean, min-max band)
  fig3_reliability.png        Fig. 3  reliability diagrams, one line per partition
  fig6_feature_importance.png Fig. 5  random-forest feature importance (mean +- sd)
  fig7_per_well_coverage.png  Fig. 4  coverage per test well vs. binomial band
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import conformal as CF  # noqa: E402

COLOR = {"rf": "#2a78d6", "transformer": "#eb6834"}
NAME = {"rf": "Random forest", "transformer": "Transformer"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
PRETTY = {"log_diff_energy": "log energy of diff.", "last": "final value", "q25": "1st quartile",
          "q75": "3rd quartile", "std": "std. dev."}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "savefig.dpi": 600, "savefig.bbox": "tight"})


def _seeds(exp):
    return sorted(exp["seeds"], key=int)


def fig_risk_coverage(exp, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    grid = np.linspace(0.02, 1.0, 200)
    for key in ("rf", "transformer"):
        curves = []
        for s in _seeds(exp):
            p = np.asarray(exp["seeds"][s]["arms"][key]["proba_test"])
            y = np.asarray(exp["seeds"][s]["y_test"])
            cov, risk, _ = CF.risk_coverage(y, p)
            curves.append(np.interp(grid, cov, risk))
        M = np.vstack(curves)
        ax.fill_between(grid, M.min(0), M.max(0), color=COLOR[key], alpha=0.15, lw=0)
        ax.plot(grid, M.mean(0), color=COLOR[key], lw=1.6, label=NAME[key])
    ax.set_xlabel("coverage (fraction of windows decided)")
    ax.set_ylabel("risk (error on decided windows)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out, "fig2_risk_coverage.png")


def fig_reliability(exp, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    edges = np.linspace(0.5, 1.0, 11)
    for ax, key in zip(axes, ("rf", "transformer")):
        ax.plot([0.45, 1.0], [0.45, 1.0], ls="--", c=MUTED, lw=0.8)
        for s in _seeds(exp):
            p = np.asarray(exp["seeds"][s]["arms"][key]["proba_test"])
            y = np.asarray(exp["seeds"][s]["y_test"])
            conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() > 20:
                    xs.append(conf[m].mean())
                    ys.append(acc[m].mean())
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.0, alpha=0.8, color=COLOR[key])
        ax.set_title(NAME[key], fontsize=8, color=INK)
        ax.set_xlabel("mean confidence")
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy")
    return _save(fig, out, "fig3_reliability.png")


def fig_importance(exp, out, top=12):
    acc = {}
    for s in _seeds(exp):
        for d in exp["seeds"][s]["arms"]["rf"].get("importance", []):
            acc.setdefault(d["feature"], []).append(d["importance"])
    items = sorted(acc.items(), key=lambda kv: -np.mean(kv[1]))[:top][::-1]

    def pretty(name):
        stat, var = name.split("(")
        return f"{PRETTY.get(stat, stat)} ({var}"
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh([pretty(k) for k, _ in items], [np.mean(v) for _, v in items],
            xerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for _, v in items],
            color=COLOR["rf"], height=0.65, error_kw={"lw": 0.7, "ecolor": INK})
    ax.set_xlabel("feature importance (random forest)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, out, "fig6_feature_importance.png")


def fig_per_well(out, alpha=0.10):
    """Funnel plot: coverage of each test well vs. its number of windows, with
    the 95% band expected if windows were exchangeable."""
    path = Path(out) / "robustness.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["per_well"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    n = np.logspace(2, np.log10(max(r["n"] for r in rows) * 1.2), 200)
    half = 1.96 * np.sqrt(alpha * (1 - alpha) / n)
    for ax, key in zip(axes, ("rf", "transformer")):
        ax.fill_between(n, 1 - alpha - half, np.minimum(1, 1 - alpha + half),
                        color=GRID, alpha=0.8, lw=0, label="95% band, exchangeable windows")
        ax.axhline(1 - alpha, color=MUTED, ls="--", lw=0.8)
        r = [x for x in rows if x["arm"] == key]
        ax.scatter([x["n"] for x in r], [x["coverage"] for x in r], s=14, color=COLOR[key],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_title(NAME[key], fontsize=8, color=INK)
        ax.set_xlabel("test windows of the well")
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    return _save(fig, out, "fig7_per_well_coverage.png")


def _save(fig, out, name):
    d = Path(out) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / name)
    plt.close(fig)
    return d / name


def run(out_dir: Path, log=print):
    exp = json.loads((Path(out_dir) / "experiment.json").read_text())
    for f in (fig_risk_coverage, fig_reliability, fig_importance):
        log(f"  {f(exp, out_dir)}")
    log(f"  {fig_per_well(out_dir)}")
