"""Asset-level calibration of the three-state rule  (task T23, RQ2).

Post-hoc analysis of the stored per-window outputs (arrays/seed*_*.npz); no
model is refitted.  For every seed and arm, both thresholds of the rule (tau
for the conformal set, rho for the atypicality test) are recomputed with
three calibration methods on the same calibration windows:

  window         the TII article protocol: split-conformal quantile over
                 windows (Proposition 1; guarantee marginal over windows).
  well_quantile  quantile over calibration wells of each well's within-well
                 quantile (conformal.well_quantile_threshold): with
                 probability >= 1 - delta over a new well, its coverage is
                 >= 1 - alpha and its atypicality firing rate is <= beta.
  well_balanced  quantile of the well-balanced CDF (each well weight 1/K):
                 "pooling CDFs" of Dunn, Wasserman and Ramdas (2023);
                 asymptotic, targets coverage averaged over wells.
  well_subsample repeated subsampling of Dunn et al. (2023): one window per
                 calibration well per subsample; coverage of a new window of
                 a new well >= 1 - 2 alpha (finite sample).

Two targets are therefore compared: coverage AVERAGED over wells (Dunn et
al.; well_balanced, well_subsample; summary coverage_well_mean) and the
FRACTION of wells covered at 1 - alpha (well_quantile; summary
share_wells_covered).

For each method it reports what the guarantee buys (share of test wells with
coverage >= 1 - alpha, worst-well coverage, share of wells whose firing rate
stays <= beta) and what it costs (mean set size, abstention), together with
novelty recall and its lift over a random abstainer (recall minus the
abstention rate on known-regime windows).

Writes asset.json, figures/fig_asset_per_well.png, and adds hier.* keys to
results.json (when present).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import conformal as CF

# Sensitivity of the well-level guarantee to delta.  With K calibration
# wells the threshold is the ceil((K+1)(1-delta))-th smallest per-well
# quantile; for delta = 0.10 and 9 <= K <= 18 this is the WORST calibration
# well, so a single hard well decides the threshold.  Larger delta trades the
# probability of covering a new well for informativeness.
DELTA_SWEEP = (0.20, 0.30)
METHODS = ("window", "well_quantile", "well_balanced", "well_subsample") + tuple(
    f"well_quantile_d{d:.2f}" for d in DELTA_SWEEP)
ARMS = (("rf", "rf"), ("transformer", "tr"))   # default order; see _arms()


def _arms(rows):
    """(arm, tag) of every arm present in the rows, RF and Transformer first."""
    present = {r["arm"] for r in rows}
    order = [a for a in C.ARM_TAGS if a in present] + sorted(present - set(C.ARM_TAGS))
    return [(a, C.ARM_TAGS.get(a, a)) for a in order]
MIN_WELL_WINDOWS = 100   # wells shown in the per-well summaries (as in R2)


def _load(arr_dir):
    arrs = {}
    for f in sorted(Path(arr_dir).glob("seed*_*.npz")):
        seed, arm = f.stem.split("_", 1)
        arrs[(int(seed[4:]), arm)] = dict(np.load(f))
    return arrs


def thresholds(d, method, alpha, beta, delta):
    """(tau, rho, K) of one method on the calibration windows of d."""
    s_cal = CF.nonconformity(d["p_cal"].astype(float), d["y_cal"].astype(int))
    r_cal, w_cal = d["r_cal"].astype(float), d["well_cal"]
    if method == "window":
        tau = CF.conformal_threshold(d["p_cal"].astype(float), d["y_cal"].astype(int), alpha)
        rho = float(np.quantile(r_cal, CF.conformal_quantile_level(len(r_cal), beta),
                                method="higher"))
        return tau, rho, int(np.unique(w_cal).size)
    if method.startswith("well_quantile"):
        dlt = float(method.split("_d")[1]) if "_d" in method else delta
        tau, K = CF.well_quantile_threshold(s_cal, w_cal, alpha, dlt)
        rho, _ = CF.well_quantile_threshold(r_cal, w_cal, beta, dlt)
        return tau, rho, K
    if method == "well_balanced":
        tau, K = CF.well_balanced_threshold(s_cal, w_cal, alpha)
        rho, _ = CF.well_balanced_threshold(r_cal, w_cal, beta)
        return tau, rho, K
    if method == "well_subsample":
        tau, K = CF.well_subsample_threshold(s_cal, w_cal, alpha)
        rho, _ = CF.well_subsample_threshold(r_cal, w_cal, beta)
        return tau, rho, K
    raise ValueError(method)


def _cal_wells(d, alpha):
    """Within-well (1 - alpha) score quantile and error rate of each
    calibration well: the values the well-level threshold is chosen from."""
    s = CF.nonconformity(d["p_cal"].astype(float), d["y_cal"].astype(int))
    w, y = d["well_cal"], d["y_cal"]
    return [{"well": int(g), "n": int((w == g).sum()), "n_fault": int(((w == g) & (y == 1)).sum()),
             "q": CF.within_group_quantile(s[w == g], alpha),
             "error_rate": float((s[w == g] > 0.5).mean())} for g in np.unique(w)]


def evaluate(d, tau, rho, alpha, beta):
    """Guarantee and cost of one (tau, rho) pair on the test wells of d."""
    p, y, w = d["p_test"].astype(float), d["y_test"].astype(int), d["well_test"]
    r = d["r_test"].astype(float)
    sets = CF.prediction_sets(p, tau)
    card = sets.sum(1)
    hit = sets[np.arange(len(y)), y]
    fire = r > rho
    dec = CF.three_state_decision(p, tau, r, rho)
    dec_un = CF.three_state_decision(d["p_unseen"].astype(float), tau,
                                     d["r_unseen"].astype(float), rho)
    wells = []
    for g in np.unique(w):
        m = w == g
        wells.append({"well": int(g), "n": int(m.sum()), "coverage": float(hit[m].mean()),
                      "firing": float(fire[m].mean()), "set_size": float(card[m].mean())})
    big = [x for x in wells if x["n"] >= MIN_WELL_WINDOWS]
    abst = float((dec < 0).mean())
    rec = float((dec_un < 0).mean())
    return {
        "coverage": float(hit.mean()),
        "coverage_well_mean": float(np.mean([x["coverage"] for x in wells])),
        "share_wells_covered": float(np.mean([x["coverage"] >= 1 - alpha for x in big])),
        "coverage_worst_well": float(min(x["coverage"] for x in big)),
        "share_wells_firing_le_beta": float(np.mean([x["firing"] <= beta for x in big])),
        "firing_rate": float(fire.mean()),
        "mean_set_size": float(card.mean()),
        "singleton_rate": float((card == 1).mean()),
        "empty_set_rate": float((card == 0).mean()),
        "abstention": abst,
        "wrong_label_rate": float(((dec >= 0) & (dec != y)).mean()),
        "novelty_recall": rec,
        "novelty_lift": rec - abst,
        "n_wells_summarized": len(big),
        "wells": wells,
    }


def analyse(arrs, alpha=None, beta=None, delta=None):
    alpha = C.ALPHA if alpha is None else alpha
    beta = C.BETA if beta is None else beta
    delta = C.WELL_DELTA if delta is None else delta
    rows = []
    for (seed, arm), d in sorted(arrs.items()):
        for method in METHODS:
            tau, rho, K = thresholds(d, method, alpha, beta, delta)
            rows.append({"seed": seed, "arm": arm, "method": method, "tau": tau, "rho": rho,
                         "K_cal_wells": K, **evaluate(d, tau, rho, alpha, beta)})
        rows[-1]["cal_wells"] = _cal_wells(d, alpha)
    return rows


SUMMARY = ("tau", "coverage", "coverage_well_mean", "share_wells_covered",
           "coverage_worst_well", "share_wells_firing_le_beta", "firing_rate",
           "mean_set_size", "singleton_rate", "abstention", "wrong_label_rate",
           "novelty_recall", "novelty_lift")


def summarize(rows):
    """Mean, min and max over seeds of each summary quantity."""
    out = {}
    for arm, tag in _arms(rows):
        for method in METHODS:
            r = [x for x in rows if x["arm"] == arm and x["method"] == method]
            if not r:
                continue
            base = f"hier.{method}.{tag}"
            for k in SUMMARY:
                v = np.array([x[k] for x in r], float)
                out[f"{base}.{k}"] = float(np.mean(v))
                out[f"{base}.{k}.min"] = float(np.min(v))
                out[f"{base}.{k}.max"] = float(np.max(v))
            out[f"{base}.n_infinite_tau"] = int(sum(np.isinf(x["tau"]) for x in r))
            out[f"{base}.K_cal_wells.min"] = int(min(x["K_cal_wells"] for x in r))
    return out


def figure(rows, out_png, alpha):
    """Per-well coverage vs. number of test windows, window-level vs.
    well-quantile calibration, one panel per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (arm, _), title in zip(axes, ARMS, ("Random forest", "Transformer")):
        for method, mk, col, lab in (("window", "o", "#1f77b4", "Window-level"),
                                     ("well_quantile", "^", "#d62728", "Well-level")):
            pts = [(x["n"], x["coverage"]) for row in rows
                   if row["arm"] == arm and row["method"] == method
                   for x in row["wells"] if x["n"] >= MIN_WELL_WINDOWS]
            if pts:
                n, c = zip(*pts)
                ax.scatter(n, c, s=14, marker=mk, color=col, alpha=0.75, label=lab)
        ax.axhline(1 - alpha, ls="--", lw=0.8, color="k")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test windows in the well", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Coverage of the well", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def run(out_dir: Path, log=print, arrays_dir: Path | None = None):
    out_dir = Path(out_dir)
    arrs = _load(arrays_dir or out_dir / "arrays")
    if not arrs:
        log("no stored arrays; asset-level analysis skipped")
        return None
    rows = analyse(arrs)
    summ = summarize(rows)
    (out_dir / "asset.json").write_text(json.dumps(
        {"alpha": C.ALPHA, "beta": C.BETA, "delta": C.WELL_DELTA,
         "min_well_windows": MIN_WELL_WINDOWS, "summary": summ, "rows": rows}, indent=1))
    figure(rows, out_dir / "figures" / "fig_asset_per_well.png", C.ALPHA)
    res = out_dir / "results.json"
    if res.exists():
        R = json.loads(res.read_text())
        R.update(summ)
        res.write_text(json.dumps(R, indent=1))
    for arm, tag in _arms(rows):
        for m in METHODS:
            b = f"hier.{m}.{tag}"
            if f"{b}.coverage" in summ:
                log(f"  {tag} {m:13s} cov={summ[b + '.coverage']:.3f} "
                    f"wells>=1-a={summ[b + '.share_wells_covered']:.2f} "
                    f"worst={summ[b + '.coverage_worst_well']:.2f} "
                    f"|C|={summ[b + '.mean_set_size']:.2f} abst={summ[b + '.abstention']:.3f} "
                    f"lift={summ[b + '.novelty_lift']:+.3f} inf_tau={summ[b + '.n_infinite_tau']}")
    return summ
