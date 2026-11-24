#!/usr/bin/env python3
"""Exploratory re-analysis of the stored per-window outputs (2026-11-19).

Reads results/cpu_reference/arrays/seed*_{rf,transformer}.npz (written by the
experiment stage) and, without refitting any model, checks four things that
the TII manuscript did not report:

  (a) Lift of novelty recall over a random abstainer.  A rule that abstains on
      a fraction q of all windows, at random, has novelty recall q.  The
      informative quantity is therefore recall minus the abstention rate on
      known-regime test windows, not recall alone.
  (b) Well confound of the atypicality score r.  AUROC of unseen vs. known
      windows computed (i) over all test windows and (ii) only over the test
      wells that also contain unseen windows.  If (ii) collapses towards 0.5,
      r is recognising the well, not the regime.
  (c) Shift of r between calibration and test wells.  AUROC of known test
      windows vs. calibration windows on r alone; 0.5 if exchangeable.
  (d) Prototype of well-level calibration.  tau is the conformal quantile,
      over calibration WELLS, of each well's (1 - alpha) score quantile, so the
      well is treated as the exchangeable unit.  Reports the share of test
      wells (>= 100 windows) with coverage >= 1 - alpha and the mean set size.

Usage:  python reanalysis_20261119.py results/cpu_reference/arrays
Output goes to stdout; the run of 2026-11-19 is stored in
evidencias/reanalysis_20261119/.
"""
import glob
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ALPHA = 0.10           # same level as the article (config.ALPHA = config.BETA)
MIN_WELL_WINDOWS = 100  # same threshold as robustness.per_well_coverage


def conformal_quantile(scores, level):
    """Split-conformal quantile with the finite-sample correction."""
    n = len(scores)
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - level)) / n),
                       method="higher")


def auroc(neg, pos):
    """AUROC with `pos` as the positive class."""
    return roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])


def main(arr_dir):
    files = sorted(glob.glob(f"{arr_dir}/*.npz"))
    print("== (a) recall vs random abstainer, (b) well confound, (c) shift of r")
    for f in files:
        d = np.load(f)
        # Window-level thresholds, exactly as in the article.
        s_cal = 1 - d["p_cal"][np.arange(len(d["y_cal"])), d["y_cal"]]
        tau, rho = conformal_quantile(s_cal, ALPHA), conformal_quantile(d["r_cal"], ALPHA)

        def unknown(p, r):
            # Three-state rule: Unknown if the set is not a singleton or r is atypical.
            sets = (1 - p) <= tau
            return (sets.sum(1) != 1) | (r > rho)

        abst = unknown(d["p_test"], d["r_test"]).mean()
        rec = unknown(d["p_unseen"], d["r_unseen"]).mean()
        # (b) restrict known windows to the wells that also hold unseen windows.
        wu = np.unique(d["well_unseen"])
        same = np.isin(d["well_test"], wu)
        au_same = auroc(d["r_test"][same], d["r_unseen"]) if same.any() else float("nan")
        au_all = auroc(d["r_test"], d["r_unseen"])
        au_shift = auroc(d["r_cal"], d["r_test"])
        print(f"{f.split('/')[-1]:22s} abst={abst:.3f} rec={rec:.3f} lift={rec - abst:+.3f} | "
              f"AUROC_r all={au_all:.3f} same-wells={au_same:.3f} "
              f"(known windows in unseen wells={same.sum()}, unseen wells={wu.tolist()}) | "
              f"AUROC cal-vs-test(r)={au_shift:.3f}")

    print("\n== (d) well-level calibration prototype: tau = conformal quantile over cal "
          "WELLS of per-well (1-a) score quantile")
    for f in files:
        d = np.load(f)
        s_cal = 1 - d["p_cal"][np.arange(len(d["y_cal"])), d["y_cal"]]
        w_cal = d["well_cal"]
        tau_win = conformal_quantile(s_cal, ALPHA)
        per_well = np.sort([np.quantile(s_cal[w_cal == w], 1 - ALPHA) for w in np.unique(w_cal)])
        k = len(per_well)
        # Order statistic ceil((k+1)(1-alpha)) over wells; undefined with one well.
        tau_well = per_well[min(k - 1, int(np.ceil((k + 1) * (1 - ALPHA))) - 1)] if k > 1 else np.nan
        y, p, w = d["y_test"], d["p_test"], d["well_test"]
        s_test = 1 - p[np.arange(len(y)), y]
        out = []
        for t in (tau_win, tau_well):
            if np.isnan(t):
                out.append("n/a")
                continue
            cov = np.array([(s_test[w == ww] <= t).mean() for ww in np.unique(w)
                            if (w == ww).sum() >= MIN_WELL_WINDOWS])
            size = ((1 - p) <= t).sum(1).mean()
            out.append(f"tau={t:.3f} wells>=0.9:{np.mean(cov >= 1 - ALPHA):.2f} "
                       f"minWellCov={cov.min():.2f} |C|={size:.2f}")
        print(f"{f.split('/')[-1]:22s} calWells={k:2d}  window-level: {out[0]}  ||  well-level: {out[1]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/cpu_reference/arrays")
