"""Post-hoc robustness analyses from the stored per-window outputs.

R1  Sensitivity to alpha and beta.  For alpha, beta in {0.05, 0.10, 0.20},
    the conformal threshold and the atypicality threshold are recalibrated on
    the same calibration windows (no refitting).  Checks the claim that any
    alpha below the error rate keeps tau >= 0.5 and the empty set unreachable,
    and shows how abstention and novelty recall trade off.

R2  Coverage per test well at alpha = 0.10.  If windows were exchangeable,
    the coverage of a well with n windows would scatter around 1 - alpha with
    binomial SD sqrt(alpha(1-alpha)/n).  Overdispersion relative to that band
    is direct evidence that the well, not the window, is the exchangeable unit.

R3  Threshold-free novelty detection: AUROC of the maximum-softmax-
    probability baseline and of the atypicality score.

Writes robustness.json and adds robust.* keys to results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import conformal as CF

LEVELS = [0.05, 0.10, 0.20]
MIN_WELL_WINDOWS = 100


def _load(out_dir):
    arrs = {}
    for f in sorted((Path(out_dir) / "arrays").glob("seed*_*.npz")):
        seed, arm = f.stem.split("_", 1)
        arrs[(int(seed[4:]), arm)] = dict(np.load(f))
    return arrs


def _beta_rho(r_cal, beta):
    return float(np.quantile(r_cal, CF.conformal_quantile_level(len(r_cal), beta),
                             method="higher"))


def alpha_beta_grid(arrs):
    rows = []
    for arm in ("rf", "transformer"):
        for alpha in LEVELS:
            for beta in LEVELS:
                per = []
                for (seed, a), d in arrs.items():
                    if a != arm:
                        continue
                    tau = CF.conformal_threshold(d["p_cal"].astype(float), d["y_cal"].astype(int), alpha)
                    rho = _beta_rho(d["r_cal"].astype(float), beta)
                    p, y = d["p_test"].astype(float), d["y_test"].astype(int)
                    sets = CF.prediction_sets(p, tau)
                    dec = CF.three_state_decision(p, tau, d["r_test"], rho)
                    dec_un = CF.three_state_decision(d["p_unseen"].astype(float), tau, d["r_unseen"], rho)
                    card_un = CF.set_size_only_decision(d["p_unseen"].astype(float), tau)
                    per.append({
                        "tau": tau,
                        "coverage": sets[np.arange(len(y)), y].mean(),
                        "empty_set_rate": (sets.sum(1) == 0).mean(),
                        "abstention": (dec < 0).mean(),
                        "bound": (sets.sum(1) != 1).mean() + beta,
                        "wrong_label_rate": ((dec >= 0) & (dec != y)).mean(),
                        "novelty_recall": (dec_un < 0).mean(),
                        "novelty_recall_set_size": (card_un < 0).mean(),
                    })
                agg = {k: float(np.mean([r[k] for r in per])) for k in per[0]}
                agg.update(arm=arm, alpha=alpha, beta=beta,
                           tau_min=float(min(r["tau"] for r in per)),
                           empty_set_max=float(max(r["empty_set_rate"] for r in per)),
                           bound_ok=bool(all(r["abstention"] <= r["bound"] + 1e-12 for r in per)),
                           wrong_le_alpha=bool(np.mean([r["wrong_label_rate"] for r in per]) <= alpha))
                rows.append(agg)
    return rows


def per_well_coverage(arrs, alpha=None):
    alpha = C.ALPHA if alpha is None else alpha
    rows = []
    for (seed, arm), d in arrs.items():
        tau = CF.conformal_threshold(d["p_cal"].astype(float), d["y_cal"].astype(int), alpha)
        p, y, w = d["p_test"].astype(float), d["y_test"].astype(int), d["well_test"]
        hit = CF.prediction_sets(p, tau)[np.arange(len(y)), y]
        for well in np.unique(w):
            m = w == well
            if m.sum() >= MIN_WELL_WINDOWS:
                rows.append({"seed": seed, "arm": arm, "well": int(well), "n": int(m.sum()),
                             "coverage": float(hit[m].mean()),
                             "fault_share": float((y[m] == 1).mean())})
    out = {"rows": rows}
    for arm in ("rf", "transformer"):
        r = [x for x in rows if x["arm"] == arm]
        cov = np.array([x["coverage"] for x in r])
        n = np.array([x["n"] for x in r])
        z = (cov - (1 - alpha)) / np.sqrt(alpha * (1 - alpha) / n)
        out[arm] = {"n_well_partitions": len(r), "coverage_min": float(cov.min()),
                    "coverage_max": float(cov.max()), "coverage_sd": float(cov.std(ddof=1)),
                    "share_outside_binomial_95": float((np.abs(z) > 1.96).mean()),
                    "dispersion_ratio": float(np.mean(z ** 2)),   # 1 if exchangeable
                    "share_below_0.80": float((cov < 0.80).mean())}
    return out


def novelty_auroc(arrs):
    """R3: threshold-free novelty detection.  AUROC of unseen-regime windows
    (positive) vs. known-regime test windows of the same partition, for the
    maximum-softmax-probability baseline (score 1 - max p) and for the
    atypicality score r of the representation."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for arm in ("rf", "transformer"):
        msp, atyp = [], []
        for (seed, a), d in arrs.items():
            if a != arm:
                continue
            lab = np.r_[np.zeros(len(d["p_test"])), np.ones(len(d["p_unseen"]))]
            s_msp = np.r_[1 - d["p_test"].max(1), 1 - d["p_unseen"].max(1)]
            s_r = np.r_[d["r_test"], d["r_unseen"]]
            msp.append(roc_auc_score(lab, s_msp))
            atyp.append(roc_auc_score(lab, s_r))
        out[arm] = {"auroc_msp_mean": float(np.mean(msp)), "auroc_msp_sd": float(np.std(msp, ddof=1)),
                    "auroc_atyp_mean": float(np.mean(atyp)), "auroc_atyp_sd": float(np.std(atyp, ddof=1)),
                    "atyp_beats_msp": int(np.sum(np.array(atyp) > np.array(msp)))}
    return out


def run(out_dir: Path, log=print):
    out_dir = Path(out_dir)
    arrs = _load(out_dir)
    if not arrs:
        log("no stored arrays; robustness skipped")
        return None
    grid = alpha_beta_grid(arrs)
    wells = per_well_coverage(arrs)
    auroc = novelty_auroc(arrs)
    (out_dir / "robustness.json").write_text(json.dumps(
        {"alpha_beta": grid, "per_well": wells, "novelty_auroc": auroc}, indent=1))
    R = json.loads((out_dir / "results.json").read_text())
    for g in grid:
        tag = "rf" if g["arm"] == "rf" else "tr"
        base = f"robust.a{g['alpha']:.2f}.b{g['beta']:.2f}.{tag}"
        for k in ("tau", "coverage", "abstention", "novelty_recall", "novelty_recall_set_size",
                  "wrong_label_rate", "empty_set_max", "tau_min"):
            R[f"{base}.{k}"] = g[k]
        R[f"{base}.bound_ok"] = g["bound_ok"]
    R["robust.all_empty_set_zero"] = bool(all(g["empty_set_max"] == 0 for g in grid))
    R["robust.all_tau_ge_half"] = bool(all(g["tau_min"] >= 0.5 for g in grid))
    R["robust.all_bounds_ok"] = bool(all(g["bound_ok"] for g in grid))
    R["robust.all_wrong_le_alpha"] = bool(all(g["wrong_le_alpha"] for g in grid))
    for arm, tag in (("rf", "rf"), ("transformer", "tr")):
        for k, v in wells[arm].items():
            R[f"robust.wells.{tag}.{k}"] = v
    for arm, tag in (("rf", "rf"), ("transformer", "tr")):
        for k, v in auroc[arm].items():
            R[f"robust.auroc.{tag}.{k}"] = v
    R["robust.bound_fail_cells"] = int(sum(not g["bound_ok"] for g in grid))
    gaps = [g["novelty_recall"] - g["novelty_recall_set_size"] for g in grid]
    R["robust.full_minus_size_min"], R["robust.full_minus_size_max"] = min(gaps), max(gaps)
    for arm, tag in (("rf", "rf"), ("transformer", "tr")):
        n_low = 0
        for (seed, a), d in arrs.items():
            if a == arm and CF.conformal_threshold(d["p_cal"].astype(float),
                                                   d["y_cal"].astype(int), 0.20) < 0.5:
                n_low += 1
        R[f"robust.a0.20.partitions_tau_below_half.{tag}"] = n_low
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    log(f"robustness: {len(grid)} (alpha, beta) cells; per-well: "
        f"rf {wells['rf']['n_well_partitions']}, tr {wells['transformer']['n_well_partitions']}")
    return grid, wells
