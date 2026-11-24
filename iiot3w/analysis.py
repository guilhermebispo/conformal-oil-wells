"""Aggregation of every value printed in the article  [stage S7 of Fig. 1].

Reads audit.json and experiment.json and writes results.json, a flat
dictionary key -> value.  The keys are the ones used by compare.py and
reference_values.py; the article section of each block is noted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import stats as ST

PERF = ["balanced_accuracy", "f1", "f1_macro", "mcc", "roc_auc", "pr_auc"]
CONF = [("conformal", "tau"), ("conformal", "coverage"), ("conformal", "mean_set_size"),
        ("conformal", "abstention_rate"), ("conformal", "balanced_accuracy_emitted"),
        ("conformal", "mcc_emitted"), ("closed", "ece"), ("closed", "aurc")]
ARMS = {"rf": "rf", "transformer": "tr"}


def _vals(exp, arm, *path):
    out = []
    for s in sorted(exp["seeds"], key=int):
        d = exp["seeds"][s]["arms"][arm]
        for p in path:
            d = d[p]
        out.append(d)
    return np.asarray(out, float)


def _dataset(aud, R):
    """Table II (tab:dataset) and Sec. IV-B/C."""
    R["dataset.n_instances"] = aud["n_instances"]
    R["dataset.n_wells"] = aud["n_wells"]
    R["dataset.n_observations"] = aud["n_observations"]
    R["dataset.first_day"] = aud["first_timestamp"]
    R["dataset.last_day"] = aud["last_timestamp"]
    R["dataset.duplicate_timestamps"] = aud["duplicate_timestamps"]
    for v, pct in aud["missing_pct"].items():
        R[f"dataset.missing_pct.{v}"] = pct
    R["dataset.n_windows"] = aud["n_windows"]
    R["dataset.n_wells_with_windows"] = aud["n_wells_with_windows"]
    for c, n in aud["instances_per_code"].items():
        R[f"dataset.instances.code{c}"] = n
    for c, n in aud["windows_per_code"].items():
        R[f"dataset.windows.code{c}"] = n
    R["dataset.instances.known_fault"] = aud["instances_known_fault"]
    R["dataset.instances.unseen"] = aud["instances_unseen"]
    R["dataset.windows.known_fault"] = aud["n_windows_known_fault"]
    R["dataset.windows.unseen"] = aud["n_windows_unseen"]
    R["dataset.windows.normal"] = aud["n_windows_normal"]
    R["dataset.instances_with_windows"] = sum(aud["instances_with_a_window"].values())


def _splits(exp, R):
    """Table III (tab:splits)."""
    for s in sorted(exp["seeds"], key=int):
        i = exp["seeds"][s]["split"]
        for k in ("n_train", "n_cal", "n_test", "n_unseen"):
            R[f"split.seed{s}.{k}"] = i[k]
        R[f"split.seed{s}.prevalence_test"] = i["prevalence_test"]
        R[f"split.seed{s}.fault_types_test"] = ", ".join(map(str, i["fault_types_test"]))
        R[f"split.seed{s}.inner_validation"] = i["inner_validation"]["type"]
    un = [exp["seeds"][s]["split"]["n_unseen"] for s in exp["seeds"]]
    R["split.n_unseen_min"], R["split.n_unseen_max"] = min(un), max(un)
    for part in ("train", "cal", "test"):
        n = [len(exp["seeds"][s]["split"][f"wells_{part}"]) for s in exp["seeds"]]
        R[f"split.wells_{part}_min"], R[f"split.wells_{part}_max"] = min(n), max(n)
    R["split.seed0.wells_train"] = len(exp["seeds"]["0"]["split"]["wells_train"])
    for s in sorted(exp["seeds"], key=int):
        i = exp["seeds"][s]["split"]
        R[f"split.seed{s}.test_types_absent_from_train"] = ", ".join(
            str(c) for c in i["fault_types_test"] if c not in i["fault_types_train"])


def _performance(exp, R):
    """Table V (tab:performance) and Sec. V-A (RQ1)."""
    seeds = sorted(exp["seeds"], key=int)
    n_tr = float(np.mean([exp["seeds"][s]["split"]["n_train"] for s in seeds]))
    n_te = float(np.mean([exp["seeds"][s]["split"]["n_test"] for s in seeds]))
    R["rq1.n_test_over_n_train"] = n_te / n_tr
    half, ps = [], []
    for m in PERF:
        a, b = _vals(exp, "rf", "closed", m), _vals(exp, "transformer", "closed", m)
        R[f"perf.{m}.rf_mean"], R[f"perf.{m}.rf_sd"] = a.mean(), a.std(ddof=1)
        R[f"perf.{m}.tr_mean"], R[f"perf.{m}.tr_sd"] = b.mean(), b.std(ddof=1)
        t = ST.corrected_resampled_ttest(b - a, n_tr, n_te)
        R[f"perf.{m}.diff"], R[f"perf.{m}.ci_low"] = t["mean"], t["ci_low"]
        R[f"perf.{m}.ci_high"], R[f"perf.{m}.p"] = t["ci_high"], t["p"]
        R[f"perf.{m}.diff_sd"] = t["sd"]
        R[f"perf.{m}.sd_exceeds_mean"] = bool(t["sd"] > abs(t["mean"]))
        half.append((t["ci_high"] - t["ci_low"]) / 2)
        ps.append(t["p"])
        if m in ("balanced_accuracy", "mcc"):
            dd = ST.detectable_difference(t["sd"], n_te, n_tr)
            R[f"rq1.sigma.{m}"] = t["sd"]
            R[f"rq1.detectable.{m}"] = dd["detectable"]
            R["rq1.se_factor_limit"] = dd["se_factor"]
    R["rq1.min_p"], R["rq1.ci_halfwidth_min"], R["rq1.ci_halfwidth_max"] = (
        min(ps), min(half), max(half))
    R["rq1.all_sd_exceed_mean"] = all(R[f"perf.{m}.sd_exceeds_mean"] for m in PERF)
    d = _vals(exp, "transformer", "closed", "mcc") - _vals(exp, "rf", "closed", "mcc")
    R["rq1.mcc.rf_wins"] = int((d < 0).sum())
    R["rq1.mcc.tr_wins"] = int((d > 0).sum())
    R["rq1.mcc.rf_margin_min"] = float((-d[d < 0]).min()) if (d < 0).any() else float("nan")
    R["rq1.mcc.rf_margin_max"] = float((-d[d < 0]).max()) if (d < 0).any() else float("nan")
    R["rq1.mcc.tr_margins"] = ", ".join(f"{x:.2f}" for x in sorted(d[d > 0], reverse=True))


def _novelty(exp, R):
    """Table VI (tab:novelty) and Sec. V-B (RQ2)."""
    for rule in ("full", "set_size"):
        for q in ("novelty_recall", "novelty_precision", "undue_abstention"):
            for arm, tag in ARMS.items():
                R[f"novelty.{rule}.{q}.{tag}"] = _vals(exp, arm, f"novelty_{rule}", q).mean()
    for tag in ARMS.values():
        full, card = R[f"novelty.full.novelty_recall.{tag}"], R[f"novelty.set_size.novelty_recall.{tag}"]
        R[f"rq2.recall_drop_pct.{tag}"] = 100 * (1 - card / full)


def _conformal(exp, R):
    """Table VII (tab:conformal) and Sec. V-C (RQ3)."""
    for path in CONF:
        for arm, tag in ARMS.items():
            v = _vals(exp, arm, *path)
            R[f"conf.{path[1]}.{tag}_mean"] = float(np.nanmean(v))
            R[f"conf.{path[1]}.{tag}_sd"] = float(np.nanstd(v, ddof=1))
    seeds = sorted(exp["seeds"], key=int)
    theory = [ST.coverage_sd_exchangeable(exp["seeds"][s]["split"]["n_cal"],
                                          exp["seeds"][s]["split"]["n_test"]) for s in seeds]
    R["rq3.coverage_sd_theory_max"] = max(theory)
    R["rq3.n_cal_min"] = min(exp["seeds"][s]["split"]["n_cal"] for s in seeds)
    R["rq3.n_cal_max"] = max(exp["seeds"][s]["split"]["n_cal"] for s in seeds)
    R["rq3.n_test_min"] = min(exp["seeds"][s]["split"]["n_test"] for s in seeds)
    ratios = []
    for arm, tag in ARMS.items():
        sd = R[f"conf.coverage.{tag}_sd"]
        ratios.append(sd / R["rq3.coverage_sd_theory_max"])
        R[f"rq3.effective_units.{tag}"] = ST.effective_units(sd)
        R[f"rq3.set_size_minus_one.{tag}"] = R[f"conf.mean_set_size.{tag}_mean"] - 1
        R[f"rq3.not_singleton.{tag}"] = _vals(exp, arm, "conformal", "not_singleton_rate").mean()
        R[f"rq3.bound_prop3.{tag}"] = _vals(exp, arm, "conformal", "bound_prop3").mean()
        R[f"rq3.atypical_adds.{tag}"] = _vals(exp, arm, "conformal", "atypical_only_rate").mean()
        R[f"rq3.ambiguous_rate.{tag}"] = _vals(exp, arm, "conformal", "ambiguous_set_rate").mean()
        R[f"rq3.empty_set_rate.{tag}"] = _vals(exp, arm, "conformal", "empty_set_rate").max()
        R[f"rq3.wrong_label_rate_max.{tag}"] = _vals(exp, arm, "conformal", "wrong_label_rate").max()
        R[f"rq3.bound_satisfied.{tag}"] = bool(np.all(
            _vals(exp, arm, "conformal", "abstention_rate")
            <= _vals(exp, arm, "conformal", "bound_prop3") + 1e-12))
        R[f"rq3.tau_min.{tag}"] = _vals(exp, arm, "conformal", "tau").min()
    R["rq3.coverage_sd_ratio_min"] = min(ratios)
    atyp = [exp["seeds"][s]["arms"][a]["conformal"]["atypical_rate"]
            for s in seeds for a in ARMS]
    R["rq3.atypical_rate_min"], R["rq3.atypical_rate_max"] = min(atyp), max(atyp)
    for arm, tag in ARMS.items():
        ab = _vals(exp, arm, "conformal", "abstention_rate")
        R[f"rq3.abstention_min.{tag}"], R[f"rq3.abstention_max.{tag}"] = ab.min(), ab.max()
        fs = _vals(exp, arm, "fit_seconds")
        R[f"cost.fit_seconds_min.{tag}"], R[f"cost.fit_seconds_max.{tag}"] = fs.min(), fs.max()
        h = [exp["seeds"][s]["arms"][arm]["history"] for s in seeds]
        if h[0]:
            R[f"train.{tag}.best_epoch_max"] = max(
                int(np.argmin([x["val_loss"] for x in hh])) + 1 for hh in h)
            R[f"train.{tag}.epochs_run_max"] = max(len(hh) for hh in h)


def _selection(exp, R):
    base = exp["seeds"][str(C.SELECT_ON_SEED)]["arms"]
    for arm, tag in ARMS.items():
        R[f"selection.{tag}"] = json.dumps(base[arm]["params"], sort_keys=True)
    # Per-seed selection (differs across seeds only under nested selection, T21).
    for sd in sorted(exp["seeds"], key=int):
        for arm, tag in ARMS.items():
            rec = exp["seeds"][sd]["arms"][arm]
            R[f"selection.{tag}.seed{sd}"] = json.dumps(rec["params"], sort_keys=True)
            best = max((r.get("val_roc_auc", float("nan")) for r in rec.get("selection", [])),
                       default=float("nan"))
            R[f"selection.{tag}.seed{sd}.best_val_roc_auc"] = best


def run(out_dir: Path, log=print):
    out_dir = Path(out_dir)
    aud = json.loads((out_dir / "audit.json").read_text())
    exp = json.loads((out_dir / "experiment.json").read_text())
    R = {}
    if not aud.get("synthetic"):
        _dataset(aud, R)
    _splits(exp, R)
    _performance(exp, R)
    _novelty(exp, R)
    _conformal(exp, R)
    _selection(exp, R)
    R = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in R.items()}
    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    log(f"results.json: {len(R)} values")
    return R
