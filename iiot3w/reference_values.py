"""Values REPORTED in the article (tii-articles-template.tex), used ONLY for the
automatic comparison in compare.py.  Nothing here is ever written as a result.

Tolerances are declared, not tuned: a value printed with k decimals matches if
the computed value rounds to it (|diff| <= 0.5 * 10^-k, plus 1e-9).  Counts and
strings must match exactly.
"""

ARTICLE = {
    # ---------------- Table II / Sec. IV-B, IV-C ----------------
    "dataset.n_instances": 1119, "dataset.n_wells": 40,
    "dataset.n_observations": 32_872_250,
    "dataset.first_day": "2011-08-30", "dataset.last_day": "2023-09-26",
    "dataset.duplicate_timestamps": 0,
    "dataset.missing_pct.T-TPT": 13.6, "dataset.missing_pct.P-TPT": 16.4,
    "dataset.missing_pct.P-PDG": 25.0,
    "dataset.n_windows": 45_548, "dataset.n_wells_with_windows": 28,
    "dataset.instances.code0": 594, "dataset.windows.code0": 31_552,
    "dataset.instances.code2": 22, "dataset.windows.code2": 102,
    "dataset.instances.code3": 32, "dataset.windows.code3": 3_476,
    "dataset.instances.code4": 343, "dataset.windows.code4": 9_498,
    "dataset.instances.code5": 11, "dataset.windows.code5": 65,
    "dataset.instances.code6": 6, "dataset.windows.code6": 73,
    "dataset.instances.code7": 36, "dataset.windows.code7": 31,
    "dataset.instances.code1": 4, "dataset.windows.code1": 50,
    "dataset.instances.code8": 14, "dataset.windows.code8": 596,
    "dataset.instances.code9": 57, "dataset.windows.code9": 105,
    "dataset.instances.known_fault": 450, "dataset.windows.known_fault": 13_245,
    "dataset.instances.unseen": 75, "dataset.windows.unseen": 751,
    # ---------------- Table III ----------------
    "split.seed0.n_train": 28_967, "split.seed0.n_cal": 7_380, "split.seed0.n_test": 8_450,
    "split.seed0.n_unseen": 607, "split.seed0.prevalence_test": 0.29,
    "split.seed0.fault_types_test": "2, 4, 5, 7",
    "split.seed1.n_train": 27_024, "split.seed1.n_cal": 7_257, "split.seed1.n_test": 10_516,
    "split.seed1.n_unseen": 68, "split.seed1.prevalence_test": 0.18,
    "split.seed1.fault_types_test": "2, 4, 5, 7",
    "split.seed2.n_train": 30_346, "split.seed2.n_cal": 4_604, "split.seed2.n_test": 9_847,
    "split.seed2.n_unseen": 385, "split.seed2.prevalence_test": 0.19,
    "split.seed2.fault_types_test": "2, 4",
    "split.seed3.n_train": 29_027, "split.seed3.n_cal": 5_016, "split.seed3.n_test": 10_754,
    "split.seed3.n_unseen": 312, "split.seed3.prevalence_test": 0.33,
    "split.seed3.fault_types_test": "2, 3, 4, 5, 7",
    "split.seed4.n_train": 30_144, "split.seed4.n_cal": 4_459, "split.seed4.n_test": 10_194,
    "split.seed4.n_unseen": 187, "split.seed4.prevalence_test": 0.19,
    "split.seed4.fault_types_test": "2, 4, 5",
    "split.n_unseen_min": 68, "split.n_unseen_max": 607,
    # ---------------- Table V ----------------
    "perf.balanced_accuracy.rf_mean": 0.813, "perf.balanced_accuracy.rf_sd": 0.069,
    "perf.balanced_accuracy.tr_mean": 0.798, "perf.balanced_accuracy.tr_sd": 0.075,
    "perf.balanced_accuracy.diff": -0.016, "perf.balanced_accuracy.ci_low": -0.247,
    "perf.balanced_accuracy.ci_high": 0.215, "perf.balanced_accuracy.p": 0.860,
    "perf.f1.rf_mean": 0.650, "perf.f1.rf_sd": 0.053, "perf.f1.tr_mean": 0.686,
    "perf.f1.tr_sd": 0.081, "perf.f1.diff": 0.036, "perf.f1.ci_low": -0.233,
    "perf.f1.ci_high": 0.305, "perf.f1.p": 0.730,
    "perf.f1_macro.rf_mean": 0.750, "perf.f1_macro.rf_sd": 0.040,
    "perf.f1_macro.tr_mean": 0.794, "perf.f1_macro.tr_sd": 0.051,
    "perf.f1_macro.diff": 0.043, "perf.f1_macro.ci_low": -0.113,
    "perf.f1_macro.ci_high": 0.199, "perf.f1_macro.p": 0.486,
    "perf.mcc.rf_mean": 0.546, "perf.mcc.rf_sd": 0.091, "perf.mcc.tr_mean": 0.593,
    "perf.mcc.tr_sd": 0.104, "perf.mcc.diff": 0.047, "perf.mcc.ci_low": -0.281,
    "perf.mcc.ci_high": 0.376, "perf.mcc.p": 0.709,
    "perf.roc_auc.rf_mean": 0.853, "perf.roc_auc.rf_sd": 0.112,
    "perf.roc_auc.tr_mean": 0.904, "perf.roc_auc.tr_sd": 0.043,
    "perf.roc_auc.diff": 0.051, "perf.roc_auc.ci_low": -0.155,
    "perf.roc_auc.ci_high": 0.257, "perf.roc_auc.p": 0.529,
    "perf.pr_auc.rf_mean": 0.640, "perf.pr_auc.rf_sd": 0.050,
    "perf.pr_auc.tr_mean": 0.653, "perf.pr_auc.tr_sd": 0.087,
    "perf.pr_auc.diff": 0.012, "perf.pr_auc.ci_low": -0.203,
    "perf.pr_auc.ci_high": 0.228, "perf.pr_auc.p": 0.880,
    # ---------------- Sec. V-A (RQ1) text ----------------
    "rq1.ci_halfwidth_min": 0.16, "rq1.ci_halfwidth_max": 0.33,
    "rq1.mcc.rf_wins": 3, "rq1.mcc.tr_wins": 2,
    "rq1.mcc.rf_margin_min": 0.03, "rq1.mcc.rf_margin_max": 0.09,
    "rq1.mcc.tr_margins": "0.25, 0.19",
    "rq1.all_sd_exceed_mean": True,
    "rq1.n_test_over_n_train": 0.34, "rq1.se_factor_limit": 0.58,
    "rq1.sigma.balanced_accuracy": 0.11, "rq1.sigma.mcc": 0.16,
    "rq1.detectable.balanced_accuracy": 0.19, "rq1.detectable.mcc": 0.26,
    # ---------------- Table VI / Sec. V-B (RQ2) ----------------
    "novelty.full.novelty_recall.rf": 0.677, "novelty.full.novelty_recall.tr": 0.447,
    "novelty.full.novelty_precision.rf": 0.054, "novelty.full.novelty_precision.tr": 0.126,
    "novelty.full.undue_abstention.rf": 0.483, "novelty.full.undue_abstention.tr": 0.176,
    "novelty.set_size.novelty_recall.rf": 0.267, "novelty.set_size.novelty_recall.tr": 0.107,
    "novelty.set_size.novelty_precision.rf": 0.019, "novelty.set_size.novelty_precision.tr": 0.036,
    "novelty.set_size.undue_abstention.rf": 0.459, "novelty.set_size.undue_abstention.tr": 0.138,
    "rq2.recall_drop_pct.rf": 61, "rq2.recall_drop_pct.tr": 76,
    # ---------------- Table VII ----------------
    "conf.tau.rf_mean": 0.755, "conf.tau.rf_sd": 0.132,
    "conf.tau.tr_mean": 0.620, "conf.tau.tr_sd": 0.120,
    "conf.coverage.rf_mean": 0.918, "conf.coverage.rf_sd": 0.048,
    "conf.coverage.tr_mean": 0.893, "conf.coverage.tr_sd": 0.048,
    "conf.mean_set_size.rf_mean": 1.459, "conf.mean_set_size.rf_sd": 0.269,
    "conf.mean_set_size.tr_mean": 1.138, "conf.mean_set_size.tr_sd": 0.141,
    "conf.abstention_rate.rf_mean": 0.483, "conf.abstention_rate.rf_sd": 0.290,
    "conf.abstention_rate.tr_mean": 0.176, "conf.abstention_rate.tr_sd": 0.137,
    "conf.balanced_accuracy_emitted.rf_mean": 0.851, "conf.balanced_accuracy_emitted.rf_sd": 0.108,
    "conf.balanced_accuracy_emitted.tr_mean": 0.827, "conf.balanced_accuracy_emitted.tr_sd": 0.095,
    "conf.mcc_emitted.rf_mean": 0.668, "conf.mcc_emitted.rf_sd": 0.169,
    "conf.mcc_emitted.tr_mean": 0.681, "conf.mcc_emitted.tr_sd": 0.122,
    "conf.ece.rf_mean": 0.101, "conf.ece.rf_sd": 0.015,
    "conf.ece.tr_mean": 0.132, "conf.ece.tr_sd": 0.035,
    "conf.aurc.rf_mean": 0.192, "conf.aurc.rf_sd": 0.102,
    "conf.aurc.tr_mean": 0.135, "conf.aurc.tr_sd": 0.046,
    # ---------------- Sec. V-C (RQ3) text ----------------
    "rq3.n_cal_min": 4_459, "rq3.n_cal_max": 7_380, "rq3.n_test_min": 8_450,
    "rq3.coverage_sd_theory_max": 0.006,     # stated as "below 0.006" (checked as <=)
    "rq3.coverage_sd_ratio_min": 8,          # stated as "at least eight times" (checked as >=)
    "rq3.set_size_minus_one.rf": 0.459, "rq3.set_size_minus_one.tr": 0.138,
    "rq3.bound_prop3.rf": 0.559, "rq3.bound_prop3.tr": 0.238,
    "rq3.atypical_adds.rf": 0.024, "rq3.atypical_adds.tr": 0.038,
    "rq3.ambiguous_rate.rf": 0.46,
    "rq3.empty_set_rate.rf": 0.0, "rq3.empty_set_rate.tr": 0.0018,
    "rq3.tau_min.tr": 0.498, "rq3.wrong_label_rate_max.rf": 0.137,
    "rq3.bound_satisfied.rf": True, "rq3.bound_satisfied.tr": True,
    # ---------------- Sec. V-D (robustness) ----------------
    "robust.wells.rf.n_well_partitions": 39, "robust.wells.tr.n_well_partitions": 39,
    "robust.wells.rf.share_outside_binomial_95": 0.82, "robust.wells.tr.share_outside_binomial_95": 0.85,
    "robust.wells.rf.dispersion_ratio": 104, "robust.wells.tr.dispersion_ratio": 228,
    "robust.wells.rf.coverage_min": 0.38, "robust.wells.tr.coverage_min": 0.22,
    "robust.wells.rf.coverage_max": 1.00, "robust.wells.tr.coverage_max": 1.00,
    "robust.a0.05.b0.10.rf.tau_min": 0.72, "robust.a0.05.b0.10.tr.tau_min": 0.72,
    "robust.a0.05.b0.10.rf.empty_set_max": 0.0, "robust.a0.05.b0.10.tr.empty_set_max": 0.0,
    "robust.a0.20.b0.10.rf.empty_set_max": 0.19, "robust.a0.20.b0.10.tr.empty_set_max": 0.22,
    "robust.a0.10.b0.05.rf.novelty_recall": 0.537, "robust.a0.10.b0.20.rf.novelty_recall": 0.706,
    "robust.a0.10.b0.05.tr.novelty_recall": 0.288, "robust.a0.10.b0.20.tr.novelty_recall": 0.637,
    "robust.a0.10.b0.05.rf.abstention": 0.475, "robust.a0.10.b0.20.rf.abstention": 0.544,
    "robust.a0.10.b0.05.tr.abstention": 0.160, "robust.a0.10.b0.20.tr.abstention": 0.303,
    "robust.bound_fail_cells": 8, "robust.all_wrong_le_alpha": True,
    "robust.a0.10.b0.10.rf.bound_ok": True, "robust.a0.10.b0.10.tr.bound_ok": True,
    "robust.auroc.rf.auroc_msp_mean": 0.333, "robust.auroc.rf.auroc_msp_sd": 0.061,
    "robust.auroc.tr.auroc_msp_mean": 0.352, "robust.auroc.tr.auroc_msp_sd": 0.162,
    "robust.auroc.rf.auroc_atyp_mean": 0.748, "robust.auroc.rf.auroc_atyp_sd": 0.164,
    "robust.auroc.tr.auroc_atyp_mean": 0.788, "robust.auroc.tr.auroc_atyp_sd": 0.094,
    "robust.auroc.rf.atyp_beats_msp": 5, "robust.auroc.tr.atyp_beats_msp": 5,
    # ---------------- added by the full review ----------------
    "split.seed0.wells_train": 4,
    "split.wells_train_min": 4, "split.wells_train_max": 13,
    "split.wells_cal_min": 1, "split.wells_cal_max": 13,
    "split.wells_test_min": 6, "split.wells_test_max": 23,
    "split.seed0.test_types_absent_from_train": "2, 5, 7",
    "split.seed3.test_types_absent_from_train": "2, 5, 7",
    "split.seed1.test_types_absent_from_train": "",
    "dataset.instances_with_windows": 815,
    "rq3.effective_units.rf": 77, "rq3.effective_units.tr": 78,
    "rq3.atypical_rate_min": 0.008, "rq3.atypical_rate_max": 0.153,
    "rq3.abstention_min.rf": 0.25, "rq3.abstention_max.rf": 0.93,
    "cost.fit_seconds_min.rf": 13, "cost.fit_seconds_max.rf": 16,
    "cost.fit_seconds_min.tr": 287, "cost.fit_seconds_max.tr": 318,
    "train.tr.best_epoch_max": 1, "train.tr.epochs_run_max": 5,
    "robust.full_minus_size_min": 0.17, "robust.full_minus_size_max": 0.53,
    "robust.a0.20.partitions_tau_below_half.rf": 2,
    "robust.a0.20.partitions_tau_below_half.tr": 4,
    # ---------------- R4: early stopping on held-out wells ----------------
    "r4.tr_best_epoch_min": 1, "r4.tr_best_epoch_max": 2,
    "r4.rq1.min_p": 0.13,
    "r4.perf.mcc.diff": 0.099, "r4.perf.mcc.ci_low": -0.157, "r4.perf.mcc.ci_high": 0.354,
    "r4.rq1.mcc.rf_wins": 1, "r4.rq1.mcc.tr_wins": 4,
    "r4.novelty.full.novelty_recall.rf": 0.719, "r4.novelty.set_size.novelty_recall.rf": 0.484,
    "r4.novelty.full.novelty_recall.tr": 0.586, "r4.novelty.set_size.novelty_recall.tr": 0.229,
    "r4.rq2.recall_drop_pct.rf": 33, "r4.rq2.recall_drop_pct.tr": 61,
    "r4.rq3.coverage_sd_ratio_min": 9,
}

# Table VIII (alpha sensitivity, beta = 0.10), printed with three decimals.
_T7 = {"rf": [(0.05, 0.883, 0.966, 0.806, 0.845, 0.528), (0.10, 0.755, 0.918, 0.483, 0.677, 0.267),
              (0.20, 0.535, 0.791, 0.201, 0.565, 0.098)],
       "tr": [(0.05, 0.816, 0.963, 0.596, 0.697, 0.379), (0.10, 0.620, 0.893, 0.176, 0.447, 0.107),
              (0.20, 0.401, 0.779, 0.175, 0.516, 0.180)]}
for _arm, _rows in _T7.items():
    for _a, _tau, _cov, _abst, _rec, _recs in _rows:
        _b = f"robust.a{_a:.2f}.b0.10.{_arm}"
        ARTICLE.update({f"{_b}.tau": _tau, f"{_b}.coverage": _cov, f"{_b}.abstention": _abst,
                        f"{_b}.novelty_recall": _rec, f"{_b}.novelty_recall_set_size": _recs})

# Keys whose article statement is an inequality, not a rounded value.
UPPER_BOUND = {"rq3.coverage_sd_theory_max"}
LOWER_BOUND = {"rq3.coverage_sd_ratio_min", "r4.rq3.coverage_sd_ratio_min", "r4.rq1.min_p",
               "robust.a0.05.b0.10.rf.tau_min", "robust.a0.05.b0.10.tr.tau_min"}


def decimals(key):
    """Number of decimals with which the article prints each value."""
    if key.startswith("dataset.missing_pct"):
        return 1
    if key.startswith(("rq2.recall_drop_pct", "r4.rq2.recall_drop_pct")) or key.endswith("dispersion_ratio"):
        return 0
    if key.startswith(("rq3.effective_units", "cost.fit_seconds")):
        return 0
    if key.startswith(("rq3.abstention_m", "robust.full_minus")):
        return 2
    if key == "rq3.empty_set_rate.tr":
        return 4          # printed as 0.18%
    if key.startswith("robust.wells") or key.endswith("empty_set_max"):
        return 2
    if key.endswith("prevalence_test") or key.startswith("rq1.") or key == "rq3.ambiguous_rate.rf":
        return 2
    return 3
