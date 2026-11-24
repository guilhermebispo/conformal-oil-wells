"""Unit tests of the claims the article proves or relies on.  No 3W data needed.

Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iiot3w import config as C  # noqa: E402
from iiot3w import conformal as CF  # noqa: E402
from iiot3w import data as D  # noqa: E402
from iiot3w import splits as SP  # noqa: E402
from iiot3w import stats as ST  # noqa: E402


def _binary_problem(n, rng, shift=1.2):
    y = rng.integers(0, 2, n)
    logit = shift * (2 * y - 1) + rng.normal(0, 1.5, n)
    p1 = 1 / (1 + np.exp(-logit))
    return np.column_stack([1 - p1, p1]), y


# --------------------------------------------------------------------------
# Proposition 1: marginal coverage >= 1 - alpha under exchangeability
# --------------------------------------------------------------------------
def test_split_conformal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        p_cal, y_cal = _binary_problem(500, rng)
        p_te, y_te = _binary_problem(500, rng)
        tau = CF.conformal_threshold(p_cal, y_cal, alpha=0.1)
        sets = CF.prediction_sets(p_te, tau)
        covs.append(sets[np.arange(len(y_te)), y_te].mean())
    assert np.mean(covs) >= 0.9 - 0.005
    # and not grossly conservative: E[coverage] <= 1 - alpha + 1/(n+1)
    assert np.mean(covs) <= 0.9 + 1 / 501 + 0.01


# --------------------------------------------------------------------------
# Proposition 2: in the binary case the empty set is unreachable when
# tau >= 0.5, so set-size-only abstention comes from ambiguity alone
# --------------------------------------------------------------------------
def test_empty_set_unreachable_binary():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1, 1], 10000)
    for tau in (0.5, 0.62, 0.755, 0.9):
        assert (CF.prediction_sets(p, tau).sum(1) >= 1).all()
    # below 0.5 the empty set can occur
    assert (CF.prediction_sets(p, 0.3).sum(1) == 0).any()


def test_tau_at_least_half_when_alpha_below_error_rate():
    rng = np.random.default_rng(2)
    p_cal, y_cal = _binary_problem(2000, rng, shift=0.5)   # error rate > alpha
    assert CF.conformal_threshold(p_cal, y_cal, alpha=0.1) >= 0.5


# --------------------------------------------------------------------------
# Proposition 3: (i) wrong emitted labels <= alpha in expectation;
# (ii) abstention <= P(|C| != 1) + beta
# --------------------------------------------------------------------------
def test_proposition3_bounds():
    rng = np.random.default_rng(3)
    wrong, abst, bound = [], [], []
    for _ in range(100):
        p_cal, y_cal = _binary_problem(800, rng)
        p_te, y_te = _binary_problem(800, rng)
        Ztr, Zca, Zte = (rng.normal(size=(800, 4)) for _ in range(3))
        tau = CF.conformal_threshold(p_cal, y_cal, 0.1)
        det = CF.AtypicalityDetector(beta=0.1).fit(Ztr).calibrate(Zca)
        dec = CF.three_state_decision(p_te, tau, det.score(Zte), det.rho)
        card = CF.prediction_sets(p_te, tau).sum(1)
        wrong.append(((dec >= 0) & (dec != y_te)).mean())
        abst.append((dec < 0).mean())
        bound.append((card != 1).mean() + 0.1)
    assert np.mean(wrong) <= 0.1 + 0.01
    assert np.mean(abst) <= np.mean(bound) + 0.01


def test_atypicality_detector_level_and_power():
    rng = np.random.default_rng(4)
    det = CF.AtypicalityDetector(beta=0.1).fit(rng.normal(size=(3000, 5)))
    det.calibrate(rng.normal(size=(3000, 5)))
    assert abs((det.score(rng.normal(size=(5000, 5))) > det.rho).mean() - 0.1) < 0.02
    assert (det.score(rng.normal(3, 1, size=(500, 5))) > det.rho).mean() > 0.9


def test_set_size_only_rule_matches_full_rule_without_detector():
    rng = np.random.default_rng(5)
    p, _ = _binary_problem(1000, rng)
    a = CF.set_size_only_decision(p, 0.7)
    b = CF.three_state_decision(p, 0.7, np.zeros(len(p)), 1.0)
    assert (a == b).all()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_corrected_ttest_is_wider_than_naive():
    from scipy import stats
    d = np.array([0.03, -0.05, 0.25, -0.09, 0.19])
    r = ST.corrected_resampled_ttest(d, n_train=29000, n_test=10000)
    naive = stats.ttest_1samp(d, 0)
    assert r["p"] > naive.pvalue
    se_naive = d.std(ddof=1) / np.sqrt(5)
    assert np.isclose(r["se"], se_naive * np.sqrt(1 + 5 * 10000 / 29000))


def test_detectable_difference_floor():
    a = ST.detectable_difference(0.16, 10000, 29000, k=5)["detectable"]
    b = ST.detectable_difference(0.16, 10000, 29000, k=10**9)["detectable"]
    assert a > b > 0
    assert np.isclose(ST.detectable_difference(1, 34, 100)["se_factor"], np.sqrt(0.34))


def test_coverage_sd_theory_matches_simulation():
    rng = np.random.default_rng(6)
    n_cal, n_te, covs = 2000, 4000, []
    for _ in range(400):
        s_cal, s_te = rng.random(n_cal), rng.random(n_te)
        q = np.quantile(s_cal, CF.conformal_quantile_level(n_cal, 0.1), method="higher")
        covs.append((s_te <= q).mean())
    assert abs(np.std(covs) - ST.coverage_sd_exchangeable(n_cal, n_te)) < 0.0015


# --------------------------------------------------------------------------
# Data and partition
# --------------------------------------------------------------------------
def test_windows_are_causal_homogeneous_and_capped():
    n = 64 * 200
    X = np.random.default_rng(7).normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    cls = np.zeros(n)
    cls[5000:5100] = 103          # transient
    cls[5100:9000] = 3            # steady fault
    Xw, yw, cw = D.windows_from_instance(X, cls)
    assert (yw == 0).sum() <= C.MAX_WINDOWS_PER_INSTANCE
    assert set(np.unique(cw)) <= {0, 3}
    # every window equals the raw samples ending at its index (causal)
    ends = range(C.WINDOW_T - 1, n, C.WINDOW_STEP)
    raw = {tuple(X[e - C.WINDOW_T + 1:e + 1, 0][:3]) for e in ends}
    assert all(tuple(w[:3, 0]) in raw for w in Xw)
    # no window overlaps the transient
    for w, c in zip(Xw, cw):
        assert c in (0, 3)


def test_label_mapping():
    y = D.label_of(np.array([0, 1, 2, 7, 8, 9, 101, np.nan]))
    assert y.tolist() == [0, 2, 1, 1, 2, 2, -1, -1]


def test_split_is_well_disjoint_and_constrained():
    rng = np.random.default_rng(8)
    well = np.repeat(np.arange(30), 200)
    code = np.where(rng.random(well.size) < 0.7, 0, rng.choice([2, 3, 4, 5], well.size))
    code[well % 7 == 0] = np.where(rng.random((well % 7 == 0).sum()) < 0.2, 8, 0)
    y = D.label_of(code).astype(np.int8)
    i_tr, i_ca, i_te, i_un, info = SP.make_split(y, code, well, seed=0)
    sets = [set(well[i]) for i in (i_tr, i_ca, i_te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert (y[i_tr] < 2).all() and (y[i_ca] < 2).all() and (y[i_un] == 2).all()
    assert set(well[i_un]) <= sets[2]
    for i in (i_tr, i_ca, i_te):
        assert len(set(code[i][y[i] == 1]) & set(C.CLASS_KNOWN_FAULT)) >= 2
        assert C.MIN_PREVALENCE <= (y[i] == 1).mean() <= C.MAX_PREVALENCE


def test_split_is_deterministic():
    rng = np.random.default_rng(9)
    well = np.repeat(np.arange(30), 100)
    code = np.where(rng.random(well.size) < 0.7, 0, rng.choice([2, 3, 4, 8], well.size))
    y = D.label_of(code).astype(np.int8)
    a = SP.make_split(y, code, well, seed=3)[4]
    b = SP.make_split(y, code, well, seed=3)[4]
    assert a == b


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_quantile_level(alpha):
    n = 4459
    assert CF.conformal_quantile_level(n, alpha) == min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


# --------------------------------------------------------------------------
# Task T20: well-level partition
# --------------------------------------------------------------------------
def _synthetic_wells(seed, n_wells=30, per_well=200):
    """Wells with a well-specific mix of known fault types and some unseen."""
    rng = np.random.default_rng(seed)
    well = np.repeat(np.arange(n_wells), per_well)
    code = np.where(rng.random(well.size) < 0.7, 0, rng.choice([2, 3, 4, 5], well.size))
    code[well % 7 == 0] = np.where(rng.random((well % 7 == 0).sum()) < 0.2, 8, 0)
    return D.label_of(code).astype(np.int8), code, well


def test_well_split_constraints():
    y, code, well = _synthetic_wells(8)
    i_tr, i_ca, i_te, i_un, info = SP.make_split_wells(y, code, well, seed=0)
    sets = [set(well[i]) for i in (i_tr, i_ca, i_te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert (y[i_tr] < 2).all() and (y[i_ca] < 2).all() and (y[i_te] < 2).all()
    assert (y[i_un] == 2).all() and set(well[i_un]) <= sets[2]
    known_total = (y < 2).sum()
    for name, i in zip(("train", "cal", "test"), (i_tr, i_ca, i_te)):
        assert len(set(well[i])) >= C.WS_MIN_WELLS[name]
        assert i.size / known_total >= C.WS_MIN_SHARE[name]
        assert C.MIN_PREVALENCE <= (y[i] == 1).mean() <= C.MAX_PREVALENCE
    t = lambda i: set(code[i][y[i] == 1].tolist())
    assert (t(i_ca) | t(i_te)) <= t(i_tr)
    assert info["mode"] == "wells"


def test_partition_dispatch_keeps_article_protocol(monkeypatch):
    y, code, well = _synthetic_wells(8)
    a = SP.partition(y, code, well, seed=0)[4]
    assert a == SP.make_split(y, code, well, seed=0)[4]   # default: windows
    monkeypatch.setattr(C, "SPLIT_MODE", "wells")
    assert SP.partition(y, code, well, seed=0)[4]["mode"] == "wells"


# --------------------------------------------------------------------------
# Task T23: well-level calibration
# --------------------------------------------------------------------------
def test_within_group_quantile_is_deterministic_bound():
    rng = np.random.default_rng(1)
    for n in (1, 7, 60, 333):
        s = rng.random(n)
        q = CF.within_group_quantile(s, 0.1)
        assert (s <= q).mean() >= 0.9


def test_well_quantile_needs_enough_wells():
    rng = np.random.default_rng(2)
    for K, finite in ((8, False), (9, True)):
        g = np.repeat(np.arange(K), 50)
        t, k = CF.well_quantile_threshold(rng.random(g.size), g, 0.1, delta=0.1)
        assert k == K and np.isfinite(t) == finite


def test_well_quantile_per_well_guarantee_under_well_heterogeneity():
    """Wells differ (random shift of the score); windows within a well are
    correlated.  Window-level calibration then fails for many new wells,
    whereas the well-level threshold covers >= 1 - alpha of the windows of a
    new well with probability >= 1 - delta."""
    rng = np.random.default_rng(3)
    alpha = delta = 0.1
    ok_well, ok_win = [], []
    for _ in range(400):
        K = 15
        shift = rng.normal(0, 1, K + 1)
        n = rng.integers(30, 300, K + 1)
        s = [shift[g] + rng.normal(0, 0.3, n[g]) for g in range(K + 1)]
        s_cal = np.concatenate(s[:K])
        g_cal = np.concatenate([np.full(n[g], g) for g in range(K)])
        t_well, _ = CF.well_quantile_threshold(s_cal, g_cal, alpha, delta)
        t_win = np.quantile(s_cal, CF.conformal_quantile_level(len(s_cal), alpha), method="higher")
        ok_well.append((s[K] <= t_well).mean() >= 1 - alpha)
        ok_win.append((s[K] <= t_win).mean() >= 1 - alpha)
    assert np.mean(ok_well) >= 1 - delta - 0.03     # guarantee (MC tolerance)
    assert np.mean(ok_win) < np.mean(ok_well) - 0.1  # window-level is not per well


def test_well_balanced_equal_weight():
    # One huge well with low scores and nine small ones with high scores:
    # the window quantile follows the huge well, the balanced one does not.
    s = np.r_[np.zeros(10000), np.ones(9 * 10)]
    g = np.r_[np.zeros(10000), np.repeat(np.arange(1, 10), 10)]
    t_bal, K = CF.well_balanced_threshold(s, g, 0.1)
    t_win = np.quantile(s, CF.conformal_quantile_level(len(s), 0.1), method="higher")
    assert K == 10 and t_win == 0.0 and t_bal == 1.0


# --------------------------------------------------------------------------
# Task T21: nested, well-grouped selection
# --------------------------------------------------------------------------
def test_grouped_folds_hold_whole_wells_and_both_classes():
    y, code, well = _synthetic_wells(11)
    idx = np.where(y < 2)[0]
    folds, info = SP.grouped_folds(y, well, idx, seed=0, k=3)
    seen = []
    for i_fit, i_val in folds:
        assert not set(well[i_fit]) & set(well[i_val])
        assert np.bincount(y[i_val], minlength=2)[:2].min() >= C.NESTED_MIN_PER_CLASS
        seen.extend(i_val.tolist())
    assert sorted(seen) == sorted(idx.tolist())   # every window validated once
    assert info["k"] == 3


def test_grouped_folds_reject_impossible_constraint(monkeypatch):
    y, code, well = _synthetic_wells(12)
    idx = np.where(y < 2)[0]
    monkeypatch.setattr(C, "NESTED_MIN_PER_CLASS", 10 ** 6)
    monkeypatch.setattr(C, "NESTED_MAX_ATTEMPTS", 50)
    with pytest.raises(RuntimeError):
        SP.grouped_folds(y, well, idx, seed=0, k=3)


# --------------------------------------------------------------------------
# Task T25: novelty scores and controls
# --------------------------------------------------------------------------
def test_fit_time_scores_rank_shifted_windows_higher():
    from iiot3w import novelty as NV
    rng = np.random.default_rng(5)
    Z_ref = rng.normal(0, 1, (2000, 6))
    Zs = {"test": rng.normal(0, 1, (300, 6)), "unseen": rng.normal(3, 1, (300, 6))}
    p = {k: np.full((len(v), 2), 0.5) for k, v in Zs.items()}
    lg = {k: np.zeros((len(v), 2)) for k, v in Zs.items()}
    out = NV.fit_time_scores(Z_ref, Zs, p, lg, seed=0)
    for s in ("knn", "iforest"):
        assert NV._auroc(out[f"nov_{s}_test"], out[f"nov_{s}_unseen"]) > 0.95
    assert {"nov_msp_test", "nov_energy_unseen"} <= set(out)
    assert "nov_energy_test" not in NV.fit_time_scores(Z_ref, Zs, p, None, seed=0)


def test_same_well_auroc_removes_well_confound():
    """A score that only encodes the well separates unseen windows globally
    but is at chance within the wells that hold them."""
    from iiot3w import novelty as NV
    rng = np.random.default_rng(6)
    n = 400
    # Well 0 (no unseen windows) holds 3/4 of the known windows; well 1 holds
    # the unseen ones.  Expected global AUROC about 3/4 * 0.98 + 1/4 * 0.5.
    well_test = np.r_[np.zeros(3 * n), np.ones(n)].astype(int)
    s_test = np.r_[rng.normal(0, 1, 3 * n), rng.normal(3, 1, n)]  # score = well level
    s_un = rng.normal(3, 1, 200)
    d = {"r_cal": rng.normal(0, 1, 500), "r_test": s_test, "r_unseen": s_un,
         "well_test": well_test, "well_unseen": np.ones(200, int),
         "well_cal": np.repeat(np.arange(10), 50)}
    ev = NV.evaluate(d, "maha")
    assert ev["auroc"] > 0.8 and abs(ev["auroc_same_wells"] - 0.5) < 0.1


def test_well_subsample_well_averaged_coverage():
    """Repeated subsampling (Dunn et al. 2023): coverage of a new window from a
    new well, averaged over wells, is >= 1 - 2 alpha (and near 1 - alpha)."""
    rng = np.random.default_rng(7)
    alpha, covs = 0.1, []
    for _ in range(200):
        K = 20
        shift = rng.normal(0, 1, K + 1)
        n = rng.integers(20, 200, K + 1)
        s = [shift[g] + rng.normal(0, 0.5, n[g]) for g in range(K + 1)]
        g_cal = np.concatenate([np.full(n[g], g) for g in range(K)])
        t, k = CF.well_subsample_threshold(np.concatenate(s[:K]), g_cal, alpha, B=50, seed=0)
        covs.append((s[K] <= t).mean())
    assert k == 20 and np.mean(covs) >= 1 - 2 * alpha
    assert np.mean(covs) >= 1 - alpha - 0.05


# --------------------------------------------------------------------------
# Tasks T24 and T28: online calibration and event metrics
# --------------------------------------------------------------------------
def test_aci_restores_long_run_coverage_of_a_shifted_well():
    """Calibration scores from one distribution, test well shifted: the fixed
    threshold under-covers the well, ACI brings its coverage near 1 - alpha."""
    from iiot3w import temporal as TM
    rng = np.random.default_rng(10)
    alpha = 0.1
    cal = np.sort(rng.beta(2, 8, 2000))            # nonconformity scores in [0, 1]
    n = 3000
    y = rng.integers(0, 2, n)
    s_true = np.clip(rng.beta(2, 8, n) + 0.25, 0, 1)   # shifted: harder well
    p = np.zeros((n, 2))
    p[np.arange(n), y] = 1 - s_true
    p[np.arange(n), 1 - y] = s_true
    hit_fixed = (s_true <= TM._threshold_at(cal, alpha)).mean()
    hit, size, _ = TM.aci_well(p, y, cal, alpha, gamma=0.01, delay=10)
    assert hit_fixed < 0.75
    assert abs(hit.mean() - (1 - alpha)) < 0.03


def test_event_delay_and_detection():
    from iiot3w import temporal as TM
    # One known-fault event of 5 windows (alarm from the 3rd), one normal instance.
    p_test = np.array([[0.1, 0.9]] * 5 + [[0.9, 0.1]] * 3)
    p_test[:2] = [0.9, 0.1]                 # first two windows look normal
    d = {"p_test": p_test, "r_test": np.zeros(8), "y_test": np.array([1] * 5 + [0] * 3),
         "p_unseen": np.array([[0.5, 0.5]]), "r_unseen": np.array([10.0])}
    extra = {"inst_test": np.array([0] * 5 + [1] * 3), "t_test": np.arange(8) * 160,
             "inst_unseen": np.array([2]), "t_unseen": np.array([0])}
    ev = TM.event_analysis(d, extra, tau=0.5, rho=1.0)
    assert ev["known_fault"]["detected"] == 1.0
    assert ev["known_fault"]["delay_median_min"] == pytest.approx(2 * 160 / 60)
    assert ev["unseen_unknown"]["detected"] == 1.0
    assert ev["normal"]["instances_with_alarm"] == 0.0


def test_attach_refuses_a_mismatched_partition():
    from iiot3w import temporal as TM
    y, code, well = _synthetic_wells(8)
    inst = np.arange(len(y)); t_end = np.arange(len(y))
    i_tr, i_ca, i_te, i_un, _ = SP.partition(y, code, well, seed=0)
    good = {"well_test": well[i_te], "y_test": y[i_te], "code_test": code[i_te],
            "well_cal": well[i_ca], "well_unseen": well[i_un]}
    TM.attach(good, 0, (y, code, well, inst, t_end))
    bad = dict(good, y_test=y[i_te][::-1])
    with pytest.raises(RuntimeError):
        TM.attach(bad, 0, (y, code, well, inst, t_end))


# --------------------------------------------------------------------------
# Tasks T27 (multiclass), T29 (baselines), T26 (rotation)
# --------------------------------------------------------------------------
def _toy_windows(n=400, K=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    X = rng.normal(0, 1, (n, C.WINDOW_T, 3)).astype(np.float32)
    X += (y[:, None, None] * 0.8 * np.linspace(0, 1, C.WINDOW_T)[None, :, None]).astype(np.float32)
    return X, y


@pytest.mark.parametrize("arm", ["rf", "lgbm", "cnn", "transformer"])
def test_arms_multiclass_probabilities(monkeypatch, arm):
    from iiot3w import experiment as E
    monkeypatch.setattr(C, "N_CLASSES", 3)
    monkeypatch.setattr(C, "TR_EPOCHS", 1)
    cls, grid, is_nn = E.ARM_REGISTRY[arm]
    X, y = _toy_windows()
    m = cls(getattr(C, grid)[0], 0).fit(X, y, X[:50], y[:50], epochs=1 if is_nn else None)
    p = m.predict_proba(X[:20])
    assert p.shape == (20, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert m.represent(X[:5]).shape[0] == 5


def test_full_proba_pads_classes_absent_from_training(monkeypatch):
    from iiot3w import models as M
    monkeypatch.setattr(C, "N_CLASSES", 4)
    X, y = _toy_windows(K=3)            # class 3 never seen
    m = M.RFArm(C.RF_GRID[1], 0).fit(X, y)
    p = m.predict_proba(X[:10])
    assert p.shape == (10, 4) and np.all(p[:, 3] == 0)


def test_model_labels_and_rotation(monkeypatch):
    from iiot3w import experiment as E
    code = np.array([0, 2, 3, 7, 8, 1])
    monkeypatch.setattr(C, "LABEL_MODE", "multiclass")
    y = D.label_of(code)
    assert E.model_labels(y, code).tolist() == [0, 1, 2, 6, -1, -1]
    monkeypatch.setattr(C, "CLASS_UNOBSERVED", [3])
    monkeypatch.setattr(C, "CLASS_KNOWN_FAULT", [1, 2, 4, 5, 6, 7, 8, 9])
    y = D.label_of(code)
    assert y.tolist() == [0, 1, 2, 1, 1, 1]              # code 3 is now unseen
    assert E.model_labels(y, code).tolist() == [0, 2, -1, 6, 7, 1]


def test_multiclass_closed_metrics_collapse_to_fault_vs_normal():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]])
    y = np.array([0, 1, 2, 0])
    m = CF.closed_set_metrics(y, p)
    assert m["balanced_accuracy"] == 1.0 and m["roc_auc"] == 1.0
    assert m["mc_balanced_accuracy"] == 1.0


# --------------------------------------------------------------------------
# Audit 2026-11-23: order statistic of the thresholds and Holm correction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [100, 999, 2619, 9816])
def test_conformal_threshold_is_conservative_by_at_most_one_order_statistic(n):
    rng = np.random.default_rng(n)
    p1 = rng.random(n)
    proba = np.column_stack([1 - p1, p1])
    y = np.zeros(n, int)
    s = np.sort(1.0 - proba[np.arange(n), y])
    k = int(np.ceil((n + 1) * (1 - 0.10)))
    tau = CF.conformal_threshold(proba, y, 0.10)
    assert s[k - 1] <= tau <= s[min(k, n - 1)]


def test_holm_step_down():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from families_holm import holm
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.09, 0.09 (monotone), 0.20
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.20])
    assert np.all(holm(p) >= p) and np.all(holm(p) <= 1)
