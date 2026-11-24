"""Three-state decision rule and evaluation metrics  [Sec. III].

The Unknown state is not a learned class.  It is the union of two criteria
with explicit error control: the split conformal predictor (Proposition 1:
marginal coverage >= 1 - alpha) and a conformal atypicality test on the
internal representation, at level beta.  By Proposition 2, the probability of
a wrong emitted label is <= alpha, and abstention on the known regime is
<= P(|C(X)| != 1) + beta.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from . import config as C


# ---------------------------------------------------------------------------
# Least-ambiguous split conformal predictor (Sadinle et al., 2019)
# ---------------------------------------------------------------------------
def conformal_quantile_level(n, level):
    """Finite-sample corrected quantile level ceil((n+1)(1-level))/n."""
    return min(1.0, np.ceil((n + 1) * (1 - level)) / n)


def conformal_threshold(proba_cal, y_cal, alpha=None):
    """Split-conformal threshold on s = 1 - p(y|x).

    np.quantile(..., method="higher") at level ceil((n+1)(1-alpha))/n returns
    the order statistic just ABOVE the ceil((n+1)(1-alpha))-th (for k < n).
    The threshold is therefore conservative by at most one order statistic
    (coverage changes by <= 1/n); kept as is so that every released number
    is reproduced exactly.  temporal._threshold_at uses the exact statistic.
    """
    alpha = C.ALPHA if alpha is None else alpha
    s = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    return float(np.quantile(s, conformal_quantile_level(len(s), alpha), method="higher"))


def nonconformity(proba, y):
    """s(X, y) = 1 - p(y | X) at the true label."""
    return 1.0 - proba[np.arange(len(y)), y]


# ---------------------------------------------------------------------------
# Well-level calibration (task T23).  The exchangeable unit is the well, not
# the window.  Both functions take any score (nonconformity s or atypicality
# r) and the well of each calibration window.
# ---------------------------------------------------------------------------
def within_group_quantile(scores, level):
    """Order statistic ceil(n (1 - level)) of one group's scores.

    By construction at least a fraction 1 - level of the group's scores is
    <= the returned value (a deterministic property, no assumption needed).
    """
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil(len(s) * (1 - level)))
    return float(s[min(max(k, 1), len(s)) - 1])


def well_quantile_threshold(scores, groups, level, delta=None):
    """Threshold with a per-well guarantee.

    q_g is the within-well (1 - level) order statistic of well g.  The
    threshold is the ceil((K+1)(1-delta))-th smallest of the K values q_g.

    Guarantee.  If the K calibration wells and a new well are exchangeable as
    whole units, and the score is fitted independently of them, then q_new is
    exchangeable with q_1..q_K, so P(q_new <= threshold) >= 1 - delta; and
    q_new <= threshold implies that at least a fraction 1 - level of the new
    well's windows have score <= threshold.  Hence, with probability >= 1 -
    delta over the draw of the new well, its empirical coverage (for s) or its
    non-firing rate (for r) is >= 1 - level.

    The construction (a within-group set, then an order statistic across
    groups) is that of the "double conformal" method of Dunn, Wasserman and
    Ramdas (2023, Method 0), and with level -> 0 it is the per-unit block
    calibration on the maximum score of Busico (2027).  The guarantee differs
    from Dunn et al.: theirs is marginal coverage of one new observation from
    a new group (and needs equal group sizes); this one bounds the fraction
    of windows covered WITHIN the new well, with probability >= 1 - delta.

    The order statistic exists only if ceil((K+1)(1-delta)) <= K, that is
    K >= (1 - delta)/delta (9 wells at delta = 0.10); otherwise the threshold
    is +inf (the set always contains every label; the detector never fires).
    Returns (threshold, K).
    """
    delta = C.WELL_DELTA if delta is None else delta
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    q = np.sort([within_group_quantile(scores[groups == g], level)
                 for g in np.unique(groups)])
    K = len(q)
    r = int(np.ceil((K + 1) * (1 - delta)))
    return (float(q[r - 1]) if r <= K else float("inf")), K


def well_balanced_threshold(scores, groups, level):
    """Quantile of the well-balanced empirical CDF (each well weight 1/K).

    Targets coverage averaged over wells rather than over windows, so that the
    largest wells no longer dominate the calibration.  This is the "pooling
    CDFs" method of Dunn, Wasserman and Ramdas (2023, JASA, Method 1), which
    is asymptotically valid as K grows; here the level is also inflated by
    (K+1)/K as in split conformal.  No finite-sample guarantee is claimed.
    Returns (threshold, K).
    """
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    ug = np.unique(groups)
    K = len(ug)
    w = np.zeros(len(scores))
    for g in ug:
        m = groups == g
        w[m] = 1.0 / (K * m.sum())
    order = np.argsort(scores, kind="stable")
    cdf = np.cumsum(w[order])
    target = min(1.0, (1 - level) * (K + 1) / K)
    i = int(np.searchsorted(cdf, target - 1e-12))
    return (float(scores[order][i]) if i < len(scores) else float("inf")), K


def well_subsample_threshold(scores, groups, level, B=100, seed=0):
    """Repeated-subsampling threshold of Dunn, Wasserman and Ramdas (2023,
    JASA, Method 3), for a one-sided score.

    Each of B subsamples takes one window uniformly at random from every
    calibration well, so it holds K i.i.d. draws from the mixture over wells.
    For a candidate score u, the conformal p-value in subsample b is
    pi_b(u) = (1 + #{subsample scores >= u}) / (K + 1); the threshold is the
    largest u with mean_b pi_b(u) >= level.  Target: coverage of a new window
    from a NEW well (averaged over wells, each well weight 1), guaranteed
    >= 1 - 2 level (their Theorem 6), close to 1 - level in practice.
    Contrast with well_quantile_threshold, whose target is the fraction of
    windows covered within the new well.  Returns (threshold, K).
    """
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    rng = np.random.default_rng(5000 + seed)
    ug = np.unique(groups)
    K = len(ug)
    idx = [np.where(groups == g)[0] for g in ug]
    sub = np.array([[scores[rng.choice(i)] for i in idx] for _ in range(B)])   # (B, K)
    cand = np.unique(sub)
    # mean p-value at each candidate, vectorized: count of scores >= u per subsample
    ge = (sub[:, :, None] >= cand[None, None, :]).sum(1)          # (B, n_cand)
    pbar = ((1 + ge) / (K + 1)).mean(0)
    ok = cand[pbar >= level]
    return (float(ok.max()) if ok.size else float("-inf")), K


def prediction_sets(proba, tau):
    """Boolean (n, 2): label y is in C(x) iff 1 - p(y|x) <= tau."""
    return (1.0 - proba) <= tau


# ---------------------------------------------------------------------------
# Conformal atypicality test on the representation
# ---------------------------------------------------------------------------
class AtypicalityDetector:
    """Mahalanobis-type quadratic form under a shrunk covariance, calibrated
    at level beta on the calibration subset."""

    def __init__(self, beta=None, shrinkage=None):
        self.beta = C.BETA if beta is None else beta
        self.shrinkage = C.SHRINKAGE if shrinkage is None else shrinkage

    def fit(self, Z_train):
        Z = np.asarray(Z_train, np.float64)
        self.mu = Z.mean(0)
        Zc = Z - self.mu
        S = (Zc.T @ Zc) / max(len(Z) - 1, 1)
        d = S.shape[0]
        S = (1 - self.shrinkage) * S + self.shrinkage * np.trace(S) / d * np.eye(d)
        self.P = np.linalg.pinv(S)
        return self

    def score(self, Z):
        Zc = np.asarray(Z, np.float64) - self.mu
        return np.einsum("ij,jk,ik->i", Zc, self.P, Zc)

    def calibrate(self, Z_cal):
        s = self.score(Z_cal)
        self.rho = float(np.quantile(s, conformal_quantile_level(len(s), self.beta),
                                     method="higher"))
        return self


def three_state_decision(proba, tau, r, rho):
    """Decision in {0, 1, -1}; -1 is Unknown.  A label is emitted iff the
    conformal set is a singleton and the input is not atypical."""
    sets = prediction_sets(proba, tau)
    single = sets.sum(1) == 1
    dec = np.full(len(proba), -1, np.int8)
    dec[single] = np.argmax(sets[single], axis=1)
    dec[r > rho] = -1
    return dec


def set_size_only_decision(proba, tau):
    """Ablation: Unknown whenever |C(X)| != 1 (no atypicality test)."""
    sets = prediction_sets(proba, tau)
    card = sets.sum(1)
    return np.where(card == 1, proba.argmax(1), -1).astype(np.int8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y, p, bins=None):
    bins = bins or C.ECE_BINS
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage(y, p):
    """Risk-coverage curve (sorted by decreasing confidence) and its area."""
    conf = p.max(1)
    err = (p.argmax(1) != y).astype(float)
    e = err[np.argsort(-conf)]   # numpy default sort, as in the original runs
    k = np.arange(1, len(e) + 1)
    cov, risk = k / len(e), np.cumsum(e) / k
    return cov, risk, float(np.trapezoid(risk, cov))


def closed_set_metrics(y, p):
    """Closed-set metrics of normal vs. fault.  With K > 2 classes (task T27)
    the multiclass output is collapsed to fault = 1 - p(normal), so that every
    metric stays comparable with the binary formulation, and the multiclass
    balanced accuracy and macro F1 are added."""
    if p.shape[1] > 2:
        extra = {"mc_balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(1))),
                 "mc_f1_macro": float(f1_score(y, p.argmax(1), average="macro", zero_division=0))}
        y = (np.asarray(y) > 0).astype(int)
        p = np.column_stack([p[:, 0], 1.0 - p[:, 0]])
        out = closed_set_metrics(y, p)
        out.update(extra)
        return out
    pred = p.argmax(1)
    out = {"accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
           "mcc": matthews_corrcoef(y, pred)}
    two = len(np.unique(y)) > 1
    out["roc_auc"] = roc_auc_score(y, p[:, 1]) if two else float("nan")
    out["pr_auc"] = average_precision_score(y, p[:, 1]) if two else float("nan")
    out["ece"] = expected_calibration_error(y, p)
    out["aurc"] = risk_coverage(y, p)[2]
    return {k: float(v) for k, v in out.items()}


def selective_metrics(y, dec):
    """Performance restricted to emitted decisions."""
    emitted = dec >= 0
    out = {"abstention_rate": float(1.0 - emitted.mean())}
    if emitted.sum() and len(np.unique(y[emitted])) > 1:
        out["balanced_accuracy_emitted"] = float(balanced_accuracy_score(y[emitted], dec[emitted]))
        out["mcc_emitted"] = float(matthews_corrcoef(y[emitted], dec[emitted]))
        out["error_emitted"] = float((dec[emitted] != y[emitted]).mean())
    else:
        out.update(balanced_accuracy_emitted=float("nan"), mcc_emitted=float("nan"),
                   error_emitted=float("nan"))
    out["wrong_label_rate"] = float(((dec >= 0) & (dec != y)).mean())
    return out


def novelty_metrics(dec_known, dec_unseen):
    """Recall/precision of the Unknown state on unseen windows and undue
    abstention on known-regime windows."""
    tp = int((dec_unseen < 0).sum())
    fn_ = int((dec_unseen >= 0).sum())
    fp = int((dec_known < 0).sum())
    rec = tp / max(tp + fn_, 1)
    prec = tp / max(tp + fp, 1)
    return {"novelty_recall": float(rec), "novelty_precision": float(prec),
            "novelty_f1": float(2 * rec * prec / max(rec + prec, 1e-12)),
            "undue_abstention": float(fp / max(len(dec_known), 1))}
