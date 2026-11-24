"""Statistical comparison and finite-sample theory  [Sec. IV-F, V-A, V-C]."""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import config as C


def corrected_resampled_ttest(diffs, n_train, n_test, ci=None):
    """Corrected resampled t-test (Nadeau and Bengio, 2003).

    The k partitions reuse the same data, so the naive paired t-test inflates
    the type I error.  The variance of the mean difference is inflated by the
    train/test overlap term: s^2 (1/k + n_test/n_train).  Returns the mean,
    t, two-sided p, and the CI half-width with the t quantile on k-1 df.
    """
    ci = C.CI if ci is None else ci
    d = np.asarray(diffs, float)
    k = len(d)
    if k < 2:
        return {"mean": float(d.mean()) if k else float("nan"), "t": float("nan"),
                "p": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    se = d.std(ddof=1) * np.sqrt(1.0 / k + n_test / max(n_train, 1))
    t = d.mean() / se if se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=k - 1) if se > 0 else float("nan")
    q = stats.t.ppf(0.5 + ci / 2.0, df=k - 1)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "se": float(se),
            "t": float(t), "p": float(p),
            "ci_low": float(d.mean() - q * se), "ci_high": float(d.mean() + q * se)}


def detectable_difference(sigma, n_test, n_train, k=None,
                          alpha=None, power=None):
    """Smallest difference detectable at the given significance and power.

    Var(mean difference) = sigma^2 (1/k + n_te/n_tr).  With k -> infinity the
    standard error tends to sigma * sqrt(n_te/n_tr) > 0, so more repartitions
    of the same wells cannot resolve smaller effects (normal approximation).
    """
    alpha = C.SIGNIFICANCE if alpha is None else alpha
    power = C.POWER if power is None else power
    r = n_test / n_train
    se = sigma * np.sqrt(r + (0.0 if k is None else 1.0 / k))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return {"se_factor": float(np.sqrt(r)), "detectable": float(z * se)}


def coverage_sd_exchangeable(n_cal, n_test, alpha=None):
    """SD of empirical test coverage across partitions if windows were
    exchangeable.

    Conditional on calibration, coverage ~ Beta(n+1-l, l) with
    l = floor((n+1) alpha) (Vovk, 2012; Angelopoulos and Bates, 2023), whose
    variance is about alpha(1-alpha)/(n+2); the test set adds binomial
    sampling variance cov(1-cov)/n_test.
    """
    alpha = C.ALPHA if alpha is None else alpha
    n = n_cal
    l = np.floor((n + 1) * alpha)
    a, b = n + 1 - l, l
    var_beta = a * b / ((a + b) ** 2 * (a + b + 1))
    cov = a / (a + b)
    var_test = cov * (1 - cov) / n_test
    return float(np.sqrt(var_beta + var_test))


def effective_units(observed_sd, alpha=None):
    """Number of exchangeable units (same for calibration and test) that would
    explain the observed SD of coverage: 2 alpha(1-alpha)/n = sd^2."""
    alpha = C.ALPHA if alpha is None else alpha
    return float(2 * alpha * (1 - alpha) / observed_sd ** 2)
