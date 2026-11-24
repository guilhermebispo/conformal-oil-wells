"""Novelty scores and their controlled evaluation  (task T25, RQ3).

Fit-time part (called by experiment.run_seed): every score is computed on the
calibration, known-regime test and unseen windows and stored per window in
arrays/seed*_<arm>.npz as nov_<score>_{cal,test,unseen}.  Higher = more novel.

  msp    1 - max softmax probability (confidence baseline; Hendrycks & Gimpel)
  energy -logsumexp of the logits (Transformer only; Liu et al., 2020)
  maha   squared Mahalanobis distance on the representation, shrunk covariance
         (the atypicality score r of the article)
  knn    mean distance to the k nearest training representations (standardized)
  iforest  Isolation Forest anomaly score on the representation

kNN and Isolation Forest use a seeded subsample of at most NOVELTY_REF_MAX
training representations, so that their cost is bounded.

Analysis part (stage "novelty", from the stored arrays, no refitting).  For
every score it reports
  auroc            unseen vs. all known-regime test windows
  auroc_same_wells unseen vs. known windows of the same test wells only; this
                   removes the well confound (a score that recognizes the well
                   rather than the regime falls to about 0.5 here)
  recall, false_alarm, lift
                   the score thresholded as a conformal outlier test at level
                   beta on the calibration windows (window-level) and with the
                   well-level threshold of T23: recall on unseen windows, firing
                   rate on known test windows, and lift = recall - false alarm
                   (a random abstainer has lift 0)
  recall_at_fa     recall when the threshold is set so that the false-alarm
                   rate on known TEST windows is NOVELTY_FA (oracle, threshold
                   free comparison across scores; not deployable)

Writes novelty.json and adds nov.* keys to results.json (when present).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import conformal as CF

NOVELTY_FA = 0.10
ARMS = (("rf", "rf"), ("transformer", "tr"))   # default order; see _arms()


def _arms(rows):
    """(arm, tag) of every arm present in the rows, RF and Transformer first."""
    present = {r["arm"] for r in rows}
    order = [a for a in C.ARM_TAGS if a in present] + sorted(present - set(C.ARM_TAGS))
    return [(a, C.ARM_TAGS.get(a, a)) for a in order]


# ---------------------------------------------------------------------------
# Fit-time scores
# ---------------------------------------------------------------------------
def fit_time_scores(Z_ref, Zs, probas, logits, seed):
    """Novelty scores of several window sets.

    Z_ref   training representation (fit windows of the final model)
    Zs      dict name -> representation of a window set (cal, test, unseen)
    probas  dict name -> predicted probabilities of the same sets
    logits  dict name -> logits, or None when the model has none (RF)
    Returns dict "nov_<score>_<set>" -> float32 array.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(4000 + seed)
    Z_ref = np.asarray(Z_ref, np.float64)
    if len(Z_ref) > C.NOVELTY_REF_MAX:
        Z_ref = Z_ref[rng.choice(len(Z_ref), C.NOVELTY_REF_MAX, replace=False)]
    mu, sd = Z_ref.mean(0), Z_ref.std(0) + 1e-8
    ref = (Z_ref - mu) / sd
    nn = NearestNeighbors(n_neighbors=C.NOVELTY_KNN_K).fit(ref)
    iso = IsolationForest(n_estimators=C.NOVELTY_IF_TREES, random_state=seed,
                          n_jobs=1).fit(ref)
    out = {}
    for name, Z in Zs.items():
        z = (np.asarray(Z, np.float64) - mu) / sd
        out[f"nov_knn_{name}"] = nn.kneighbors(z)[0].mean(1)
        out[f"nov_iforest_{name}"] = -iso.score_samples(z)
        out[f"nov_msp_{name}"] = 1.0 - np.asarray(probas[name]).max(1)
        if logits is not None:
            lg = np.asarray(logits[name], np.float64)
            m = lg.max(1, keepdims=True)
            out[f"nov_energy_{name}"] = -(m[:, 0] + np.log(np.exp(lg - m).sum(1)))
    return {k: v.astype(np.float32) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def score_names(d):
    """Scores available in one array file (maha is the stored r)."""
    names = sorted({k.split("_")[1] for k in d if k.startswith("nov_")})
    return ["maha"] + names


def _get(d, score, part):
    if score == "maha":
        return d[{"cal": "r_cal", "test": "r_test", "unseen": "r_unseen"}[part]].astype(float)
    return d[f"nov_{score}_{part}"].astype(float)


def evaluate(d, score, beta=None, delta=None):
    beta = C.BETA if beta is None else beta
    delta = C.WELL_DELTA if delta is None else delta
    cal, te, un = (_get(d, score, p) for p in ("cal", "test", "unseen"))
    same = np.isin(d["well_test"], np.unique(d["well_unseen"]))
    out = {"auroc": _auroc(te, un), "auroc_same_wells": _auroc(te[same], un),
           "n_known_same_wells": int(same.sum())}
    thr = {"window": float(np.quantile(cal, CF.conformal_quantile_level(len(cal), beta),
                                       method="higher")),
           "well": CF.well_quantile_threshold(cal, d["well_cal"], beta, delta)[0]}
    for name, t in thr.items():
        rec, fa = float((un > t).mean()), float((te > t).mean())
        out[f"{name}.threshold"] = t
        out[f"{name}.recall"], out[f"{name}.false_alarm"], out[f"{name}.lift"] = rec, fa, rec - fa
    t_or = float(np.quantile(te, 1 - NOVELTY_FA))
    out["recall_at_fa"] = float((un > t_or).mean())
    return out


SUMMARY = ("auroc", "auroc_same_wells", "window.recall", "window.false_alarm", "window.lift",
           "well.recall", "well.false_alarm", "well.lift", "recall_at_fa")


def analyse(arrs):
    rows = []
    for (seed, arm), d in sorted(arrs.items()):
        if not any(k.startswith("nov_") for k in d):
            continue
        for score in score_names(d):
            rows.append({"seed": seed, "arm": arm, "score": score, **evaluate(d, score)})
    return rows


def summarize(rows):
    out = {}
    for arm, tag in _arms(rows):
        for score in sorted({r["score"] for r in rows if r["arm"] == arm}):
            r = [x for x in rows if x["arm"] == arm and x["score"] == score]
            for k in SUMMARY:
                v = np.array([x[k] for x in r], float)
                out[f"nov.{tag}.{score}.{k}"] = float(np.nanmean(v))
                out[f"nov.{tag}.{score}.{k}.sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
            out[f"nov.{tag}.{score}.n_seeds"] = len(r)
    return out


def run(out_dir: Path, log=print, arrays_dir: Path | None = None):
    from .asset import _load
    out_dir = Path(out_dir)
    rows = analyse(_load(arrays_dir or out_dir / "arrays"))
    if not rows:
        log("no stored novelty scores (run the experiment with this version); skipped")
        return None
    summ = summarize(rows)
    (out_dir / "novelty.json").write_text(json.dumps(
        {"beta": C.BETA, "delta": C.WELL_DELTA, "fa_oracle": NOVELTY_FA,
         "summary": summ, "rows": rows}, indent=1))
    res = out_dir / "results.json"
    if res.exists():
        R = json.loads(res.read_text())
        R.update(summ)
        res.write_text(json.dumps(R, indent=1))
    for arm, tag in _arms(rows):
        for score in sorted({r["score"] for r in rows if r["arm"] == arm}):
            b = f"nov.{tag}.{score}"
            log(f"  {tag} {score:8s} AUROC={summ[b + '.auroc']:.3f} same-wells={summ[b + '.auroc_same_wells']:.3f} "
                f"lift(win)={summ[b + '.window.lift']:+.3f} lift(well)={summ[b + '.well.lift']:+.3f} "
                f"rec@FA10={summ[b + '.recall_at_fa']:.3f}")
    return summ
