"""Pairwise comparison of the model families with Holm correction  [RQ4].

Reads a merged experiment.json with several arms (run03_families) and writes
families.json: per-arm means and SDs, all pairwise corrected resampled t-tests
(Nadeau and Bengio, 2003) and Holm-adjusted p-values within each metric.

    python scripts/families_holm.py <experiment.json> <out.json>
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iiot3w import stats as ST  # noqa: E402

METRICS = {"mcc": ("closed", "mcc"), "roc_auc": ("closed", "roc_auc"),
           "balanced_accuracy": ("closed", "balanced_accuracy"),
           "coverage": ("conformal", "coverage"),
           "abstention": ("conformal", "abstention_rate"),
           "mcc_emitted": ("conformal", "mcc_emitted"),
           "novelty_recall": ("novelty_full", "novelty_recall")}
TESTED = ("mcc", "roc_auc", "balanced_accuracy")


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main(src, dst):
    exp = json.loads(Path(src).read_text())
    seeds = sorted(exp["seeds"], key=int)
    arms = list(exp["seeds"][seeds[0]]["arms"])
    n_tr = float(np.mean([exp["seeds"][s]["split"]["n_train"] for s in seeds]))
    n_te = float(np.mean([exp["seeds"][s]["split"]["n_test"] for s in seeds]))

    def vals(arm, path):
        out = []
        for s in seeds:
            d = exp["seeds"][s]["arms"][arm]
            for k in path:
                d = d[k]
            out.append(d)
        return np.asarray(out, float)

    R = {"seeds": seeds, "arms": arms, "n_train_mean": n_tr, "n_test_mean": n_te,
         "per_seed": {}, "summary": {}, "pairs": {}}
    for m, path in METRICS.items():
        R["per_seed"][m] = {a: vals(a, path).tolist() for a in arms}
        R["summary"][m] = {a: {"mean": float(np.nanmean(vals(a, path))),
                               "sd": float(np.nanstd(vals(a, path), ddof=1))} for a in arms}
    for m in TESTED:
        rows = []
        for a, b in itertools.combinations(arms, 2):
            d = vals(b, METRICS[m]) - vals(a, METRICS[m])
            t = ST.corrected_resampled_ttest(d, n_tr, n_te)
            rows.append({"a": a, "b": b, "diff_b_minus_a": t["mean"], "ci_low": t["ci_low"],
                         "ci_high": t["ci_high"], "p": t["p"],
                         "b_wins": int((d > 0).sum()), "a_wins": int((d < 0).sum())})
        for r, q in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = float(q)
        R["pairs"][m] = rows
    Path(dst).write_text(json.dumps(R, indent=1))
    for m in TESTED:
        print(f"== {m}")
        for a in arms:
            s = R["summary"][m][a]
            print(f"  {a:12s} {s['mean']:.3f} +- {s['sd']:.3f}")
        for r in R["pairs"][m]:
            print(f"  {r['b']:>11s} - {r['a']:<11s} {r['diff_b_minus_a']:+.3f} "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p']:.3f} "
                  f"p_holm={r['p_holm']:.3f} wins {r['b_wins']}-{r['a_wins']}")
    for m in ("coverage", "abstention", "mcc_emitted", "novelty_recall"):
        print(f"== {m}: " + ", ".join(f"{a} {R['summary'][m][a]['mean']:.3f}" for a in arms))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
