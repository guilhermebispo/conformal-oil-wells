"""Robustness R4: Transformer early stopping on held-out wells.

In the article protocol the early-stopping wells of the Transformer are part
of its training set.  The variant fits both models on the inner-fit wells
only and early-stops the Transformer on the held-out internal validation
wells; hyperparameters are those selected in the main run.  The five seeds
are run independently (one process each) and merged here.

    for s in 0 1 2 3 4; do
      python run_all.py --stages experiment --seeds $s --threads 1 \
        --holdout-early-stopping --selection-from outputs/experiment.json \
        --out outputs_holdout/s$s &
    done; wait
    python scripts/variant_r4.py

Writes outputs_holdout/experiment.json, results.json and adds r4.* keys to
outputs/results.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from iiot3w import analysis  # noqa: E402

MAIN, VAR = HERE / "outputs", HERE / "outputs_holdout"


def merge():
    res = None
    for s in range(5):
        part = json.loads((VAR / f"s{s}" / "experiment.json").read_text())
        if res is None:
            res = {"config": part["config"], "seeds": {}}
        res["seeds"].update(part["seeds"])
    res["config"]["HOLDOUT_EARLY_STOPPING"] = True
    (VAR / "experiment.json").write_text(json.dumps(res))
    shutil.copy(MAIN / "audit.json", VAR / "audit.json")
    return res


def main():
    exp = merge()
    R = analysis.run(VAR)
    main_R = json.loads((MAIN / "results.json").read_text())
    keys = [k for k in R if k.startswith(("perf.", "novelty.", "conf.", "rq1.", "rq2.", "rq3."))]
    for k in keys:
        main_R[f"r4.{k}"] = R[k]
    best = []
    for s, d in exp["seeds"].items():
        h = d["arms"]["transformer"]["history"]
        best.append(int(np.argmin([x["val_loss"] for x in h])) + 1)
        main_R[f"r4.seed{s}.tr_epochs_run"] = len(h)
    main_R["r4.tr_best_epoch_min"], main_R["r4.tr_best_epoch_max"] = min(best), max(best)
    (MAIN / "results.json").write_text(json.dumps(main_R, indent=1))
    print("best epochs:", best)
    for m in ("balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc"):
        print(f"{m:18s} RF {R[f'perf.{m}.rf_mean']:.3f}  TR {R[f'perf.{m}.tr_mean']:.3f}  "
              f"diff {R[f'perf.{m}.diff']:+.3f} ({R[f'perf.{m}.ci_low']:.3f}, "
              f"{R[f'perf.{m}.ci_high']:.3f}) p={R[f'perf.{m}.p']:.3f}")
    for q in ("novelty_recall", "undue_abstention"):
        for rule in ("full", "set_size"):
            print(rule, q, f"RF {R[f'novelty.{rule}.{q}.rf']:.3f} TR {R[f'novelty.{rule}.{q}.tr']:.3f}")
    print("coverage", R["conf.coverage.rf_mean"], R["conf.coverage.tr_mean"],
          "sd", R["conf.coverage.rf_sd"], R["conf.coverage.tr_sd"])
    print("mcc wins rf/tr", R["rq1.mcc.rf_wins"], R["rq1.mcc.tr_wins"])


if __name__ == "__main__":
    main()
