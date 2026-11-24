#!/usr/bin/env python3
"""Reproduce every number of the article from the public 3W Dataset 2.0.0.

Stages (in order):
  audit       audit of the real instances + causal windowing  -> audit.json, windows.npz
  experiment  5 well-disjoint partitions x {random forest, Transformer}
                                                               -> experiment.json
  analysis    aggregation of every value printed in the article -> results.json
  robustness  alpha/beta sensitivity and per-well coverage       -> robustness.json
  asset       well-level calibration of the rule (task T23)       -> asset.json
  novelty     controlled evaluation of novelty scores (task T25)  -> novelty.json
  temporal    online ACI per well (T24) and event metrics (T28)  -> temporal.json
  figures     figures of the article                            -> figures/*.png
  compare     computed vs. reported values                      -> COMPARISON.md

Examples
  python run_all.py --dataset-dir data/3W/dataset            # all stages
  python run_all.py --stages analysis,figures,compare        # from cached runs
  python run_all.py --smoke-test                             # synthetic data, minutes
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iiot3w import config as C  # noqa: E402
from iiot3w import models as M  # noqa: E402

STAGES = ["audit", "experiment", "analysis", "robustness", "asset", "novelty", "temporal", "figures", "compare"]


def environment():
    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "torch": torch.__version__, "threads": M.N_THREADS}


def synthetic_windows(n_wells=30, seed=0):
    """Small synthetic stand-in with the structure of the real data (wells,
    known faults of several codes, unseen codes) for the smoke test."""
    rng = np.random.default_rng(seed)
    Xs, ys, cs, ws = [], [], [], []
    for w in range(n_wells):
        level = rng.normal(0, 5, 3)
        for code, n in ((0, 120), (2 + w % 4, 40), (3 + (w + 1) % 3, 25),
                        ([1, 8, 9][w % 3], 10 if w % 2 else 0)):
            if n == 0:
                continue
            t = np.linspace(0, 1, C.WINDOW_T)[None, :, None]
            base = level + rng.normal(0, 1, (n, C.WINDOW_T, 3))
            if code != 0:
                base = base + (1.5 + 0.3 * code) * t * rng.choice([-1, 1], 3)
            Xs.append(base.astype(np.float32))
            ys.append(np.full(n, 0 if code == 0 else (2 if code in C.CLASS_UNOBSERVED else 1), np.int8))
            cs.append(np.full(n, code, np.int16))
            ws.append(np.full(n, w, np.int16))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(ws))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=C.DATASET_DIR)
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--seeds", default=None, help="comma-separated subset of seeds")
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU threads for PyTorch and scikit-learn (article: 1)")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--holdout-early-stopping", action="store_true",
                    help="variant R4: fit on inner-fit wells, early-stop on held-out wells")
    ap.add_argument("--split-mode", choices=["windows", "wells"], default="windows",
                    help="windows: TII protocol (default); wells: well-level partition (T20)")
    ap.add_argument("--selection", choices=["inner", "nested"], default="inner",
                    help="inner: TII protocol (default); nested: grouped K-fold per seed (T21)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arms: rf,transformer (default),lgbm,cnn (T29)")
    ap.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary",
                    help="binary: TII protocol (default); multiclass: one class per known fault (T27)")
    ap.add_argument("--withheld", default=None,
                    help="comma-separated 3W codes withheld as unseen (default 1,8,9; T26 rotation)")
    ap.add_argument("--provenance-dir", type=Path, default=None,
                    help="folder with windows.npz (with provenance) and instances.json for the "
                         "temporal stage; default: --out")
    ap.add_argument("--selection-from", type=Path, default=None,
                    help="experiment.json whose seed-0 hyperparameter selection is reused")
    a = ap.parse_args(argv)

    M.N_THREADS = a.threads
    C.HOLDOUT_EARLY_STOPPING = a.holdout_early_stopping
    C.SPLIT_MODE = a.split_mode
    C.SELECTION = a.selection
    C.LABEL_MODE = a.label_mode
    if a.arms:
        C.ARMS_ENABLED = a.arms.split(",")
    if a.withheld:
        C.CLASS_UNOBSERVED = sorted(int(c) for c in a.withheld.split(","))
        C.CLASS_KNOWN_FAULT = [c for c in range(1, 10) if c not in C.CLASS_UNOBSERVED]
    out = a.out
    if a.smoke_test:
        out = HERE / "outputs_smoke"
        C.TR_EPOCHS, C.TR_SELECT_EPOCHS = 2, 1
        C.SEEDS = [0, 1]
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    todo = STAGES if a.stages == "all" else a.stages.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    (out / "environment.json").write_text(json.dumps(environment(), indent=2))
    import importlib
    data = importlib.import_module("iiot3w.data")
    experiment = importlib.import_module("iiot3w.experiment")

    for stage in STAGES:
        if stage in ("analysis", "robustness", "asset", "novelty", "temporal", "figures", "compare") and stage in todo:
            globals()[stage] = importlib.import_module(
                "iiot3w." + ("compare" if stage == "compare" else stage))
        analysis = globals().get("analysis")
        robustness = globals().get("robustness")
        asset = globals().get("asset")
        novelty = globals().get("novelty")
        temporal = globals().get("temporal")
        figures = globals().get("figures")
        compare = globals().get("compare")
        if stage not in todo:
            continue
        t0 = time.time()
        log(f"== stage {stage}")
        if stage == "audit":
            if a.smoke_test:
                X, y, code, well = synthetic_windows()
                np.savez_compressed(out / "windows.npz", X=X, y=y, code=code, well=well)
                (out / "audit.json").write_text(json.dumps({"synthetic": True}))
            else:
                data.audit_and_window(a.dataset_dir, out, log)
        elif stage == "experiment":
            experiment.run(data.load_windows(out if (out / "windows.npz").exists()
                                              else C.OUT_DIR), out, log, seeds,
                           selection_from=a.selection_from)
        elif stage == "analysis":
            analysis.run(out, log)
        elif stage == "robustness":
            robustness.run(out, log)
        elif stage == "asset":
            asset.run(out, log)
        elif stage == "novelty":
            novelty.run(out, log)
        elif stage == "temporal":
            temporal.run(out, log, provenance_dir=a.provenance_dir)
        elif stage == "figures":
            figures.run(out, log)
        elif stage == "compare":
            if a.smoke_test:
                log("compare skipped in the smoke test (synthetic data)")
            else:
                compare.run(out, log)
        log(f"   {stage} finished in {time.time() - t0:.0f} s")
    log("exit=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
