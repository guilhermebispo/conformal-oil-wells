#!/usr/bin/env python3
"""Merge seeds run in separate processes into one output folder.

Each seed of a run can be executed by its own process (one CPU thread each),
writing to <run>/s<seed>/.  This script joins them into <run>/ so that the
analysis, robustness, asset and figures stages can run on the whole run:

  <run>/experiment.json   union of the per-seed "seeds" blocks; the config
                          block is taken from the selection seed and checked
                          to be identical across seeds
  <run>/arrays/           all per-window arrays
  <run>/audit.json, windows.npz, environment.json   from the selection seed

Usage:  python scripts/merge_seeds.py outputs/run01_wells
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from iiot3w import config as C  # noqa: E402


def main(run_dir):
    run = Path(run_dir)
    parts = sorted(run.glob("s[0-9]*/experiment.json"))
    if not parts:
        raise SystemExit(f"no s<seed>/experiment.json under {run}")
    merged, config = {"seeds": {}}, None
    (run / "arrays").mkdir(parents=True, exist_ok=True)
    for f in parts:
        e = json.loads(f.read_text())
        cfg = {k: v for k, v in e["config"].items() if k not in ("OUT_DIR",)}
        if config is None:
            config = cfg
        elif cfg != config:   # every seed must use the same settings
            diff = sorted(k for k in set(cfg) | set(config) if cfg.get(k) != config.get(k))
            raise SystemExit(f"config differs in {f}: {diff}")
        for s, rec in e["seeds"].items():
            if s in merged["seeds"]:
                raise SystemExit(f"seed {s} appears twice")
            merged["seeds"][s] = rec
        for a in (f.parent / "arrays").glob("seed*_*.npz"):
            shutil.copy2(a, run / "arrays" / a.name)
    merged["config"] = config
    sel = run / f"s{C.SELECT_ON_SEED}"
    for name in ("audit.json", "windows.npz", "environment.json"):
        if (sel / name).exists():
            shutil.copy2(sel / name, run / name)
    (run / "experiment.json").write_text(json.dumps(merged))
    print(f"merged seeds {sorted(merged['seeds'], key=int)} into {run}")


if __name__ == "__main__":
    main(sys.argv[1])
