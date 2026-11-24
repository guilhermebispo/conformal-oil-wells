"""Audit of the real 3W instances and causal windowing  [Sec. IV-B/C, Table II].

One pass over the Parquet files produces
  outputs/audit.json     characterization of the data (counts, missingness,
                         duplicates, time span, coverage of event types)
  outputs/windows.npz    window tensor X (n, T, 3), label y in {0,1,2},
                         3W event code, and well identifier.

No statistic from calibration or test data enters this stage.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C

WELL_RE = re.compile(r"WELL-(\d+)_")


def instance_files(dataset_dir: Path):
    """(path, event_code, well_id) of every real instance, in a fixed order."""
    out = []
    for code_dir in sorted(Path(dataset_dir).iterdir()):
        if not code_dir.is_dir() or not code_dir.name.isdigit():
            continue
        code = int(code_dir.name)
        for f in sorted(code_dir.glob(f"{C.INSTANCE_PREFIX}*.parquet")):
            m = WELL_RE.search(f.name)
            if m is not None:
                out.append((f, code, int(m.group(1))))
    return out


def read_instance(path):
    """Timestamps, the three variables (sentinels -> NaN) and point labels."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=C.VARIABLES + ["class", "timestamp"])
    ts = tbl.column("timestamp").to_numpy(zero_copy_only=False)
    cls = tbl.column("class").to_pandas().to_numpy(dtype="float64")
    X = np.column_stack([tbl.column(v).to_pandas().to_numpy(dtype="float64")
                         for v in C.VARIABLES])
    X[np.abs(X) > C.SENTINEL_ABS] = np.nan
    return ts, X, cls


def label_of(code_array):
    """3W class code -> 0 (normal), 1 (known fault), 2 (unseen), -1 (dropped:
    transient, unknown, or not used)."""
    y = np.full(np.shape(code_array), -1, dtype=np.int8)
    y[code_array == C.CLASS_NORMAL] = 0
    for c in C.CLASS_KNOWN_FAULT:
        y[code_array == c] = 1
    for c in C.CLASS_UNOBSERVED:
        y[code_array == c] = 2
    return y


def windows_from_instance(X, cls, return_ends=False):
    """Valid causal windows of one (already decimated) instance.

    A window ending at t uses samples t-T+1..t only.  It is kept if it has no
    missing value, a single label, no transient code, and a usable label.
    Normal windows are capped per instance by uniform subsampling over the
    whole instance (never by truncation); fault and unseen windows are kept
    in full because they are scarce.

    With return_ends=True a fourth array is returned: the index, in the
    (decimated) instance, of the last sample of each window (tasks T24, T28).
    """
    T, S, F = C.WINDOW_T, C.WINDOW_STEP, len(C.VARIABLES)
    empty = (np.empty((0, T, F), np.float32), np.empty(0, np.int8),
             np.empty(0, np.int16)) + ((np.empty(0, np.int64),) if return_ends else ())
    n = X.shape[0]
    if n < T:
        return empty
    y_point = label_of(cls)
    Xs, ys, codes, ends = [], [], [], []
    for end in range(T - 1, n, S):
        sl = slice(end - T + 1, end + 1)
        w = X[sl]
        if np.isnan(w).any():
            continue
        yy = y_point[sl]
        if yy[-1] < 0:
            continue
        if C.REQUIRE_LABEL_HOMOGENEITY and not np.all(yy == yy[-1]):
            continue
        if C.DROP_TRANSIENTS and np.any(cls[sl] >= 100):
            continue
        Xs.append(w)
        ys.append(yy[-1])
        codes.append(int(cls[end]))
        ends.append(end)
    if not Xs:
        return empty
    ys_arr = np.asarray(ys)
    i_norm = np.where(ys_arr == 0)[0]
    i_other = np.where(ys_arr != 0)[0]
    if i_norm.size > C.MAX_WINDOWS_PER_INSTANCE:
        keep = i_norm[np.linspace(0, i_norm.size - 1,
                                  C.MAX_WINDOWS_PER_INSTANCE).astype(int)]
        sel = np.sort(np.concatenate([keep, i_other]))
        Xs = [Xs[i] for i in sel]
        ys = [ys[i] for i in sel]
        codes = [codes[i] for i in sel]
        ends = [ends[i] for i in sel]
    out = (np.asarray(Xs, np.float32), np.asarray(ys, np.int8),
           np.asarray(codes, np.int16))
    return out + ((np.asarray(ends, np.int64),) if return_ends else ())


def audit_and_window(dataset_dir: Path, out_dir: Path, log=print):
    files = instance_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"no real 3W instances under {dataset_dir}")
    log(f"real instances found: {len(files)}")

    n_obs = n_dup = miss_den = 0
    wells, miss_num = set(), defaultdict(int)
    t_min = t_max = None
    inst_per_code, win_per_code = defaultdict(int), defaultdict(int)
    inst_with_window, inst_with_vars = defaultdict(int), defaultdict(int)
    Xall, yall, codeall, wellall = [], [], [], []
    # Provenance of each window (T24, T28): instance index and the time
    # (seconds since the epoch) of its last sample.  Additive: X, y, code and
    # well are unchanged, so released runs are unaffected.
    install, tendall, instances = [], [], []

    for i, (path, code, well) in enumerate(files, 1):
        ts, X, cls = read_instance(path)
        inst_per_code[code] += 1
        wells.add(well)
        n_obs += X.shape[0]
        miss_den += X.shape[0]
        for j, v in enumerate(C.VARIABLES):
            miss_num[v] += int(np.isnan(X[:, j]).sum())
        if not np.isnan(X).all(axis=0).any():
            inst_with_vars[code] += 1
        if ts.size:
            lo, hi = ts.min(), ts.max()
            t_min = lo if t_min is None else min(t_min, lo)
            t_max = hi if t_max is None else max(t_max, hi)
            n_dup += int(ts.size - np.unique(ts).size)

        Xw, yw, cw, ew = windows_from_instance(X[::C.DECIMATION], cls[::C.DECIMATION],
                                               return_ends=True)
        if Xw.shape[0]:
            Xall.append(Xw)
            yall.append(yw)
            codeall.append(cw)
            wellall.append(np.full(yw.shape, well, np.int16))
            ts_dec = ts[::C.DECIMATION].astype("datetime64[s]").astype(np.int64)
            install.append(np.full(yw.shape, len(instances), np.int32))
            tendall.append(ts_dec[ew])
            instances.append({"file": Path(path).name, "code": int(code), "well": int(well)})
            inst_with_window[code] += 1
            for cc, nn in zip(*np.unique(cw, return_counts=True)):
                win_per_code[int(cc)] += int(nn)
        if i % 100 == 0:
            log(f"  {i}/{len(files)} instances processed")

    X = np.concatenate(Xall)
    y = np.concatenate(yall)
    code = np.concatenate(codeall)
    well = np.concatenate(wellall)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=X, y=y, code=code, well=well,
                        inst=np.concatenate(install), t_end=np.concatenate(tendall))
    (out_dir / "instances.json").write_text(json.dumps(instances))

    k = lambda d: {str(c): int(n) for c, n in sorted(d.items())}
    audit = {
        "n_instances": len(files),
        "n_wells": len(wells),
        "n_observations": int(n_obs),
        "sampling_hz": 1,
        "first_timestamp": str(np.datetime_as_string(t_min, unit="D")),
        "last_timestamp": str(np.datetime_as_string(t_max, unit="D")),
        "duplicate_timestamps": int(n_dup),
        "missing_pct": {v: round(100.0 * miss_num[v] / max(miss_den, 1), 1)
                        for v in C.VARIABLES},
        "instances_per_code": k(inst_per_code),
        "instances_with_three_variables": k(inst_with_vars),
        "instances_with_a_window": k(inst_with_window),
        "windows_per_code": k(win_per_code),
        "n_windows": int(X.shape[0]),
        "n_wells_with_windows": int(np.unique(well).size),
        "n_windows_normal": int((y == 0).sum()),
        "n_windows_known_fault": int((y == 1).sum()),
        "n_windows_unseen": int((y == 2).sum()),
        "instances_known_fault": int(sum(inst_per_code[c] for c in C.CLASS_KNOWN_FAULT)),
        "instances_unseen": int(sum(inst_per_code[c] for c in C.CLASS_UNOBSERVED)),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def load_windows(out_dir: Path):
    d = np.load(Path(out_dir) / "windows.npz")
    return d["X"], d["y"], d["code"], d["well"]


def load_provenance(out_dir: Path):
    """Instance index and end time (s) of every window, and the instance table
    (T24, T28).  Requires windows.npz written by this version of the audit."""
    d = np.load(Path(out_dir) / "windows.npz")
    if "inst" not in d:
        raise KeyError("windows.npz has no provenance; re-run the audit stage")
    return d["inst"], d["t_end"], json.loads((Path(out_dir) / "instances.json").read_text())
