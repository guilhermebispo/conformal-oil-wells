"""Well-disjoint partitioning under constraints  [Sec. IV-D, Table III].

Unconstrained well-disjoint draws often place one fault type only in training
and another only in test, so the comparison would measure transfer across
event types rather than across learning paradigms.  Each subset must therefore
contain at least MIN_FAULT_TYPES_PER_SPLIT known fault types and a fault
prevalence in [MIN_PREVALENCE, MAX_PREVALENCE]; draws violating either are
rejected and redrawn.  Shares are computed on windows, not wells, because 3W
wells contribute very unequal numbers of windows.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def _valid(y, code, idx):
    if idx.size == 0:
        return False
    yy, cc = y[idx], code[idx]
    known = yy < 2
    if known.sum() == 0:
        return False
    prev = float((yy[known] == 1).mean())
    if not (C.MIN_PREVALENCE <= prev <= C.MAX_PREVALENCE):
        return False
    types = set(np.unique(cc[yy == 1]).tolist())
    return len(types & set(C.CLASS_KNOWN_FAULT)) >= C.MIN_FAULT_TYPES_PER_SPLIT


def make_split(y, code, well, seed):
    """Return (idx_train, idx_cal, idx_test, idx_unseen, info).

    Unseen-regime windows (y == 2) never enter training or calibration; those
    of the test wells form the unseen evaluation set.
    """
    rng = np.random.default_rng(seed)
    wells = np.unique(well)
    known = y < 2
    n_by_well = {int(w): int(((well == w) & known).sum()) for w in wells}
    total = sum(n_by_well.values())
    target = np.array([C.FRAC_TRAIN, C.FRAC_CAL, C.FRAC_TEST])

    for attempt in range(C.MAX_SPLIT_ATTEMPTS):
        perm = rng.permutation(wells)
        acc, w_tr, w_ca, w_te = 0, [], [], []
        for w in perm:
            frac = acc / max(total, 1)
            if frac < C.FRAC_TRAIN:
                w_tr.append(w)
            elif frac < C.FRAC_TRAIN + C.FRAC_CAL:
                w_ca.append(w)
            else:
                w_te.append(w)
            acc += n_by_well[int(w)]
        if not (w_tr and w_ca and w_te):
            continue
        m_tr, m_ca, m_te = (np.isin(well, w) for w in (w_tr, w_ca, w_te))
        unseen = y == 2
        i_tr = np.where(m_tr & ~unseen)[0]
        i_ca = np.where(m_ca & ~unseen)[0]
        i_te = np.where(m_te & ~unseen)[0]
        i_un = np.where(m_te & unseen)[0]

        fr = np.array([i_tr.size, i_ca.size, i_te.size], float)
        fr /= max(fr.sum(), 1)
        if np.abs(fr - target).max() > C.FRAC_TOL:
            continue
        if i_un.size == 0 or not all(_valid(y, code, i) for i in (i_tr, i_ca, i_te)):
            continue
        info = {
            "seed": int(seed),
            "attempts": attempt + 1,
            "wells_train": sorted(int(w) for w in w_tr),
            "wells_cal": sorted(int(w) for w in w_ca),
            "wells_test": sorted(int(w) for w in w_te),
            "n_train": int(i_tr.size),
            "n_cal": int(i_ca.size),
            "n_test": int(i_te.size),
            "n_unseen": int(i_un.size),
            "shares": [float(x) for x in fr],
            "prevalence_train": float((y[i_tr] == 1).mean()),
            "prevalence_test": float((y[i_te] == 1).mean()),
            "fault_types_train": sorted(int(c) for c in np.unique(code[i_tr][y[i_tr] == 1])),
            "fault_types_test": sorted(int(c) for c in np.unique(code[i_te][y[i_te] == 1])),
            "unseen_types_test": sorted(int(c) for c in np.unique(code[i_un])),
        }
        return i_tr, i_ca, i_te, i_un, info

    raise RuntimeError(f"no partition met the constraints in "
                       f"{C.MAX_SPLIT_ATTEMPTS} attempts (seed {seed})")


def _split_info(seed, attempt, w_tr, w_ca, w_te, i_tr, i_ca, i_te, i_un, y, code, mode):
    """Provenance of a partition, identical in both modes (plus the mode)."""
    fr = np.array([i_tr.size, i_ca.size, i_te.size], float)
    fr /= max(fr.sum(), 1)
    return {
        "mode": mode,
        "seed": int(seed),
        "attempts": attempt + 1,
        "wells_train": sorted(int(w) for w in w_tr),
        "wells_cal": sorted(int(w) for w in w_ca),
        "wells_test": sorted(int(w) for w in w_te),
        "n_train": int(i_tr.size),
        "n_cal": int(i_ca.size),
        "n_test": int(i_te.size),
        "n_unseen": int(i_un.size),
        "shares": [float(x) for x in fr],
        "prevalence_train": float((y[i_tr] == 1).mean()),
        "prevalence_test": float((y[i_te] == 1).mean()),
        "fault_types_train": sorted(int(c) for c in np.unique(code[i_tr][y[i_tr] == 1])),
        "fault_types_cal": sorted(int(c) for c in np.unique(code[i_ca][y[i_ca] == 1])),
        "fault_types_test": sorted(int(c) for c in np.unique(code[i_te][y[i_te] == 1])),
        "unseen_types_test": sorted(int(c) for c in np.unique(code[i_un])),
    }


def make_split_wells(y, code, well, seed):
    """Well-level partition (task T20).  Same return value as make_split.

    Each well is proposed to train / calibration / test with probabilities
    WS_ASSIGN_PROB; the draw is accepted only if
      - every subset has at least WS_MIN_WELLS wells that hold known-regime
        windows and at least WS_MIN_SHARE of those windows;
      - every subset satisfies the fault-type and prevalence constraints of
        the article protocol (_valid);
      - (WS_REQUIRE_TYPES_IN_TRAIN) every known fault type present in
        calibration or test also occurs in training;
      - the test wells contain unseen-regime windows.
    Wells without known-regime windows carry no information for fitting or
    calibration; they are proposed like the others, and their unseen windows
    count only if they fall in the test subset.
    """
    rng = np.random.default_rng(seed)
    wells = np.unique(well)
    known = y < 2
    unseen = y == 2
    n_known = {int(w): int(((well == w) & known).sum()) for w in wells}
    total = sum(n_known.values())
    names = ("train", "cal", "test")
    p = np.asarray(C.WS_ASSIGN_PROB, float)
    p /= p.sum()

    for attempt in range(C.WS_MAX_ATTEMPTS):
        lab = rng.choice(3, size=wells.size, p=p)
        groups = [wells[lab == k] for k in range(3)]
        ok = True
        for k, name in enumerate(names):
            with_known = [w for w in groups[k] if n_known[int(w)] > 0]
            share = sum(n_known[int(w)] for w in groups[k]) / max(total, 1)
            if len(with_known) < C.WS_MIN_WELLS[name] or share < C.WS_MIN_SHARE[name]:
                ok = False
                break
        if not ok:
            continue
        m_tr, m_ca, m_te = (np.isin(well, g) for g in groups)
        i_tr = np.where(m_tr & ~unseen)[0]
        i_ca = np.where(m_ca & ~unseen)[0]
        i_te = np.where(m_te & ~unseen)[0]
        i_un = np.where(m_te & unseen)[0]
        if i_un.size == 0 or not all(_valid(y, code, i) for i in (i_tr, i_ca, i_te)):
            continue
        if C.WS_REQUIRE_TYPES_IN_TRAIN:
            types = lambda i: set(np.unique(code[i][y[i] == 1]).tolist())
            if not (types(i_ca) | types(i_te)) <= types(i_tr):
                continue
        info = _split_info(seed, attempt, *groups, i_tr, i_ca, i_te, i_un, y, code, "wells")
        return i_tr, i_ca, i_te, i_un, info

    raise RuntimeError(f"no well-level partition met the constraints in "
                       f"{C.WS_MAX_ATTEMPTS} attempts (seed {seed})")


def partition(y, code, well, seed):
    """Dispatch on SPLIT_MODE.  'windows' is the TII protocol, unchanged, so
    the released results keep matching the article."""
    if C.SPLIT_MODE == "wells":
        return make_split_wells(y, code, well, seed)
    return make_split(y, code, well, seed)


def grouped_folds(y, well, idx, seed, k=None):
    """Grouped K-fold over the wells of idx, for nested selection (T21).

    Wells are assigned at random to k folds; an assignment is accepted only if
    every validation fold has >= NESTED_MIN_PER_CLASS windows of each class and
    its complement contains both classes.  Among the
    accepted assignments (up to NESTED_MAX_ATTEMPTS draws), the one whose fold
    sizes (in windows) are most even is kept, since 3W wells are very unequal.
    Returns a list of (i_fit, i_val) and a description.
    """
    k = C.NESTED_FOLDS if k is None else k
    rng = np.random.default_rng(3000 + seed)
    w = np.unique(well[idx])
    if w.size < k:
        raise RuntimeError(f"{w.size} training wells < {k} folds")
    best = None
    for _ in range(C.NESTED_MAX_ATTEMPTS):
        lab = rng.integers(0, k, w.size)
        if np.unique(lab).size < k:
            continue
        folds, ok = [], True
        for f in range(k):
            m = np.isin(well[idx], w[lab == f])
            i_fit, i_val = idx[~m], idx[m]
            n_val = np.bincount(y[i_val].astype(int), minlength=2)[:2]
            if n_val.min() < C.NESTED_MIN_PER_CLASS or len(np.unique(y[i_fit])) < 2:
                ok = False
                break
            folds.append((i_fit, i_val))
        if not ok:
            continue
        sizes = np.array([v.size for _, v in folds], float) / idx.size
        spread = float(np.abs(sizes - 1.0 / k).max())
        if best is None or spread < best[0]:
            best = (spread, folds, {int(x): int(f) for x, f in zip(w, lab)})
    if best is None:
        raise RuntimeError("no grouped fold assignment meets NESTED_MIN_PER_CLASS in every fold")
    return best[1], {"type": "grouped_kfold", "k": k, "max_share_deviation": best[0],
                     "fold_of_well": best[2],
                     "val_shares": [float(v.size / idx.size) for _, v in best[1]]}


def inner_split(y, well, idx, seed):
    """Internal split of the training set for hyperparameter selection.

    Well-disjoint when some subset of training wells approximates the target
    share; otherwise a stratified random split, and the fallback is recorded
    in the output (never silent).  Calibration and test data are not touched.
    """
    rng = np.random.default_rng(1000 + seed)
    w = np.unique(well[idx])
    best = None
    for _ in range(2000):
        perm = rng.permutation(w)
        n_val = rng.integers(1, max(2, w.size))
        m = np.isin(well[idx], perm[:n_val])
        i_fit, i_val = idx[~m], idx[m]
        if i_fit.size == 0 or i_val.size == 0:
            continue
        if len(np.unique(y[i_fit])) < 2 or len(np.unique(y[i_val])) < 2:
            continue
        frac = i_val.size / idx.size
        err = abs(frac - C.INNER_VAL_FRAC)
        if best is None or err < best[0]:
            best = (err, i_fit, i_val, frac)
    if best is not None and best[0] <= C.INNER_VAL_TOL:
        return best[1], best[2], {"type": "well_disjoint", "val_share": float(best[3])}

    rng2 = np.random.default_rng(2000 + seed)
    i_fit, i_val = [], []
    for cls in np.unique(y[idx]):
        p = rng2.permutation(idx[y[idx] == cls])
        k = max(1, int(round(C.INNER_VAL_FRAC * p.size)))
        i_val.append(p[:k])
        i_fit.append(p[k:])
    i_fit, i_val = np.concatenate(i_fit), np.concatenate(i_val)
    return i_fit, i_val, {"type": "stratified_random",
                          "reason": "no subset of training wells approached the target share",
                          "val_share": float(i_val.size / idx.size)}
