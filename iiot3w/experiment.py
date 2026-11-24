"""Experiment runner  [stage S5-S6 of Fig. 1].

For each seed: well-disjoint partition under constraints; hyperparameter
selection of both arms on an internal split of the training set (done once,
on SELECT_ON_SEED, and reused); final fit; calibration of the conformal
predictor and of the atypicality test on the calibration subset; evaluation
on the known-regime test windows and on the unseen windows of the test wells.

Writes outputs/experiment.json, resumable by whole seed (a partial seed is
never stored).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from . import config as C
from . import conformal as CF
from . import data as D
from . import novelty as NV
from . import models as M
from . import splits as SP

# Registry of arms: key -> (class, grid name in config, is a neural network).
# config.ARMS_ENABLED selects which run (TII protocol: rf, transformer).
ARM_REGISTRY = {"rf": (M.RFArm, "RF_GRID", False),
                "transformer": (M.TransformerArm, "TR_GRID", True),
                "lgbm": (M.LGBMArm, "LGBM_GRID", False),
                "cnn": (M.CNNArm, "CNN_GRID", True)}


def _fault_score(p):
    """Score of 'fault' vs. 'normal'.  Binary: the fault column, exactly as in
    the TII protocol; multiclass: 1 - p(normal)."""
    return p[:, 1] if p.shape[1] == 2 else 1.0 - p[:, 0]


def model_labels(y, code):
    """Labels the models are trained on (task T27).  Binary: y itself (0
    normal, 1 known fault).  Multiclass: 0 normal and 1..K-1 one per known
    fault type, in the order of config.CLASS_KNOWN_FAULT; unseen windows get
    -1 (they never enter training, calibration or the known test set)."""
    if C.LABEL_MODE == "binary":
        return y.astype(int)
    ym = np.full(len(y), -1, int)
    ym[y == 0] = 0
    for k, c in enumerate(C.CLASS_KNOWN_FAULT, start=1):
        ym[(y == 1) & (code == c)] = k
    return ym


def _select(arm_cls, grid, Xfit, yfit, Xval, yval, seed, is_tr, norm, log):
    """Grid search on the internal split, scored by ROC-AUC (balanced accuracy
    collapses to 0.5 for any majority-class predictor under strong imbalance
    and no longer orders configurations)."""
    best, best_s, rows = None, -np.inf, []
    for params in grid:
        arm = arm_cls(params, seed)
        arm.fit(Xfit, yfit, Xval, yval, epochs=C.TR_SELECT_EPOCHS if is_tr else None,
                norm=norm)
        pv = arm.predict_proba(Xval)
        s = roc_auc_score(yval > 0, _fault_score(pv))
        rows.append({"params": params, "val_roc_auc": float(s),
                     "val_balanced_accuracy": float(balanced_accuracy_score(yval, pv.argmax(1)))})
        log(f"      {params} -> val ROC-AUC {s:.3f}")
        if s > best_s:
            best, best_s = params, s
    return best, rows


def _select_nested(arm_cls, grid, X, y, well, i_tr, seed, is_tr, norm, log, ym=None):
    """Nested selection (T21): grouped K-fold over the training wells of this
    seed, scored by the ROC-AUC of the pooled out-of-fold predictions.  Tries
    NESTED_FOLDS folds and falls back to fewer (down to 2) when no assignment
    meets NESTED_MIN_PER_CLASS; the number used is recorded.  As in the TII
    selection, the Transformer early-stops on the validation fold with the
    reduced epoch budget."""
    folds = info = None
    for k in range(C.NESTED_FOLDS, 1, -1):
        try:
            folds, info = SP.grouped_folds(y, well, i_tr, seed, k=k)
            break
        except RuntimeError:
            continue
    if folds is None:
        raise RuntimeError(f"nested selection impossible for seed {seed}")
    info["fallback"] = info["k"] < C.NESTED_FOLDS
    yy = (y if ym is None else ym).astype(int)    # folds are built on y; models fit on ym
    best, best_s, rows = None, -np.inf, []
    for params in grid:
        oof = np.full(len(y), np.nan)
        per_fold = []
        for i_fit, i_val in folds:
            arm = arm_cls(params, seed)
            arm.fit(X[i_fit], yy[i_fit], X[i_val], yy[i_val],
                    epochs=C.TR_SELECT_EPOCHS if is_tr else None, norm=norm)
            pv = _fault_score(arm.predict_proba(X[i_val]))
            oof[i_val] = pv
            per_fold.append(float(roc_auc_score(yy[i_val] > 0, pv)))
        s = float(roc_auc_score(yy[i_tr] > 0, oof[i_tr]))
        rows.append({"params": params, "oof_roc_auc": s, "fold_roc_auc": per_fold,
                     "val_roc_auc": s})
        log(f"      {params} -> pooled OOF ROC-AUC {s:.3f} (folds {[round(x, 3) for x in per_fold]})")
        if s > best_s:
            best, best_s = params, s
    return best, rows, info


def _decomposition(p_te, yte, tau, r_te, rho):
    """Quantities of Proposition 3 on the known-regime test windows."""
    sets = CF.prediction_sets(p_te, tau)
    card = sets.sum(1)
    atyp = r_te > rho
    not_single = card != 1
    return {
        "empty_set_rate": float((card == 0).mean()),
        "ambiguous_set_rate": float((card >= 2).mean()),
        "not_singleton_rate": float(not_single.mean()),
        "atypical_rate": float(atyp.mean()),
        "atypical_only_rate": float((atyp & ~not_single).mean()),
        "bound_prop3": float(not_single.mean() + C.BETA),
        "coverage": float(sets[np.arange(len(yte)), yte].mean()),
        "mean_set_size": float(card.mean()),
    }


def run_seed(X, y, code, well, seed, selection_cache, log, out_dir, ym=None):
    t0 = time.time()
    i_tr, i_ca, i_te, i_un, info = SP.partition(y, code, well, seed)
    log(f"  split: train={i_tr.size} cal={i_ca.size} test={i_te.size} "
        f"unseen={i_un.size} prev_test={info['prevalence_test']:.3f}")
    ym = y.astype(int) if ym is None else ym
    ytr, yca, yte = (ym[i].astype(int) for i in (i_tr, i_ca, i_te))
    Xtr, Xca, Xte, Xun = X[i_tr], X[i_ca], X[i_te], X[i_un]
    norm = M.fit_normalizer(Xtr)

    i_fit, i_val, inner = SP.inner_split(y, well, i_tr, seed)
    info["inner_validation"] = inner
    Xfit, yfit, Xval, yval = X[i_fit], ym[i_fit].astype(int), X[i_val], ym[i_val].astype(int)

    out = {"split": info, "arms": {}, "y_test": yte.tolist()}
    for key in C.ARMS_ENABLED:
        arm_cls, grid_name, is_tr = ARM_REGISTRY[key]
        if C.SELECTION == "nested":
            log(f"    nested selection of {key} ...")
            params, sel, folds_info = _select_nested(arm_cls, getattr(C, grid_name), X, y, well,
                                                     i_tr, seed, is_tr, norm, log, ym=ym)
            info["nested_folds"] = folds_info
        elif key in selection_cache:
            params, sel = selection_cache[key]
        else:
            log(f"    selecting {key} ...")
            params, sel = _select(arm_cls, getattr(C, grid_name), Xfit, yfit, Xval, yval,
                                  seed, is_tr, norm, log)
            selection_cache[key] = (params, sel)
        log(f"    {key}: {params}")
        t_fit = time.time()
        arm = arm_cls(params, seed)
        # Article protocol: both arms fit on all training windows; the
        # Transformer early-stops on the internal validation wells, which are
        # part of those training windows.  Variant (HOLDOUT_EARLY_STOPPING):
        # both arms fit only on the inner-fit wells, so early stopping uses
        # wells the model never trained on.
        X_fit_final, y_fit_final = (Xfit, yfit) if C.HOLDOUT_EARLY_STOPPING else (Xtr, ytr)
        if is_tr:
            arm.fit(X_fit_final, y_fit_final, Xval, yval, norm=norm)
        else:
            arm.fit(X_fit_final, y_fit_final, norm=norm)
        fit_s = time.time() - t_fit

        p_ca, p_te, p_un = arm.predict_proba(Xca), arm.predict_proba(Xte), arm.predict_proba(Xun)
        tau = CF.conformal_threshold(p_ca, yca)
        Z_ref = arm.represent(X_fit_final)
        Z = {"cal": arm.represent(Xca), "test": arm.represent(Xte), "unseen": arm.represent(Xun)}
        det = CF.AtypicalityDetector().fit(Z_ref).calibrate(Z["cal"])
        r_ca, r_te, r_un = (det.score(Z[k]) for k in ("cal", "test", "unseen"))
        # Novelty scores for T25 (stored per window; analysed post hoc).
        lg = {"cal": arm.logits(Xca), "test": arm.logits(Xte), "unseen": arm.logits(Xun)}
        nov = NV.fit_time_scores(Z_ref, Z, {"cal": p_ca, "test": p_te, "unseen": p_un},
                                 None if lg["cal"] is None else lg, seed)
        del Z_ref, Z
        # Raw per-window outputs, for the post-hoc robustness analyses
        # (alpha/beta sensitivity and per-well coverage) without refitting.
        arr_dir = Path(out_dir) / "arrays"
        arr_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arr_dir / f"seed{seed}_{key}.npz",
            p_cal=p_ca.astype(np.float32), y_cal=yca.astype(np.int8),
            r_cal=r_ca.astype(np.float32), well_cal=well[i_ca],
            p_test=p_te.astype(np.float32), y_test=yte.astype(np.int8),
            r_test=r_te.astype(np.float32), well_test=well[i_te], code_test=code[i_te],
            p_unseen=p_un.astype(np.float32), r_unseen=r_un.astype(np.float32),
            well_unseen=well[i_un], code_unseen=code[i_un], **nov)

        dec_te = CF.three_state_decision(p_te, tau, r_te, det.rho)
        dec_un = CF.three_state_decision(p_un, tau, r_un, det.rho)
        dec_te_card = CF.set_size_only_decision(p_te, tau)
        dec_un_card = CF.set_size_only_decision(p_un, tau)

        rec = {
            "params": params,
            "selection": sel,
            "fit_seconds": fit_s,
            "closed": CF.closed_set_metrics(yte, p_te),
            "conformal": {"tau": tau, "rho": float(det.rho),
                          **_decomposition(p_te, yte, tau, r_te, det.rho),
                          **CF.selective_metrics(yte, dec_te)},
            "novelty_full": CF.novelty_metrics(dec_te, dec_un),
            "novelty_set_size": CF.novelty_metrics(dec_te_card, dec_un_card),
            "unseen_empty_set_rate": float((CF.prediction_sets(p_un, tau).sum(1) == 0).mean()),
            "proba_test": np.round(p_te, 6).tolist(),
            "history": arm.history,
        }
        if key == "rf":
            names = M.feature_labels(C.VARIABLES)
            rec["importance"] = sorted(
                ({"feature": n, "importance": float(v)}
                 for n, v in zip(names, arm.model.feature_importances_)),
                key=lambda d: -d["importance"])
        out["arms"][key] = rec
        log(f"    {key}: MCC={rec['closed']['mcc']:.3f} AUC={rec['closed']['roc_auc']:.3f} "
            f"abst={rec['conformal']['abstention_rate']:.3f} "
            f"nov_rec={rec['novelty_full']['novelty_recall']:.3f} ({fit_s:.0f} s fit)")
    log(f"  seed {seed} done in {time.time() - t0:.0f} s")
    return out


def run(windows, out_dir: Path, log=print, seeds=None, selection_from=None):
    X, y, code, well = windows
    # Task T26: when the withheld codes differ from those used at audit time,
    # relabel every window from its own 3W code (the windows themselves do not
    # change: only normal windows are capped, and their label is unaffected).
    y = D.label_of(code).astype(np.int8)
    ym = model_labels(y, code)
    C.N_CLASSES = 2 if C.LABEL_MODE == "binary" else 1 + len(C.CLASS_KNOWN_FAULT)
    log(f"labels: {C.LABEL_MODE} ({C.N_CLASSES} classes); known {C.CLASS_KNOWN_FAULT}; "
        f"withheld {C.CLASS_UNOBSERVED}; arms {C.ARMS_ENABLED}")
    path = Path(out_dir) / "experiment.json"
    res = {"config": C.as_dict(), "seeds": {}}
    selection_cache = {}
    if selection_from is not None:   # reuse the selection of another run
        base = json.loads(Path(selection_from).read_text())["seeds"][str(C.SELECT_ON_SEED)]
        for k, rec in base["arms"].items():
            selection_cache[k] = (rec["params"], rec.get("selection", []))
        log(f"selection taken from {selection_from}")
    if path.exists():
        prev = json.loads(path.read_text())
        res["seeds"].update(prev.get("seeds", {}))
        base = res["seeds"].get(str(C.SELECT_ON_SEED))
        if base is not None:   # selection is inherited, never redone on resume
            for k, rec in base["arms"].items():
                selection_cache[k] = (rec["params"], rec.get("selection", []))
        if res["seeds"]:
            log(f"resuming; seeds already done: {sorted(res['seeds'], key=int)}")
    log(f"windows={X.shape} normal={(y == 0).sum()} fault={(y == 1).sum()} "
        f"unseen={(y == 2).sum()} wells={np.unique(well).size}")
    order = sorted(seeds or C.SEEDS, key=lambda s: s != C.SELECT_ON_SEED)
    for seed in order:
        if str(seed) in res["seeds"]:
            continue
        log(f"=== seed {seed} ===")
        res["seeds"][str(seed)] = run_seed(X, y, code, well, seed, selection_cache, log, out_dir, ym=ym)
        path.write_text(json.dumps(res))
    return res
