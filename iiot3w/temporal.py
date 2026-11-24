"""Online calibration (T24, RQ2) and event-level metrics (T28).

Post-hoc analyses of the stored per-window outputs; no model is refitted.
Both need the instance and the end time of every test window, which the
arrays do not store.  They are recovered by regenerating the windows with
provenance (audit stage of this version, windows.npz with "inst" and
"t_end") and redoing the deterministic partition of each seed: the order of
the test, calibration and unseen windows then matches the arrays exactly,
which `attach` verifies before any computation.

T24  Adaptive conformal inference (Gibbs and Candes, 2021) per test well.
     The windows of a well are processed in time order.  At step t the set is
     built with the window-level calibration quantile at level 1 - alpha_t,
     and the level is updated with the miscoverage observed h windows earlier
     (labels arrive with a delay of h windows, e.g. after an operator's
     confirmation):  alpha_{t+1} = alpha_t + gamma (alpha - err_{t-h}).
     The long-run miscoverage of each well then tends to alpha whatever the
     shift, at the price of the labels.

T28  Event-level metrics.  An event is a test-well instance of a known fault
     or of an unseen regime.  An alarm is any decision other than "normal"
     (label 1 or Unknown); for unseen events the Unknown state is reported
     separately.  Reported: fraction of events with at least one alarm, delay
     from the first window of the event to the first alarm, alarm rate on
     normal windows, and fraction of normal instances with at least one
     alarm.  The first window of an event is the first steady-state window:
     transients are not windowed, so delays are measured from the start of
     the steady state, not from the onset of the event.

Writes temporal.json and adds aci.* and event.* keys to results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import conformal as CF
from . import data as D
from . import splits as SP

ACI_GAMMAS = (0.005, 0.01, 0.05)
ACI_DELAYS = (0, 10, 60)          # windows; 60 windows of stride 160 s ~ 2.7 h
MIN_WELL_WINDOWS = 100
ARMS = (("rf", "rf"), ("transformer", "tr"))   # default order; see _arms()


def _arms(rows):
    """(arm, tag) of every arm present in the rows, RF and Transformer first."""
    present = {r["arm"] for r in rows}
    order = [a for a in C.ARM_TAGS if a in present] + sorted(present - set(C.ARM_TAGS))
    return [(a, C.ARM_TAGS.get(a, a)) for a in order]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def attach(d, seed, prov):
    """Instance and end time of the calibration, test and unseen windows of
    one stored array file, after checking that the regenerated partition has
    the same order as the stored arrays."""
    y, code, well, inst, t_end = prov
    i_tr, i_ca, i_te, i_un, _ = SP.partition(y, code, well, seed)
    # y_test holds the model labels: 0/1 (binary) or 0..K-1 (multiclass, T27);
    # in both cases "label > 0" is the known-fault indicator y of the windows.
    checks = (np.array_equal(d["well_test"], well[i_te]),
              np.array_equal((d["y_test"] > 0).astype(int), y[i_te].astype(int)),
              np.array_equal(d["code_test"], code[i_te]), np.array_equal(d["well_cal"], well[i_ca]),
              np.array_equal(d["well_unseen"], well[i_un]))
    if not all(checks):
        raise RuntimeError(f"seed {seed}: regenerated partition does not match the stored arrays")
    return {"inst_test": inst[i_te], "t_test": t_end[i_te],
            "inst_unseen": inst[i_un], "t_unseen": t_end[i_un]}


# ---------------------------------------------------------------------------
# T24: adaptive conformal inference per well
# ---------------------------------------------------------------------------
def _threshold_at(sorted_cal, level):
    """Split-conformal quantile of sorted calibration scores at miscoverage
    `level` (+inf for level <= 0: full set; -inf for level >= 1: empty set)."""
    n = len(sorted_cal)
    if level <= 0:
        return np.inf
    if level >= 1:
        return -np.inf
    k = int(np.ceil((n + 1) * (1 - level)))
    return np.inf if k > n else sorted_cal[k - 1]


def aci_well(p, y, s_cal_sorted, alpha, gamma, delay):
    """ACI over the windows of one well, given in time order.  Returns the
    per-window coverage indicator and set size."""
    n = len(y)
    a = alpha
    err = np.zeros(n)
    size = np.zeros(n, int)
    single_label = np.full(n, -1)
    for t in range(n):
        tau = _threshold_at(s_cal_sorted, a)
        sets = (1.0 - p[t]) <= tau
        err[t] = 0.0 if sets[y[t]] else 1.0
        size[t] = int(sets.sum())
        if size[t] == 1:
            single_label[t] = int(np.argmax(sets))
        if t - delay >= 0:              # the label of window t - delay arrives now
            a = a + gamma * (alpha - err[t - delay])
    return 1.0 - err, size, single_label


def aci_analysis(d, extra, alpha):
    p, y, w = d["p_test"].astype(float), d["y_test"].astype(int), d["well_test"]
    r, t = d["r_test"].astype(float), extra["t_test"]
    s_cal = np.sort(CF.nonconformity(d["p_cal"].astype(float), d["y_cal"].astype(int)))
    rho = float(np.quantile(d["r_cal"].astype(float),
                            CF.conformal_quantile_level(len(d["r_cal"]), C.BETA), method="higher"))
    out = []
    for gamma in ACI_GAMMAS:
        for delay in ACI_DELAYS:
            cov_w, size_all, abst_all, hit_all = [], [], [], []
            for g in np.unique(w):
                m = np.where(w == g)[0]
                m = m[np.argsort(t[m], kind="stable")]
                hit, size, single = aci_well(p[m], y[m], s_cal, alpha, gamma, delay)
                if len(m) >= MIN_WELL_WINDOWS:
                    cov_w.append(hit.mean())
                hit_all.append(hit)
                size_all.append(size)
                abst_all.append((single < 0) | (r[m] > rho))
            cov_w = np.array(cov_w)
            out.append({"gamma": gamma, "delay": delay,
                        "coverage": float(np.concatenate(hit_all).mean()),
                        "coverage_well_mean": float(cov_w.mean()),
                        "share_wells_covered": float((cov_w >= 1 - alpha - 1e-12).mean()),
                        "coverage_worst_well": float(cov_w.min()),
                        "mean_set_size": float(np.concatenate(size_all).mean()),
                        "abstention": float(np.concatenate(abst_all).mean())})
    return out


# ---------------------------------------------------------------------------
# T28: event-level metrics
# ---------------------------------------------------------------------------
def event_analysis(d, extra, tau, rho, stride_s=C.WINDOW_STEP * C.DECIMATION):
    """Event metrics of the three-state rule with thresholds (tau, rho)."""
    dec_te = CF.three_state_decision(d["p_test"].astype(float), tau, d["r_test"].astype(float), rho)
    dec_un = CF.three_state_decision(d["p_unseen"].astype(float), tau, d["r_unseen"].astype(float), rho)
    y = (d["y_test"].astype(int) > 0).astype(int)     # fault vs. normal, any label mode
    out = {}
    for name, dec, inst, tt, mask, alarm_fn in (
            ("known_fault", dec_te, extra["inst_test"], extra["t_test"], y == 1, lambda v: v != 0),
            ("unseen_any_alarm", dec_un, extra["inst_unseen"], extra["t_unseen"],
             np.ones(len(dec_un), bool), lambda v: v != 0),
            ("unseen_unknown", dec_un, extra["inst_unseen"], extra["t_unseen"],
             np.ones(len(dec_un), bool), lambda v: v < 0)):
        det, delays = [], []
        for i in np.unique(inst[mask]):
            m = np.where(mask & (inst == i))[0]
            m = m[np.argsort(tt[m], kind="stable")]
            a = alarm_fn(dec[m])
            det.append(bool(a.any()))
            if a.any():
                delays.append((tt[m][np.argmax(a)] - tt[m][0]) / 60.0)   # minutes
        out[name] = {"n_events": len(det), "detected": float(np.mean(det)) if det else float("nan"),
                     "delay_median_min": float(np.median(delays)) if delays else float("nan"),
                     "delay_p90_min": float(np.percentile(delays, 90)) if delays else float("nan")}
    normal = y == 0
    alarms = dec_te[normal] != 0
    inst_n = extra["inst_test"][normal]
    per_inst = [alarms[inst_n == i].any() for i in np.unique(inst_n)]
    out["normal"] = {
        "n_windows": int(normal.sum()), "n_instances": len(per_inst),
        "alarm_rate_per_window": float(alarms.mean()),
        # Split of the alarms on normal operation: a wrong fault label versus a
        # referral to inspection (Unknown state).
        "fault_label_rate_per_window": float((dec_te[normal] == 1).mean()),
        "unknown_rate_per_window": float((dec_te[normal] < 0).mean()),
        # One decision per stride; normal windows are capped per instance, so
        # this is a rate per hour of *monitored decisions*, not of calendar time.
        "alarms_per_hour_of_decisions": float(alarms.mean() * 3600.0 / stride_s),
        "instances_with_alarm": float(np.mean(per_inst))}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(out_dir: Path, log=print, provenance_dir: Path | None = None):
    from .asset import _load, thresholds
    out_dir = Path(out_dir)
    pdir = Path(provenance_dir or out_dir)
    X, y, code, well = D.load_windows(pdir)
    try:
        inst, t_end, _ = D.load_provenance(pdir)
    except (KeyError, FileNotFoundError) as e:
        log(f"no window provenance in {pdir} ({e}); temporal analyses skipped")
        return None
    # The partition must be redone with the labels of this run (T26 rotation).
    y = D.label_of(code).astype(np.int8)
    del X
    prov = (y, code, well, inst, t_end)
    arrs = _load(out_dir / "arrays")
    if not arrs:
        log("no stored arrays; temporal analyses skipped")
        return None
    rows_aci, rows_ev = [], []
    for (seed, arm), d in sorted(arrs.items()):
        extra = attach(d, seed, prov)
        for r in aci_analysis(d, extra, C.ALPHA):
            rows_aci.append({"seed": seed, "arm": arm, **r})
        for method in ("window", "well_quantile"):
            tau, rho, _ = thresholds(d, method, C.ALPHA, C.BETA, C.WELL_DELTA)
            rows_ev.append({"seed": seed, "arm": arm, "method": method,
                            **event_analysis(d, extra, tau, rho)})
    summ = {}
    for arm, tag in _arms(rows_aci):
        for gamma in ACI_GAMMAS:
            for delay in ACI_DELAYS:
                r = [x for x in rows_aci if x["arm"] == arm and x["gamma"] == gamma and x["delay"] == delay]
                if not r:
                    continue
                b = f"aci.g{gamma}.h{delay}.{tag}"
                for k in ("coverage", "coverage_well_mean", "share_wells_covered",
                          "coverage_worst_well", "mean_set_size", "abstention"):
                    summ[f"{b}.{k}"] = float(np.mean([x[k] for x in r]))
        for method in ("window", "well_quantile"):
            r = [x for x in rows_ev if x["arm"] == arm and x["method"] == method]
            if not r:
                continue
            b = f"event.{method}.{tag}"
            for ev in ("known_fault", "unseen_any_alarm", "unseen_unknown"):
                for k in ("detected", "delay_median_min"):
                    summ[f"{b}.{ev}.{k}"] = float(np.nanmean([x[ev][k] for x in r]))
                summ[f"{b}.{ev}.n_events"] = int(sum(x[ev]["n_events"] for x in r))
            for k in ("alarm_rate_per_window", "fault_label_rate_per_window", "unknown_rate_per_window",
                      "alarms_per_hour_of_decisions", "instances_with_alarm"):
                summ[f"{b}.normal.{k}"] = float(np.mean([x["normal"][k] for x in r]))
    (out_dir / "temporal.json").write_text(json.dumps(
        {"alpha": C.ALPHA, "beta": C.BETA, "gammas": ACI_GAMMAS, "delays": ACI_DELAYS,
         "summary": summ, "aci": rows_aci, "events": rows_ev}, indent=1))
    res = out_dir / "results.json"
    if res.exists():
        R = json.loads(res.read_text())
        R.update(summ)
        res.write_text(json.dumps(R, indent=1))
    for arm, tag in _arms(rows_aci):
        b = f"aci.g0.01.h10.{tag}"
        if f"{b}.coverage" in summ:
            log(f"  {tag} ACI(g=0.01,h=10): cov={summ[b + '.coverage']:.3f} "
                f"wells>=1-a={summ[b + '.share_wells_covered']:.2f} |C|={summ[b + '.mean_set_size']:.2f} "
                f"abst={summ[b + '.abstention']:.3f}")
        b = f"event.window.{tag}"
        if f"{b}.known_fault.detected" in summ:
            log(f"  {tag} events(window): known detected={summ[b + '.known_fault.detected']:.2f} "
                f"delay={summ[b + '.known_fault.delay_median_min']:.0f} min | unseen Unknown="
                f"{summ[b + '.unseen_unknown.detected']:.2f} | normal alarm rate="
                f"{summ[b + '.normal.alarm_rate_per_window']:.3f}, instances with alarm="
                f"{summ[b + '.normal.instances_with_alarm']:.2f}")
    return summ
