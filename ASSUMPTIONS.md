# Implementation decisions

Each item states a choice the article does not fix explicitly, the decision
taken, and where it lives in the code.

**A1 – Data version.** 3W Dataset 2.0.0, real instances only (`WELL-*`),
pinned to commit `826fec47d5ba59e93b1b91be572bb48641043a2e` of
`petrobras/3W` (`scripts/fetch_3w.sh`). Later commits of the repository may
add or relabel instances; results are tied to this commit.

**A2 – Hyperparameter selection.** In the primary protocol (nested selection,
`run02_wells_nested`), hyperparameter selection is performed per partition
using internal well-disjoint folds. For the fixed-selection variant
(`config.SELECT_ON_SEED`), the grid search is run once on the partition of
seed 0, and the chosen configuration is reused for seeds 1–4. On resume the
choice is read back from `experiment.json`, never redone (`experiment.run`).

**A3 – Selection criterion.** Configurations are ranked by ROC-AUC on the
internal well-disjoint validation split. Balanced accuracy is not used because
it collapses to 0.5 for any majority-class predictor under the strong class
imbalance of the internal split.

**A4 – Threads and determinism.** The released outputs use one CPU thread
(`--threads 1`), as the original reference runs did. Seeds fix NumPy,
scikit-learn and PyTorch; with more threads PyTorch may sum in a different order
and the Transformer results can differ in the last digits.

**A5 – Unseen windows.** Windows of codes 1, 8 and 9 never enter training,
calibration or selection in the primary evaluation; only those of the test
wells are evaluated (`splits.make_split`). In the rotation protocol (T26,
`--withheld <c>`), each fault code is withheld in turn to evaluate novelty
detection across diverse physical regimes.

**A6 – Corrected t-test.** `n_test/n_train` is the mean over the five
partitions of the known-regime test and training window counts. The 95%
interval uses the t quantile with k − 1 = 4 degrees of freedom and the
Nadeau–Bengio standard error s·sqrt(1/k + n_test/n_train) (`stats.py`).
Pairwise comparisons across the 4 model families are adjusted via Holm's
step-down procedure (`scripts/families_holm.py`).

**A7 – Detectable difference.** Normal approximation,
(z₀.₉₇₅ + z₀.₈) · σ · sqrt(n_test/n_train), the limit k → ∞ of the corrected
standard error; σ is the SD of the per-partition difference.

**A8 – Theoretical SD of coverage.** Conditional on the calibration
set, coverage of split conformal prediction with exchangeable data follows
Beta(n+1−l, l), l = ⌊(n+1)α⌋; the test set adds binomial variance
cov(1−cov)/n_test. The largest value over the five partitions is reported
(`stats.coverage_sd_exchangeable`).

**A9 – Per-well coverage.** Only test wells with at least 100 known-regime
windows in a partition are analysed, so that the binomial band is informative
(`robustness.MIN_WELL_WINDOWS`).

**A10 – α/β sensitivity.** Thresholds are recalibrated on the same calibration
windows for each (α, β); the models are not refit.

**A11 – Comparison tolerance.** A printed value matches when the computed
value rounds to it at the printed number of decimals; counts and strings must
match exactly; values printed as inequalities ("below 0.006", "at least eight
times") are checked as inequalities (`reference_values.py`, `compare.py`).

**A12 – Early stopping.** In the standard protocol
(`config.HOLDOUT_EARLY_STOPPING = False`), the final models are fitted on
all training windows, and the Transformer early-stops on the internal
validation wells. The held-out variant (`--holdout-early-stopping`) fits both
models on the inner-fit wells only and early-stops on the held-out validation
wells, reusing the hyperparameters selected in the main run.

**A13 – Asset-level calibration (Proposition 3).** Split conformal quantiles
computed at the window level fail under well shift. `iiot3w/asset.py` computes
the well-quantile threshold: with K calibration wells, the threshold is the
ceil((K+1)(1-delta))-th smallest within-well conformal quantile. This guarantees
that with probability >= 1 - delta over a new well, its coverage is >= 1 - alpha.
Baselines include the well-balanced pooled CDF and repeated well subsampling
(Dunn, Wasserman & Ramdas, 2023).

**A14 – Online Adaptive Conformal Inference (ACI).** To adapt to unseen well
dynamics in deployment, `iiot3w/temporal.py` implements Gibbs & Candès (2021)
online ACI per well with feedback delay of h windows, updating
alpha_{t+1} = alpha_t + gamma (alpha - err_{t-h}).

**A15 – Model family parity.** The four model families (Random Forest,
Transformer encoder, LightGBM, and 1D-CNN) are trained under identical 5-fold
well-disjoint splits with nested selection. LightGBM runs with single threading
to avoid CPU contention with PyTorch OpenMP runtimes.
