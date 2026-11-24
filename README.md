# Asset-Level Conformal Monitoring of Offshore Oil Wells: Open-Set Guarantees When the Exchangeable Unit Is the Well

Reproducibility package and experimental pipeline for the article:

> **Asset-Level Conformal Monitoring of Offshore Oil Wells: Open-Set Guarantees When the Exchangeable Unit Is the Well**  
> André Luiz Marques Serrano and Guilherme Dantas Bispo  
> Submitted to *Reliability Engineering & System Safety* (RESS), Elsevier.  
> Repository: <https://github.com/guilhermebispo/conformal-oil-wells>

This package implements, verifies, and reproduces every empirical result, table, figure, and statistical test reported in the manuscript, using real multivariate sensor data from the public **3W Dataset 2.0.0** (Petrobras).

---

## 1. Overview and Problem Formulation

In industrial Internet of Things (IIoT) monitoring of offshore oil wells:
1. **Exchangeability fails across wells:** windows of one well are dependent and wells differ. When whole wells are held out, window-level split conformal calibration covers 0.75 (random forest) and 0.79 (Transformer) of the test windows instead of the nominal 0.90, and only 65% and 39% of the test wells reach the nominal level (per-well coverage from 0.07 to 1.00).
2. **Asset-level guarantee (Proposition 3):** with the well as the exchangeable unit, the threshold is the $\lceil(K+1)(1-\delta)\rceil$-th smallest of the $K$ within-well quantiles of the calibration wells. With probability at least $1-\delta$ over a new well, at least a fraction $1-\alpha$ of its windows is covered. It requires $K \ge (1-\delta)/\delta$ calibration wells.
3. **Open-set detection:** a three-state rule (normal, fault, Unknown) combines the conformal set with a conformal atypicality test on the representation. Five novelty scores (Mahalanobis, kNN, Isolation Forest, energy, MSP) are evaluated against random abstention (lift) and within wells, which controls for the identity of the well. Separability depends strongly on which fault type is withheld.
4. **Model families:** random forest (30 window statistics), Transformer encoder, LightGBM and 1D-CNN, compared over five random well-disjoint partitions with the corrected resampled $t$-test (Nadeau and Bengio) and Holm's step-down correction.

---

## 2. Package Architecture and File Layout

```
code/
├── run_all.py                 # Master entry point orchestrating all execution stages
├── requirements.txt           # Python dependencies
├── CITATION.cff               # Citation metadata (CFF format)
├── LICENSE                    # MIT License
├── ASSUMPTIONS.md             # Implementation decisions and protocol specifications
│
├── iiot3w/                    # Core Python package
│   ├── config.py              # Configuration constants, hyperparameter grids, and paths
│   ├── data.py                # Audit of 3W instances, causal windowing, and feature extraction
│   ├── splits.py              # 5-fold well-disjoint partitioning under operational constraints
│   ├── models.py              # Model architectures: RF (30 features), Transformer, LightGBM, 1D-CNN
│   ├── conformal.py           # Conformal prediction sets, atypicality detector, 3-state decision rule
│   ├── asset.py               # Asset-level calibration (Proposition 3: well-quantile, pooling, subsampling)
│   ├── novelty.py             # Controlled novelty evaluation (Mahalanobis, kNN, iForest, energy, MSP)
│   ├── temporal.py            # Online Adaptive Conformal Inference (ACI) and event-level metrics
│   ├── stats.py               # Nadeau-Bengio corrected resampled t-test, Holm adjustment, coverage SD
│   ├── experiment.py          # Training, nested hyperparameter selection, inference caching
│   ├── analysis.py            # Statistical aggregation of results for manuscript tables
│   ├── robustness.py          # Robustness analyses (alpha/beta sensitivity, per-well coverage, AUROC)
│   ├── figures.py             # Publication figure generation
│   ├── reference_values.py    # Reference values from the manuscript for automated verification
│   └── compare.py             # Comparison engine generating COMPARISON.md
│
├── scripts/                   # Standalone analysis and utility scripts
│   ├── families_holm.py       # Pairwise comparison across 4 model families with Holm adjustment
│   ├── merge_seeds.py         # Merges parallel seed outputs into unified experiment.json
│   ├── variant_r4.py          # Variant R4: held-out validation early stopping
│   ├── fetch_3w.sh            # Downloads real 3W instances at the pinned Petrobras commit
│   └── exploratory/           # Exploratory re-analyses
│
├── colab/                     # Self-contained Google Colab notebooks
│   ├── 01_run_all_colab.ipynb     # Batch 1: reference, well-disjoint splits, and nested selection
│   └── 02_run_batch2_colab.ipynb  # Batch 2: model families, multiclasse, and fault rotation
│
├── tests/                     # Unit test suite (no dataset required)
│   └── test_core.py           # Verification of Propositions 1-3, coverage theory, split logic
│
└── results/                   # Experimental outputs and evidence archives
    ├── cpu_reference/         # Reference CPU run (303/303 MATCH in COMPARISON.md)
    ├── cpu_reference_holdout_r4/ # Reference outputs for held-out early stopping (R4)
    ├── run01_wells/           # Well-disjoint split, local exploratory run
    └── colab/                 # Runs reported in the article (ref_colab, run01-run05)
```

---

## 3. Installation and Setup

### 3.1 Python Environment

The article's numbers were produced on Google Colab (Linux, Python 3.13.18, CPU, one thread per seed); `requirements-lock.txt` pins the exact versions of that environment, and `requirements.txt` gives the minimum compatible versions. Tree ensembles can differ in the third decimal across operating systems even with identical library versions (the reference run on Windows and its rerun on Colab differ there for the random forest, while the Transformer agrees to the eighth decimal), so use the pinned versions on Linux to reproduce the published values. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt   # or requirements.txt for minimum versions
```

### 3.2 Dependencies

The package requires standard scientific Python libraries:
- `numpy`, `scipy`, `pandas`, `pyarrow`
- `scikit-learn`, `lightgbm`, `torch`
- `matplotlib`, `pytest`

### 3.3 Fetching the 3W Dataset

Download the real instances (`WELL-*`) of the **3W Dataset 2.0.0** (Petrobras), pinned to commit `826fec47d5ba59e93b1b91be572bb48641043a2e`:

```bash
bash scripts/fetch_3w.sh data/3W
```
This downloads approximately 0.5 GB of real sensor data into `data/3W/dataset`.

---

## 4. Execution Guide

### 4.1 Unit Testing (Fast Verification)

Run the unit tests to verify mathematical propositions (Proposition 1, 2, and 3), conformal thresholds, and statistical corrections without needing the 3W dataset:

```bash
pytest tests/ -v
```
All unit tests execute in seconds on synthetic distributions.

### 4.2 Smoke Test (End-to-End Pipeline)

To verify the entire 9-stage pipeline quickly on synthetic data (~1 minute):

```bash
python run_all.py --smoke-test
```

### 4.3 Full Reproduction from 3W Dataset

**The default options reproduce the earlier window-level protocol (TII version), not this article.** Each run of the article needs explicit options (seeds 0-4, one thread per seed):

| Run | Article | Options |
|---|---|---|
| `ref_colab` | contrast: split by windows | *(defaults)* |
| `run02_wells_nested` | RQ1, RQ2, RQ3 (default withheld set), RF vs Transformer | `--split-mode wells --selection nested` |
| `run03_families` | RQ4: four families | `--split-mode wells --selection nested --arms rf,transformer,lgbm,cnn` |
| `run04_multiclass` | RQ4: multiclass | `--split-mode wells --selection nested --label-mode multiclass` |
| `run05_rot{3,4,6,8}` | RQ3: rotation | `--split-mode wells --selection inner --withheld <code>` |

For example:

```bash
python run_all.py --dataset-dir data/3W/dataset --out outputs/run02_wells_nested \
    --split-mode wells --selection nested --threads 1
```

With `--selection inner` the hyperparameters are selected on seed 0 and reused (pass `--selection-from <seed-0 experiment.json>` when seeds run in separate processes). The Colab notebooks in `colab/` run every configuration exactly as published.

### 4.4 Running Specific Stages

`run_all.py` supports selective execution of pipeline stages via `--stages`:

```bash
# Stages: audit, experiment, analysis, robustness, asset, novelty, temporal, figures, compare
python run_all.py --dataset-dir data/3W/dataset --stages audit,experiment --threads 1
python run_all.py --stages analysis,robustness,asset,novelty,temporal,figures,compare
```

Stage descriptions:
- `audit`: Audits raw CSVs and builds causal window tensor (`windows.npz`).
- `experiment`: Fits models across 5 well-disjoint seeds with hyperparameter selection.
- `analysis`: Aggregates metrics and checks coverage guarantees.
- `robustness`: Runs $\alpha/\beta$ sensitivity, per-well coverage, and AUROC evaluations.
- `asset`: Evaluates Proposition 3 well-quantile calibration vs. pooling and subsampling.
- `novelty`: Evaluates controlled novelty scores (Mahalanobis, kNN, iForest, energy, MSP).
- `temporal`: Evaluates online Adaptive Conformal Inference (ACI) and event-level alarms.
- `figures`: Renders publication figures into `figures/`.
- `compare`: Compares computed numbers against reference values in `COMPARISON.md`.

### 4.5 Model Families & Holm Correction

To run pairwise comparisons across the four model families (Random Forest, Transformer, LightGBM, 1D-CNN) with Holm step-down correction:

```bash
python scripts/families_holm.py <run03_families>/experiment.json outputs/families_holm.json
```

The input is the merged `experiment.json` of `run03_families` (about 6 MB, written by notebook 02 to `runs/run03_families/merged/`; not stored in this repository because of its size). Its output for the article is `results/colab/run03_families/families.json`.

### 4.6 Google Colab Workflows

For cloud reproduction without local resource constraints:
1. `colab/01_run_all_colab.ipynb`: Executes reference run, well-disjoint splits, and nested selection.
2. `colab/02_run_batch2_colab.ipynb`: Executes model family benchmarking (RF, Transformer, LightGBM, CNN), multiclass evaluation, and withheld fault code rotation (codes 3, 4, 6, 8).

---

## 5. Summary of Key Theoretical and Empirical Results

- **Proposition 1 (Marginal coverage):** split conformal prediction guarantees $P(Y \in C(X)) \ge 1 - \alpha$ under window exchangeability. The implementation takes the order statistic just above $\lceil (n+1)(1-\alpha)\rceil$ (`np.quantile(..., method="higher")`), which is conservative by at most $1/n$ (below 0.04% for the calibration sizes of the article).
- **Proposition 2 (Error and abstention):** a wrong emitted label has probability at most $\alpha$, and abstention on the known regime is at most $P(|C(X)| \ne 1) + \beta$.
- **Remark (Empty set):** in binary classification the conformal set cannot be empty when $\tau \ge 1/2$; with $K$ classes it can be empty only if $\tau < 1 - 1/K$.
- **Proposition 3 (Asset-Level Coverage):** Taking the $\lceil(K+1)(1-\delta)\rceil$-th quantile across $K$ calibration wells guarantees that with probability $\ge 1 - \delta$ over a new well, its coverage is $\ge 1 - \alpha$.
- **Controlled novelty evaluation:** confidence (MSP) is at chance; representation scores reach AUROC 0.62-0.73 within wells for the default withheld set and 0.90-0.95 for severe slugging and flow instability, but chance for a choke restriction, and for hydrate formation the separation over all windows vanishes within wells. The within-well AUROC pools the windows of all test wells that carry unseen events.

---

## 6. Citation and License

If you use this code or experimental framework in your research, please cite:

```bibtex
@article{serrano2026conformal,
  title   = {Asset-Level Conformal Monitoring of Offshore Oil Wells: Open-Set Guarantees When the Exchangeable Unit Is the Well},
  author  = {Serrano, Andr{\'e} Luiz Marques and Bispo, Guilherme Dantas},
  journal = {Reliability Engineering \& System Safety},
  year    = {2026},
  note    = {Submitted}
}
```

- **Code License:** [MIT License](LICENSE)
- **Data License:** The 3W Dataset is distributed by Petrobras under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
