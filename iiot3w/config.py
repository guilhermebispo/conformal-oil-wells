"""Single source of every design choice of the experiment.

No module defines its own constants: every value printed in the article is
traceable either to this file or to a JSON file written by the pipeline.
The section of the article that states each value is given in brackets.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (overridable from the command line of run_all.py)
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PACKAGE_ROOT / "data" / "3W" / "dataset"   # 3W Dataset 2.0.0 root
OUT_DIR = PACKAGE_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data selection  [Sec. IV-B]
# ---------------------------------------------------------------------------
# Real instances only.  Simulated instances come from one phenomenological
# model of one well and hand-drawn ones reproduce idealized signatures; either
# would make results depend on provenance mix and break the well-disjoint
# partition, since they have no genuine well of origin.
INSTANCE_PREFIX = "WELL-"

# Temperature and pressure at the subsea Christmas-tree transducer (TPT) and
# pressure at the permanent downhole gauge (PDG).
VARIABLES = ["T-TPT", "P-TPT", "P-PDG"]

# 3W label taxonomy: 0 normal; 1..9 steady state of an undesirable event;
# 101..109 transient; null unknown.
CLASS_NORMAL = 0
CLASS_KNOWN_FAULT = [2, 3, 4, 5, 6, 7]   # the known-fault class
CLASS_UNOBSERVED = [1, 8, 9]             # withheld from training/calibration

SENTINEL_ABS = 1e12   # magnitudes incompatible with the physical units -> NaN

# ---------------------------------------------------------------------------
# Causal windowing  [Sec. IV-C, Table IV]
# ---------------------------------------------------------------------------
DECIMATION = 5             # 1 Hz -> 0.2 Hz
WINDOW_T = 64              # decimated samples (320 s)
WINDOW_STEP = 32           # stride (160 s)
REQUIRE_LABEL_HOMOGENEITY = True
DROP_TRANSIENTS = True
MAX_WINDOWS_PER_INSTANCE = 60   # cap applied to normal windows only

# "offset": subtract the first sample of the window (causal) and divide by a
# per-channel scale estimated on the training set.  "raw": no normalization.
NORMALIZATION = "offset"

# ---------------------------------------------------------------------------
# Well-disjoint partitioning  [Sec. IV-D, Table III]
# ---------------------------------------------------------------------------
FRAC_TRAIN, FRAC_CAL, FRAC_TEST = 0.60, 0.15, 0.25
MIN_FAULT_TYPES_PER_SPLIT = 2
MIN_PREVALENCE, MAX_PREVALENCE = 0.05, 0.60
FRAC_TOL = 0.08
MAX_SPLIT_ATTEMPTS = 5000
SEEDS = [0, 1, 2, 3, 4]

# Partition mode (task T20 of the reformulation).
#   "windows" (TII article protocol, default): target shares of WINDOWS, which
#             in 3W leaves very few wells in some subsets (seed 0: one
#             calibration well), because five wells hold 86% of the windows.
#   "wells":  wells are the sampling unit.  Each subset needs a minimum number
#             of wells and a minimum share of known windows; fault-type and
#             prevalence constraints are kept; optionally every known fault
#             type in calibration and test must also occur in training.
SPLIT_MODE = "windows"
# Probabilities with which each well is proposed to train / cal / test.
WS_ASSIGN_PROB = (0.40, 0.35, 0.25)
# Minimum number of wells per subset.  Nine calibration wells is the smallest
# K for which the well-level conformal quantile is finite at delta = 0.10,
# since ceil((K+1)(1-delta)) <= K requires K >= (1-delta)/delta  [T23].
WS_MIN_WELLS = {"train": 5, "cal": 9, "test": 5}
# Minimum share of the known-regime windows per subset (models need data).
WS_MIN_SHARE = {"train": 0.40, "cal": 0.05, "test": 0.10}
# Fixes defect D4 of the TII protocol: no fault type in calibration or test
# that the model never saw in training.
WS_REQUIRE_TYPES_IN_TRAIN = True
WS_MAX_ATTEMPTS = 200000

# ---------------------------------------------------------------------------
# Three-state decision rule  [Sec. III, Table IV]
# ---------------------------------------------------------------------------
ALPHA = 0.10       # conformal predictor level
BETA = 0.10        # atypicality detector level
SHRINKAGE = 0.10   # Ledoit-Wolf-type shrinkage toward a scaled identity
# Well-level calibration (task T23): with probability >= 1 - WELL_DELTA over
# the draw of a new well, the fraction of its windows covered is >= 1 - alpha.
WELL_DELTA = 0.10

# ---------------------------------------------------------------------------
# Models  [Sec. IV-E, Table IV]
# ---------------------------------------------------------------------------
RF_GRID = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2},
]
TR_GRID = [
    {"d_model": 48, "nhead": 4, "layers": 2, "lr": 3e-4, "dropout": 0.1},
    {"d_model": 48, "nhead": 4, "layers": 2, "lr": 1e-4, "dropout": 0.1},
    {"d_model": 64, "nhead": 4, "layers": 2, "lr": 3e-4, "dropout": 0.2},
]
# Baselines of task T29 (same budget of three configurations).
LGBM_GRID = [
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
    {"n_estimators": 500, "num_leaves": 15, "learning_rate": 0.05},
    {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.10},
]
CNN_GRID = [
    {"channels": 32, "kernel": 7, "lr": 1e-3, "dropout": 0.1},
    {"channels": 64, "kernel": 5, "lr": 1e-3, "dropout": 0.1},
    {"channels": 64, "kernel": 9, "lr": 3e-4, "dropout": 0.2},
]
# Arms run by the experiment; the TII protocol uses the first two.
ARMS_ENABLED = ["rf", "transformer"]
# Short tag of each arm in results.json keys.
ARM_TAGS = {"rf": "rf", "transformer": "tr", "lgbm": "lgbm", "cnn": "cnn"}

# Label formulation (task T27).  "binary": normal vs. any known fault (TII
# protocol); "multiclass": normal plus one class per known fault type.
LABEL_MODE = "binary"
N_CLASSES = 2          # set by experiment.run from LABEL_MODE and the known types

TR_EPOCHS = 15
TR_SELECT_EPOCHS = 6     # reduced budget during selection
SELECT_ON_SEED = 0       # selection done once and reused for all seeds
TR_PATIENCE = 4
TR_BATCH = 256
TR_WEIGHT_DECAY = 1e-2
TR_CLIP = 1.0
INNER_VAL_FRAC = 0.20
INNER_VAL_TOL = 0.08
# Hyperparameter selection (task T21).
#   "inner"  (TII protocol, default): one internal well-disjoint split, on
#            SELECT_ON_SEED only, reused for every seed.
#   "nested": in EVERY seed, grouped K-fold over the training wells (each
#            validation fold holds whole wells and >= NESTED_MIN_PER_CLASS
#            windows of each class); a configuration is scored by the ROC-AUC
#            of its pooled out-of-fold predictions (folds are very unequal
#            in 3W, so a mean of per-fold AUCs would over-weight tiny folds).  Calibration and test wells are never touched.
SELECTION = "inner"
NESTED_FOLDS = 3
# Minimum windows of EACH class in every validation fold; folds with a
# handful of faults give a meaningless ROC-AUC.
NESTED_MIN_PER_CLASS = 30
NESTED_MAX_ATTEMPTS = 5000

# Novelty scores computed at fit time and stored per window (task T25).
# kNN and Isolation Forest use a seeded subsample of the training
# representation as reference.
NOVELTY_REF_MAX = 5000
NOVELTY_KNN_K = 10
NOVELTY_IF_TREES = 200
# False (article protocol): final fit on all training windows, Transformer
# early-stopped on internal validation wells that are part of them.
# True (robustness variant R4): final fit on the inner-fit wells only.
HOLDOUT_EARLY_STOPPING = False

# ---------------------------------------------------------------------------
# Statistics  [Sec. IV-F]
# ---------------------------------------------------------------------------
CI = 0.95
POWER = 0.80
SIGNIFICANCE = 0.05
ECE_BINS = 15


def as_dict():
    """Upper-case settings, for provenance in the output JSON."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, list, dict, Path))}
