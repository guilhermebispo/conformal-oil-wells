"""The two experimental arms  [Sec. IV-E].

Both receive the same window X_t, the same partition, the same treatment of
class imbalance, and the same hyperparameter-search budget on an internal
split of the training set.  The only deliberate difference is the inductive
bias.  Each arm exposes

    fit(X, y, ...)       training
    predict_proba(X)     probability estimate, shape (n, 2)
    represent(X)         internal representation phi(.) for the atypicality test
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C

N_THREADS = 1   # set by run_all.py (--threads)


# ---------------------------------------------------------------------------
# Window normalization, identical for both arms  [Sec. IV-C]
# ---------------------------------------------------------------------------
def fit_normalizer(X_train):
    """Per-channel scale estimated on the training set only."""
    ref = X_train if C.NORMALIZATION == "raw" else X_train - X_train[:, :1, :]
    sd = ref.reshape(-1, X_train.shape[2]).std(axis=0)
    return {"scale": np.maximum(sd, 1e-8).astype(np.float64)}


def normalize(X, norm):
    """Subtract the first sample of each window (causal; removes the absolute,
    well-specific level) and divide by the training scale (keeps the amplitude
    relation between channels, where event signatures reside)."""
    Z = X if C.NORMALIZATION == "raw" else X - X[:, :1, :]
    return (Z / norm["scale"]).astype(np.float32)


FEATURE_NAMES = ["mean", "std", "min", "max", "median",
                 "q25", "q75", "last", "slope", "log_diff_energy"]


def window_features(X):
    """(n, T, F) window -> (n, 10 F) per-channel statistics (D = 30)."""
    n, T, F = X.shape
    t = np.arange(T, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    d = np.diff(X, axis=1)
    feats = [X.mean(1), X.std(1), X.min(1), X.max(1), np.median(X, 1),
             np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
             X[:, -1, :],
             np.einsum("t,ntf->nf", t, X) / T,
             np.log1p((d ** 2).sum(1))]
    return np.concatenate(feats, axis=1).astype(np.float64)


def n_classes():
    """Number of model classes: 2 (binary, the TII protocol) or 1 + the number
    of known fault types (multiclass, task T27)."""
    return C.N_CLASSES


def full_proba(model, P):
    """predict_proba re-indexed to all n_classes() columns, in case a class is
    absent from the training windows of a partition (its column is zero)."""
    K = n_classes()
    if P.shape[1] == K:
        return P
    out = np.zeros((P.shape[0], K))
    out[:, np.asarray(model.classes_, int)] = P
    return out


def feature_labels(variables):
    return [f"{s}({v})" for s in FEATURE_NAMES for v in variables]


class RFArm:
    name = "Random forest"

    def __init__(self, params, seed):
        self.params, self.seed, self.model = dict(params), seed, None
        self.history = []

    def fit(self, X, y, Xval=None, yval=None, epochs=None, norm=None):
        self.norm = norm if norm is not None else fit_normalizer(X)
        self.model = RandomForestClassifier(class_weight="balanced_subsample",
                                            random_state=self.seed,
                                            n_jobs=N_THREADS, **self.params)
        self.model.fit(window_features(normalize(X, self.norm)), y)
        return self

    def predict_proba(self, X):
        return full_proba(self.model, self.model.predict_proba(window_features(normalize(X, self.norm))))

    def represent(self, X):
        return window_features(normalize(X, self.norm))

    def logits(self, X):
        """A random forest has no logits; the energy score is not defined."""
        return None


class LGBMArm(RFArm):
    """Gradient-boosted trees on the same window statistics as the random
    forest (task T29): a strong tabular baseline with the same representation,
    so that the atypicality test and the novelty scores are comparable."""
    name = "LightGBM"

    def fit(self, X, y, Xval=None, yval=None, epochs=None, norm=None):
        from lightgbm import LGBMClassifier
        self.norm = norm if norm is not None else fit_normalizer(X)
        # n_jobs=1 always: LightGBM's and PyTorch's OpenMP runtimes deadlock when
        # both use several threads in one process (reproduced on macOS,
        # 2026-11-20).  The article runs use one thread per seed anyway.
        self.model = LGBMClassifier(class_weight="balanced", random_state=self.seed,
                                    n_jobs=1, verbose=-1, **self.params)
        self.model.fit(window_features(normalize(X, self.norm)), y)
        return self


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(N_THREADS)
    return torch


class TransformerArm:
    name = "Transformer"

    def __init__(self, params, seed):
        self.params, self.seed, self.net = dict(params), seed, None
        self.history = []

    def _build(self, F):
        torch = _torch()
        nn = torch.nn
        p = self.params
        K = n_classes()

        class PosEnc(nn.Module):
            def __init__(self, d, T):
                super().__init__()
                pe = torch.zeros(T, d)
                pos = torch.arange(T).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, : x.shape[1]]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                d = p["d_model"]
                self.proj = nn.Linear(F, d)
                self.pos = PosEnc(d, C.WINDOW_T)
                layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=p["nhead"], dim_feedforward=4 * d,
                    dropout=p["dropout"], activation="gelu",
                    batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, p["layers"],
                                                 enable_nested_tensor=False)
                self.attn = nn.Linear(d, 1)    # attention pooling
                self.head = nn.Linear(d, K)

            def encode(self, x):
                h = self.enc(self.pos(self.proj(x)))
                w = torch.softmax(self.attn(h), dim=1)
                return (w * h).sum(1)

            def forward(self, x):
                return self.head(self.encode(x))

        torch.manual_seed(self.seed)
        return Net()

    def fit(self, X, y, Xval=None, yval=None, epochs=None, norm=None):
        torch = _torch()
        nn = torch.nn
        epochs = epochs or C.TR_EPOCHS
        self.norm = norm if norm is not None else fit_normalizer(X)
        Xn = normalize(X, self.norm)
        self.net = self._build(Xn.shape[2])

        K = n_classes()
        w = np.bincount(y, minlength=K).astype(np.float64)
        w = w.sum() / (float(K) * np.maximum(w, 1))   # = the binary formula for K = 2
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"],
                                weight_decay=C.TR_WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xt, yt = torch.tensor(Xn), torch.tensor(y.astype(np.int64))
        n = Xt.shape[0]
        rng = np.random.default_rng(self.seed)
        has_val = Xval is not None
        if has_val:
            Xv = torch.tensor(normalize(Xval, self.norm))
            yv = torch.tensor(yval.astype(np.int64))

        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            order = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, C.TR_BATCH):
                b = order[i:i + C.TR_BATCH]
                opt.zero_grad()
                loss = crit(self.net(Xt[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), C.TR_CLIP)
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            rec = {"epoch": ep + 1, "train_loss": tot / n}
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv), yv).item()
                rec["val_loss"] = vl
                self.history.append(rec)
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= C.TR_PATIENCE:
                        break
            else:
                self.history.append(rec)
        if has_val and best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _batched(self, X, fn):
        torch = _torch()
        Xn = torch.tensor(normalize(X, self.norm))
        out = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], 512):
                out.append(fn(Xn[i:i + 512]).numpy())
        return np.concatenate(out) if out else np.empty((0, 2))

    def predict_proba(self, X):
        torch = _torch()
        return self._batched(X, lambda b: torch.softmax(self.net(b), dim=1))

    def represent(self, X):
        return self._batched(X, lambda b: self.net.encode(b))

    def logits(self, X):
        """Pre-softmax outputs, for the energy novelty score (T25)."""
        return self._batched(X, lambda b: self.net(b))


class CNNArm(TransformerArm):
    """One-dimensional convolutional network with global average pooling
    (task T29): the standard deep baseline for time-series classification.
    Same training loop, loss, class weights and early stopping as the
    Transformer; the pooled feature vector serves as phi(.)."""
    name = "1D-CNN"

    def _build(self, F):
        torch = _torch()
        nn = torch.nn
        p = self.params
        K = n_classes()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                c, k = p["channels"], p["kernel"]
                layers, cin = [], F
                for _ in range(3):
                    layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.GELU(),
                               nn.Dropout(p["dropout"])]
                    cin = c
                self.body = nn.Sequential(*layers)
                self.head = nn.Linear(c, K)

            def encode(self, x):            # x: (batch, T, F)
                return self.body(x.transpose(1, 2)).mean(-1)

            def forward(self, x):
                return self.head(self.encode(x))

        torch.manual_seed(self.seed)
        return Net()
