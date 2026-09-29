"""Model ladder on host-bucket windows.

  lr        logistic regression on the current bucket
  cur       LightGBM on the current bucket
  lag       LightGBM on current + 3 explicit lags + 10-bucket mean/max
  world     GRU dynamics model P(S_t+1 | history) rolled forward K steps; a LightGBM head reads
            [current state, forecast mean, forecast max, latent state] to output P(attack in next K)
  stage     LightGBM multiclass: which stage the forecast attack falls in (trained on onsets only)

`shuffle=True` permutes the order of the history buckets inside each window (fixed seed, applied
identically at fit and predict). Models that really use temporal order must lose skill.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .features import FEATURES, Table, human

F = len(FEATURES)
LGB = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=15, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=4)


class Scaler:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), 1e-6)

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def windows(X: np.ndarray, idx: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Gather (N, L, F) windows; index -1 selects the pad row (no-traffic state)."""
    return np.vstack([X, pad[None]])[idx]


def shuffled_idx(idx: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute history buckets (all but the current one) independently per row."""
    rng = np.random.default_rng(seed)
    perm = rng.random(idx[:, :-1].shape).argsort(1)
    out = idx.copy()
    out[:, :-1] = np.take_along_axis(idx[:, :-1], perm, 1)
    return out


def lag_matrix(W: np.ndarray):
    """Order-sensitive lags 1-3 plus order-invariant 10-bucket mean and max."""
    parts = [W[:, -1], W[:, -2], W[:, -3], W[:, -4], W.mean(1), W.max(1)]
    tags = ["now", "1 min ago", "2 min ago", "3 min ago", "10-min mean", "10-min max"]
    names = [f"{h}|{t}" for t in tags for h in FEATURES]
    return np.hstack(parts), names


def pretty(name: str) -> str:
    f, _, tag = name.partition("|")
    if f.startswith("latent"):
        return f"learned latent state, dimension {f[6:]}"
    return f"{human(f)} ({tag})" if tag else human(f)


class Dyn(nn.Module):
    """Residual GRU: next state = current state + f(history)."""

    def __init__(self, f=F, h=48):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.out = nn.Linear(h, f)
        self.h = h

    def encode(self, x):
        o, _ = self.gru(x)
        return o[:, -1]

    def forward(self, x):
        return x[:, -1] + self.out(self.encode(x))


@torch.no_grad()
def rollout(model: Dyn, W: np.ndarray, K: int, chunk: int = 8192):
    """Forward simulation: feed each predicted state back in for K steps."""
    model.eval()
    preds, zs = [], []
    for s in range(0, len(W), chunk):
        w = torch.from_numpy(W[s:s + chunk])
        zs.append(model.encode(w).numpy())
        step = []
        for _ in range(K):
            nxt = model(w)
            step.append(nxt.numpy())
            w = torch.cat([w[:, 1:], nxt[:, None]], 1)
        preds.append(np.stack(step, 1))
    return np.concatenate(preds), np.concatenate(zs)


def head_matrix(cur, preds, z):
    X = np.hstack([cur, preds.mean(1), preds.max(1), z])
    names = ([f"{f}|now" for f in FEATURES] + [f"{f}|forecast mean" for f in FEATURES]
             + [f"{f}|forecast max" for f in FEATURES] + [f"latent{i}|state" for i in range(z.shape[1])])
    return X, names


def _lgb(y, X, **kw):
    return lgb.LGBMClassifier(**{**LGB, **kw}).fit(X, y)


def fit_dyn(Xs: np.ndarray, tab: Table, idx: np.ndarray, rows: np.ndarray, pad, epochs=12, seed=0) -> Dyn:
    """Dynamics learning: predict the next scaled state from the L-bucket history."""
    torch.manual_seed(seed)
    has_next = (rows + 1) < tab.end[rows]
    rows = rows[has_next]
    W = windows(Xs, idx[rows], pad)
    Y = Xs[rows + 1]
    model, opt = Dyn(), None
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    Wt, Yt = torch.from_numpy(W), torch.from_numpy(Y)
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(Wt)), max(1, len(Wt) // 1024)):
            loss = ((model(Wt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def fit(tab: Table, train_rows: np.ndarray, parts=("lr", "cur", "lag", "world", "stage"), shuffle=False, seed=0):
    """Fit on `train_rows` (row indices of the training days). Scaler and models see only these rows."""
    b = {"shuffle": shuffle, "seed": seed, "K": tab.K, "L": tab.L}
    scaler = Scaler(tab.X[train_rows])
    b["scaler"] = scaler
    pad_raw = np.zeros(F, dtype=np.float32)
    pad_sc = scaler(pad_raw[None])[0]
    idx = shuffled_idx(tab.idx, seed) if shuffle else tab.idx
    tr = train_rows[tab.valid[train_rows]]
    y = tab.fut[tr].astype(int)
    if "lr" in parts:
        b["lr"] = LogisticRegression(max_iter=2000, C=0.5).fit(scaler(tab.X[tr]), y)
    if "cur" in parts:
        b["cur"] = _lgb(y, tab.X[tr], random_state=seed)
    if "lag" in parts or "stage" in parts:
        Wr = windows(tab.X, idx[tr], pad_raw)
        Xl, names = lag_matrix(Wr)
        b["lag_names"] = names
        if "lag" in parts:
            b["lag"] = _lgb(y, Xl, random_state=seed)
        if "stage" in parts:
            on = (~tab.attack[tr]) & tab.fut[tr]
            cls = np.unique(tab.fut_stage[tr][on])
            b["stage_classes"] = cls
            if len(cls) >= 2:
                b["stage"] = lgb.LGBMClassifier(**{**LGB, "n_estimators": 80, "min_child_samples": 5,
                                                   "random_state": seed}).fit(Xl[on], tab.fut_stage[tr][on])
    if "world" in parts:
        Xs = scaler(tab.X)
        dyn = fit_dyn(Xs, tab, idx, train_rows, pad_sc, seed=seed)
        b["dyn"] = dyn
        preds, z = rollout(dyn, windows(Xs, idx[tr], pad_sc), tab.K)
        Xh, hn = head_matrix(Xs[tr], preds, z)
        b["head_names"] = hn
        b["head"] = _lgb(y, Xh, random_state=seed)
    return b


def predict(b, tab: Table, rows: np.ndarray, which=("lr", "cur", "lag", "world"), explain=False):
    """P(attack in next K buckets) for `rows` from each requested model."""
    out = {}
    scaler = b["scaler"]
    pad_raw = np.zeros(F, dtype=np.float32)
    pad_sc = scaler(pad_raw[None])[0]
    idx = shuffled_idx(tab.idx, b["seed"]) if b["shuffle"] else tab.idx
    if "lr" in which and "lr" in b:
        out["lr"] = b["lr"].predict_proba(scaler(tab.X[rows]))[:, 1]
    if "cur" in which and "cur" in b:
        out["cur"] = b["cur"].predict_proba(tab.X[rows])[:, 1]
    if ("lag" in which and "lag" in b) or "stage" in which:
        Xl, _ = lag_matrix(windows(tab.X, idx[rows], pad_raw))
        if "lag" in b and "lag" in which:
            out["lag"] = b["lag"].predict_proba(Xl)[:, 1]
            if explain:
                out["lag_contrib"] = b["lag"].predict(Xl, pred_contrib=True)
        if "stage" in which:
            k = len(b["stage_classes"])
            if "stage" in b:
                pr = b["stage"].predict_proba(Xl)
            else:
                pr = np.ones((len(rows), max(k, 1)))
            full = np.zeros((len(rows), 5))
            for j, c in enumerate(b["stage_classes"]):
                full[:, c] = pr[:, j]
            out["stage"] = full
    if "world" in which and "world" not in out and "head" in b:
        Xs = scaler(tab.X)
        preds, z = rollout(b["dyn"], windows(Xs, idx[rows], pad_sc), tab.K)
        Xh, _ = head_matrix(Xs[rows], preds, z)
        out["world"] = b["head"].predict_proba(Xh)[:, 1]
        out["world_rollout"] = preds  # (n, K, F) in scaled space
        if explain:
            out["world_contrib"] = b["head"].predict(Xh, pred_contrib=True)
    return out
