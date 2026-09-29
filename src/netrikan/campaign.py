"""ZeekData24 models.

Task A  recognizer: which ATT&CK techniques are active in this host-minute (multi-label, current
        minute's behaviour only). One LightGBM per technique.
Task B  campaign forecaster: for an attacker, P(technique T fires within the next K minutes), given
        the attacker's recent technique activity. Inputs are *history only* (no clock, no IPs):
          since/n60/n150 per technique from the last 150 minutes of flags,
          plus the current minute's behaviour.
        Ladder: renewal hazard | state only | own-technique history | all-technique history |
        GRU world model (next-state dynamics rolled K steps, read by a GBDT head).
        Ablation (NOT a shipped model, breaks the no-timestamp rule): + minute-of-hour.
        Controls: window-shuffled history (LightGBM), block-shuffled sequences (world model).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from .features import FEATURES, LOG_IDX, signed_log1p
from .zeek import TECH_LABEL, TECH_NAME, TECHS

K, WIN, NT = 5, 150, len(TECHS)
LGB = dict(n_estimators=150, learning_rate=0.06, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=4)


def log_state(grid: pd.DataFrame) -> np.ndarray:
    X = grid[FEATURES].to_numpy(np.float64)
    X[:, LOG_IDX] = signed_log1p(X[:, LOG_IDX])
    return X.astype(np.float32)


def _fit(X, y, seed=0, **kw):
    if y.sum() == 0 or y.sum() == len(y):
        return SimpleNamespace(const=float(y.mean()) if len(y) else 0.0)
    return lgb.LGBMClassifier(**{**LGB, "random_state": seed, **kw}).fit(X, y)


def _proba(m, X):
    return np.full(len(X), m.const) if hasattr(m, "const") else m.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ Task A: recognizer
def fit_recognizer(X, Y, rows, seed=0):
    return [_fit(X[rows], Y[rows, j], seed) for j in range(NT)]


def predict_recognizer(models, X, traffic):
    P = np.zeros((len(X), NT))
    idx = np.flatnonzero(traffic)
    for j, m in enumerate(models):
        P[idx, j] = _proba(m, X[idx])
    return P


# ------------------------------------------------------------------ Task B: attacker sequences
@dataclass
class Seqs:
    host: np.ndarray
    wk: np.ndarray
    bucket: np.ndarray   # absolute minute
    X: np.ndarray        # (N, 38) current-minute behaviour, log space
    vol: np.ndarray      # (N, 2) log1p outbound / inbound flow counts
    F: np.ndarray        # (N, 5) true technique flags
    Fp: np.ndarray       # (N, 5) recognizer-predicted flags
    start: np.ndarray    # (N,) first row of this row's sequence
    end: np.ndarray      # (N,) one past last row

    @property
    def n(self):
        return len(self.host)

    def seq_bounds(self):
        s = np.unique(self.start)
        return [(a, int(self.end[a])) for a in s]

    def take(self, wks):
        m = np.isin(self.wk, wks)
        return self.subset(np.flatnonzero(m))

    def subset(self, rows):
        # rows must be a union of whole sequences, in order
        sid = np.cumsum(np.r_[True, self.start[rows][1:] != self.start[rows][:-1]]) - 1
        first = np.flatnonzero(np.r_[True, np.diff(sid) != 0])
        lens = np.diff(np.r_[first, len(rows)])
        start = np.repeat(first, lens)
        return Seqs(self.host[rows], self.wk[rows], self.bucket[rows], self.X[rows], self.vol[rows], self.F[rows], self.Fp[rows],
                    start, start + np.repeat(lens, lens))


def build_seqs(grid: pd.DataFrame, X: np.ndarray, F: np.ndarray, Fp: np.ndarray, day2wk: dict, select: np.ndarray, min_active: int = 30) -> Seqs:
    """Per (host, week) contiguous minute sequences for hosts that *send* traffic flagged with a technique.
    `select` (N, 5) chooses which flags define an attacker (true flags for evaluation, predicted for deployment)."""
    wk = grid["day"].map(day2wk).to_numpy()
    out_flows = grid["out_n_flows"].to_numpy()
    active = ((out_flows > 0)[:, None] & select).any(1)
    df = pd.DataFrame({"host": grid["host"].to_numpy(), "wk": wk, "a": active})
    cnt = df.groupby(["host", "wk"])["a"].sum()
    keep = set(cnt[cnt >= min_active].index)
    idx = df.groupby(["host", "wk"]).indices
    vol_all = np.log1p(np.c_[out_flows, grid["in_n_flows"].to_numpy()]).astype(np.float32)
    buckets = grid["bucket"].to_numpy()
    parts = {k: [] for k in ("host", "wk", "bucket", "X", "vol", "F", "Fp", "start", "end")}
    n = 0
    for key in sorted(keep, key=lambda k: (str(k[1]), str(k[0]))):
        r = idx[key]
        b = buckets[r]
        T = int(b.max() - b.min() + 1)
        pos = (b - b.min()).astype(int)
        x = np.zeros((T, X.shape[1]), np.float32); x[pos] = X[r]
        v = np.zeros((T, 2), np.float32); v[pos] = vol_all[r]
        f = np.zeros((T, NT), bool); f[pos] = F[r]
        fp = np.zeros((T, NT), bool); fp[pos] = Fp[r]
        parts["host"].append(np.full(T, key[0], dtype=object)); parts["wk"].append(np.full(T, key[1], dtype=object))
        parts["bucket"].append(np.arange(b.min(), b.max() + 1)); parts["X"].append(x); parts["vol"].append(v)
        parts["F"].append(f); parts["Fp"].append(fp)
        parts["start"].append(np.full(T, n)); parts["end"].append(np.full(T, n + T))
        n += T
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return Seqs(**cat)


def targets(S: Seqs):
    """fut[i, j]: technique j active in the next K minutes; valid[i]: the whole horizon lies inside the sequence."""
    n = S.n
    i = np.arange(n)
    cs = np.vstack([np.zeros((1, NT), int), np.cumsum(S.F, 0)])
    hi = np.minimum(i + K, S.end - 1)
    fut = (cs[hi + 1] - cs[i + 1]) > 0
    valid = (i + K) <= (S.end - 1)
    return fut & valid[:, None], valid


def view(S: Seqs, j: int, fut, valid, bucket_s=60):
    """Adapter so metrics.evaluate / block_bootstrap can score technique j."""
    return SimpleNamespace(fut=fut[:, j] & valid, attack=S.F[:, j], start=S.start, end=S.end, K=K, valid=valid, bucket_s=bucket_s)


SCHED_NAMES = [f"{kind}|{TECH_NAME[t]}" for kind in ("since", "n60", "n150") for t in TECHS]


def sched_feats(S: Seqs, flags: np.ndarray, shuffle: bool = False, seed: int = 0) -> np.ndarray:
    """Per technique: minutes since last active (150 = none in window), bursts-minutes in last 60 and 150.
    shuffle=True permutes the order of the 150-minute window before summarising (control)."""
    out = np.zeros((S.n, 3 * NT), np.float32)
    rng = np.random.default_rng(seed)
    for a, b in S.seq_bounds():
        f = flags[a:b]
        W = sliding_window_view(np.vstack([np.zeros((WIN - 1, NT), bool), f]), WIN, axis=0)  # (T, NT, WIN)
        if shuffle:
            perm = rng.random((b - a, WIN)).argsort(1)
            W = np.take_along_axis(W, perm[:, None, :], 2)
        last = np.where(W, np.arange(WIN), -1).max(-1)
        out[a:b, :NT] = np.where(last >= 0, WIN - 1 - last, WIN)
        out[a:b, NT:2 * NT] = W[..., WIN - 60:].sum(-1)
        out[a:b, 2 * NT:] = W.sum(-1)
    return out


class Hazard:
    """Renewal baseline: P(burst within K | minutes since the last burst), nonparametric, 3-minute bins."""
    B = 3

    def fit(self, since, y, prior_n=20.0):
        b = np.minimum(since.astype(int), WIN) // self.B
        nb = WIN // self.B + 1
        tot, pos = np.bincount(b, minlength=nb), np.bincount(b, weights=y, minlength=nb)
        p0 = y.mean()
        self.rate = (pos + prior_n * p0) / (tot + prior_n)
        return self

    def predict(self, since):
        return self.rate[np.minimum(since.astype(int), WIN) // self.B]


# ------------------------------------------------------------------ GRU world model
class SeqDyn(nn.Module):
    """Predicts the next minute's [technique flags (logits), log flow counts (residual)] from the whole past."""

    def __init__(self, f=NT + 2, h=48):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.fl = nn.Linear(h, NT)
        self.cn = nn.Linear(h, 2)
        self.h = h

    def heads(self, o, x):
        return self.fl(o), x[..., NT:] + self.cn(o)


def _block_perm(bucket: np.ndarray, seed: int) -> np.ndarray:
    """Random order within each clock-hour block (blocks stay in place)."""
    return np.lexsort((np.random.default_rng(seed).random(len(bucket)), bucket // 60))


class World:
    def __init__(self, block_shuffle=False, seed=0, epochs=6):
        self.block_shuffle, self.seed, self.epochs = block_shuffle, seed, epochs

    def _vec(self, S: Seqs, flags):
        V = np.hstack([flags.astype(np.float32), (S.vol - self.mu) / self.sd]).astype(np.float32)
        if self.block_shuffle:
            for a, b in S.seq_bounds():
                V[a:b] = V[a:b][_block_perm(S.bucket[a:b], self.seed)]
        return V

    def fit(self, S: Seqs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = S.vol.mean(0), np.maximum(S.vol.std(0), 1e-6)
        V = self._vec(S, S.F)
        self.net = SeqDyn()
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        L, B = 256, 32
        bounds = [(a, b) for a, b in S.seq_bounds() if b - a > L]
        for _ in range(self.epochs):
            chunks = []
            for a, b in bounds:
                off = int(rng.integers(0, L))
                chunks += [V[s:s + L] for s in range(a + off, b - L, L)]
            chunks = np.stack(chunks)
            for bi in np.array_split(rng.permutation(len(chunks)), max(1, len(chunks) // B)):
                x = torch.from_numpy(chunks[bi])
                o, _ = self.net.gru(x)
                logit, cnt = self.net.heads(o[:, :-1], x[:, :-1])
                loss = nn.functional.binary_cross_entropy_with_logits(logit, x[:, 1:, :NT]) + ((cnt - x[:, 1:, NT:]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def features(self, S: Seqs, flags: np.ndarray):
        """(N, 3*NT + h): rolled-out P(flag) mean / max / any over K steps per technique, plus the latent state."""
        self.net.eval()
        V = self._vec(S, flags)
        summ, lat = np.zeros((S.n, 3 * NT), np.float32), np.zeros((S.n, self.net.h), np.float32)
        for a, b in S.seq_bounds():
            x = torch.from_numpy(V[a:b])
            o, _ = self.net.gru(x[None])
            H = o[0]
            lat[a:b] = H.numpy()
            cur = x
            probs = []
            for _ in range(K):
                logit, cnt = self.net.heads(H, cur)
                p = torch.sigmoid(logit)
                probs.append(p)
                cur = torch.cat([p, cnt], 1)
                H = self.net.gru(cur[:, None, :], H[None].contiguous())[0][:, 0]
            P = torch.stack(probs, 1)  # (T, K, NT)
            summ[a:b, :NT] = P.mean(1).numpy()
            summ[a:b, NT:2 * NT] = P.max(1).values.numpy()
            summ[a:b, 2 * NT:] = (1 - (1 - P).prod(1)).numpy()
        return summ, lat


WORLD_NAMES = ([f"sim mean|{TECH_NAME[t]}" for t in TECHS] + [f"sim max|{TECH_NAME[t]}" for t in TECHS]
               + [f"sim any|{TECH_NAME[t]}" for t in TECHS] + [f"latent{i}|state" for i in range(48)])
STATE_NAMES = [f for f in FEATURES]


def clock_col(S: Seqs) -> np.ndarray:
    return (S.bucket % 60).astype(np.float32)[:, None]


def fit_forecasters(S: Seqs, fut, valid, stride=2, seed=0, world: World | None = None, shuffle=False,
                    parts=("renewal", "state", "own", "sched", "clock", "world")):
    """Fit every ladder rung for every technique on training sequences (true flags as history)."""
    pop_rows = np.flatnonzero(valid)[::stride]
    sch = sched_feats(S, S.F, shuffle=shuffle, seed=seed)
    models = {p: {} for p in parts}
    wf = world.features(S, S.F) if ("world" in parts and world is not None) else None
    for j in range(NT):
        rows = pop_rows[~S.F[pop_rows, j]]
        y = fut[rows, j].astype(int)
        own = [j, NT + j, 2 * NT + j]
        if "renewal" in parts:
            models["renewal"][j] = Hazard().fit(sch[rows, j], y)
        if "state" in parts:
            models["state"][j] = _fit(S.X[rows], y, seed)
        if "own" in parts:
            models["own"][j] = _fit(np.c_[sch[rows][:, own], S.X[rows]], y, seed)
        if "sched" in parts:
            models["sched"][j] = _fit(np.c_[sch[rows], S.X[rows]], y, seed)
        if "clock" in parts:
            models["clock"][j] = _fit(np.c_[sch[rows], S.X[rows], clock_col(S)[rows]], y, seed)
        if wf is not None:
            models["world"][j] = _fit(np.c_[sch[rows], wf[0][rows], wf[1][rows]], y, seed)
    return models


def predict_forecasters(models, S: Seqs, flags, shuffle=False, seed=0, world: World | None = None):
    """(N, NT) probabilities per ladder rung. `flags` are the history the forecaster sees."""
    sch = sched_feats(S, flags, shuffle=shuffle, seed=seed)
    wf = world.features(S, flags) if ("world" in models and models["world"] and world is not None) else None
    out = {p: np.zeros((S.n, NT)) for p in models if models[p]}
    for j in range(NT):
        own = [j, NT + j, 2 * NT + j]
        if "renewal" in out:
            out["renewal"][:, j] = models["renewal"][j].predict(sch[:, j])
        if "state" in out:
            out["state"][:, j] = _proba(models["state"][j], S.X)
        if "own" in out:
            out["own"][:, j] = _proba(models["own"][j], np.c_[sch[:, own], S.X])
        if "sched" in out:
            out["sched"][:, j] = _proba(models["sched"][j], np.c_[sch, S.X])
        if "clock" in out:
            out["clock"][:, j] = _proba(models["clock"][j], np.c_[sch, S.X, clock_col(S)])
        if "world" in out and wf is not None:
            out["world"][:, j] = _proba(models["world"][j], np.c_[sch, wf[0], wf[1]])
    return out
