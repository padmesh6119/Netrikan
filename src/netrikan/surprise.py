"""World-model surprise: a GRU trained to predict a host's next-minute state from its history, on benign minutes
only. The prediction error of an arriving minute is an attack-label-free anomaly score."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.out = nn.Linear(h, f)


class Surprise:
    def __init__(self, h=32, chunk=64, epochs=5, max_chunks=4000, seed=0):
        self.h, self.chunk, self.epochs, self.max_chunks, self.seed = h, chunk, epochs, max_chunks, seed

    def fit(self, seqs):
        """seqs: list of (X (T, F) log-space features, attack (T,) bool). Attack minutes are excluded from the loss."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        allx = np.vstack([x[~a] for x, a in seqs if (~a).any()])
        self.mu, self.sd = allx.mean(0), np.maximum(allx.std(0), 1e-6)
        S = [((x - self.mu) / self.sd).astype(np.float32) for x, _ in seqs]
        A = [a for _, a in seqs]
        self.net = Net(allx.shape[1], self.h)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        L = self.chunk
        pool = [i for i, s in enumerate(S) if len(s) > L]
        for _ in range(self.epochs):
            xs, ms = [], []
            for i in rng.permutation(pool):
                for st in range(int(rng.integers(0, L)), len(S[i]) - L, L):
                    xs.append(S[i][st:st + L]); ms.append(~A[i][st:st + L])
                if len(xs) >= self.max_chunks:
                    break
            xs, ms = np.stack(xs[:self.max_chunks]), np.stack(ms[:self.max_chunks])
            for b in np.array_split(rng.permutation(len(xs)), max(1, len(xs) // 64)):
                x, m = torch.from_numpy(xs[b]), torch.from_numpy(ms[b][:, 1:])
                o, _ = self.net.gru(x)
                pred = x[:, :-1] + self.net.out(o[:, :-1])
                err = ((pred - x[:, 1:]) ** 2).mean(-1)
                loss = (err * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-minute prediction error of one sequence, from its own past only."""
        x = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))[None]
        o, _ = self.net.gru(x)
        pred = x[:, :-1] + self.net.out(o[:, :-1])
        err = ((pred - x[:, 1:]) ** 2).mean(-1)[0].numpy()
        return np.r_[float((x[0, 0] ** 2).mean()), err]

    def score_grid(self, Xd: np.ndarray, gid: np.ndarray) -> np.ndarray:
        out = np.zeros(len(Xd))
        for g in np.unique(gid):
            idx = np.flatnonzero(gid == g)
            out[idx] = self.score(Xd[idx])
        return out
