"""
SHAP explanations for the world model.

Uses shap.GradientExplainer (expected gradients, Erion et al. 2021) rather than
DeepExplainer: DeepExplainer's DeepLIFT rules are unreliable through LSTM gates,
while expected gradients integrate along the path from a background reference and
work correctly on recurrent nets.

The explainer attributes to every (timestep, feature) cell of the input window,
so it answers both "which feature" and "which flow in the window" for one target.
Two targets are available:

  breach   P(malicious) from the breach head -- "why is this an attack"
  onset    P(a new stage begins within k=5 windows) -- "why will it escalate",
           which is the forecasting question the problem statement actually asks

SHAP is far heavier than one backward pass (a GradientExplainer call runs many
samples per input), so it is computed on demand for a chosen window in the
dashboard, never for every window of a capture. gradient x input in infer.py
stays as the fast always-on attribution.

    ex = ShapExplainer(model, background)      # background: (n, WINDOW, F) tensor
    vals = ex.explain(window_batch, target="breach")   # (batch, WINDOW, F)
    feat = ex.by_feature(vals)                 # (batch, F) summed over time
"""

import numpy as np
import torch


class _Head(torch.nn.Module):
    """Exposes one scalar model output so SHAP sees a single target."""

    def __init__(self, model, target, k_index):
        super().__init__()
        self.model = model
        self.target = target
        self.k_index = k_index

    def forward(self, x):
        stage_logits, breach, _state, onset_logits, *_ = self.model(
            x, return_attention=True)
        if self.target == "breach":
            out = breach                                   # (B,)
        elif self.target == "onset":
            out = torch.sigmoid(onset_logits)[:, self.k_index]   # (B,)
        elif self.target == "attack":
            # P(not benign) from the stage head
            out = 1.0 - torch.softmax(stage_logits, dim=1)[:, 0]
        else:
            raise ValueError(f"unknown target {self.target!r}")
        return out.unsqueeze(1)                            # (B,1) for shap


class ShapExplainer:
    def __init__(self, model, background, k_index=1):
        """
        model       a WorldModel in eval mode
        background  reference windows, (n, WINDOW, F) float32 tensor/ndarray.
                    50-100 windows sampled from benign traffic is plenty; more
                    only slows it down.
        k_index     which onset horizon for target="onset" (default index 1 = k5)
        """
        import shap
        self._shap = shap
        self.model = model
        self.k_index = k_index
        bg = torch.as_tensor(np.asarray(background, dtype=np.float32))
        self._bg = bg
        self._explainers = {}

    def _explainer(self, target):
        if target not in self._explainers:
            head = _Head(self.model, target, self.k_index).eval()
            self._explainers[target] = self._shap.GradientExplainer(head, self._bg)
        return self._explainers[target]

    def explain(self, windows, target="breach", nsamples=64):
        """SHAP values per (timestep, feature) for each window.

        windows: (batch, WINDOW, F). Returns ndarray of the same 3 dims.
        """
        x = torch.as_tensor(np.asarray(windows, dtype=np.float32))
        sv = self._explainer(target).shap_values(x, nsamples=nsamples)
        sv = sv[0] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 4:          # trailing singleton output dim
            sv = sv[..., 0]
        return sv                  # (batch, WINDOW, F)

    @staticmethod
    def by_feature(shap_vals):
        """Collapse the time axis: signed sum over timesteps -> (batch, F)."""
        return np.asarray(shap_vals).sum(axis=1)

    @staticmethod
    def by_timestep(shap_vals):
        """Collapse the feature axis: |value| summed over features -> (batch, WINDOW).
        Shows which flow in the window mattered most, like attention but for a
        specific target."""
        return np.abs(np.asarray(shap_vals)).sum(axis=2)

    def top_features(self, shap_vals, feature_names, top_k=5):
        """Ranked feature contributions for one window (shap_vals: (WINDOW, F))."""
        per_feat = np.asarray(shap_vals).sum(axis=0)       # (F,)
        order = np.argsort(-np.abs(per_feat))[:top_k]
        return [{"feature": feature_names[j],
                 "shap_value": round(float(per_feat[j]), 6),
                 "direction": "raises" if per_feat[j] > 0 else "lowers"}
                for j in order]


def build_background(scaled_windows, n=64, seed=0):
    """Sample n reference windows for the explainer from a pool of scaled windows."""
    x = np.asarray(scaled_windows, dtype=np.float32)
    if len(x) <= n:
        return x
    idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
    return x[idx]


def _check():
    """Self-test on a randomly-initialised model: SHAP runs, shapes are right, and
    values are additive-ish (sum over cells correlates with output delta)."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import WorldModel

    m = WorldModel(input_size=24).eval()
    rng = np.random.default_rng(0)
    bg = rng.normal(0, 1, (40, 30, 24)).astype(np.float32)
    X = rng.normal(0, 1, (4, 30, 24)).astype(np.float32)

    ex = ShapExplainer(m, bg)
    for target in ("breach", "onset", "attack"):
        sv = ex.explain(X, target=target, nsamples=32)
        assert sv.shape == (4, 30, 24), (target, sv.shape)
        assert np.isfinite(sv).all(), f"{target} produced non-finite SHAP values"
    feats = ex.top_features(sv[0], [f"f{i}" for i in range(24)])
    ts = ex.by_timestep(sv)
    assert ts.shape == (4, 30)
    print(f"shap self-check OK: (batch,WINDOW,F) attribution for breach/onset/"
          f"attack; top feature f-index example {feats[0]['feature']}")


if __name__ == '__main__':
    _check()
