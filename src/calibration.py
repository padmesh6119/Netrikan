"""
Temperature scaling, ECE, and a reliability diagram.

The rollout in model.rollout() is free-running: each step's prediction feeds the
next. An overconfident model compounds that overconfidence, so by the last step
the trajectory is confidently wrong. Temperature scaling fits one scalar T on a
held-out split and divides the logits by it before the softmax, which leaves the
argmax untouched and only rescales confidence.

Fitting on a split that is later used for evaluation would leak, so the
calibration split is carved out of the validation indices and the remainder is
what ECE-after is measured on.

  fit_temperature(logits, labels)  -> T minimising NLL
  compute_ece(probs, labels)       -> expected calibration error, equal-mass bins
  reliability(probs, labels)       -> (confidence, accuracy, weight) per bin

Usage:
    python3 src/calibration.py --data /storage/netrikan-base-w30 \\
        --model models/base_w30.pt
    # writes models/temperature.json, which infer.py loads automatically
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WorldModel                   # noqa: E402
from train_v2 import blocked_split, STAGES     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar T minimising cross-entropy of softmax(logits / T)."""
    lt = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    ce = torch.nn.CrossEntropyLoss()

    def nll(t):
        return float(ce(lt / float(t), yt))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return float(res.x)


def reliability(probs, labels, n_bins=10, equal_mass=True):
    """Per-bin mean confidence, accuracy and share of samples.

    Equal-mass bins by default: equal-width bins leave the high-confidence bins
    nearly empty on a confident model, which makes ECE read artificially low.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    order = np.argsort(conf)
    if equal_mass:
        chunks = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        chunks = [np.flatnonzero((conf >= edges[b]) & (conf < edges[b + 1]))
                  for b in range(n_bins)]
    out = []
    for c in chunks:
        if len(c) == 0:
            continue
        out.append((float(conf[c].mean()), float(correct[c].mean()),
                    len(c) / len(conf)))
    return out


def compute_ece(probs, labels, n_bins=10, equal_mass=True) -> float:
    return float(sum(w * abs(c - a)
                     for c, a, w in reliability(probs, labels, n_bins, equal_mass)))


def save_reliability_plot(before, after, path):
    """Reliability diagram. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], '--', color='#999', lw=1, label='perfect calibration')
    for rows, label, colour in ((before, 'before scaling', '#ef4444'),
                                (after, 'after scaling', '#2563eb')):
        ax.plot([c for c, _, _ in rows], [a for _, a, _ in rows],
                'o-', color=colour, label=label, ms=4)
    ax.set_xlabel('mean predicted confidence')
    ax.set_ylabel('observed accuracy')
    ax.set_title('Reliability diagram')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'base_w30.pt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'temperature.json'))
    ap.add_argument('--plot', default=os.path.join(ROOT, 'models', 'reliability.png'))
    ap.add_argument('--cal-frac', type=float, default=0.5,
                    help='share of the val split used to FIT T; the rest scores it')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))
    n = min(n_all, len(y))

    _, va_idx = blocked_split(y[:n], purge=W - 1)
    va_idx = va_idx[va_idx < n]

    # split validation into fit / score halves so ECE-after is not measured on
    # the same rows T was fitted on
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(va_idx))
    cut = int(len(va_idx) * args.cal_frac)
    fit_idx, score_idx = va_idx[perm[:cut]], va_idx[perm[cut:]]
    print(f"fit on {len(fit_idx):,} windows, score on {len(score_idx):,}", flush=True)

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    def logits_for(idx):
        out = np.zeros((len(idx), len(STAGES)), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(idx), args.batch):
                sl = idx[i:i + args.batch]
                xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32))
                out[i:i + len(sl)] = model(xb)[0].numpy()
        return out

    lf, ls = logits_for(fit_idx), logits_for(score_idx)
    yf, ys = y[fit_idx], y[score_idx]

    T = fit_temperature(lf, yf)

    def softmax(z, t=1.0):
        z = z / t
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    p_before = softmax(ls, 1.0)
    p_after = softmax(ls, T)
    ece_before = compute_ece(p_before, ys, args.bins)
    ece_after = compute_ece(p_after, ys, args.bins)
    rel_before = reliability(p_before, ys, args.bins)
    rel_after = reliability(p_after, ys, args.bins)

    acc_before = float((p_before.argmax(1) == ys).mean())
    acc_after = float((p_after.argmax(1) == ys).mean())

    print(f"\nT = {T:.4f}")
    print(f"ECE  before {ece_before:.4f}   after {ece_after:.4f}   "
          f"({(1 - ece_after / max(ece_before, 1e-12)):.1%} reduction)")
    print(f"accuracy unchanged by construction: {acc_before:.4f} -> {acc_after:.4f}")
    print(f"mean confidence {p_before.max(1).mean():.4f} -> {p_after.max(1).mean():.4f}")

    plotted = save_reliability_plot(rel_before, rel_after, args.plot)
    payload = {
        'T': T,
        'model': args.model,
        'data': args.data,
        'n_fit': int(len(fit_idx)),
        'n_score': int(len(score_idx)),
        'bins': args.bins,
        'binning': 'equal_mass',
        'ece_before': round(ece_before, 4),
        'ece_after': round(ece_after, 4),
        'accuracy_before': round(acc_before, 4),
        'accuracy_after': round(acc_after, 4),
        'mean_confidence_before': round(float(p_before.max(1).mean()), 4),
        'mean_confidence_after': round(float(p_after.max(1).mean()), 4),
        'reliability_before': [{'confidence': round(c, 4), 'accuracy': round(a, 4),
                                'weight': round(w, 4)} for c, a, w in rel_before],
        'reliability_after': [{'confidence': round(c, 4), 'accuracy': round(a, 4),
                               'weight': round(w, 4)} for c, a, w in rel_after],
        'plot': args.plot if plotted else None,
        'note': ('T is fitted on one half of the blocked-split validation set and '
                 'ECE is measured on the other half, so ece_after is not fitted '
                 'on its own scoring data. Temperature scaling cannot change the '
                 'argmax, so accuracy is identical before and after -- only '
                 'confidence is rescaled.'),
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {args.out}" + (f" and {args.plot}" if plotted else
                                    "  (matplotlib absent, no plot)"))


def _check():
    """Self-test on synthetic logits: a deliberately overconfident model must be
    corrected towards T > 1, and ECE must fall."""
    rng = np.random.default_rng(0)
    n, c = 20000, 5
    labels = rng.integers(0, c, n)
    logits = rng.normal(0, 1, (n, c))
    logits[np.arange(n), labels] += 1.2
    logits *= 4.0                       # inflate -> overconfident
    T = fit_temperature(logits, labels)

    def sm(z, t):
        z = z / t
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    before = compute_ece(sm(logits, 1.0), labels)
    after = compute_ece(sm(logits, T), labels)
    assert T > 1.5, f"overconfident logits should need T>1.5, got {T:.3f}"
    assert after < before, f"ECE did not improve: {before:.4f} -> {after:.4f}"
    assert (sm(logits, 1.0).argmax(1) == sm(logits, T).argmax(1)).all(), \
        "temperature must not change the argmax"
    print(f"self-check OK: T={T:.3f}  ECE {before:.4f} -> {after:.4f}  "
          f"argmax unchanged")


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        _check()
    else:
        main()
