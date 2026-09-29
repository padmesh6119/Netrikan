"""
Transition-only evaluation: score the model where the stage actually changes.

All-window macro-F1 on this data is dominated by persistence. Roughly 91% of
windows carry the same label as the previous one, so "repeat the last label"
scores ~0.906 with no model at all. That number measures label autocorrelation,
not forecasting skill.

The honest arena for a forecaster is the subset of windows where the label
changes. A persistence oracle names the wrong stage on every one of them by
construction, so its multi-class macro-F1 there is exactly 0.000.

Note that its *binary* attack-vs-benign SEDI on that subset is not necessarily
poor: a transition from Initial Access to DoS is attack-to-attack, so persistence
still gets the binary call right while getting the stage wrong. The macro-F1 row
is the one that carries the argument; the SEDI rows are reported for completeness
and should not be quoted as if persistence collapsed on both.

Reported here, on identical held-out indices for both:

  all windows        macro-F1, model vs persistence  (the metric that flatters persistence)
  transition windows macro-F1, model vs persistence  (the metric that matters)
  Brier score        model vs persistence, plus Brier Skill Score
  SEDI               binary attack/benign, both subsets

The caveat, stated because it is the first thing a careful reader asks:
conditioning on transitions measures discrimination *given* that something
changed. It does not measure how often the model cries transition when nothing
happened. That is what the all-window rows and the Brier scores are for. Read
the table, not one cell of it.

Usage:
    python3 src/transition_eval.py --data /storage/netrikan-base-w30 \\
        --model models/base_w30.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bench'))

from model import WorldModel                   # noqa: E402
from train_v2 import blocked_split, STAGES     # noqa: E402
from sedi import sedi                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brier(probs, labels, n_classes):
    """Multi-class Brier score: mean squared error against the one-hot truth.
    Lower is better. A proper scoring rule, unlike accuracy or F1."""
    onehot = np.zeros((len(labels), n_classes), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def binary_rates(y_true, y_pred):
    """Hit rate and false alarm rate for attack-vs-benign."""
    t = (y_true > 0).astype(int)
    p = (y_pred > 0).astype(int)
    tp = int(((p == 1) & (t == 1)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    H = tp / max(tp + fn, 1)
    F = fp / max(fp + tn, 1)
    return H, F, dict(tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'base_w30.pt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models',
                                                 'transition_eval.json'))
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    xpath = os.path.join(args.data, 'X.npy')
    X = np.load(xpath, mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))
    n = min(n_all, len(y))
    n_classes = len(STAGES)

    # the same split the model was trained under, so this is genuinely held out
    _, va_idx = blocked_split(y[:n], purge=W - 1)
    va_idx = va_idx[va_idx < n]
    va_idx = va_idx[va_idx > 0]          # need y[t-1] to define a transition
    print(f"val windows: {len(va_idx):,}  (window={W}, features={F})", flush=True)

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    probs = np.zeros((len(va_idx), n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(va_idx), args.batch):
            sl = va_idx[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32))
            probs[i:i + len(sl)] = torch.softmax(model(xb)[0], dim=1).numpy()
    preds = probs.argmax(1)

    y_true = y[va_idx]
    y_prev = y[va_idx - 1]               # the persistence prediction
    is_trans = y_true != y_prev

    print(f"transition windows: {is_trans.sum():,} of {len(va_idx):,} "
          f"({is_trans.mean():.2%})", flush=True)
    if is_trans.sum() == 0:
        raise SystemExit("no transition windows in the validation split")

    # persistence as a probability forecast: all mass on the previous label
    persist_probs = np.zeros_like(probs)
    persist_probs[np.arange(len(y_prev)), y_prev] = 1.0

    rows = {}
    for name, mask in (('all_windows', np.ones(len(va_idx), bool)),
                       ('transition_windows', is_trans),
                       ('steady_windows', ~is_trans)):
        if mask.sum() == 0:
            continue
        mf1 = float(f1_score(y_true[mask], preds[mask], average='macro',
                             zero_division=0))
        pf1 = float(f1_score(y_true[mask], y_prev[mask], average='macro',
                             zero_division=0))
        bm = brier(probs[mask], y_true[mask], n_classes)
        bp = brier(persist_probs[mask], y_true[mask], n_classes)
        Hm, Fm, cm_m = binary_rates(y_true[mask], preds[mask])
        Hp, Fp, cm_p = binary_rates(y_true[mask], y_prev[mask])
        rows[name] = {
            'n': int(mask.sum()),
            'model_macro_f1': round(mf1, 4),
            'persistence_macro_f1': round(pf1, 4),
            'gap': round(mf1 - pf1, 4),
            'model_brier': round(bm, 4),
            'persistence_brier': round(bp, 4),
            # BSS > 0 means better-calibrated probabilities than persistence,
            # even where discrete macro-F1 does not flip
            'brier_skill_score': round(1.0 - bm / bp, 4) if bp > 0 else None,
            'model_sedi': round(sedi(Hm, Fm), 4),
            'persistence_sedi': round(sedi(Hp, Fp), 4),
            'model_binary': {'hit_rate': round(Hm, 4), 'fpr': round(Fm, 4), **cm_m},
            'persistence_binary': {'hit_rate': round(Hp, 4), 'fpr': round(Fp, 4),
                                   **cm_p},
        }
        print(f"\n{name}  (n={mask.sum():,})")
        print(f"  macro-F1     model {mf1:.4f}   persistence {pf1:.4f}   "
              f"gap {mf1 - pf1:+.4f}")
        print(f"  Brier        model {bm:.4f}   persistence {bp:.4f}   "
              f"BSS {rows[name]['brier_skill_score']}")
        print(f"  SEDI         model {sedi(Hm, Fm):+.4f}  persistence "
              f"{sedi(Hp, Fp):+.4f}")

    print("\nper-class F1 on transition windows only:")
    print(classification_report(y_true[is_trans], preds[is_trans],
                                labels=list(range(n_classes)),
                                target_names=STAGES, digits=3,
                                zero_division=0), flush=True)

    payload = {
        'model': args.model,
        'data': args.data,
        'window': int(W),
        'n_val': int(len(va_idx)),
        'transition_rate': round(float(is_trans.mean()), 4),
        'note': ('Persistence names the wrong stage on every transition window '
                 'by construction, so its transition macro-F1 is exactly 0. Its '
                 'binary attack/benign SEDI on that subset can still be high, '
                 'because an attack-to-attack transition keeps the binary call '
                 'correct -- do not quote the SEDI row as if persistence '
                 'collapsed. Conditioning on transitions measures discrimination '
                 'given that a change occurred, not how often the model claims a '
                 'change that did not happen; read it alongside the all_windows '
                 'and steady_windows rows.'),
        'subsets': rows,
        'per_class_transition': classification_report(
            y_true[is_trans], preds[is_trans], labels=list(range(n_classes)),
            target_names=STAGES, output_dict=True, zero_division=0),
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
