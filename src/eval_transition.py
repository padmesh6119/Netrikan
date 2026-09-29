"""
Transition-window F1, swept across forecast horizons.

The single most important number the project did not have (TRAINER_BACKLOG.md
P0 §1): persistence beats the LSTM on all-window macro-F1 at every horizon
(persistence_baseline.json), but ~91% of windows carry no label change, so
"copy the last label forward" scores ~0.90-0.92 by autocorrelation alone. The
honest arena for a forecaster is the subset of windows where the stage
actually changes -- persistence names the wrong stage on every one of them by
construction (transition macro-F1 exactly 0), so any positive LSTM score there
is a genuine forecasting result, not label copying.

This script requires one checkpoint PER HORIZON k, trained with
`train_v2.py --horizon k --fixed-split` -- the horizon is baked into the
label each checkpoint was trained to predict (target for window i is
y_full[i+k]), so a single checkpoint cannot be evaluated at multiple k. The
`--fixed-split` flag pins the blocked train/val split to the UNSHIFTED labels,
so every horizon's checkpoint is scored on the same held-out row indices --
that is what makes the k values comparable to each other.

Definitions, matching bench/persistence_baseline.py's convention exactly:
  target window i, horizon k:  true  = y_full[i + k]
                                 persistence pred = y_full[i - 1]   (the last
                                 label the operator actually observed, since
                                 y_full[j] is the label of the flow at the end
                                 of window j, i.e. row j + W - 1)
  transition window:  true != persistence pred
  onset window (subset of transition): persistence pred == Benign (0) and
                                        true != Benign  (a benign->attack
                                        transition specifically)

Usage:
    python3 src/eval_transition.py --data /tmp/netrikan-cic-full-w30 \\
        --checkpoints models/htz_k0.pt=0 models/htz_k30.pt=30 \\
                      models/htz_k90.pt=90 models/htz_k180.pt=180 \\
                      models/htz_k360.pt=360
    # or, with the default naming convention models/htz_k<K>.pt:
    python3 src/eval_transition.py --data /tmp/netrikan-cic-full-w30
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WorldModel               # noqa: E402
from train_v2 import blocked_split, STAGES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [0, 30, 90, 180, 360]


def load_model(path, n_features):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=n_features, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def predict(model, X, idx, batch=8192):
    n_classes = len(STAGES)
    probs = np.zeros((len(idx), n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sl = idx[i:i + batch]
            xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32))
            probs[i:i + len(sl)] = torch.softmax(model(xb)[0], dim=1).numpy()
    return probs


def eval_horizon(data_dir, ckpt_path, k, batch=8192):
    X = np.load(os.path.join(data_dir, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y_full = np.load(os.path.join(data_dir, 'y.npy'))
    n = min(n_all, len(y_full)) - k
    if n <= 0:
        raise SystemExit(f"k={k}: horizon exceeds dataset length")

    # fixed split: blocks are drawn from the UNSHIFTED labels, so the same
    # held-out row indices are used at every horizon (matches train_v2.py
    # --fixed-split exactly -- same seed/n_blocks/val_frac/purge defaults)
    _, va_idx = blocked_split(y_full[:n], purge=W - 1, n_blocks=500,
                             val_frac=0.2, seed=42)
    va_idx = va_idx[(va_idx < n) & (va_idx > 0)]   # need i-1 for persistence

    y_true = y_full[va_idx + k]
    y_persist = y_full[va_idx - 1]
    is_trans = y_true != y_persist
    is_onset = is_trans & (y_persist == 0) & (y_true != 0)
    n_transitions = int(is_trans.sum())
    print(f"k={k}: {len(va_idx):,} val windows, {n_transitions:,} transitions "
          f"({is_trans.mean():.2%}), {int(is_onset.sum()):,} onsets", flush=True)
    if n_transitions == 0:
        raise SystemExit(f"k={k}: no transition windows in the fixed val split")

    model = load_model(ckpt_path, F)
    probs = predict(model, X, va_idx, batch=batch)
    preds = probs.argmax(1)

    lstm_trans_f1 = float(f1_score(y_true[is_trans], preds[is_trans],
                                   average='macro', zero_division=0))
    # persistence's prediction on the transition subset is, by definition of
    # is_trans, never equal to the truth -- its macro-F1 there is exactly 0.
    persist_trans_f1 = float(f1_score(y_true[is_trans], y_persist[is_trans],
                                      average='macro', zero_division=0))

    n_onset = int(is_onset.sum())
    if n_onset > 0:
        lstm_onset_recall = float((preds[is_onset] != 0).mean())
        # persistence never predicts a change, so it never calls an onset:
        # recall is 0 by construction, reported rather than assumed.
        persistence_onset_recall = float((y_persist[is_onset] != 0).mean())
    else:
        lstm_onset_recall = None
        persistence_onset_recall = None

    return {
        'k': k,
        'checkpoint': ckpt_path,
        'n_val': int(len(va_idx)),
        'n_transitions': n_transitions,
        'transition_rate': round(float(is_trans.mean()), 4),
        'n_onsets': n_onset,
        'lstm_transition_f1': round(lstm_trans_f1, 4),
        'persistence_transition_f1': round(persist_trans_f1, 4),
        'gap': round(lstm_trans_f1 - persist_trans_f1, 4),
        'lstm_onset_recall': (round(lstm_onset_recall, 4)
                              if lstm_onset_recall is not None else None),
        'persistence_onset_recall': (round(persistence_onset_recall, 4)
                                     if persistence_onset_recall is not None
                                     else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/tmp/netrikan-cic-full-w30')
    ap.add_argument('--checkpoints', nargs='*', default=None,
                    help='path=k pairs, e.g. models/htz_k30.pt=30. Defaults to '
                         'models/htz_k<K>.pt for K in ' + str(HORIZONS))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models',
                                                 'transition_f1.json'))
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    if args.checkpoints:
        pairs = []
        for spec in args.checkpoints:
            path, k = spec.rsplit('=', 1)
            pairs.append((path, int(k)))
    else:
        pairs = [(os.path.join(ROOT, 'models', f'htz_k{k}.pt'), k)
                 for k in HORIZONS]

    missing = [p for p, _ in pairs if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing checkpoints (train them first with "
                         f"train_v2.py --horizon k --fixed-split): {missing}")

    rows = [eval_horizon(args.data, path, k, batch=args.batch)
            for path, k in sorted(pairs, key=lambda pk: pk[1])]

    lstm_wins_transition = all(r['gap'] > 0 for r in rows)
    n_wins = sum(1 for r in rows if r['gap'] > 0)
    payload = {
        'lstm_wins_transition': lstm_wins_transition,
        'n_horizons_lstm_wins': n_wins,
        'n_horizons_total': len(rows),
        'verdict': (
            'LSTM beats persistence on transition-only macro-F1 at every '
            f'horizon tested ({n_wins}/{len(rows)}) -- the forecasting claim '
            'stands.' if lstm_wins_transition else
            'LSTM does NOT beat persistence at every horizon '
            f'({n_wins}/{len(rows)} won) -- do not claim unconditional '
            'forecasting superiority; report per-horizon results.'
        ),
        'note': ('Persistence names the wrong stage on every transition '
                'window by construction, so persistence_transition_f1 is '
                'exactly 0 at every horizon -- that is expected, not a bug. '
                'The gap column is what carries the argument. Onset recall '
                'is reported on the benign->attack subset of transitions '
                'only; persistence_onset_recall is 0 by construction for '
                'the same reason.'),
        'horizons': rows,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n{payload['verdict']}\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
