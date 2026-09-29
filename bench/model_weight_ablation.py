"""
Sweep infer.MODEL_WEIGHT on the DAPT 2020 cross-dataset corpus.

infer.MODEL_WEIGHT decides how much of each prediction comes from the learned
model versus the 14 hand-written rules in signals.py. It was set by hand and
never measured. This answers the question the problem statement forces:
can the world model carry the decision on its own?

  weight = 0.0   rules only, no model
  weight = 0.30  the shipped default
  weight = 1.0   model only, no rules

Reports macro-F1, binary attack F1, recall, FPR and SEDI per weight so the
choice can be defended either way. Writes models/model_weight_ablation.json.

Usage:
    python3 bench/model_weight_ablation.py --data /storage/netrikan-base-w30 \
        --model models/base_w30.pt
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dapt                                    # noqa: E402
import signals as sig                          # noqa: E402
from model import WorldModel                   # noqa: E402
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features  # noqa: E402
from eval_dapt import load_feature_list        # noqa: E402
from attck_map import N_STAGES, MODEL_TO_CHAIN, BENIGN  # noqa: E402
from sedi import sedi                          # noqa: E402

WEIGHTS = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0]


def build_windows_with_ports(df, features, window):
    """Same windowing as eval_dapt.build_windows, but also returns the per-window
    destination ports. signals.detect() needs them -- most of the 14 rules are
    port-keyed, so dropping ports would silence the rule channel and make the
    sweep measure nothing."""
    Xs, ys, Ps = [], [], []
    has_port = 'Dst Port' in df.columns
    for _, grp in df.groupby('capture', sort=False):
        g = grp.sort_values('ts', kind='mergesort')
        v = g[features].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        p = (pd.to_numeric(g['Dst Port'], errors='coerce').fillna(0)
             .to_numpy(np.int64) if has_port else np.zeros(len(v), np.int64))
        if len(v) <= window:
            continue
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])
            Ps.append(p[i:i + window])
    return (np.asarray(Xs, np.float32), np.asarray(ys),
            np.asarray(Ps, np.int64))


def fuse(model_probs5, signal_score, signal_hint, weight):
    """Mirrors infer._fuse() exactly, with the weight as a free parameter."""
    p = np.zeros(N_STAGES)
    for mi, ci in MODEL_TO_CHAIN.items():
        p[ci] += float(model_probs5[mi])

    ev = np.full(N_STAGES, 0.02)
    if signal_score > 0:
        ev[signal_hint] += signal_score
    else:
        ev[BENIGN] += 0.9
    ev = ev / ev.sum()

    fused = weight * p + (1.0 - weight) * ev
    return fused / fused.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'processed'))
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'base_w30.pt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models',
                                                 'model_weight_ablation.json'))
    args = ap.parse_args()

    features = load_feature_list(args.data)
    needs_ports = any(f in features for f in PORT_FEATURES)

    Xh = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    window, n_feat = Xh.shape[1], Xh.shape[2]
    del Xh
    assert n_feat == len(features), f"feature mismatch: {n_feat} vs {len(features)}"

    df = dapt.load_all()
    if needs_ports:
        df = add_port_features(df)

    X_raw, y_true, ports = build_windows_with_ports(df, features, window)
    print(f"windows={len(X_raw):,}  window={window}  features={n_feat}", flush=True)

    with open(os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    base_idx = [features.index(f) for f in BASE_FEATURES]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)

    model = WorldModel(input_size=n_feat)
    model.load_state_dict(torch.load(args.model, map_location='cpu',
                                     weights_only=True), strict=False)
    model.eval()

    logits = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out = model(torch.from_numpy(X[i:i + 8192]))
            logits.append(out[0].numpy())
    model_probs5 = torch.softmax(torch.from_numpy(np.concatenate(logits)),
                                 dim=1).numpy()

    # the rule channel does not depend on the weight, so detect once and reuse
    print("running rule detectors…", flush=True)
    scores = np.zeros(len(X_raw))
    hints = np.zeros(len(X_raw), dtype=np.int64)
    for i in range(len(X_raw)):
        s = sig.detect(X_raw[i], ports[i], None)
        scores[i], hints[i] = s['score'], s['stage_hint']
    print(f"rules fired on {(scores > 0).mean():.1%} of windows", flush=True)

    true_att = (y_true > 0).astype(int)
    results = []
    for w in WEIGHTS:
        preds = np.array([np.argmax(fuse(model_probs5[i], scores[i], hints[i], w))
                          for i in range(len(X))])
        pred_att = (preds > 0).astype(int)
        tp = int(((pred_att == 1) & (true_att == 1)).sum())
        fp = int(((pred_att == 1) & (true_att == 0)).sum())
        fn = int(((pred_att == 0) & (true_att == 1)).sum())
        tn = int(((pred_att == 0) & (true_att == 0)).sum())
        H = tp / max(tp + fn, 1)
        F = fp / max(fp + tn, 1)
        row = {
            'weight': w,
            'macro_f1': float(f1_score(y_true, preds, average='macro',
                                       zero_division=0)),
            'binary_f1': float(f1_score(true_att, pred_att, zero_division=0)),
            'recall': round(H, 4),
            'fpr': round(F, 4),
            'precision': round(tp / max(tp + fp, 1), 4),
            'sedi': round(sedi(H, F), 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        results.append(row)
        print(f"weight={w:.2f}  macro_f1={row['macro_f1']:.4f}  "
              f"binary_f1={row['binary_f1']:.4f}  H={H:.4f}  F={F:.4f}  "
              f"SEDI={row['sedi']:+.4f}", flush=True)

    best_f1 = max(results, key=lambda r: r['macro_f1'])
    best_sedi = max(results, key=lambda r: r['sedi'])
    payload = {
        'model': args.model,
        'data': args.data,
        'n_windows': int(len(X)),
        'shipped_weight': 0.30,
        'rule_fire_rate': float((scores > 0).mean()),
        'best_by_macro_f1': best_f1['weight'],
        'best_by_sedi': best_sedi['weight'],
        'sweep': results,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nbest macro-F1 at weight={best_f1['weight']}  "
          f"best SEDI at weight={best_sedi['weight']}")
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
