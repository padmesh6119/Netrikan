#!/usr/bin/env python3
"""
Does host identity in the training windows improve forecasting?

Same data (DAPT 2020), same 24 features, same window, same train_v2 recipe.
Only the window construction differs:

  segment   30 consecutive flows of one capture, any host (what we train on now)
  perhost   30 consecutive flows of one (capture, Src IP)

Evaluation is leave-one-capture-out: train_v2 --lodo-val-file holds out a whole
capture, so no held-out window's host or time range is seen in training. Only
captures with at least 50 held-out onsets at k=5 are scored (train_v2's floor).

Onset labels never cross a group boundary (a different host or capture); those
rows get -1 and train_v2 drops them.

Usage:
    python3 bench/dapt_identity_ab.py --work /path/with/2GB/free
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import dapt                                   # noqa: E402
from model import ONSET_HORIZONS              # noqa: E402
from pipeline_v2 import BASE_FEATURES         # noqa: E402

MODES = {'segment': ['capture'], 'perhost': ['capture', 'Src IP']}


def build(df, keys, window):
    captures = sorted(df['capture'].unique(),
                      key=lambda c: (df.loc[df['capture'] == c, 'day_order'].iloc[0], c))
    cap_id = {c: i for i, c in enumerate(captures)}
    Xs, ys, fids, gids = [], [], [], []
    for gi, (_, g) in enumerate(sorted(df.groupby(keys, sort=False),
                                       key=lambda kv: cap_id[kv[1]['capture'].iloc[0]])):
        g = g.sort_values('ts', kind='mergesort')
        v = g[BASE_FEATURES].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        m = len(v) - window
        if m <= 0:
            continue
        idx = np.arange(m)[:, None] + np.arange(window)
        Xs.append(v[idx])
        ys.append(s[window:window + m])
        fids.append(np.full(m, cap_id[g['capture'].iloc[0]]))
        gids.append(np.full(m, gi))
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.int64)
    fid = np.concatenate(fids).astype(np.int64)
    gid = np.concatenate(gids)
    onset = {}
    for k in ONSET_HORIZONS:
        o = np.full(len(y), -1, np.int8)
        same = np.zeros(len(y), bool)
        same[:-k] = gid[:-k] == gid[k:]
        o[same] = (y[k:][same[:-k]] != y[:-k][same[:-k]])
        onset[k] = o
    return X, y, fid, onset, captures


def write(out, X, y, fid, onset):
    os.makedirs(out, exist_ok=True)
    flat = X.reshape(-1, X.shape[2])
    sc = StandardScaler().fit(flat)
    Xs = ((flat - sc.mean_) / np.where(sc.scale_ > 0, sc.scale_, 1)).astype(np.float32)
    np.save(os.path.join(out, 'X.npy'), Xs.reshape(X.shape))
    np.save(os.path.join(out, 'y.npy'), y)
    np.save(os.path.join(out, 'file_id.npy'), fid)
    for k, o in onset.items():
        np.save(os.path.join(out, f'onset_k{k}.npy'), o)


def scorable(fid, onset, f):
    o = onset[5][fid == f]
    o = o[o >= 0]
    return 50 <= o.sum() < len(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--samples', type=int, default=200000)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'dapt_identity_ab.json'))
    args = ap.parse_args()

    df = dapt.load_all()
    df = df[df['ts'].notna()]
    runs, meta = {}, {}
    for mode, keys in MODES.items():
        X, y, fid, onset, captures = build(df, keys, args.window)
        d = os.path.join(args.work, f'dapt_{mode}_w{args.window}')
        write(d, X, y, fid, onset)
        folds = [f for f in range(len(captures)) if scorable(fid, onset, f)]
        meta[mode] = {'n_windows': int(len(y)), 'class_counts': np.bincount(y, minlength=5).tolist(),
                      'onset_rate_k5': float((onset[5][onset[5] >= 0]).mean())}
        print(f"[{mode}] {len(y):,} windows, scorable folds {folds}", flush=True)
        for f in folds:
            tag = f'dab_{mode}_f{f}'
            mp = os.path.join(ROOT, 'models', f'{tag}_metrics.json')
            if not os.path.exists(mp):
                subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'train_v2.py'),
                                '--data', d, '--tag', tag, '--lodo-val-file', str(f),
                                '--epochs', str(args.epochs), '--samples', str(args.samples),
                                '--select-on', 'onset_auc_k5'], check=True,
                               env={**os.environ, 'NETRIKAN_WORKERS': '0'})
            m = json.load(open(mp))
            runs.setdefault(mode, {})[captures[f]] = {
                'onset_auc': {k: v['auc'] for k, v in m['onset_auc'].items()},
                'macro_f1': m.get('final_macro_f1')}

    common = sorted(set(runs.get('segment', {})) & set(runs.get('perhost', {})))
    summary = {}
    for k in [f'k{h}' for h in ONSET_HORIZONS]:
        seg = np.array([runs['segment'][c]['onset_auc'][k] for c in common])
        hst = np.array([runs['perhost'][c]['onset_auc'][k] for c in common])
        summary[k] = {'segment_mean': round(float(seg.mean()), 4),
                      'perhost_mean': round(float(hst.mean()), 4),
                      'perhost_minus_segment': round(float((hst - seg).mean()), 4),
                      'perhost_wins': int((hst > seg).sum()), 'folds': len(common)}
    out = {'experiment': 'host identity in training windows, DAPT 2020, leave-one-capture-out',
           'window': args.window, 'epochs': args.epochs, 'samples_per_epoch': args.samples,
           'datasets': meta, 'onset_auc_summary': summary, 'per_capture': runs,
           'caveat': ('onset horizons count flows of the group: for perhost, k flows of '
                      'the same host; for segment, k flows of the network. Scaler is fit '
                      'on all windows (unsupervised, no labels).')}
    json.dump(out, open(args.out, 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
