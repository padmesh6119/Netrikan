"""
CTU-13 dataset builder.

CTU-13 is named in the problem statement and both CSVs are committed to
`data/`, but nothing in the repo read them. This builds the same on-disk layout
pipeline_v2.py produces (X.npy, y.npy, onset_k*.npy, scaler.pkl, features.txt),
so train_v2.py, transition_eval.py, calibration.py and baseline.py all work
against it unchanged.

What CTU-13 gives, and what it does not:

  * 23 of the 24 BASE_FEATURES directly. 'PSH Flag Cnt' is absent; the CSVs carry
    'Bwd PSH Flags' instead, which is used in its place and recorded in
    `feature_notes.json` so the substitution is not invisible.
  * Binary labels only: botnet vs normal. Botnet maps to class 4 (Botnet/C2) in
    pipeline_v2.LABEL_MAP terms, so the 5-class head trains with classes 1-3
    empty. This is a 2-class problem wearing a 5-class head.
  * No Timestamp, no Dst Port, no Src/Dst IP. Row order within each file is the
    only ordering signal, and there is no identity, so per-host windowing is
    impossible here.
  * The two files are separate captures. Windows are built inside each file and
    never span the seam between them, so no window mixes normal and botnet flows
    from unrelated captures.

The consequence for the onset head, stated plainly: concatenating one benign
capture and one botnet capture yields exactly ONE label transition in 92k flows.
Onset labels are written for pipeline compatibility but they are degenerate, and
train_v2 will fall back to macro-F1 selection because the onset AUC is undefined.
Training the onset head needs CIC-IDS-2018 or DAPT 2020, where attack phases
alternate.

Usage:
    python3 src/ctu13.py --out /tmp/netrikan-ctu13-w30 --window 30
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model import ONSET_HORIZONS
from pipeline_v2 import BASE_FEATURES, make_onset_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

# CTU-13 lacks 'PSH Flag Cnt'. 'Bwd PSH Flags' is the closest available column;
# it counts backward-direction PSH flags rather than both directions, so it is a
# partial substitute, not an equivalent.
SUBSTITUTIONS = {'PSH Flag Cnt': 'Bwd PSH Flags'}

# CTU-13 is botnet traffic. Class 4 is Botnet/C2 in pipeline_v2.LABEL_MAP.
BOTNET_CLASS = 4

FILES = [
    ('CTU13_Normal_Traffic.csv', 0),
    ('CTU13_Attack_Traffic.csv', BOTNET_CLASS),
]


def load_file(path, stage):
    df = pd.read_csv(path, low_memory=False)
    cols = {}
    notes = {}
    for f in BASE_FEATURES:
        if f in df.columns:
            cols[f] = df[f]
        elif f in SUBSTITUTIONS and SUBSTITUTIONS[f] in df.columns:
            cols[f] = df[SUBSTITUTIONS[f]]
            notes[f] = f"substituted from '{SUBSTITUTIONS[f]}'"
        else:
            cols[f] = pd.Series(0.0, index=df.index)
            notes[f] = "absent in CTU-13, zero-filled"
    out = pd.DataFrame(cols)
    out = out.apply(pd.to_numeric, errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out['stage'] = stage
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--data-dir', default=DATA_DIR)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    W = args.window

    parts, all_notes = [], {}
    for name, stage in FILES:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        d, notes = load_file(path, stage)
        all_notes.update(notes)
        print(f"  {name}: {len(d):,} flows -> class {stage}", flush=True)
        parts.append(d)

    # Scaler is fit across both captures, matching pipeline_v2's convention of
    # fitting on all rows before windowing.
    allrows = pd.concat([p[BASE_FEATURES] for p in parts], ignore_index=True)
    scaler = StandardScaler().fit(allrows)
    del allrows

    # Windows are built inside each capture. Building across the concatenation
    # would produce W-1 windows that mix normal and botnet flows from unrelated
    # captures and label them by whichever flow happened to land at t+W.
    Xs, ys = [], []
    for (name, stage), d in zip(FILES, parts):
        v = scaler.transform(d[BASE_FEATURES]).astype(np.float32)
        s = d['stage'].to_numpy(np.int64)
        if len(v) <= W:
            print(f"  {name}: shorter than window, skipped", flush=True)
            continue
        for i in range(len(v) - W):
            Xs.append(v[i:i + W])
            ys.append(s[i + W])
        print(f"  {name}: {len(v) - W:,} windows", flush=True)
    del parts

    X = np.asarray(Xs, np.float32)
    y = np.asarray(ys, np.int64)
    del Xs, ys
    print(f"total {len(X):,} windows x {W} x {len(BASE_FEATURES)}", flush=True)
    print(f"class counts: {dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))}",
          flush=True)

    np.save(os.path.join(args.out, 'X.npy'), X)
    np.save(os.path.join(args.out, 'y.npy'), y)
    with open(os.path.join(args.out, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out, 'features.txt'), 'w') as f:
        f.write("\n".join(BASE_FEATURES))

    onset, n_onset = make_onset_labels(y, ONSET_HORIZONS)
    for k, o in onset.items():
        np.save(os.path.join(args.out, f'onset_k{k}.npy'), o)
    rates = {k: round(float(o.mean()), 6) for k, o in onset.items()}
    n_trans = int((np.diff(y) != 0).sum())
    print(f"onset labels: {n_onset:,} rows, positive rate {rates}", flush=True)
    print(f"label transitions in the whole corpus: {n_trans}", flush=True)
    if n_trans <= 2:
        print("  -> DEGENERATE for onset training, as expected for CTU-13. "
              "train_v2 will select on macro_f1.", flush=True)

    with open(os.path.join(args.out, 'feature_notes.json'), 'w') as f:
        json.dump({
            'source': 'CTU-13',
            'window': W,
            'n_windows': int(len(X)),
            'classes': {'0': 'Benign', str(BOTNET_CLASS): 'Botnet/C2'},
            'binary_only': True,
            'feature_substitutions': all_notes,
            'label_transitions': n_trans,
            'onset_usable': n_trans > 2,
            'caveats': [
                'Binary labels only: classes 1 (InitialAccess), 2 (DoS) and 3 '
                '(Infiltration) have zero examples, so per-class F1 for those is '
                'meaningless and macro-F1 over 5 classes is not comparable to a '
                'CIC-IDS-2018 macro-F1.',
                'No Timestamp, Dst Port or IP columns: the rule layer in '
                'signals.py cannot use ports, and per-host windowing is '
                'impossible.',
                'Two separate captures concatenated, so the corpus contains one '
                'label transition and cannot train the onset head.',
            ],
        }, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
