"""
Reconstruct file_id.npy for a dataset built before pipeline_v2 saved it.

Replays the exact deterministic windowing pipeline_v2 uses (same load_all, same
per-file `starts`) and writes file_id.npy = the capture-file id of each window's
label position, aligned to X.npy/y.npy. Reads the source CSVs only; never touches
X.npy, so it is safe to run while a training job has X.npy memory-mapped.

Verifies the reconstruction against y.npy (same length, same label sequence)
before writing, and refuses if they disagree.

    python3 bench/_make_file_id.py --data /tmp/netrikan-cic-full-w30
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from pipeline_v2 import load_all, BASE_FEATURES, PORT_FEATURES, ONSET_HORIZONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    args = ap.parse_args()

    feats_path = os.path.join(args.data, 'features.txt')
    features = ([l.strip() for l in open(feats_path) if l.strip()]
                if os.path.exists(feats_path) else list(BASE_FEATURES))
    with_ports = any(f in features for f in PORT_FEATURES)

    y_ref = np.load(os.path.join(args.data, 'y.npy'))
    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    W = X.shape[1]
    del X

    df = load_all(features, with_ports)
    labels = df['stage'].to_numpy(np.int64)
    file_id = df['_file'].to_numpy(np.int64)
    del df

    # identical starts logic to pipeline_v2.main()
    starts, b, N = [], 0, len(labels)
    while b < N:
        e = b
        while e < N and file_id[e] == file_id[b]:
            e += 1
        starts.append(np.arange(b, max(b, e - W)))
        b = e
    starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)

    y_rebuilt = labels[starts + W]
    if len(y_rebuilt) != len(y_ref) or not np.array_equal(y_rebuilt, y_ref):
        raise SystemExit(f"reconstruction does not match y.npy "
                         f"(rebuilt {len(y_rebuilt)}, ref {len(y_ref)}) — refusing "
                         f"to write a misaligned file_id.npy")

    file_at = file_id[starts + W]
    out = os.path.join(args.data, 'file_id.npy')
    np.save(out, file_at)
    counts = {int(f): int((file_at == f).sum()) for f in np.unique(file_at)}
    print(f"verified against y.npy ({len(y_ref):,} windows). "
          f"file_id.npy written: {len(counts)} capture files, per-file window "
          f"counts {counts}")
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
