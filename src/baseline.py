import os
import json
import time
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from train_v2 import blocked_split, STAGES

MODEL_DIR = os.path.expanduser("~/netrikan/models")


def subsample(idx, y, per_class, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(STAGES)):
        c_idx = idx[y[idx] == c]
        if len(c_idx) == 0:
            continue
        take = min(per_class, len(c_idx))
        keep.append(rng.choice(c_idx, take, replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def gather(X, idx, mode, chunk=100_000):
    """mode 'last' = final flow only (no temporal context); 'flat' = whole window."""
    parts = []
    for i in range(0, len(idx), chunk):
        sl = np.sort(idx[i:i + chunk])
        blk = np.asarray(X[sl], dtype=np.float32)
        parts.append(blk[:, -1, :] if mode == 'last' else blk.reshape(len(sl), -1))
    return np.concatenate(parts)


def evaluate(clf, X, va_idx, y, mode, chunk=100_000):
    P = []
    order = np.sort(va_idx)
    for i in range(0, len(order), chunk):
        sl = order[i:i + chunk]
        blk = np.asarray(X[sl], dtype=np.float32)
        feats = blk[:, -1, :] if mode == 'last' else blk.reshape(len(sl), -1)
        P.append(clf.predict(feats))
    return np.concatenate(P), y[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--per-class', type=int, default=60_000)
    args = ap.parse_args()

    t0 = time.time()
    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    y = np.load(os.path.join(args.data, 'y.npy'))
    W = X.shape[1]
    tr_idx, va_idx = blocked_split(y, purge=W - 1)
    print(f"[{args.tag}] window={W} feat={X.shape[2]} train={len(tr_idx):,} val={len(va_idx):,}",
          flush=True)

    sub = subsample(tr_idx, y, args.per_class)
    print(f"balanced training subsample: {len(sub):,}", flush=True)

    sub_sorted = np.sort(sub)          # gather() reads in sorted order
    ytr = y[sub_sorted]

    results = {}
    for mode in ('last', 'flat'):
        t = time.time()
        Xtr = gather(X, sub_sorted, mode)
        n_feat = int(Xtr.shape[1])
        print(f"  [{mode}] fitting on {Xtr.shape} ...", flush=True)
        clf = LogisticRegression(max_iter=300, n_jobs=-1)
        clf.fit(Xtr, ytr)
        del Xtr
        P, T = evaluate(clf, X, va_idx, y, mode)
        f1 = f1_score(T, P, average='macro', zero_division=0)
        per = f1_score(T, P, average=None, zero_division=0, labels=list(range(len(STAGES))))
        print(f"  [{mode}] macroF1={f1:.4f}  ({(time.time()-t)/60:.1f} min)", flush=True)
        print(classification_report(T, P, target_names=STAGES, digits=3, zero_division=0),
              flush=True)
        results[mode] = {
            "macro_f1": float(f1),
            "per_class": {s: float(v) for s, v in zip(STAGES, per)},
            "n_features": n_feat,
        }

    out = {"tag": args.tag, "data": args.data, "window": int(W),
           "results": results, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(MODEL_DIR, f'{args.tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[{args.tag}] saved. last={results['last']['macro_f1']:.4f} "
          f"flat={results['flat']['macro_f1']:.4f}", flush=True)


if __name__ == '__main__':
    main()
