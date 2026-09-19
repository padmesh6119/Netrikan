"""
Persistence baseline for the horizon forecasting claim.

Persistence prediction: copy the last in-window label forward k steps.
Since y[i] = labels[W + i], the last in-window label for window i is
labels[W - 1 + i] = y[i - 1].

For horizon k: true target = y[i + k], persistence = y[i - 1].
Valid range: i in [1, len(y) - k - 1].

Usage:
    python3 bench/persistence_baseline.py --y data/processed/y.npy
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

HORIZONS = [0, 30, 90, 180, 360]
LSTM_SCORES = {
    0: 0.8889, 30: 0.8704, 90: 0.8661, 180: 0.8621, 360: 0.8622,
}
CLASS_NAMES = ["Benign", "InitialAccess", "DoS", "Infiltration", "Botnet"]


def run(y_path: str, out_dir: str):
    y = np.load(y_path)
    print(f"y shape: {y.shape}  unique: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    results = []
    for k in HORIZONS:
        start, end = 1, len(y) - k
        if end <= start:
            print(f"k={k}: not enough data")
            continue
        pred = y[start - 1:end - 1]
        true = y[start + k:end + k]

        macro = f1_score(true, pred, average="macro", zero_division=0)
        per_class = f1_score(true, pred, average=None, labels=list(range(5)), zero_division=0)
        change_rate = float(np.mean(pred != true))
        lstm = LSTM_SCORES.get(k, None)
        gap = (lstm - macro) if lstm is not None else None

        flag = ""
        if lstm is not None and macro >= lstm - 0.01:
            flag = "  *** AUTOCORRELATION LIKELY — persistence matches LSTM"

        gap_str = f"{gap:+.4f}" if gap is not None else "     ?"
        print(
            f"k={k:4d} | persistence={macro:.4f}  lstm={str(lstm or '?'):>6}  "
            f"gap={gap_str}  change_rate={change_rate:.4f}{flag}"
        )
        for i, name in enumerate(CLASS_NAMES):
            print(f"       {name:<15} persistence={per_class[i]:.4f}")
        print()

        results.append({
            "k": k, "persistence_macro_f1": round(macro, 6),
            "lstm_macro_f1": lstm, "gap": round(gap, 6) if gap is not None else None,
            "change_rate": round(change_rate, 6),
            "per_class": {n: round(float(v), 6) for n, v in zip(CLASS_NAMES, per_class)},
        })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "persistence_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="data/processed/y.npy")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    if not os.path.exists(args.y):
        sys.exit(f"y.npy not found at {args.y}")

    run(args.y, args.out)
