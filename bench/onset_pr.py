"""
Onset head: PR-AUC and recall at fixed FPR.

ROC-AUC flatters a rare-event forecaster because the true-negative pool is huge.
Precision-recall is the honest presentation. This reports, per horizon:

  positive rate + absolute positive count   (the PR denominator)
  ROC-AUC                                    (kept, for continuity)
  PR-AUC (average precision)
  recall at FPR 0.001 / 0.01 / 0.05, each with threshold and ACHIEVED FPR
  no-skill PR baseline (= positive rate), so PR-AUC is readable

Evaluated on the blocked-split validation set, onset labels boundary-safe (the -1
sentinel from pipeline_v2 is excluded per horizon).

    python3 bench/onset_pr.py --data "$DATA" --model "$CKPT" \
        --out models/onset_pr_cic_full_w30.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import WorldModel, ONSET_HORIZONS   # noqa: E402
from train_v2 import blocked_split             # noqa: E402


def recall_at_fpr(y_true, score, target_fpr):
    """Highest recall attainable while keeping FPR <= target. Returns
    (recall, threshold, achieved_fpr)."""
    neg = score[y_true == 0]
    if len(neg) == 0:
        return None, None, None
    # threshold = the target-FPR upper quantile of negative scores
    thr = float(np.quantile(neg, 1.0 - target_fpr))
    pred = score >= thr
    tp = int((pred & (y_true == 1)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    recall = tp / max(tp + fn, 1)
    achieved = fp / max(fp + tn, 1)
    return round(recall, 4), round(thr, 5), round(achieved, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'onset_pr_cic_v2_w30.json'))
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))[:n_all]
    onset = {k: np.load(os.path.join(args.data, f'onset_k{k}.npy'))
             for k in ONSET_HORIZONS}

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    if 'onset_head.weight' not in sd:
        raise SystemExit(f"{args.model} has no trained onset head")
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    _, va = blocked_split(y, purge=W - 1)
    va = np.sort(va[va < n_all])

    pred = np.zeros((n_all, len(ONSET_HORIZONS)), np.float32)
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            sl = va[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(X[sl], np.float32))
            pred[sl] = torch.sigmoid(model(xb)[3]).numpy()

    out = {"model": args.model, "data": args.data, "n_val": int(len(va)),
           "horizons": {}}
    print(f"val windows: {len(va):,}")
    for ki, k in enumerate(ONSET_HORIZONS):
        lab = onset[k][va]
        p = pred[va, ki]
        valid = lab >= 0                       # drop boundary -1 sentinels
        lab, p = lab[valid].astype(int), p[valid]
        if lab.max() == lab.min():
            out["horizons"][f"k{k}"] = {"note": "no positives in val range"}
            continue
        pos = int(lab.sum())
        rec = {f"fpr_{t}": recall_at_fpr(lab, p, t) for t in (0.001, 0.01, 0.05)}
        row = {
            "positive_rate": round(float(lab.mean()), 5),
            "positive_count": pos, "n": int(len(lab)),
            "roc_auc": round(float(roc_auc_score(lab, p)), 4),
            "pr_auc": round(float(average_precision_score(lab, p)), 4),
            "no_skill_pr_baseline": round(float(lab.mean()), 5),
            "recall_at_fpr": {t: {"recall": r[0], "threshold": r[1],
                                  "achieved_fpr": r[2]}
                              for t, r in zip((0.001, 0.01, 0.05), rec.values())},
        }
        out["horizons"][f"k{k}"] = row
        print(f"\nk={k}: pos={pos:,} ({row['positive_rate']:.4f})  "
              f"ROC-AUC {row['roc_auc']}  PR-AUC {row['pr_auc']}  "
              f"(no-skill {row['no_skill_pr_baseline']})")
        for t in (0.001, 0.01, 0.05):
            d = row['recall_at_fpr'][t]
            print(f"    recall@FPR {t}: {d['recall']} (achieved {d['achieved_fpr']})")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
