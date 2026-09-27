"""
Does cross-dataset transfer peak early and then decay as the model fits harder?

WHY. models/lodo_ensemble_dapt.json found something unplanned: 6 of 7 checkpoints
trained on SIX capture days transfer BETTER to DAPT 2020 than the deployed model
trained on all seven (member breach AUC 0.749 +/- 0.074 vs 0.694; best member
0.825). Two explanations fit that observation and they imply opposite fixes:

  (i)  holding out a DAY is what helps -- some specific day teaches CIC-specific
       habits, and dropping any one of them regularises;
  (ii) the members are simply LESS FITTED -- they early-stopped at 5-10 epochs
       while the deployed checkpoint trained longer, and less fitting transfers
       better regardless of which data is held out.

This script separates them. Train once on ALL SEVEN days with --save-epochs, then
score every epoch's checkpoint on DAPT. If transfer rises then falls while
in-dataset macro-F1 keeps climbing, (ii) is the mechanism and the fix is a
stopping criterion -- available to every future run at zero cost. If transfer
tracks in-dataset performance flat or upward, (ii) is refuted and the LODO
advantage really is about which data is held out.

HONESTY CONSTRAINT. Reporting this curve is legitimate science. CHOOSING a
deployment epoch by reading DAPT off it is test-set selection, and the output
JSON says so. To act on a peak you need a CIC-only proxy that correlates with it;
this script reports in-dataset macro-F1 per epoch alongside, so whether such a
proxy exists is itself measurable rather than assumed.

    python3 bench/transfer_vs_epoch.py --data "$DATA" --tag xfer_w30 \
        --out models/transfer_vs_epoch.json
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'bench'))

import dapt                                                      # noqa: E402
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features  # noqa: E402
from operating_points import (build_windows, load_feature_list,  # noqa: E402
                              entity_rollup, persist_mask, threshold_for_fpr)
from lodo_ensemble import score_model, onset_aucs                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', default='xfer_w30')
    ap.add_argument('--persist', type=int, default=3)
    ap.add_argument('--bucket', default='1h')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ckpts = glob.glob(os.path.join(ROOT, 'models', f'{args.tag}_ep*.pt'))
    if not ckpts:
        raise SystemExit(f"no per-epoch checkpoints for tag {args.tag}; "
                         f"train with --save-epochs")
    ckpts.sort(key=lambda p: int(re.search(r'_ep(\d+)\.pt$', p).group(1)))
    print(f"{len(ckpts)} epoch checkpoints", flush=True)

    # in-dataset macro-F1 per epoch, from the training history -- the CIC-only
    # signal that a stopping criterion would have to rely on
    hist = {}
    hp = os.path.join(ROOT, 'models', f'{args.tag}_metrics.json')
    if os.path.exists(hp):
        for h in json.load(open(hp))['history']:
            hist[h['epoch']] = {
                "macro_f1": round(h['macro_f1'], 4),
                "onset_auc_k5": round(h.get('onset_auc', {}).get('k5', float('nan')), 4)}

    features = load_feature_list(args.data)
    Xh = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    window, n_feat = Xh.shape[1], Xh.shape[2]
    del Xh

    df = dapt.load_all()
    if any(f in features for f in PORT_FEATURES):
        df = add_port_features(df)
    X_raw, y, cap, host, ts = build_windows(df, features, window)
    with open(os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    base_idx = [features.index(f) for f in BASE_FEATURES]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)
    truth = y > 0
    g_host = np.array([f"{c}|{h}" for c, h in zip(cap, host)])
    print(f"DAPT windows {len(X):,}  attack {int(truth.sum()):,}", flush=True)

    rows = []
    for c in ckpts:
        ep = int(re.search(r'_ep(\d+)\.pt$', c).group(1))
        b, o = score_model(c, X, n_feat)
        r = {"epoch": ep,
             "dapt_breach_auc": round(float(roc_auc_score(truth, b)), 4),
             "dapt_breach_pr_auc": round(float(average_precision_score(truth, b)), 4),
             "dapt_onset_auc": onset_aucs(o, y, cap),
             "cic_macro_f1": hist.get(ep, {}).get('macro_f1'),
             "cic_onset_auc_k5": hist.get(ep, {}).get('onset_auc_k5')}
        thr, cc = threshold_for_fpr(b, truth, 0.05, g_host, args.persist)
        if cc:
            pred = persist_mask(b >= thr, g_host, args.persist)
            e = entity_rollup(pred, truth, cap, host, ts, args.bucket)
            r["dapt_recall_at_fpr05"] = cc['recall']
            r["dapt_sedi_at_fpr05"] = cc['sedi']
            r["dapt_entity_recall"] = e['entity_recall']
            r["dapt_entity_fa_rate"] = e['false_alarm_rate_per_benign_cell']
        rows.append(r)
        print(f"  ep{ep:>3}  DAPT AUC {r['dapt_breach_auc']:.4f}  "
              f"PR {r['dapt_breach_pr_auc']:.4f}  "
              f"recall@5% {r.get('dapt_recall_at_fpr05')}  "
              f"| CIC macro-F1 {r['cic_macro_f1']}", flush=True)

    aucs = [r['dapt_breach_auc'] for r in rows]
    best = rows[int(np.argmax(aucs))]
    last = rows[-1]
    cic = [(r['epoch'], r['cic_macro_f1']) for r in rows if r['cic_macro_f1'] is not None]
    best_cic = max(cic, key=lambda t: t[1]) if cic else (None, None)

    # Does the CIC-only signal point at the same epoch the transfer peak wants?
    # If not, no CIC-side stopping criterion can recover the peak, and that is
    # the finding rather than a detail.
    out = {
        "data": args.data, "tag": args.tag,
        "eval_set": "DAPT 2020 (never seen in training)",
        "n_checkpoints": len(rows),
        "per_epoch": rows,
        "peak_transfer": {"epoch": best['epoch'], "dapt_breach_auc": best['dapt_breach_auc']},
        "final_epoch": {"epoch": last['epoch'], "dapt_breach_auc": last['dapt_breach_auc']},
        "decay_from_peak": round(best['dapt_breach_auc'] - last['dapt_breach_auc'], 4),
        "best_cic_epoch": {"epoch": best_cic[0], "cic_macro_f1": best_cic[1]},
        "cic_criterion_would_pick_transfer_peak": best_cic[0] == best['epoch'],
        "honesty_note": (
            "Reporting this curve is legitimate; choosing a deployment epoch by "
            "reading DAPT off it is test-set selection. Acting on a peak requires "
            "a CIC-only proxy that correlates with it -- "
            "cic_criterion_would_pick_transfer_peak reports whether the obvious "
            "proxy (best in-dataset macro-F1) does."),
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\npeak transfer at epoch {best['epoch']} (AUC {best['dapt_breach_auc']:.4f}); "
          f"final epoch {last['epoch']} (AUC {last['dapt_breach_auc']:.4f}); "
          f"decay {out['decay_from_peak']:+.4f}")
    print(f"best CIC macro-F1 at epoch {best_cic[0]}; would that pick the transfer "
          f"peak? {out['cic_criterion_would_pick_transfer_peak']}")
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
