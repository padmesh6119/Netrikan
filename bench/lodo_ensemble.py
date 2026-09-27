"""
Does averaging the 7 leave-one-day-out checkpoints beat the single deployed model?

WHY ON DAPT AND NOT ON A HELD-OUT DAY. Checkpoint `lodo_f3_w30.pt` is the only one
of the seven that never saw capture day 3; the other six trained on it. So
averaging all seven and scoring on day 3 would let six members predict data they
were trained on, and the result would be meaningless. Restricting the ensemble to
members that never saw the evaluation day leaves exactly ONE member, which is just
that fold's original number. A cross-day LODO ensemble is therefore not possible
with these checkpoints, and this file does not pretend otherwise.

DAPT 2020 is clean for every member: all seven trained only on CIC-IDS-2018, so
none has seen a single DAPT flow. That makes it a legitimate test of whether
ensembling buys cross-domain discrimination -- which is exactly what the single
model is weakest at (breach ROC-AUC 0.679, and
models/operating_points_dapt_cic_v2_w30.json shows no threshold rescues it).

The members are also diverse in a useful way rather than by random seed alone:
each one is missing a different capture day, so each carries a different blind
spot. That is closer to a bagged ensemble than to a seed ensemble.

Reported for the baseline, each member, and the mean-probability ensemble:
  breach ROC-AUC / PR-AUC          -- ranking quality, the thing thresholds cannot fix
  onset AUC per horizon            -- cross-dataset forecasting transfer
  entity roll-up at FPR 5%, persist 3  -- the operator-facing operating point

    python3 bench/lodo_ensemble.py --data "$DATA" \
        --out models/lodo_ensemble_dapt.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'bench'))

import dapt                                                     # noqa: E402
from model import WorldModel, ONSET_HORIZONS                     # noqa: E402
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features  # noqa: E402
from operating_points import (build_windows, load_feature_list, counts,  # noqa: E402
                              entity_rollup, persist_mask, threshold_for_fpr)


def score_model(path, X, n_feat, batch=8192):
    """Returns (breach prob, onset prob per horizon) for one checkpoint."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=n_feat, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()
    b_all, o_all = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            _, b, _, o = model(torch.from_numpy(X[i:i + batch]))
            b_all.append(b.numpy())
            o_all.append(torch.sigmoid(o).numpy())
    return np.concatenate(b_all), np.concatenate(o_all)


def onset_aucs(onset, y, cap):
    """Cross-dataset onset AUC per horizon, never pairing across capture seams."""
    rows = {}
    for ki, k in enumerate(ONSET_HORIZONS):
        if k >= len(cap):
            continue
        same = cap[:-k] == cap[k:]
        o_true = (y[:-k][same] != y[k:][same]).astype(int)
        o_pred = onset[:-k][same, ki]
        if len(o_true) == 0 or o_true.max() == o_true.min():
            continue
        rows[f'k{k}'] = round(float(roc_auc_score(o_true, o_pred)), 4)
    return rows


def evaluate(name, breach, onset, truth, y, cap, host, ts, persist, bucket):
    out = {
        "breach_roc_auc": round(float(roc_auc_score(truth, breach)), 4),
        "breach_pr_auc": round(float(average_precision_score(truth, breach)), 4),
        "onset_auc": onset_aucs(onset, y, cap),
    }
    g_host = np.array([f"{c}|{h}" for c, h in zip(cap, host)])
    thr, c = threshold_for_fpr(breach, truth, 0.05, g_host, persist)
    if c:
        pred = persist_mask(breach >= thr, g_host, persist)
        out["at_fpr05_persist%d" % persist] = {
            "threshold": round(thr, 5), "window_level": c,
            **entity_rollup(pred, truth, cap, host, ts, bucket)}
    print(f"  {name:<18} breach AUC {out['breach_roc_auc']:.4f}  "
          f"PR {out['breach_pr_auc']:.4f}  onset {out['onset_auc']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--members', default=os.path.join(ROOT, 'models', 'lodo_f?_w30.pt'))
    ap.add_argument('--baseline', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--persist', type=int, default=3)
    ap.add_argument('--bucket', default='1h')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    members = sorted(glob.glob(args.members))
    if not members:
        raise SystemExit(f"no member checkpoints matched {args.members}")
    print(f"members: {[os.path.basename(m) for m in members]}", flush=True)

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
    print(f"DAPT windows {len(X):,}  attack {int(truth.sum()):,}  "
          f"hosts {len(np.unique(host))}", flush=True)

    out = {
        "data": args.data, "eval_set": "DAPT 2020 (unseen by every member)",
        "why_not_heldout_day": (
            "Only lodo_fN never saw day N; the other six trained on it, so a 7-member "
            "ensemble scored on day N would leak. Restricting to clean members leaves "
            "one, which is just that fold's own result. A cross-day LODO ensemble is "
            "not possible with these checkpoints."),
        "n_windows": int(len(X)), "n_attack": int(truth.sum()),
        "members": [os.path.basename(m) for m in members],
        "per_member": {}, "baseline": None, "ensemble": None,
    }

    print("\nbaseline:")
    b_base, o_base = score_model(args.baseline, X, n_feat)
    out['baseline'] = {"checkpoint": os.path.basename(args.baseline),
                       **evaluate(os.path.basename(args.baseline), b_base, o_base,
                                  truth, y, cap, host, ts, args.persist, args.bucket)}

    print("\nmembers:")
    bs, os_ = [], []
    for m in members:
        b, o = score_model(m, X, n_feat)
        bs.append(b)
        os_.append(o)
        out['per_member'][os.path.basename(m)] = evaluate(
            os.path.basename(m), b, o, truth, y, cap, host, ts,
            args.persist, args.bucket)

    print("\nensemble (mean probability):")
    b_ens = np.mean(bs, axis=0)
    o_ens = np.mean(os_, axis=0)
    out['ensemble'] = evaluate('ensemble', b_ens, o_ens, truth, y, cap, host, ts,
                               args.persist, args.bucket)

    # Rank averaging, motivated a priori rather than chosen after seeing results.
    # Averaging seven sigmoid outputs compresses the dynamic range toward the
    # middle: members disagree on scale even when they agree on order, so a
    # probability mean can rank well globally (AUC) while placing no useful
    # decision boundary at a fixed FPR -- exactly the pathology
    # models/lodo_ensemble_dapt.json showed (+0.118 AUC, no gain at FPR 5%).
    # Averaging within-model percentile ranks is scale-free and keeps the
    # boundary usable. Both variants are reported; neither selects over members,
    # and choosing between them on DAPT would itself be test-set selection, so
    # the JSON keeps both and the decision is deferred to a CIC-only criterion.
    print("\nensemble (mean rank):")
    def _rank(v):
        r = np.empty(len(v), dtype=np.float64)
        r[np.argsort(v, kind='mergesort')] = np.arange(len(v))
        return r / max(len(v) - 1, 1)
    b_rank = np.mean([_rank(b) for b in bs], axis=0)
    o_rank = np.stack([np.mean([_rank(o[:, ki]) for o in os_], axis=0)
                       for ki in range(os_[0].shape[1])], axis=1)
    out['ensemble_rank'] = evaluate('ensemble_rank', b_rank, o_rank, truth, y,
                                    cap, host, ts, args.persist, args.bucket)

    mem_auc = [v['breach_roc_auc'] for v in out['per_member'].values()]
    out['summary'] = {
        "baseline_breach_auc": out['baseline']['breach_roc_auc'],
        "member_breach_auc_mean": round(float(np.mean(mem_auc)), 4),
        "member_breach_auc_best": max(mem_auc),
        "ensemble_breach_auc": out['ensemble']['breach_roc_auc'],
        "ensemble_gain_over_baseline": round(
            out['ensemble']['breach_roc_auc'] - out['baseline']['breach_roc_auc'], 4),
        "ensemble_gain_over_best_member": round(
            out['ensemble']['breach_roc_auc'] - max(mem_auc), 4),
        "ensemble_rank_breach_auc": out['ensemble_rank']['breach_roc_auc'],
        "ensemble_rank_gain_over_baseline": round(
            out['ensemble_rank']['breach_roc_auc']
            - out['baseline']['breach_roc_auc'], 4),
        "ensemble_rank_gain_over_best_member": round(
            out['ensemble_rank']['breach_roc_auc'] - max(mem_auc), 4),
    }
    print("\nsummary:", json.dumps(out['summary'], indent=1), flush=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
