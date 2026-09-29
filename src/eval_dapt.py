import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)

import dapt
from model import WorldModel, ONSET_HORIZONS
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DEVICE = torch.device("cpu")
CIC_STAGES = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']


def load_feature_list(data_dir):
    p = os.path.join(data_dir, 'features.txt')
    if os.path.exists(p):
        return [l.strip() for l in open(p) if l.strip()]
    return list(BASE_FEATURES)


def build_windows(df, features, window):
    """Returns X, y, and cap_id (which capture each window came from), so onset
    labels can be computed without comparing across capture-file boundaries."""
    Xs, ys, caps = [], [], []
    for ci, (_, grp) in enumerate(df.groupby('capture', sort=False)):
        g = grp.sort_values('ts', kind='mergesort')
        v = g[features].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        if len(v) <= window:
            continue
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])
            caps.append(ci)
    return (np.asarray(Xs, np.float32), np.asarray(ys),
            np.asarray(caps, np.int64))


OP_TARGET_FPRS = [0.05, 0.10, 0.20]


def operating_points(score, truth, targets=OP_TARGET_FPRS):
    """Recall/precision at fixed benign-window FPR targets. The threshold is the
    lowest one whose ACHIEVED FPR is <= target (ties at the threshold count as
    flagged), and the achieved FPR is reported next to the target."""
    score = np.asarray(score, np.float64)
    truth = np.asarray(truth).astype(bool)
    ben = np.sort(score[~truth])[::-1]            # benign scores, descending
    out = []
    for t in targets:
        n_fp = int(np.floor(t * len(ben)))
        # flag score > ben[n_fp] so at most n_fp benign windows are flagged
        thr = float(np.nextafter(ben[n_fp], np.inf)) if n_fp < len(ben) else float(ben[-1])
        pred = score >= thr
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
        out.append({'target_fpr': t, 'threshold': thr,
                    'achieved_fpr': round(fp / max(fp + tn, 1), 4),
                    'recall': round(tp / max(tp + fn, 1), 4),
                    'precision': round(tp / max(tp + fp, 1), 4),
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed'))
    ap.add_argument('--model', default=os.path.join(MODEL_DIR, 'cic_v2_w30.pt'))
    ap.add_argument('--tag', default='dapt_crossdataset')
    ap.add_argument('--scaler', default=None,
                    help='scaler pickle; defaults to <data>/scaler.pkl')
    args = ap.parse_args()

    features = load_feature_list(args.data)
    needs_ports = any(f in features for f in PORT_FEATURES)

    Xh = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    window, n_feat = Xh.shape[1], Xh.shape[2]
    del Xh
    assert n_feat == len(features), f"feature mismatch: {n_feat} vs {len(features)}"

    print(f"[{args.tag}] window={window} features={n_feat} ports={needs_ports}", flush=True)

    df = dapt.load_all()
    if needs_ports:
        df = add_port_features(df)

    X_raw, y_dapt, cap_id = build_windows(df, features, window)
    print(f"windows: {len(X_raw):,}", flush=True)

    with open(args.scaler or os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # only the continuous base features were scaled at training time
    base_idx = [features.index(f) for f in BASE_FEATURES]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)

    # strict=False: checkpoints predating the state/onset heads carry no weights
    # for them. has_onset below reports whether the onset head is real.
    sd = torch.load(args.model, map_location=DEVICE, weights_only=True)
    model = WorldModel(input_size=n_feat, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()
    has_onset = 'onset_head.weight' in sd

    preds, breach, onset, stage_att = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits, b, _, o = model(torch.from_numpy(X[i:i + 8192]))
            preds.append(logits.argmax(1).numpy())
            breach.append(b.numpy())
            onset.append(torch.sigmoid(o).numpy())
            stage_att.append(1.0 - torch.softmax(logits, 1)[:, 0].numpy())
    preds, breach = np.concatenate(preds), np.concatenate(breach)
    stage_att = np.concatenate(stage_att)
    onset = np.concatenate(onset)

    true_attack = (y_dapt > 0).astype(int)
    pred_attack = (preds > 0).astype(int)

    print("\n" + "=" * 66)
    print(f"CROSS-DATASET  CIC-IDS-2018 -> DAPT 2020   [{args.tag}]")
    print("=" * 66)
    print(classification_report(true_attack, pred_attack,
                                target_names=['Benign', 'Attack'], digits=3,
                                zero_division=0), flush=True)

    # NOTE: named breach_auc, not auc. The onset loop below used to reuse `auc`
    # as its loop variable, so the value saved to JSON as "roc_auc" was silently
    # the LAST onset AUC (k=30) rather than the breach head's ROC-AUC. Every
    # committed dapt_*.json for a checkpoint WITH an onset head carries that
    # wrong number; dapt_base_w30.json is correct only because base_w30 has no
    # onset head and the loop never ran.
    breach_auc = roc_auc_score(true_attack, breach)
    ap_ = average_precision_score(true_attack, breach)
    tn, fp, fn, tp = confusion_matrix(true_attack, pred_attack).ravel()
    print(f"breach head ROC-AUC {breach_auc:.4f}  PR-AUC {ap_:.4f}")
    print(f"TP {tp:,}  FP {fp:,}  FN {fn:,}  TN {tn:,}\n")

    # ---- fixed-FPR operating points -----------------------------------------
    # The argmax decision above is one untuned point. A SOC picks a threshold
    # for an alert budget, so report recall/precision at fixed FPR for both
    # scores the model emits.
    ops = {'breach': operating_points(breach, true_attack),
           'stage_attack_mass': operating_points(stage_att, true_attack)}
    stage_auc = roc_auc_score(true_attack, stage_att)
    print("operating points (benign-window FPR target -> recall / precision)")
    for sc, rows_ in ops.items():
        for r in rows_:
            print(f"  {sc:<18} FPR<={r['target_fpr']:.2f} (got {r['achieved_fpr']:.4f})  "
                  f"recall {r['recall']:.3f}  precision {r['precision']:.3f}  "
                  f"thr {r['threshold']:.4f}")
    best = max(ops, key=lambda k: next(r['recall'] for r in ops[k]
                                       if r['target_fpr'] == 0.10))
    at10 = next(r for r in ops[best] if r['target_fpr'] == 0.10)
    headline = (f"recall {at10['recall']:.1%} at {at10['achieved_fpr']:.1%} FPR "
                f"({best} score, precision {at10['precision']:.1%})")
    print(f"  headline: {headline}\n")

    rows = {}
    for sid, name in enumerate(dapt.DAPT_STAGES):
        m = y_dapt == sid
        if not m.any():
            continue
        rate = float((pred_attack[m] == 0).mean()) if sid == 0 else float(pred_attack[m].mean())
        print(f"  {name:<22} n={m.sum():>7,}  "
              f"{'correctly benign' if sid == 0 else 'detected'} {rate:6.1%}")
        rows[name] = {"n": int(m.sum()), "rate": rate}

    # ---- onset / hazard evaluation ------------------------------------------
    # "Does a NEW stage begin within the next k windows?" -- the forecasting
    # target proper. A persistence baseline always answers "no", so it scores
    # AUC 0.500 (no discrimination) and recall 0.000 at any threshold. Any AUC
    # above 0.500 here is forecasting skill that persistence cannot reach.
    onset_rows = {}
    if has_onset:
        print("\nOnset AUC  (persistence: AUC=0.500, recall=0.000 by construction)")
        for ki, k in enumerate(ONSET_HORIZONS):
            # onset[j] = 1 if stage changes within k windows, but only where
            # window j+k is in the SAME capture -- comparing across the 10
            # capture-file seams would invent transitions between unrelated flows
            same = cap_id[:-k] == cap_id[k:] if k < len(cap_id) else np.array([])
            o_true = (y_dapt[:-k][same] != y_dapt[k:][same]).astype(int)
            o_pred = onset[:-k][same, ki]
            if len(o_true) == 0 or o_true.max() == o_true.min():
                print(f"  k={k:>3}  no onset events in range — skipped")
                continue
            o_auc = float(roc_auc_score(o_true, o_pred))
            frac = float(o_true.mean())
            print(f"  k={k:>3}  onset_rate={frac:.4f}  AUC={o_auc:.4f}  "
                  f"gap over persistence=+{o_auc - 0.5:.4f}")
            onset_rows[f'k{k}'] = {"auc": o_auc, "onset_rate": frac,
                                   "gap_over_persistence": round(o_auc - 0.5, 4)}
    else:
        print("\nOnset head not present in this checkpoint — retrain to populate.")

    out = {
        "tag": args.tag, "model": args.model, "data": args.data,
        "has_onset_head": bool(has_onset),
        "onset_auc_dapt": onset_rows,
        "persistence_onset_auc": 0.5,
        "persistence_onset_recall": 0.0,
        "window": int(window), "features": int(n_feat),
        "n_windows": int(len(X)),
        "breach_roc_auc": float(breach_auc),
        "roc_auc": float(breach_auc),   # kept for back-compat; now correct
        "pr_auc": float(ap_),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "binary_report": classification_report(
            true_attack, pred_attack, target_names=['Benign', 'Attack'],
            output_dict=True, zero_division=0),
        "per_phase": rows,
        "stage_attack_mass_roc_auc": float(stage_auc),
        "operating_points": {
            "scores": {
                "breach": "sigmoid(breach_head)",
                "stage_attack_mass": "1 - softmax(stage_logits)[Benign]",
            },
            "fpr_definition": "false positives / benign windows",
            "curves": ops,
            "headline_at_10pct_fpr": headline,
        },
    }
    with open(os.path.join(MODEL_DIR, f'{args.tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {MODEL_DIR}/{args.tag}.json", flush=True)


if __name__ == '__main__':
    main()
