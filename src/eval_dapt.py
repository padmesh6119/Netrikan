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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed'))
    ap.add_argument('--model', default=os.path.join(MODEL_DIR, 'cic_v2_w30.pt'))
    ap.add_argument('--tag', default='dapt_crossdataset')
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

    with open(os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
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

    preds, breach, onset = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits, b, _, o = model(torch.from_numpy(X[i:i + 8192]))
            preds.append(logits.argmax(1).numpy())
            breach.append(b.numpy())
            onset.append(torch.sigmoid(o).numpy())
    preds, breach = np.concatenate(preds), np.concatenate(breach)
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
    }
    with open(os.path.join(MODEL_DIR, f'{args.tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {MODEL_DIR}/{args.tag}.json", flush=True)


if __name__ == '__main__':
    main()
