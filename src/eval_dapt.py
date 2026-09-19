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
from model import WorldModel
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features

MODEL_DIR = os.path.expanduser("~/netrikan/models")
DEVICE = torch.device("cpu")
CIC_STAGES = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']


def load_feature_list(data_dir):
    p = os.path.join(data_dir, 'features.txt')
    if os.path.exists(p):
        return [l.strip() for l in open(p) if l.strip()]
    return list(BASE_FEATURES)


def build_windows(df, features, window):
    Xs, ys = [], []
    for _, grp in df.groupby('capture', sort=False):
        g = grp.sort_values('ts', kind='mergesort')
        v = g[features].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        if len(v) <= window:
            continue
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])
    return np.asarray(Xs, np.float32), np.asarray(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/netrikan/data/processed'))
    ap.add_argument('--model', default=os.path.join(MODEL_DIR, 'lstm_world_model.pt'))
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

    X_raw, y_dapt = build_windows(df, features, window)
    print(f"windows: {len(X_raw):,}", flush=True)

    with open(os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # only the continuous base features were scaled at training time
    base_idx = [features.index(f) for f in BASE_FEATURES]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)

    model = WorldModel(input_size=n_feat)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
    model.eval()

    preds, breach = [], []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits, b, _ = model(torch.from_numpy(X[i:i + 8192]))
            preds.append(logits.argmax(1).numpy())
            breach.append(b.numpy())
    preds, breach = np.concatenate(preds), np.concatenate(breach)

    true_attack = (y_dapt > 0).astype(int)
    pred_attack = (preds > 0).astype(int)

    print("\n" + "=" * 66)
    print(f"CROSS-DATASET  CIC-IDS-2018 -> DAPT 2020   [{args.tag}]")
    print("=" * 66)
    print(classification_report(true_attack, pred_attack,
                                target_names=['Benign', 'Attack'], digits=3,
                                zero_division=0), flush=True)

    auc = roc_auc_score(true_attack, breach)
    ap_ = average_precision_score(true_attack, breach)
    tn, fp, fn, tp = confusion_matrix(true_attack, pred_attack).ravel()
    print(f"breach head ROC-AUC {auc:.4f}  PR-AUC {ap_:.4f}")
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

    out = {
        "tag": args.tag, "model": args.model, "data": args.data,
        "window": int(window), "features": int(n_feat),
        "n_windows": int(len(X)), "roc_auc": float(auc), "pr_auc": float(ap_),
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
