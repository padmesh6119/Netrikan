#!/usr/bin/env python3
"""
A1 — segment-level vs per-host windowing on DAPT 2020.

Same LSTM, same scaler, same 24 features, same window size.
Only the windowing strategy changes:

  segment  — 10 consecutive flows within a capture, regardless of source host
  per-host — 10 consecutive flows from the SAME source IP within a capture

The per-host approach lets the model see one host's own trajectory rather than
a mixture of flows from different hosts.  On DAPT's 4-day APT campaign, where
lateral movement is one attacker pivoting between machines, this should
concentrate the relevant signal.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, f1_score,
                             roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(__file__))
import dapt
from model import WorldModel
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features

MODEL_DIR = os.path.expanduser("~/netrikan/models")
DATA_DIR  = os.path.expanduser("~/netrikan/data/processed")
DEVICE    = torch.device("cpu")

CIC_STAGES  = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']
DAPT_STAGES = dapt.DAPT_STAGES   # Benign/Recon/Foothold/Lateral/Exfil


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────

def load_artefacts(data_dir, model_path):
    feat_file = os.path.join(data_dir, 'features.txt')
    features = ([l.strip() for l in open(feat_file) if l.strip()]
                if os.path.exists(feat_file) else list(BASE_FEATURES))

    X_ref = np.load(os.path.join(data_dir, 'X.npy'), mmap_mode='r')
    window, n_feat = X_ref.shape[1], X_ref.shape[2]
    del X_ref
    assert n_feat == len(features), f"feature count mismatch: {n_feat} vs {len(features)}"

    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as fh:
        scaler = pickle.load(fh)

    model = WorldModel(input_size=n_feat)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True),
        strict=False)
    model.eval()
    return model, scaler, features, window, n_feat


# ──────────────────────────────────────────────
# Window builders
# ──────────────────────────────────────────────

def _extract_windows(groups, features, window):
    Xs, ys = [], []
    for g in groups:
        g = g.sort_values('ts', kind='mergesort')
        v = g[features].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        if len(v) <= window:
            continue
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])
    if not Xs:
        return np.empty((0, window, len(features)), np.float32), np.empty(0, int)
    return np.asarray(Xs, np.float32), np.asarray(ys)


def build_segment_windows(df, features, window):
    """Current approach: consecutive flows per capture, ignoring host identity."""
    groups = [grp for _, grp in df.groupby('capture', sort=False)]
    return _extract_windows(groups, features, window)


def build_perhost_windows(df, features, window):
    """New approach: consecutive flows from the same (capture, Src IP) pair."""
    if 'Src IP' not in df.columns:
        raise ValueError("DAPT data has no 'Src IP' column")
    groups = [grp for _, grp in df.groupby(['capture', 'Src IP'], sort=False)]
    return _extract_windows(groups, features, window)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def scale_and_infer(model, scaler, X_raw, features, n_feat):
    base_idx = [features.index(f) for f in BASE_FEATURES if f in features]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)

    preds, breach = [], []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits, b, _ = model(torch.from_numpy(X[i:i + 8192]))
            preds.append(logits.argmax(1).numpy())
            breach.append(b.numpy())
    return np.concatenate(preds), np.concatenate(breach)


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def _binary_stats(y_true, preds, breach):
    ta = (y_true > 0).astype(int)
    pa = (preds  > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(ta, pa).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc  = roc_auc_score(ta, breach)
    return {
        'n':       int(len(y_true)),
        'tp':      int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'recall':  round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        'precision': round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
        'fpr':     round(fpr, 4),
        'roc_auc': round(auc, 4),
    }


def _per_stage_detection(y_dapt, preds, label=''):
    rows = {}
    for sid, name in enumerate(DAPT_STAGES):
        m = y_dapt == sid
        if not m.any():
            continue
        n = int(m.sum())
        if sid == 0:
            rate = float((preds[m] == 0).mean())
            rows[name] = {'n': n, 'correct_benign': round(rate, 4)}
        else:
            rate = float((preds[m] > 0).mean())
            rows[name] = {'n': n, 'attack_detected': round(rate, 4)}
    return rows


def _lateral_detail(y_dapt, preds):
    m = y_dapt == 3   # DAPT Lateral Movement
    if not m.any():
        return None
    detected = (preds[m] > 0).sum()
    return {
        'n_windows':       int(m.sum()),
        'detected':        int(detected),
        'recall':          round(float(detected) / m.sum(), 4),
        'as_infiltration': int((preds[m] == 3).sum()),   # CIC class 3 = Infiltration/Lateral
        'as_benign':       int((preds[m] == 0).sum()),
    }


def report(tag, y_dapt, preds, breach):
    binary = _binary_stats(y_dapt, preds, breach)
    stages = _per_stage_detection(y_dapt, preds)
    lateral = _lateral_detail(y_dapt, preds)
    return {
        'tag':     tag,
        'binary':  binary,
        'stages':  stages,
        'lateral': lateral,
    }


def print_comparison(seg, host):
    lm_s = seg['lateral']
    lm_h = host['lateral']
    print()
    print("=" * 66)
    print("A1  Segment-level  vs  Per-host  windowing   (DAPT 2020)")
    print("=" * 66)
    print(f"{'':30s} {'SEGMENT':>10} {'PER-HOST':>10}")
    print(f"{'─'*50}")
    print(f"{'Windows':30s} {seg['binary']['n']:>10,} {host['binary']['n']:>10,}")
    print(f"{'Attack recall':30s} {seg['binary']['recall']:>10.3f} {host['binary']['recall']:>10.3f}")
    print(f"{'Attack precision':30s} {seg['binary']['precision']:>10.3f} {host['binary']['precision']:>10.3f}")
    print(f"{'Benign FPR':30s} {seg['binary']['fpr']:>10.3f} {host['binary']['fpr']:>10.3f}")
    print(f"{'ROC-AUC':30s} {seg['binary']['roc_auc']:>10.4f} {host['binary']['roc_auc']:>10.4f}")
    print()
    print("Lateral Movement windows:")
    print(f"{'  n_windows':30s} {lm_s['n_windows']:>10,} {lm_h['n_windows']:>10,}")
    print(f"{'  recall':30s} {lm_s['recall']:>10.3f} {lm_h['recall']:>10.3f}")
    print(f"{'  flagged as Infiltration':30s} {lm_s['as_infiltration']:>10,} {lm_h['as_infiltration']:>10,}")
    print(f"{'  missed (as Benign)':30s} {lm_s['as_benign']:>10,} {lm_h['as_benign']:>10,}")
    print()

    for stage in DAPT_STAGES[1:]:   # skip Benign
        rs = seg['stages'].get(stage, {})
        rh = host['stages'].get(stage, {})
        ks = list(rs.keys() - {'n'})
        kh = list(rh.keys() - {'n'})
        metric = (ks or kh or ['n'])[0]
        vs = rs.get(metric, 0.0)
        vh = rh.get(metric, 0.0)
        ns = rs.get('n', 0)
        nh = rh.get('n', 0)
        delta = vh - vs
        marker = '▲' if delta > 0.005 else ('▼' if delta < -0.005 else ' ')
        print(f"  {stage:<24} seg={vs:.3f} (n={ns:,})  host={vh:.3f} (n={nh:,})  {marker}{abs(delta):.3f}")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',  default=DATA_DIR)
    ap.add_argument('--model', default=os.path.join(MODEL_DIR, 'lstm_world_model.pt'))
    ap.add_argument('--out',   default=os.path.join(MODEL_DIR, 'a1_entity_compare.json'))
    args = ap.parse_args()

    print("Loading model and scaler …", flush=True)
    model, scaler, features, window, n_feat = load_artefacts(args.data, args.model)
    print(f"  window={window}  features={n_feat}  model={os.path.basename(args.model)}", flush=True)

    needs_ports = any(f in features for f in PORT_FEATURES)

    print("Loading DAPT 2020 …", flush=True)
    df = dapt.load_all()
    if needs_ports:
        df = add_port_features(df)
    print(f"  {len(df):,} flows, {df['Src IP'].nunique()} unique source IPs", flush=True)

    print("\nBuilding segment-level windows …", flush=True)
    X_seg, y_seg = build_segment_windows(df, features, window)
    print(f"  {len(X_seg):,} windows", flush=True)

    print("Building per-host windows …", flush=True)
    X_host, y_host = build_perhost_windows(df, features, window)
    print(f"  {len(X_host):,} windows  ({df.groupby(['capture','Src IP']).size().gt(window).sum()} hosts with ≥{window+1} flows)", flush=True)

    print("\nRunning inference — segment …", flush=True)
    preds_seg, breach_seg = scale_and_infer(model, scaler, X_seg, features, n_feat)

    print("Running inference — per-host …", flush=True)
    preds_host, breach_host = scale_and_infer(model, scaler, X_host, features, n_feat)

    seg  = report('segment', y_seg,  preds_seg,  breach_seg)
    host = report('per_host', y_host, preds_host, breach_host)

    print_comparison(seg, host)

    # lateral movement: which hosts are the attackers?
    lm_hosts = df[df['stage_id'] == 3]['Src IP'].value_counts()
    if not lm_hosts.empty:
        print(f"Lateral-movement flows by source IP (top 5):")
        for ip, cnt in lm_hosts.head(5).items():
            print(f"  {ip:<18} {cnt:>6,} flows")
        print()

    out = {
        'model': os.path.basename(args.model),
        'window': int(window),
        'features': int(n_feat),
        'segment':  seg,
        'per_host': host,
        'lateral_hosts': {ip: int(n) for ip, n in lm_hosts.head(10).items()},
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
