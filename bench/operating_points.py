"""
Operating-point curve for cross-dataset detection (CIC-IDS-2018 -> DAPT 2020).

WHY THIS EXISTS. `src/eval_dapt.py` reports one number: recall 67.8% at FPR
40.4%. That FPR is not a tuned operating point -- it is `stage_logits.argmax() > 0`,
the 5-class head's hard decision, with no threshold anywhere. Reporting a single
hard decision as "our false positive rate" is indefensible when the breach head
emits a calibrated [0,1] score that can be swept, so this script sweeps it and
reports precision/recall at fixed FPR the way a detector should be reported.

It also measures two suppressors that cost nothing at inference and that any real
deployment would use:

  min_persist  A window is flagged only if it belongs to a run of >= k
               CONSECUTIVE above-threshold windows inside the same capture (and
               the same host, in host mode). Single-window flicker is the bulk of
               the FP mass; `bench/lead_time.py` already uses this idea for
               warnings, so the detection path should get the same treatment.

  entity roll-up  An operator does not read windows, they read alerts about
               hosts. Windows are grouped into (capture, Src IP, time bucket)
               cells; a cell is alerted if it contains any flagged window. The
               reported rate is then alerts per BENIGN host-hour and recall over
               ATTACK host-hours -- the numbers an operator actually lives with.
               Per-window FPR and per-host-hour alert rate are different
               quantities and this file never conflates them.

Scores compared, because which score you threshold matters and the shipped path
uses neither as a score:
  breach   sigmoid(breach_head) -- trained as "next flow is malicious"
  stage    1 - softmax(stage_logits)[Benign] -- attack mass in the 5-class head

Definitions kept fixed:
  - A window is ATTACK if its label (the flow AFTER the window) is non-benign.
  - FPR is computed over benign windows only; recall over attack windows only.
  - Runs never span a capture boundary, and in host mode never span a host.
  - Every reported operating point carries its ACHIEVED FPR next to the target,
    because the exact target is rarely attainable on a finite sample.

    python3 bench/operating_points.py --data "$DATA" --model models/cic_v2_w30.pt \
        --out models/operating_points_dapt_cic_v2_w30.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'bench'))

import dapt                                                    # noqa: E402
from model import WorldModel                                    # noqa: E402
from pipeline_v2 import BASE_FEATURES, PORT_FEATURES, add_port_features  # noqa: E402
from sedi import sedi                                           # noqa: E402

TARGET_FPRS = [0.001, 0.01, 0.05, 0.10]


def load_feature_list(data_dir):
    p = os.path.join(data_dir, 'features.txt')
    if os.path.exists(p):
        return [l.strip() for l in open(p) if l.strip()]
    return list(BASE_FEATURES)


def build_windows(df, features, window):
    """Windows plus the provenance needed for run-length and entity roll-up.

    Returns X, y, cap_id, host, ts. `host` and `ts` are taken from the LABELLED
    row (i + window), i.e. the flow the window is predicting, so an alert is
    attributed to the host and time it concerns rather than to the window's
    first row.
    """
    Xs, ys, caps, hosts, tss = [], [], [], [], []
    has_ip = 'Src IP' in df.columns
    for ci, (_, grp) in enumerate(df.groupby('capture', sort=False)):
        g = grp.sort_values('ts', kind='mergesort')
        v = g[features].apply(pd.to_numeric, errors='coerce')
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        s = g['stage_id'].to_numpy()
        ip = g['Src IP'].astype(str).to_numpy() if has_ip else np.full(len(g), '?')
        t = g['ts'].to_numpy()
        if len(v) <= window:
            continue
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])
            caps.append(ci)
            hosts.append(ip[i + window])
            tss.append(t[i + window])
    return (np.asarray(Xs, np.float32), np.asarray(ys),
            np.asarray(caps, np.int64), np.asarray(hosts),
            np.asarray(tss))


def persist_mask(flag, group, k):
    """Keep only windows inside a run of >= k consecutive True values in `flag`,
    where runs are broken by any change in `group` (capture, or capture+host).

    k=1 returns `flag` unchanged, which is the no-suppression baseline.
    """
    if k <= 1:
        return flag.copy()
    out = np.zeros_like(flag)
    n = len(flag)
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1] and group[j + 1] == group[i]:
            j += 1
        if (j - i + 1) >= k:
            out[i:j + 1] = True
        i = j + 1
    return out


def counts(pred, truth):
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(recall, 4), "fpr": round(fpr, 4),
            "precision": round(prec, 4), "f1": round(f1, 4),
            "sedi": round(sedi(recall, fpr), 4)}


def threshold_for_fpr(score, truth, target, group, k):
    """Smallest threshold whose ACHIEVED benign-window FPR is <= target, after
    min_persist suppression. Suppression changes the FP count, so the threshold
    has to be searched against the suppressed prediction, not the raw score."""
    lo, hi = 0.0, 1.0
    best = (1.0, None)
    for _ in range(40):
        mid = (lo + hi) / 2
        pred = persist_mask(score >= mid, group, k)
        c = counts(pred, truth)
        if c['fpr'] <= target:
            best = (mid, c)
            hi = mid
        else:
            lo = mid
    return best


def entity_rollup(pred, truth, cap, host, ts, bucket='1h'):
    """Group windows into (capture, host, time bucket) cells.

    A cell is ALERTED if any window in it is flagged, and is ATTACK if any window
    in it is an attack. Reports alerts per benign host-bucket (the operator's
    false-alarm rate) and recall over attack host-buckets.
    """
    b = pd.Series(pd.to_datetime(ts)).dt.floor(bucket).to_numpy()
    key = pd.MultiIndex.from_arrays([cap, host, b])
    d = pd.DataFrame({'pred': pred, 'truth': truth}, index=key)
    g = d.groupby(level=[0, 1, 2]).max()
    a_cells = int(g['truth'].sum())
    b_cells = int((~g['truth'].astype(bool)).sum())
    hit = int((g['pred'] & g['truth']).sum())
    fa = int((g['pred'] & ~g['truth'].astype(bool)).sum())
    return {"bucket": bucket,
            "attack_cells": a_cells, "benign_cells": b_cells,
            "attack_cells_detected": hit,
            "entity_recall": round(hit / a_cells, 4) if a_cells else 0.0,
            "false_alarm_cells": fa,
            "false_alarm_rate_per_benign_cell": round(fa / b_cells, 4) if b_cells else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='built CIC dir (scaler + features)')
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--persist', type=int, nargs='+', default=[1, 2, 3, 5])
    ap.add_argument('--bucket', default='1h')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    features = load_feature_list(args.data)
    Xh = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    window, n_feat = Xh.shape[1], Xh.shape[2]
    del Xh
    assert n_feat == len(features), f"feature mismatch: {n_feat} vs {len(features)}"

    df = dapt.load_all()
    if any(f in features for f in PORT_FEATURES):
        df = add_port_features(df)
    X_raw, y, cap, host, ts = build_windows(df, features, window)
    print(f"windows {len(X_raw):,}  captures {len(np.unique(cap))}  "
          f"hosts {len(np.unique(host))}", flush=True)

    with open(os.path.join(args.data, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    base_idx = [features.index(f) for f in BASE_FEATURES]
    flat = X_raw.reshape(-1, n_feat).copy()
    flat[:, base_idx] = scaler.transform(flat[:, base_idx])
    X = flat.reshape(X_raw.shape).astype(np.float32)

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=n_feat, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    breach, stage_attack, argmax_attack = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits, b, _, _ = model(torch.from_numpy(X[i:i + 8192]))
            p = torch.softmax(logits, 1)
            breach.append(b.numpy())
            stage_attack.append((1.0 - p[:, 0]).numpy())
            argmax_attack.append((logits.argmax(1) > 0).numpy())
    scores = {'breach': np.concatenate(breach),
              'stage': np.concatenate(stage_attack)}
    argmax_attack = np.concatenate(argmax_attack)
    truth = y > 0

    # group keys for run-length: runs break at a capture change (segment mode) or
    # at a capture-or-host change (host mode)
    g_cap = cap
    g_host = np.array([f"{c}|{h}" for c, h in zip(cap, host)])

    out = {
        "model": args.model, "data": args.data,
        "n_windows": int(len(X)), "n_attack_windows": int(truth.sum()),
        "n_benign_windows": int((~truth).sum()),
        "n_captures": int(len(np.unique(cap))), "n_hosts": int(len(np.unique(host))),
        "shipped_operating_point": {
            "what": "stage_logits.argmax() > 0 -- the decision eval_dapt.py reports, "
                    "a hard 5-class argmax with no threshold and so no way to trade "
                    "recall against FPR",
            **counts(argmax_attack, truth),
        },
        "curves": {}, "entity_rollup": {},
    }
    print("\nshipped (argmax):", out['shipped_operating_point'], flush=True)

    for sname, score in scores.items():
        out['curves'][sname] = {}
        for k in args.persist:
            rows = []
            for tgt in TARGET_FPRS:
                thr, c = threshold_for_fpr(score, truth, tgt, g_cap, k)
                if c is None:
                    rows.append({"target_fpr": tgt, "threshold": None,
                                 "note": "no threshold reaches this FPR"})
                    continue
                rows.append({"target_fpr": tgt, "threshold": round(thr, 5), **c})
                print(f"  {sname:<7} persist={k}  FPR<={tgt:<6} thr={thr:.4f}  "
                      f"recall={c['recall']:.4f} prec={c['precision']:.4f} "
                      f"achieved_fpr={c['fpr']:.4f} sedi={c['sedi']:.4f}", flush=True)
            out['curves'][sname][f'persist{k}'] = rows

    # Is the argmax decision better RANKED than the continuous scores, or just
    # sitting at a looser threshold? Match each score to the argmax's achieved
    # FPR and compare recall there. If the argmax wins at matched FPR it is using
    # information the breach head does not, and thresholding cannot recover it.
    amx = out['shipped_operating_point']
    out['matched_fpr_vs_argmax'] = {
        "argmax_fpr": amx['fpr'], "argmax_recall": amx['recall'],
        "argmax_sedi": amx['sedi'], "scores": {}}
    for sname, score in scores.items():
        thr, c = threshold_for_fpr(score, truth, amx['fpr'], g_cap, 1)
        out['matched_fpr_vs_argmax']['scores'][sname] = (
            {"threshold": round(thr, 5), **c} if c else {"note": "unreachable"})
        if c:
            print(f"  matched-FPR {sname:<7} fpr={c['fpr']:.4f} recall={c['recall']:.4f} "
                  f"(argmax {amx['recall']:.4f})  sedi={c['sedi']:.4f}", flush=True)

    # The shipped argmax decision rolled up the same way, so the entity numbers
    # below are compared against what already ships rather than against nothing.
    out['entity_rollup'] = {"argmax": entity_rollup(
        argmax_attack, truth, cap, host, ts, args.bucket)}
    print("  entity argmax  "
          f"entity_recall={out['entity_rollup']['argmax']['entity_recall']:.4f}  "
          f"false_alarm/benign_cell="
          f"{out['entity_rollup']['argmax']['false_alarm_rate_per_benign_cell']:.4f}",
          flush=True)

    # entity roll-up at the 5% per-window operating point, for each score, with
    # and without suppression -- the comparison that shows what an operator sees
    for sname, score in scores.items():
        out['entity_rollup'][sname] = {}
        for k in args.persist:
            thr, c = threshold_for_fpr(score, truth, 0.05, g_host, k)
            if c is None:
                continue
            pred = persist_mask(score >= thr, g_host, k)
            r = entity_rollup(pred, truth, cap, host, ts, args.bucket)
            out['entity_rollup'][sname][f'persist{k}'] = {
                "window_threshold": round(thr, 5), "window_level": c, **r}
            print(f"  entity {sname:<7} persist={k}  entity_recall={r['entity_recall']:.4f}  "
                  f"false_alarm/benign_cell={r['false_alarm_rate_per_benign_cell']:.4f}  "
                  f"({r['false_alarm_cells']} of {r['benign_cells']})", flush=True)

    out['finding'] = (
        "The shipped argmax decision is one point on a curve that was never drawn. "
        "Compare its FPR against the swept operating points at matched recall "
        "before quoting any single false-positive number.")
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
