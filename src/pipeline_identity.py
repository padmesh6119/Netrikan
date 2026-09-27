"""
Per-host dataset builder with packet-level features.

pipeline_v2.py windows CIC-IDS-2018's ML-ready CSVs, which have two structural
limits the problem statement cares about:

  1. No Src/Dst IP, so a window is a slice of network-wide time rather than one
     host's behaviour. The model can learn "at time T the network looked like
     this", never "host X moved from recon to lateral movement".
  2. No packet-level fields, so TTL variance, TCP window size, fragmentation and
     retransmissions -- the features that expose slow reconnaissance designed to
     stay under flow-level thresholds -- are absent from training entirely.

This builder fixes both by reading pcap_ingest.py output instead, which carries
Src IP, Dst IP and all eight PACKET_FEATURES. Windows are built inside a single
host's flow sequence and never span two hosts.

Label source, in order of precedence:
  1. a `Stage` or `stage_id` column already on the CSV (the DAPT path)
  2. --labels spec, a JSON list of attack windows (the re-extracted CIC path)
  3. otherwise everything is Benign, and the builder refuses to write

Usage:
    # 1. re-extract flows from raw PCAPs, preserving identity
    python3 src/pcap_ingest.py /data/cic/Wednesday.pcap /storage/cic-raw/Wednesday.csv

    # 2. build per-host windows
    python3 src/pipeline_identity.py --input-dir /storage/cic-raw \\
        --labels configs/cic_attack_windows.json \\
        --out /storage/netrikan-identity-w30 --window 30
"""

import argparse
import gc
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model import ONSET_HORIZONS
from pcap_ingest import FEATURES as FLOW_FEATURES, PACKET_FEATURES
from pipeline_v2 import LABEL_MAP, add_port_features, PORT_FEATURES, make_onset_labels

# Packet-level features join the model input here -- this is the whole point of
# this builder over pipeline_v2.
IDENTITY_FEATURES = FLOW_FEATURES + PACKET_FEATURES

MIN_FLOWS_PER_HOST = 40   # a host with fewer flows than this cannot fill a w30
                          # window plus enough context to be worth training on


def load_label_spec(path):
    """Attack windows as a JSON list. Each entry labels flows matching all of
    the fields it specifies:

        [{"stage": "Infilteration",
          "src": ["18.219.211.138"],
          "dst": ["172.31.69.24"],
          "start": "2018-02-28 10:50:00",
          "end":   "2018-02-28 12:05:00"}]

    `src`, `dst` and the time bounds are each optional; omitting one means "any".
    Flows matching no entry are Benign. Later entries win on overlap.
    """
    with open(path) as f:
        spec = json.load(f)
    out = []
    for e in spec:
        stage = e['stage']
        if stage not in LABEL_MAP:
            raise ValueError(f"stage {stage!r} not in pipeline_v2.LABEL_MAP; "
                             f"known: {sorted(LABEL_MAP)}")
        out.append({
            'stage_id': LABEL_MAP[stage],
            'stage': stage,
            'src': set(e.get('src') or []),
            'dst': set(e.get('dst') or []),
            'start': pd.Timestamp(e['start']) if e.get('start') else None,
            'end': pd.Timestamp(e['end']) if e.get('end') else None,
        })
    return out


def apply_labels(df, spec):
    """Assign stage_id from the label spec. Default Benign (0)."""
    stage = np.zeros(len(df), dtype=np.int64)
    ts = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
    src = df['Src IP'].astype(str)
    dst = df['Dst IP'].astype(str)
    for e in spec:
        m = np.ones(len(df), dtype=bool)
        if e['src']:
            m &= src.isin(e['src']).to_numpy()
        if e['dst']:
            m &= dst.isin(e['dst']).to_numpy()
        if e['start'] is not None:
            m &= (ts >= e['start']).to_numpy()
        if e['end'] is not None:
            m &= (ts <= e['end']).to_numpy()
        stage[m] = e['stage_id']
    return stage


def load_input_dir(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not paths:
        raise SystemExit(f"no CSVs in {input_dir}")
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        missing = [c for c in IDENTITY_FEATURES + ['Src IP', 'Dst IP', 'Timestamp']
                   if c not in d.columns]
        if missing:
            raise SystemExit(f"{p} is missing {missing}. This builder reads "
                             f"pcap_ingest.py output, not CIC ML-ready CSVs.")
        d['source_file'] = os.path.basename(p)
        frames.append(d)
        print(f"  {os.path.basename(p)}: {len(d):,} flows", flush=True)
    return pd.concat(frames, ignore_index=True)


def build_host_windows(df, features, window, min_flows=MIN_FLOWS_PER_HOST):
    """Sliding windows inside each Src IP's own time-ordered flow sequence.

    Returns (X, y, host_ids, host_names). A window never spans two hosts, which
    is what makes the sequence a single attacker's trajectory rather than a
    slice of unrelated network activity.
    """
    Xs, ys, hs = [], [], []
    names, skipped = [], 0
    for host, grp in df.groupby('Src IP', sort=True):
        if len(grp) < min_flows:
            skipped += 1
            continue
        g = grp.sort_values('_ts', kind='mergesort')
        v = (g[features].apply(pd.to_numeric, errors='coerce')
             .replace([np.inf, -np.inf], np.nan).fillna(0.0)
             .to_numpy(np.float32))
        s = g['stage_id'].to_numpy(np.int64)
        if len(v) <= window:
            skipped += 1
            continue
        hid = len(names)
        names.append(str(host))
        for i in range(len(v) - window):
            Xs.append(v[i:i + window])
            ys.append(s[i + window])      # the flow AFTER the window: forecasting
            hs.append(hid)
    print(f"  {len(names):,} hosts kept, {skipped:,} skipped "
          f"(<{min_flows} flows or shorter than window)", flush=True)
    if not Xs:
        raise SystemExit("no host produced a full window; lower --min-flows "
                         "or check the input")
    return (np.asarray(Xs, np.float32), np.asarray(ys, np.int64),
            np.asarray(hs, np.int64), names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True,
                    help='directory of pcap_ingest.py CSVs (with Src IP / packet features)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--labels', default=None,
                    help='JSON attack-window spec; omit if the CSVs carry Stage')
    ap.add_argument('--with-ports', action='store_true',
                    help='add the service-role port features from pipeline_v2')
    ap.add_argument('--min-flows', type=int, default=MIN_FLOWS_PER_HOST)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    print(f"reading {args.input_dir}", flush=True)
    df = load_input_dir(args.input_dir)
    print(f"total {len(df):,} flows, {df['Src IP'].nunique():,} unique Src IP", flush=True)

    # ---- labels ----
    if 'stage_id' in df.columns:
        df['stage_id'] = pd.to_numeric(df['stage_id'], errors='coerce').fillna(0).astype(np.int64)
        print("labels: existing stage_id column", flush=True)
    elif 'Stage' in df.columns:
        df['stage_id'] = df['Stage'].map(LABEL_MAP).fillna(0).astype(np.int64)
        print("labels: mapped from Stage column via pipeline_v2.LABEL_MAP", flush=True)
    elif args.labels:
        df['stage_id'] = apply_labels(df, load_label_spec(args.labels))
        print(f"labels: applied {args.labels}", flush=True)
    else:
        raise SystemExit(
            "no label source. The CSVs carry no Stage/stage_id column and no "
            "--labels spec was given, so every flow would be Benign and the "
            "resulting dataset would be unusable. Supply --labels.")

    counts = np.bincount(df['stage_id'].to_numpy(), minlength=5)
    print(f"class counts: {dict(enumerate(counts.tolist()))}", flush=True)
    if counts[1:].sum() == 0:
        raise SystemExit("every flow labelled Benign — check the label spec's "
                         "IPs and time ranges against the data")

    features = list(IDENTITY_FEATURES)
    if args.with_ports:
        df = add_port_features(df)
        features += PORT_FEATURES

    df['_ts'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
    df = df.dropna(subset=['_ts'])

    print(f"building per-host windows: window={args.window} features={len(features)}",
          flush=True)
    X, y, host_ids, host_names = build_host_windows(
        df, features, args.window, args.min_flows)
    del df
    gc.collect()
    print(f"{len(X):,} windows x {args.window} x {len(features)}", flush=True)

    # Scale only the continuous features; the port indicators are already 0/1.
    # Fit on the flattened windows, matching pipeline_v2's convention.
    cont = [f for f in features if f not in PORT_FEATURES]
    cont_idx = [features.index(f) for f in cont]
    flat = X.reshape(-1, len(features))
    scaler = StandardScaler()
    flat[:, cont_idx] = scaler.fit_transform(flat[:, cont_idx]).astype(np.float32)
    X = flat.reshape(X.shape)

    np.save(os.path.join(args.out, 'X.npy'), X)
    np.save(os.path.join(args.out, 'y.npy'), y)
    np.save(os.path.join(args.out, 'host_ids.npy'), host_ids)
    with open(os.path.join(args.out, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out, 'features.txt'), 'w') as f:
        f.write("\n".join(features))
    with open(os.path.join(args.out, 'hosts.json'), 'w') as f:
        json.dump(host_names, f, indent=2)

    # Onset labels are computed per host: a stage change across a host boundary
    # is not an onset, it is two different machines. Computing them globally
    # would manufacture a spurious positive at every boundary.
    #
    # Arrays are written at full len(X) and aligned index-for-index with X, with
    # -1 marking rows that have no valid label (the last max(k) windows of each
    # host). train_v2 drops -1 rows rather than treating them as negatives.
    max_k = max(ONSET_HORIZONS)
    onset = {k: np.full(len(y), -1, np.int8) for k in ONSET_HORIZONS}
    n_valid = 0
    for hid in range(len(host_names)):
        idx = np.flatnonzero(host_ids == hid)
        if len(idx) <= max_k:
            continue
        o, n_h = make_onset_labels(y[idx], ONSET_HORIZONS)
        for k in ONSET_HORIZONS:
            onset[k][idx[:n_h]] = o[k]
        n_valid += n_h
    for k, o in onset.items():
        np.save(os.path.join(args.out, f'onset_k{k}.npy'), o)
    rates = {k: round(float(o[o >= 0].mean()), 4) for k, o in onset.items()}
    print(f"onset labels: {n_valid:,}/{len(y):,} rows labelled "
          f"({len(y) - n_valid:,} marked -1 at host tails), "
          f"positive rate {rates}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
