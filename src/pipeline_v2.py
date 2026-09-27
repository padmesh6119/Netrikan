import os
import gc
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model import ONSET_HORIZONS

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'cic-ids2018')

BASE_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt',
    'Pkt Len Var', 'Active Mean', 'Idle Mean',
]

# Service-role features derived from Dst Port. Raw port number is useless as a
# scalar (443 and 445 are adjacent numerically, unrelated semantically), so it
# is expanded into the roles that matter for lateral movement.
PORT_GROUPS = {
    'svc_web':      {80, 443, 8080, 8443, 8000},
    'svc_smb':      {139, 445},
    'svc_rdp':      {3389},
    'svc_winrm':    {5985, 5986},
    'svc_remote':   {22, 23, 21},
    'svc_dns':      {53},
    'svc_db':       {1433, 3306, 5432, 1521, 27017},
    'svc_mail':     {25, 110, 143, 993, 995, 587},
}
PORT_FEATURES = list(PORT_GROUPS) + ['svc_ephemeral', 'svc_wellknown', 'proto_tcp', 'proto_udp']

LABEL_MAP = {
    'Benign': 0,
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Brute Force -Web': 1,
    'Brute Force -XSS': 1, 'SQL Injection': 1,
    'DoS attacks-Hulk': 2, 'DoS attacks-SlowHTTPTest': 2,
    'DoS attacks-GoldenEye': 2, 'DoS attacks-Slowloris': 2,
    # distributed variants share ATT&CK TA0040 (Impact) with single-source DoS,
    # so they join class 2 rather than forcing a 6th output class
    'DDOS attack-HOIC': 2, 'DDOS attack-LOIC-UDP': 2,
    'DDoS attacks-LOIC-HTTP': 2, 'DDOS attack-LOIC-HTTP': 2,
    'Infilteration': 3,
    'Bot': 4,
}


def add_port_features(df):
    port = pd.to_numeric(df['Dst Port'], errors='coerce').fillna(0).astype(np.int32)
    proto = pd.to_numeric(df['Protocol'], errors='coerce').fillna(0).astype(np.int16)
    for name, ports in PORT_GROUPS.items():
        df[name] = port.isin(ports).astype(np.float32)
    df['svc_ephemeral'] = (port >= 49152).astype(np.float32)
    df['svc_wellknown'] = ((port > 0) & (port < 1024)).astype(np.float32)
    df['proto_tcp'] = (proto == 6).astype(np.float32)
    df['proto_udp'] = (proto == 17).astype(np.float32)
    return df


def load_all(features, with_ports):
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.csv'))
    frames = []
    for fname in files:
        print(f"  reading {fname}", flush=True)
        df = pd.read_csv(os.path.join(DATA_DIR, fname), low_memory=False)
        df = df[df['Label'] != 'Label']
        df = df[df['Label'].isin(LABEL_MAP)]
        if not len(df):
            continue
        df['stage'] = df['Label'].map(LABEL_MAP).astype(np.int64)

        # Sort by timestamp. The CSVs as published are NOT in time order, and
        # windowing them as stored builds sequences that are not sequences: on
        # the 2018-02-14 file, row order yields 3,926 label transitions where
        # time order yields 23,948. Every temporal claim depends on this sort.
        if 'Timestamp' in df.columns:
            ts = pd.to_datetime(df['Timestamp'], errors='coerce',
                                format='mixed', dayfirst=True)
            n_bad = int(ts.isna().sum())
            # a handful of rows carry unparseable/epoch timestamps; they cannot be
            # placed in the sequence, so they are dropped rather than sorted to
            # the front where they would corrupt the first windows
            if n_bad:
                print(f"    dropping {n_bad:,} rows with unparseable timestamps",
                      flush=True)
            df = df.loc[ts.notna()].copy()
            df['_ts'] = ts[ts.notna()]
            df = df.sort_values('_ts', kind='mergesort').drop(columns=['_ts'])
        else:
            print("    WARNING: no Timestamp column; rows are windowed in file "
                  "order, which may not be time order", flush=True)

        if with_ports:
            df = add_port_features(df)
        keep = [c for c in features if c in df.columns] + ['stage']
        df = df[keep]
        for c in features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # file id, so windowing can respect day boundaries: a window must not
        # span the end of one capture day and the start of the next
        df['_file'] = len(frames)
        frames.append(df)
        print(f"    {len(df):,} rows", flush=True)
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def make_onset_labels(labels, horizons):
    """onset_k[t] = 1 if labels[t + k] != labels[t] else 0.

    Every horizon is trimmed to the same length (len(labels) - max(horizons)) so
    all onset arrays index identically against the first n rows of X and y. A
    per-horizon length would silently misalign the shorter ones.
    """
    labels = np.asarray(labels)
    max_k = max(horizons)
    n = len(labels) - max_k
    if n <= 0:
        raise ValueError(f"need more than {max_k} windows to build onset labels, "
                         f"got {len(labels)}")
    return {k: (labels[k:k + n] != labels[:n]).astype(np.int8)
            for k in horizons}, n


def _config_window(path=None):
    """Window size from configs/train_v2.yaml, so the config reproduces the
    shipped checkpoint rather than only documenting an intent."""
    cfg_path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'train_v2.yaml')
    try:
        import yaml
        with open(cfg_path) as f:
            return int((yaml.safe_load(f) or {}).get('window', 30))
    except Exception:
        return 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=None,
                    help='overrides window in configs/train_v2.yaml')
    ap.add_argument('--config', default=None,
                    help='path to configs/train_v2.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--with-ports', action='store_true')
    args = ap.parse_args()
    if args.window is None:
        args.window = _config_window(args.config)

    t0 = time.time()
    features = BASE_FEATURES + (PORT_FEATURES if args.with_ports else [])
    os.makedirs(args.out, exist_ok=True)
    print(f"features={len(features)}  window={args.window}  out={args.out}", flush=True)

    df = load_all(features, args.with_ports)
    print(f"total rows {len(df):,}", flush=True)
    print(df['stage'].value_counts().sort_index().to_string(), flush=True)

    # scaler is fit on flow rows; only the continuous base features are scaled,
    # the port/protocol indicators are already 0/1
    scaler = StandardScaler()
    df[BASE_FEATURES] = scaler.fit_transform(df[BASE_FEATURES]).astype(np.float32)
    with open(os.path.join(args.out, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out, 'features.txt'), 'w') as f:
        f.write("\n".join(features))

    vals = df[features].to_numpy(np.float32)
    labels = df['stage'].to_numpy(np.int64)
    file_id = df['_file'].to_numpy(np.int64)
    del df
    gc.collect()

    W = args.window
    max_k = max(ONSET_HORIZONS)

    # Per-file window starts. A window [s, s+W) and its label at s+W and its onset
    # horizon s+W+k must all stay inside one capture file, so a window never spans
    # a day boundary and an onset never compares two different days.
    starts = []
    b = 0
    N = len(vals)
    while b < N:
        e = b
        while e < N and file_id[e] == file_id[b]:
            e += 1
        # need s+W (label) valid, i.e. s <= e-W-1
        starts.append(np.arange(b, max(b, e - W)))
        b = e
    starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
    n = len(starts)
    print(f"writing {n:,} windows x {W} x {len(features)} "
          f"(from {file_id.max()+1} capture file(s), boundaries respected)",
          flush=True)

    # float16 on disk: every reader casts to float32 on load (WindowDataset,
    # eval_dapt, transition_eval, calibration, baseline, rollout_eval), and the
    # features are already standardised (~mean 0, std 1), so fp16 loses nothing
    # that matters. It halves the file, which is what lets all days fit on disk.
    X = np.lib.format.open_memmap(os.path.join(args.out, 'X.npy'), mode='w+',
                                  dtype=np.float16, shape=(n, W, len(features)))
    CH = 200_000
    for i in range(0, n, CH):
        j = min(i + CH, n)
        s = starts[i:j]
        for t in range(W):
            X[i:j, t, :] = vals[s + t, :].astype(np.float16)
        print(f"    {j:,}/{n:,}", flush=True)
    X.flush()
    del X
    y = labels[starts + W]
    np.save(os.path.join(args.out, 'y.npy'), y)

    # per-window capture-file id (aligned to X/y): lets evaluators keep events
    # inside a capture day. Cheap and small; every temporal bench reads it.
    file_at = file_id[starts + W]     # which file that label came from
    np.save(os.path.join(args.out, 'file_id.npy'), file_at)

    # Onset labels per file: onset_k[j] = 1 if the stage at window j's label
    # position changes within k more windows, AND that horizon stays in the same
    # file. Rows whose horizon would cross a file boundary are marked -1 (train_v2
    # drops them) rather than inventing a cross-day transition.
    onset = {k: np.full(n, -1, np.int8) for k in ONSET_HORIZONS}
    label_at = y                      # stage at each window's label position
    for k in ONSET_HORIZONS:
        # compare window j to window j+k; valid only if j+k in range and same file
        hi = n - k
        same = np.zeros(n, dtype=bool)
        same[:hi] = file_at[:hi] == file_at[k:k + hi]
        chg = np.zeros(n, dtype=np.int8)
        chg[:hi] = (label_at[:hi] != label_at[k:k + hi]).astype(np.int8)
        col = np.full(n, -1, np.int8)
        col[:hi][same[:hi]] = chg[:hi][same[:hi]]
        onset[k] = col
        np.save(os.path.join(args.out, f'onset_k{k}.npy'), col)
    rates = {k: round(float(o[o >= 0].mean()), 4) if (o >= 0).any() else None
             for k, o in onset.items()}
    n_valid = int((np.stack([onset[k] for k in ONSET_HORIZONS], 1) >= 0).all(1).sum())
    print(f"onset labels: {n_valid:,}/{n:,} rows valid across all horizons; "
          f"positive rate {rates}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
