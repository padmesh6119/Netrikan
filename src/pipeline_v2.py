import os
import gc
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.expanduser("~/netrikan/data/cic-ids2018")

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
        if with_ports:
            df = add_port_features(df)
        keep = [c for c in features if c in df.columns] + ['stage']
        df = df[keep]
        for c in features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        frames.append(df)
        print(f"    {len(df):,} rows", flush=True)
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--out', required=True)
    ap.add_argument('--with-ports', action='store_true')
    args = ap.parse_args()

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
    del df
    gc.collect()

    W = args.window
    n = len(vals) - W
    print(f"writing {n:,} windows x {W} x {len(features)}", flush=True)

    # written straight to a memmap: materialising this as a Python list would
    # need several times the array's size in RAM
    X = np.lib.format.open_memmap(os.path.join(args.out, 'X.npy'), mode='w+',
                                  dtype=np.float32, shape=(n, W, len(features)))
    CH = 200_000
    for i in range(0, n, CH):
        j = min(i + CH, n)
        for t in range(W):
            X[i:j, t, :] = vals[i + t:j + t, :]
        print(f"    {j:,}/{n:,}", flush=True)
    X.flush()
    del X
    np.save(os.path.join(args.out, 'y.npy'), labels[W:W + n])

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
