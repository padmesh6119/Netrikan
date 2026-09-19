import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

DATA_DIR = os.path.expanduser("~/netrikan/data/cic-ids2018")
OUT_DIR = os.path.expanduser("~/netrikan/data/processed")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
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

LABEL_MAP = {
    'Benign':                   0,
    'FTP-BruteForce':           1,
    'SSH-Bruteforce':           1,
    'Brute Force -Web':         1,
    'Brute Force -XSS':         1,
    'SQL Injection':            1,
    'DoS attacks-Hulk':         2,
    'DoS attacks-SlowHTTPTest': 2,
    'DoS attacks-GoldenEye':    2,
    'DoS attacks-Slowloris':    2,
    'Infilteration':            3,
    'Bot':                      4,
}

WINDOW = 10


def load_all():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    frames = []
    for fname in files:
        print(f"Loading {fname}...")
        df = pd.read_csv(os.path.join(DATA_DIR, fname), low_memory=False)
        df = df[df['Label'] != 'Label']
        df = df[df['Label'].isin(LABEL_MAP)]
        df['stage'] = df['Label'].map(LABEL_MAP)
        df = df[FEATURES + ['stage']]
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        frames.append(df)
        print(f"  {len(df)} rows, stages: {df['stage'].value_counts().to_dict()}")
    return pd.concat(frames, ignore_index=True)


def make_windows(df):
    X, y = [], []
    vals = df[FEATURES].values
    labels = df['stage'].values
    total = len(vals) - WINDOW
    for i in range(0, total, 1):
        X.append(vals[i:i + WINDOW])
        y.append(labels[i + WINDOW])
        if i % 500000 == 0:
            print(f"  windowing {i}/{total}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    print("=== Loading CSVs ===")
    df = load_all()
    print(f"\nTotal rows: {len(df)}")
    print(f"Stage distribution:\n{df['stage'].value_counts().sort_index()}\n")

    print("=== Normalising features ===")
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    print("=== Creating sliding windows ===")
    X, y = make_windows(df)
    print(f"X shape: {X.shape}  y shape: {y.shape}")

    print("=== Saving ===")
    np.save(os.path.join(OUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUT_DIR, 'y.npy'), y)
    with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
