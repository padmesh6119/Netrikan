import os
import glob
import numpy as np
import pandas as pd

DAPT_DIR = os.path.expanduser("~/netrikan/data/dapt2020")

# CICFlowMeter v4 (DAPT) -> v3 names (CIC-IDS-2018 / our pipeline)
RENAME = {
    'Total Fwd Packet': 'Tot Fwd Pkts',
    'Total Bwd packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packet': 'TotLen Fwd Pkts',
    'Total Length of Bwd Packet': 'TotLen Bwd Pkts',
    'Fwd Packet Length Max': 'Fwd Pkt Len Max',
    'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
    'Bwd Packet Length Max': 'Bwd Pkt Len Max',
    'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
    'Flow Bytes/s': 'Flow Byts/s',
    'Flow Packets/s': 'Flow Pkts/s',
    'FIN Flag Count': 'FIN Flag Cnt',
    'SYN Flag Count': 'SYN Flag Cnt',
    'RST Flag Count': 'RST Flag Cnt',
    'PSH Flag Count': 'PSH Flag Cnt',
    'ACK Flag Count': 'ACK Flag Cnt',
    'Packet Length Variance': 'Pkt Len Var',
}

# DAPT's own kill-chain phases, in campaign order
DAPT_STAGES = ['Benign', 'Reconnaissance', 'Establish Foothold',
               'Lateral Movement', 'Data Exfiltration']
DAPT_ID = {s: i for i, s in enumerate(DAPT_STAGES)}

STAGE_ALIASES = {
    'benign': 'Benign',
    'reconnaissance': 'Reconnaissance',
    'establish foothold': 'Establish Foothold',
    'lateral movement': 'Lateral Movement',
    'data exfiltration': 'Data Exfiltration',
}

# which weekday each capture belongs to, for campaign ordering
DAY_ORDER = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}


def _day_of(fname):
    low = os.path.basename(fname).lower()
    for d, i in DAY_ORDER.items():
        if d in low:
            return d, i
    return 'unknown', 99


def load_file(path, ref_cols=None):
    has_header = open(path).readline().split(',')[0].strip() == 'Flow ID'
    if has_header:
        df = pd.read_csv(path, low_memory=False)
    else:
        if ref_cols is None:
            raise ValueError(f"{path} has no header and no reference columns given")
        df = pd.read_csv(path, low_memory=False, header=None, names=ref_cols)

    df = df.rename(columns=RENAME)
    df['Stage'] = (df['Stage'].astype(str).str.strip().str.lower()
                   .map(STAGE_ALIASES))
    df = df[df['Stage'].notna()].copy()
    df['stage_id'] = df['Stage'].map(DAPT_ID).astype(int)

    day, order = _day_of(path)
    df['day'] = day
    df['day_order'] = order
    df['capture'] = os.path.basename(path)
    df['is_private'] = int(('pvt' in os.path.basename(path).lower()))
    return df


def load_all(directory=DAPT_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if not files:
        raise FileNotFoundError(f"no CSVs in {directory}")

    ref = None
    for f in files:
        if open(f).readline().split(',')[0].strip() == 'Flow ID':
            ref = list(pd.read_csv(f, nrows=1).columns)
            break

    frames = [load_file(f, ref) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df['ts'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
    return df


def transition_counts(df, within='capture'):
    """
    Count stage->stage transitions between consecutive flows, ordered by time
    inside each capture. Transitions are never counted across capture boundaries.
    """
    n = len(DAPT_STAGES)
    C = np.zeros((n, n), dtype=np.int64)
    for _, grp in df.groupby(within, sort=False):
        g = grp.sort_values('ts', kind='mergesort')
        s = g['stage_id'].to_numpy()
        if len(s) < 2:
            continue
        np.add.at(C, (s[:-1], s[1:]), 1)
    return C


def normalise(C, smoothing=1.0):
    M = C.astype(np.float64) + smoothing
    return M / M.sum(axis=1, keepdims=True)
