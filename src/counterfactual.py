"""
Counterfactual response engine.

Each intervention transforms the flow DataFrame to simulate a network control
being applied, then infer.analyze() is re-run. The delta in peak breach risk
is the intervention's measurable effect.

No new model required — works with the existing fused inference pipeline.
"""

import numpy as np
import pandas as pd

SMB_PORTS   = {139, 445}
ADMIN_PORTS = {139, 445, 3389, 5985, 5986}

_ZERO_COLS = [
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Flow Byts/s', 'Flow Pkts/s',
    'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'Flow Duration', 'Pkt Len Var',
]

INTERVENTIONS = [
    {
        'name':        'block_smb',
        'label':       'Block SMB (ports 139/445)',
        'description': 'Firewall drops all traffic to SMB — the primary lateral movement channel.',
    },
    {
        'name':        'block_admin',
        'label':       'Block all admin ports (SMB + RDP + WinRM)',
        'description': 'Drops SMB (139/445), RDP (3389) and WinRM (5985/5986).',
    },
    {
        'name':        'isolate',
        'label':       'Isolate suspected host',
        'description': 'Drops all flows from the highest-SYN source. Uses Src IP when available.',
    },
    {
        'name':        'rate_limit',
        'label':       'Rate-limit to 1 Mbps',
        'description': 'Caps throughput — effective against floods, not port-based intrusion.',
    },
]


def _port_mask(df: pd.DataFrame, ports: set) -> pd.Series:
    if 'Dst Port' not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df['Dst Port'], errors='coerce').isin(ports)


def _zero_flows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    df = df.copy()
    for col in _ZERO_COLS:
        if col in df.columns:
            df.loc[mask, col] = 0.0
    if 'Dst Port' in df.columns:
        df.loc[mask, 'Dst Port'] = 0
    return df


def apply(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == 'block_smb':
        return _zero_flows(df, _port_mask(df, SMB_PORTS))

    if name == 'block_admin':
        return _zero_flows(df, _port_mask(df, ADMIN_PORTS))

    if name == 'isolate':
        if 'Src IP' in df.columns and df['Src IP'].nunique() > 1:
            syn = 'SYN Flag Cnt' if 'SYN Flag Cnt' in df.columns else None
            top = (df.groupby('Src IP')[syn].sum().idxmax()
                   if syn else df['Src IP'].value_counts().index[0])
            mask = df['Src IP'] == top
        else:
            mask = _port_mask(df, ADMIN_PORTS)
        return _zero_flows(df, mask)

    if name == 'rate_limit':
        df = df.copy()
        cap_bps, cap_pps = 125_000.0, 800.0
        if 'Flow Byts/s' in df.columns:
            df['Flow Byts/s'] = df['Flow Byts/s'].clip(upper=cap_bps)
        if 'Flow Pkts/s' in df.columns:
            df['Flow Pkts/s'] = df['Flow Pkts/s'].clip(upper=cap_pps)
        if 'TotLen Fwd Pkts' in df.columns and 'Flow Duration' in df.columns:
            dur_s = (df['Flow Duration'] / 1e6).clip(lower=1e-3)
            cap = cap_bps * dur_s
            df['TotLen Fwd Pkts'] = df['TotLen Fwd Pkts'].clip(upper=cap)
            df['TotLen Bwd Pkts'] = df['TotLen Bwd Pkts'].clip(upper=cap)
        return df

    return df.copy()


def run_all(df: pd.DataFrame, horizon_seconds: float, analyze_fn) -> list:
    """
    Runs all interventions and returns list of result dicts sorted by risk_after.
    analyze_fn = infer.analyze, passed in to avoid circular import.
    """
    base = analyze_fn(df, horizon_seconds)
    if base is None:
        return []
    risk_before = float(np.max(base['breach']))

    results = []
    for iv in INTERVENTIONS:
        modified = apply(df, iv['name'])
        r = analyze_fn(modified, horizon_seconds)
        risk_after = float(np.max(r['breach'])) if r is not None else risk_before
        results.append({
            **iv,
            'risk_before': risk_before,
            'risk_after':  round(risk_after, 3),
            'delta':       round(risk_after - risk_before, 3),
            'sufficient':  risk_after < 0.30,
        })

    results.sort(key=lambda x: x['risk_after'])
    return results
