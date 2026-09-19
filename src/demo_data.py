import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

META = ['Dst Port', 'Protocol', 'Timestamp']

rng = np.random.default_rng(7)


def _row(dur, fwd_p, bwd_p, fwd_l, bwd_l, fmax, fmean, bmax, bmean,
         byts, pkts, iatm, iats, iatx, fiat, biat,
         fin, syn, rst, psh, ack, plvar, active, idle):
    return [dur, fwd_p, bwd_p, fwd_l, bwd_l, fmax, fmean, bmax, bmean,
            byts, pkts, iatm, iats, iatx, fiat, biat,
            fin, syn, rst, psh, ack, plvar, active, idle]


def _benign(n):
    rows, ports = [], []
    for _ in range(n):
        rows.append(_row(
            rng.integers(80000, 600000), rng.integers(6, 30), rng.integers(5, 26),
            rng.integers(800, 6000), rng.integers(600, 5000),
            rng.integers(300, 1500), rng.uniform(120, 600),
            rng.integers(300, 1500), rng.uniform(100, 500),
            rng.uniform(4000, 40000), rng.uniform(8, 60),
            rng.uniform(20000, 150000), rng.uniform(60000, 200000),
            rng.uniform(80000, 400000), rng.uniform(20000, 150000), rng.uniform(20000, 150000),
            rng.integers(0, 2), rng.integers(1, 3), 0,
            rng.integers(2, 10), rng.integers(6, 22),
            rng.uniform(2000, 25000), rng.uniform(60000, 250000), rng.uniform(150000, 600000),
        ))
        ports.append(int(rng.choice([80, 443, 443, 443, 53, 993, 25])))
    return rows, ports


def _smb_scan(n):
    """Port sweep across SMB/NetBIOS — short probes, high SYN, RST rejections."""
    rows, ports = [], []
    sweep = [445, 139, 135, 445, 139, 3389, 445, 1433, 445, 5985]
    for i in range(n):
        rows.append(_row(
            rng.integers(2000, 30000), rng.integers(2, 5), rng.integers(0, 3),
            rng.integers(120, 600), rng.integers(0, 300),
            rng.integers(60, 220), rng.uniform(40, 120),
            rng.integers(40, 200), rng.uniform(10, 90),
            rng.uniform(4000, 25000), rng.uniform(90, 350),
            rng.uniform(800, 6000), rng.uniform(300, 2500), rng.uniform(3000, 15000),
            rng.uniform(800, 6000), rng.uniform(800, 6000),
            0, rng.integers(4, 10), rng.integers(1, 4),
            rng.integers(0, 2), rng.integers(1, 5),
            rng.uniform(300, 4000), rng.uniform(1500, 12000), rng.uniform(6000, 45000),
        ))
        ports.append(sweep[i % len(sweep)])
    return rows, ports


def _lateral(n):
    """SMB sessions moving tooling between internal hosts."""
    rows, ports = [], []
    for _ in range(n):
        rows.append(_row(
            rng.integers(300000, 1800000), rng.integers(15, 60), rng.integers(10, 45),
            rng.integers(4000, 25000), rng.integers(2000, 12000),
            rng.integers(600, 1500), rng.uniform(250, 700),
            rng.integers(500, 1400), rng.uniform(180, 550),
            rng.uniform(3000, 30000), rng.uniform(4, 25),
            rng.uniform(60000, 400000), rng.uniform(40000, 250000),
            rng.uniform(250000, 900000), rng.uniform(60000, 350000), rng.uniform(60000, 350000),
            rng.integers(1, 5), rng.integers(2, 7), rng.integers(0, 2),
            rng.integers(6, 22), rng.integers(10, 35),
            rng.uniform(8000, 60000), rng.uniform(250000, 900000), rng.uniform(120000, 500000),
        ))
        ports.append(int(rng.choice([445, 445, 445, 139, 3389, 5985])))
    return rows, ports


def _c2(n):
    """Periodic beaconing — low jitter, small consistent payloads."""
    rows, ports = [], []
    base_iat = 45000
    for _ in range(n):
        iatm = base_iat * rng.uniform(0.94, 1.06)
        rows.append(_row(
            rng.integers(50000, 250000), rng.integers(3, 10), rng.integers(3, 9),
            rng.integers(250, 1400), rng.integers(200, 1200),
            rng.integers(80, 300), rng.uniform(40, 140),
            rng.integers(80, 300), rng.uniform(35, 130),
            rng.uniform(1200, 12000), rng.uniform(4, 30),
            iatm, iatm * rng.uniform(0.05, 0.22), iatm * rng.uniform(1.5, 2.6),
            iatm * rng.uniform(0.9, 1.1), iatm * rng.uniform(0.9, 1.1),
            0, rng.integers(1, 3), 0,
            rng.integers(1, 6), rng.integers(4, 14),
            rng.uniform(400, 6000), rng.uniform(15000, 90000), rng.uniform(30000, 220000),
        ))
        ports.append(int(rng.choice([443, 443, 8080, 53])))
    return rows, ports


def _exfil(n):
    """Bulk outbound transfer — long flows, heavy egress ratio."""
    rows, ports = [], []
    for _ in range(n):
        fwd_l = rng.integers(400, 2500)
        rows.append(_row(
            rng.integers(600000, 3000000), rng.integers(10, 40), rng.integers(60, 250),
            fwd_l, int(fwd_l * rng.uniform(6, 20)),
            rng.integers(200, 800), rng.uniform(80, 300),
            rng.integers(1200, 1500), rng.uniform(700, 1400),
            rng.uniform(40000, 300000), rng.uniform(25, 120),
            rng.uniform(15000, 90000), rng.uniform(8000, 60000),
            rng.uniform(120000, 600000), rng.uniform(20000, 120000), rng.uniform(10000, 80000),
            rng.integers(1, 4), rng.integers(1, 4), 0,
            rng.integers(15, 50), rng.integers(40, 140),
            rng.uniform(30000, 200000), rng.uniform(400000, 1500000), rng.uniform(80000, 400000),
        ))
        ports.append(int(rng.choice([443, 443, 22, 53])))
    return rows, ports


def _dos(n):
    rows, ports = [], []
    for _ in range(n):
        rows.append(_row(
            rng.integers(200, 6000), rng.integers(60, 500), rng.integers(0, 4),
            rng.integers(3000, 30000), rng.integers(0, 300),
            rng.integers(40, 80), rng.uniform(40, 80),
            rng.integers(40, 80), rng.uniform(0, 40),
            rng.uniform(600000, 5000000), rng.uniform(1200, 9000),
            rng.uniform(20, 400), rng.uniform(10, 200), rng.uniform(600, 5000),
            rng.uniform(20, 400), rng.uniform(10000, 90000),
            rng.integers(0, 2), rng.integers(25, 90), rng.integers(0, 6),
            0, rng.integers(10, 50),
            rng.uniform(50, 600), rng.uniform(150, 2500), rng.uniform(500, 6000),
        ))
        ports.append(int(rng.choice([80, 80, 443])))
    return rows, ports


def _assemble(chunks, interval=1.2):
    rows, ports = [], []
    for r, p in chunks:
        rows.extend(r)
        ports.extend(p)
    df = pd.DataFrame(rows, columns=FEATURES)
    df['Dst Port'] = ports
    df['Protocol'] = 6
    t0 = datetime(2026, 9, 7, 13, 40, 0)
    df['Timestamp'] = [
        (t0 + timedelta(seconds=i * interval)).strftime('%d/%m/%Y %H:%M:%S')
        for i in range(len(df))
    ]
    return df


SCENARIOS = {
    "Intrusion — scan to exfiltration": lambda n: _assemble([
        _benign(int(n * 0.30)),
        _smb_scan(int(n * 0.22)),
        _lateral(int(n * 0.20)),
        _c2(int(n * 0.16)),
        _exfil(n - int(n * 0.30) - int(n * 0.22) - int(n * 0.20) - int(n * 0.16)),
    ]),
    "Denial of service": lambda n: _assemble([
        _benign(int(n * 0.45)),
        _dos(n - int(n * 0.45)),
    ]),
    "Benign baseline": lambda n: _assemble([_benign(n)]),
}


def generate(scenario: str, n_flows: int = 260, seed: int = 7) -> pd.DataFrame:
    global rng
    rng = np.random.default_rng(seed)   # reseed so a scenario is reproducible
    return SCENARIOS[scenario](n_flows)
