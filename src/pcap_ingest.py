import os
import shutil
import subprocess
import numpy as np
import pandas as pd

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

PACKET_FEATURES = ['TTL Mean', 'TTL Std', 'Win Size Mean', 'Win Size Min',
                   'Frag Count', 'Payload Len Mean', 'Payload Len Std',
                   'Retrans Count']


def _nfstream_available() -> bool:
    try:
        import nfstream  # noqa: F401
        return True
    except ImportError:
        return False


def _load_via_nfstream(path: str, min_packets: int) -> pd.DataFrame:
    from nfstream import NFStreamer
    raw = NFStreamer(source=path, statistical_analysis=True,
                    n_dissections=0).to_pandas()
    if raw.empty:
        return pd.DataFrame(columns=FEATURES + PACKET_FEATURES +
                            ['Dst Port', 'Protocol', 'Timestamp', 'Src IP', 'Dst IP'])

    raw = raw[raw['bidirectional_packets'] >= min_packets].copy()
    dur_s = (raw['bidirectional_duration_ms'] / 1000.0).clip(lower=1e-3)
    dur_us = dur_s * 1e6

    out = pd.DataFrame(index=raw.index)
    out['Flow Duration']    = dur_us
    out['Tot Fwd Pkts']     = raw['src2dst_packets']
    out['Tot Bwd Pkts']     = raw['dst2src_packets']
    out['TotLen Fwd Pkts']  = raw['src2dst_bytes']
    out['TotLen Bwd Pkts']  = raw['dst2src_bytes']
    out['Fwd Pkt Len Max']  = raw['src2dst_max_ps']
    out['Fwd Pkt Len Mean'] = raw['src2dst_mean_ps']
    out['Bwd Pkt Len Max']  = raw['dst2src_max_ps']
    out['Bwd Pkt Len Mean'] = raw['dst2src_mean_ps']
    out['Flow Byts/s']      = raw['bidirectional_bytes'] / dur_s
    out['Flow Pkts/s']      = raw['bidirectional_packets'] / dur_s
    out['Flow IAT Mean']    = raw['bidirectional_mean_piat_ms'] * 1000
    out['Flow IAT Std']     = raw['bidirectional_stddev_piat_ms'] * 1000
    out['Flow IAT Max']     = raw['bidirectional_max_piat_ms'] * 1000
    out['Fwd IAT Mean']     = raw['src2dst_mean_piat_ms'] * 1000
    out['Bwd IAT Mean']     = raw['dst2src_mean_piat_ms'] * 1000
    out['FIN Flag Cnt']     = raw['bidirectional_fin_packets']
    out['SYN Flag Cnt']     = raw['bidirectional_syn_packets']
    out['RST Flag Cnt']     = raw['bidirectional_rst_packets']
    out['PSH Flag Cnt']     = raw['bidirectional_psh_packets']
    out['ACK Flag Cnt']     = raw['bidirectional_ack_packets']
    out['Pkt Len Var']      = raw['bidirectional_stddev_ps'] ** 2
    out['Active Mean']      = 0.0
    out['Idle Mean']        = 0.0

    for col in PACKET_FEATURES:
        out[col] = 0.0

    out['Dst Port']   = raw['dst_port']
    out['Protocol']   = raw['protocol']
    out['Timestamp']  = (pd.to_datetime(raw['bidirectional_first_seen_ms'], unit='ms')
                         .dt.strftime('%d/%m/%Y %H:%M:%S'))
    out['Src IP']     = raw['src_ip']
    out['Dst IP']     = raw['dst_ip']

    return out.sort_values('Timestamp').reset_index(drop=True)


# ── tshark fallback (kept for packet-level features when nfstream unavailable) ─

TSHARK_FIELDS = ['frame.time_epoch', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
                 'udp.srcport', 'udp.dstport', 'ip.proto', 'frame.len', 'tcp.flags',
                 'ip.ttl', 'tcp.window_size_value', 'ip.frag_offset',
                 'tcp.analysis.retransmission', 'tcp.len']

IDLE_GAP = 1.0
FLAGS = {'FIN': 0x01, 'SYN': 0x02, 'RST': 0x04, 'PSH': 0x08, 'ACK': 0x10, 'URG': 0x20}


def _read_packets_tshark(path: str) -> pd.DataFrame:
    if not shutil.which('tshark'):
        raise RuntimeError("tshark not found")
    cmd = ['tshark', '-r', path, '-T', 'fields', '-E', 'separator=,']
    for f in TSHARK_FIELDS:
        cmd += ['-e', f]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    rows = []
    for line in res.stdout.splitlines():
        p = line.split(',')
        if len(p) < 10:
            continue
        ts, src, dst, tsp, tdp, usp, udp_, proto, ln, flags = p[:10]
        ttl, win, frag, retrans, plen = (p[10:15] + [''] * 5)[:5]
        if not src or not dst or not ts:
            continue
        sport = tsp or usp
        dport = tdp or udp_

        def _i(v, d=0):
            try:
                return int(str(v).split(',')[0])
            except (ValueError, TypeError):
                return d

        try:
            rows.append((float(ts), src, dst,
                         _i(sport), _i(dport), _i(proto), _i(ln),
                         int(flags, 16) if flags else 0,
                         _i(ttl, -1), _i(win, -1), _i(frag, 0),
                         1 if str(retrans).strip() else 0, _i(plen, 0)))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=['ts', 'src', 'dst', 'sport', 'dport',
                                       'proto', 'len', 'flags', 'ttl', 'win',
                                       'frag', 'retrans', 'plen'])


def _stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return (0.0, 0.0) if a.size == 0 else (float(a.max()), float(a.mean()))


def _iat(times):
    if len(times) < 2:
        return 0.0, 0.0, 0.0
    d = np.diff(np.asarray(times, dtype=np.float64)) * 1e6
    return float(d.mean()), float(d.std()), float(d.max())


def _active_idle(times):
    if len(times) < 2:
        return 0.0, 0.0
    t = np.asarray(times, dtype=np.float64)
    gaps = np.diff(t)
    idle = gaps[gaps > IDLE_GAP]
    actives, start = [], t[0]
    for i, g in enumerate(gaps):
        if g > IDLE_GAP:
            actives.append(t[i] - start)
            start = t[i + 1]
    actives.append(t[-1] - start)
    actives = [a for a in actives if a > 0]
    return (float(np.mean(actives) * 1e6) if actives else 0.0,
            float(np.mean(idle) * 1e6) if idle.size else 0.0)


def _load_via_tshark(path: str, min_packets: int) -> pd.DataFrame:
    df = _read_packets_tshark(path)
    if df.empty:
        return pd.DataFrame(columns=FEATURES + PACKET_FEATURES +
                            ['Dst Port', 'Protocol', 'Timestamp'])
    a = list(zip(df['src'], df['sport']))
    b = list(zip(df['dst'], df['dport']))
    key = [(x, y, p) if x <= y else (y, x, p) for x, y, p in zip(a, b, df['proto'])]
    df = df.assign(_key=key)
    out = []
    for _, g in df.groupby('_key', sort=False):
        if len(g) < min_packets:
            continue
        g = g.sort_values('ts', kind='mergesort')
        first_src, first_sport = g.iloc[0]['src'], g.iloc[0]['sport']
        fwd = (g['src'] == first_src) & (g['sport'] == first_sport)
        t = g['ts'].to_numpy()
        dur_s = float(t[-1] - t[0])
        dur_us = max(dur_s * 1e6, 1.0)
        fl, bl = g['len'][fwd].to_numpy(), g['len'][~fwd].to_numpy()
        f_max, f_mean = _stats(fl)
        b_max, b_mean = _stats(bl)
        iat_m, iat_s, iat_x = _iat(t)
        f_iat, _, _ = _iat(g['ts'][fwd].to_numpy())
        b_iat, _, _ = _iat(g['ts'][~fwd].to_numpy())
        act, idl = _active_idle(t)
        fbits = g['flags'].to_numpy()
        counts = {n: int(((fbits & m) > 0).sum()) for n, m in FLAGS.items()}
        ttl_v = g['ttl'][g['ttl'] >= 0].to_numpy(dtype=np.float64)
        win_v = g['win'][g['win'] >= 0].to_numpy(dtype=np.float64)
        pl_v = g['plen'].to_numpy(dtype=np.float64)
        out.append({
            'Flow Duration': dur_us,
            'Tot Fwd Pkts': int(fwd.sum()), 'Tot Bwd Pkts': int((~fwd).sum()),
            'TotLen Fwd Pkts': float(fl.sum()), 'TotLen Bwd Pkts': float(bl.sum()),
            'Fwd Pkt Len Max': f_max, 'Fwd Pkt Len Mean': f_mean,
            'Bwd Pkt Len Max': b_max, 'Bwd Pkt Len Mean': b_mean,
            'Flow Byts/s': float(g['len'].sum()) / max(dur_s, 1e-3),
            'Flow Pkts/s': len(g) / max(dur_s, 1e-3),
            'Flow IAT Mean': iat_m, 'Flow IAT Std': iat_s, 'Flow IAT Max': iat_x,
            'Fwd IAT Mean': f_iat, 'Bwd IAT Mean': b_iat,
            'FIN Flag Cnt': counts['FIN'], 'SYN Flag Cnt': counts['SYN'],
            'RST Flag Cnt': counts['RST'], 'PSH Flag Cnt': counts['PSH'],
            'ACK Flag Cnt': counts['ACK'], 'URG Flag Cnt': counts['URG'],
            'Pkt Len Var': float(np.var(g['len'].to_numpy(dtype=np.float64))),
            'Active Mean': act, 'Idle Mean': idl,
            'Fwd/Bwd Byte Ratio': float(fl.sum()) / (float(bl.sum()) + 1.0),
            'Fwd/Bwd Pkt Ratio': float(fwd.sum()) / (float((~fwd).sum()) + 1.0),
            'TTL Mean': float(ttl_v.mean()) if ttl_v.size else 0.0,
            'TTL Std': float(ttl_v.std()) if ttl_v.size else 0.0,
            'Win Size Mean': float(win_v.mean()) if win_v.size else 0.0,
            'Win Size Min': float(win_v.min()) if win_v.size else 0.0,
            'Frag Count': int((g['frag'].to_numpy() > 0).sum()),
            'Payload Len Mean': float(pl_v.mean()) if pl_v.size else 0.0,
            'Payload Len Std': float(pl_v.std()) if pl_v.size else 0.0,
            'Retrans Count': int(g['retrans'].sum()),
            'Dst Port': int(g.iloc[0]['dport']), 'Protocol': int(g.iloc[0]['proto']),
            'Timestamp': pd.to_datetime(t[0], unit='s').strftime('%d/%m/%Y %H:%M:%S'),
            '_start': t[0], 'Src IP': g.iloc[0]['src'], 'Dst IP': g.iloc[0]['dst'],
        })
    flows = pd.DataFrame(out)
    if flows.empty:
        return flows
    return flows.sort_values('_start').drop(columns=['_start']).reset_index(drop=True)


def load_pcap(path: str, min_packets: int = 2) -> pd.DataFrame:
    if _nfstream_available():
        return _load_via_nfstream(path, min_packets)
    return _load_via_tshark(path, min_packets)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    f = load_pcap(path)
    print(f"{len(f)} flows")
    if out_path:
        f.to_csv(out_path, index=False)
        print(f"saved -> {out_path}")
    else:
        print(f.head(8).to_string())
