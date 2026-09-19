import numpy as np
from attck_map import INITIAL_ACCESS, DOS, LATERAL, C2, EXFIL, BENIGN

PORT_NAMES = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 135: "RPC", 139: "NetBIOS", 143: "IMAP",
    443: "HTTPS", 445: "SMB", 993: "IMAPS", 1433: "MSSQL", 1521: "Oracle",
    1337: "backdoor", 3306: "MySQL", 3389: "RDP", 4444: "Meterpreter",
    5432: "PostgreSQL", 5900: "VNC", 8080: "HTTP-alt",
}

F = {name: i for i, name in enumerate([
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
])}


def _svc(ports):
    if len(ports) == 0:
        return None, None
    vals, counts = np.unique(ports, return_counts=True)
    top = int(vals[np.argmax(counts)])
    return top, PORT_NAMES.get(top, f"port {top}")


def detect(raw_window: np.ndarray, ports: np.ndarray = None,
           pkt: dict = None) -> dict:
    """
    raw_window: (WINDOW, 24) unscaled feature rows for one window
    ports:      (WINDOW,) destination ports, optional
    pkt:        optional packet-level stats for this window, keys matching
                pcap_ingest.PACKET_FEATURES. Only available for PCAP input --
                CIC-IDS-2018 ships flow records, which cannot supply TTL,
                window size, fragmentation or retransmission counts.
    returns     {name, score, stage_hint, evidence}
    """
    w = raw_window
    m = w.mean(axis=0)

    dur      = m[F['Flow Duration']]
    fwd_pkts = m[F['Tot Fwd Pkts']]
    bwd_len  = m[F['TotLen Bwd Pkts']]
    fwd_len  = m[F['TotLen Fwd Pkts']]
    byts_s   = m[F['Flow Byts/s']]
    pkts_s   = m[F['Flow Pkts/s']]
    iat_mean = m[F['Flow IAT Mean']]
    iat_std  = m[F['Flow IAT Std']]
    syn      = m[F['SYN Flag Cnt']]
    rst      = m[F['RST Flag Cnt']]
    fin      = m[F['FIN Flag Cnt']]
    psh      = m[F['PSH Flag Cnt']]

    ports = np.asarray(ports) if ports is not None else np.array([])
    n_uniq_ports = len(np.unique(ports)) if len(ports) else 0
    top_port, svc = _svc(ports)
    svc_label = svc or "service"

    cands = []

    # ---- reconnaissance / scanning ----
    if n_uniq_ports >= 3 and dur < 50000 and syn >= 3:
        breadth = min(n_uniq_ports / 10.0, 1.0)
        name = f"{svc} port scan" if svc and n_uniq_ports < 8 else "Port sweep"
        cands.append((name, 0.55 + 0.35 * breadth, INITIAL_ACCESS,
                      f"{n_uniq_ports} distinct ports, {syn:.0f} SYN/flow, {dur/1000:.0f}ms flows"))

    if top_port in (445, 139) and dur < 200000 and syn >= 3:
        cands.append(("SMB port scan", 0.88, INITIAL_ACCESS,
                      f"port 445 probing, {syn:.0f} SYN/flow, {rst:.0f} RST"))

    # ---- brute force ----
    if top_port in (22, 21, 23, 1337, 3389, 5900) and rst >= 1 and dur < 60000 and pkts_s > 50:
        cands.append((f"{svc_label} brute force", 0.85, INITIAL_ACCESS,
                      f"{rst:.0f} RST/flow on {svc_label}, {pkts_s:.0f} pkt/s, {dur/1000:.0f}ms flows"))

    if top_port in (80, 443, 8080) and dur < 40000 and psh >= 2 and pkts_s > 80:
        cands.append(("Web credential stuffing", 0.72, INITIAL_ACCESS,
                      f"{psh:.0f} PSH/flow on {svc_label}, {pkts_s:.0f} pkt/s"))

    # ---- denial of service ----
    if syn > 15 and pkts_s > 500 and dur < 20000:
        cands.append(("SYN flood", 0.92, DOS,
                      f"{syn:.0f} SYN/flow, {pkts_s:.0f} pkt/s, {byts_s/1e6:.1f} MB/s"))

    if byts_s > 400000 and fwd_pkts > 40:
        cands.append(("Volumetric flood", 0.84, DOS,
                      f"{byts_s/1e6:.1f} MB/s, {fwd_pkts:.0f} fwd pkts/flow"))

    if top_port in (80, 443) and dur > 500000 and pkts_s < 10 and iat_mean > 80000:
        cands.append(("Slow HTTP DoS", 0.78, DOS,
                      f"{dur/1e6:.1f}s flows, {pkts_s:.1f} pkt/s, held connections"))

    # ---- reverse shell / backdoor ports ----
    if top_port in (4444, 1337) and syn >= 1:
        cands.append((f"{svc_label} reverse shell attempt", 0.91, C2,
                      f"connection attempt to {svc_label} (port {top_port}), "
                      f"{syn:.0f} SYN/flow, {rst:.0f} RST"))

    # ---- command and control ----
    periodicity = iat_std / (iat_mean + 1e-6)
    if 0 < periodicity < 0.25 and 5000 < iat_mean < 150000 and fwd_len < 2000 and pkts_s < 40:
        cands.append(("C2 beaconing", 0.80, C2,
                      f"periodic callback (IAT jitter {periodicity:.2f}), {fwd_len:.0f}B payloads"))

    if top_port == 53 and byts_s > 80000 and fwd_len > 2000:
        cands.append(("DNS tunneling", 0.83, C2,
                      f"{byts_s/1000:.0f} KB/s over DNS, {fwd_len:.0f}B queries — abnormal for port 53"))

    # ---- lateral movement / exfil ----
    if top_port in (445, 139, 3389, 5985) and fwd_len > 3000 and dur > 100000:
        cands.append((f"{svc_label} lateral transfer", 0.76, LATERAL,
                      f"{fwd_len/1000:.0f}KB over {svc_label}, {dur/1e6:.1f}s sessions"))

    ratio = bwd_len / (fwd_len + 1.0)
    if dur > 400000 and bwd_len > 5000 and ratio > 2.5:
        cands.append(("Bulk outbound transfer", 0.81, EXFIL,
                      f"{ratio:.1f}:1 outbound ratio, {bwd_len/1000:.0f}KB per flow"))

    # ---- stealth / low-and-slow ----
    if iat_mean > 150000 and pkts_s < 15 and dur > 300000 and fin >= 1:
        cands.append(("Low-and-slow probing", 0.68, LATERAL,
                      f"{iat_mean/1000:.0f}ms between packets — evades rate thresholds"))

    # ---- packet-level rules (PCAP input only) --------------------------------
    if pkt:
        ttl_std = float(pkt.get('TTL Std', 0) or 0)
        ttl_mean = float(pkt.get('TTL Mean', 0) or 0)
        win_min = float(pkt.get('Win Size Min', -1) or 0)
        frag = float(pkt.get('Frag Count', 0) or 0)
        retrans = float(pkt.get('Retrans Count', 0) or 0)

        # packets claiming one conversation but arriving with very different hop
        # counts -- classic spoofing / injected-traffic indicator
        if ttl_std > 25 and ttl_mean > 0:
            cands.append(("TTL anomaly — possible spoofing", 0.74, INITIAL_ACCESS,
                          f"TTL varies by {ttl_std:.0f} within one flow "
                          f"(mean {ttl_mean:.0f}) — inconsistent hop counts"))

        # receiver advertising a zero window is out of buffer: the signature of
        # slow-read / connection-exhaustion attacks
        if win_min == 0 and dur > 400000:
            cands.append(("Zero-window exhaustion", 0.79, DOS,
                          f"receiver window hit 0 over {dur/1e6:.1f}s flows — "
                          f"buffer exhausted, connection held open"))

        if retrans >= 3 and pkts_s > 40:
            cands.append(("Retransmission storm", 0.70, DOS,
                          f"{retrans:.0f} retransmissions at {pkts_s:.0f} pkt/s — "
                          f"link saturated or receiver overwhelmed"))

        # fragmenting traffic is a long-standing way to slip past signature IDS
        if frag >= 2:
            cands.append(("IP fragmentation evasion", 0.72, INITIAL_ACCESS,
                          f"{frag:.0f} fragmented packets — signature evasion"))

    if not cands:
        return {"name": "Normal traffic", "score": 0.0, "stage_hint": BENIGN,
                "evidence": "no anomalous pattern in window"}

    cands.sort(key=lambda c: -c[1])
    name, score, hint, ev = cands[0]
    return {"name": name, "score": float(score), "stage_hint": hint, "evidence": ev,
            "alternatives": [{"name": c[0], "score": float(c[1])} for c in cands[1:4]]}
