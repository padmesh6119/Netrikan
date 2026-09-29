"""Cross-corpus loaders for the zero-shot experiment. Every corpus is mapped onto the flow columns
`features.build_grid` needs, plus one boolean `fam_*` column per attack family present.

Common feature subset: only what all four corpora carry with the same meaning. CTU-13 (Argus) has no
TCP flag counters and no packet split, so packet counts, flags and packet length are excluded.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import zeek
from .campaign import log_state
from .dapt import load_dapt_dir, normalise
from .features import FEATURES, PRIVATE, build_grid

CORPORA = ("dapt", "z24", "cic17", "ctu13")
MONITORED = {"dapt": PRIVATE, "z24": zeek.MONITORED, "cic17": r"^192\.168\.10\.", "ctu13": r"^147\.32\."}
FAMILIES = {
    "dapt": {"fam_recon": "Reconnaissance", "fam_foothold": "Initial Access (foothold)", "fam_lateral": "Lateral Movement", "fam_exfil": "Exfiltration"},
    "z24": {"fam_scan": "Active Scanning", "fam_exploit": "Exploit Public-Facing App", "fam_validacct": "Valid Accounts", "fam_brute": "Brute Force", "fam_exfil": "Exfil Over Alt. Protocol"},
    "cic17": {"fam_portscan": "Port scan", "fam_ddos": "DDoS", "fam_heartbleed": "Heartbleed"},
    "ctu13": {"fam_botnet": "Botnet (spam / ICMP / C&C)"},
}
LABEL = {"dapt": "DAPT2020 (CICFlowMeter, 2019)", "z24": "UWF-ZeekData24 (Zeek, 2024)", "cic17": "CIC-IDS2017 slices (CICFlowMeter/NTLFlowLyzer, 2017)",
         "ctu13": "CTU-13 scenario 4 (Argus, 2011)"}
_UNSUPPORTED = ("pkts_fwd", "pkts_bwd", "syn", "rst", "ack", "psh", "fin", "pktlen_mean")
CF = [f for f in FEATURES if not any(f.endswith("_" + u) for u in _UNSUPPORTED)]
CF_IDX = np.array([FEATURES.index(f) for f in CF])


def load_cic17(root: str = "data/new-datasets") -> pd.DataFrame:
    parts = []
    for f in ("cic17_friday_scan_ddos.csv.gz", "cic17_wednesday_heartbleed.csv.gz"):
        d = pd.read_csv(os.path.join(root, f), low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        parts.append(d[d["Attempted Category"] == -1])  # drop flows the corrected labels mark as failed attempts
    n = normalise(pd.concat(parts, ignore_index=True))
    n["fam_portscan"], n["fam_ddos"], n["fam_heartbleed"] = n["Label"].eq("Portscan"), n["Label"].eq("DDoS"), n["Label"].eq("Heartbleed")
    return n


def load_ctu13(path: str = "data/new-datasets/ctu13_s04_c2_ddos.binetflow.gz") -> pd.DataFrame:
    d = pd.read_csv(path)
    lab = d["Label"].str.replace("flow=", "", regex=False)
    tot, src = d["TotBytes"].fillna(0), d["SrcBytes"].fillna(0)
    pk = d["TotPkts"].fillna(0)
    out = pd.DataFrame({
        "ts": pd.to_datetime(d["StartTime"], format="%Y/%m/%d %H:%M:%S.%f"), "Src IP": d["SrcAddr"], "Dst IP": d["DstAddr"],
        "Dst Port": pd.to_numeric(d["Dport"], errors="coerce").fillna(0), "Protocol": d["Proto"].map({"tcp": 6, "udp": 17, "icmp": 1}).fillna(0),
        "Flow Duration": d["Dur"].fillna(0) * 1e6, "Total Fwd Packet": pk, "Total Bwd packets": 0.0,
        "Total Length of Fwd Packet": src, "Total Length of Bwd Packet": (tot - src).clip(lower=0),
        "SYN Flag Count": 0.0, "RST Flag Count": 0.0, "ACK Flag Count": 0.0, "PSH Flag Count": 0.0, "FIN Flag Count": 0.0,
        "Packet Length Mean": np.divide(tot, pk, out=np.zeros(len(d)), where=pk.to_numpy() > 0)})
    out["fam_botnet"] = lab.str.contains("Botnet").to_numpy()
    out["stage"], out["labelled"] = 0, True
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def load_corpus(name: str) -> pd.DataFrame:
    if name == "dapt":
        df, _ = load_dapt_dir()
        for col, s in (("fam_recon", 1), ("fam_foothold", 2), ("fam_lateral", 3), ("fam_exfil", 4)):
            df[col] = df["stage"] == s
        return df
    if name == "z24":
        df = zeek.load_z24()
        for col, t in (("fam_scan", "T1595"), ("fam_exploit", "T1190"), ("fam_validacct", "T1078"), ("fam_brute", "T1110"), ("fam_exfil", "T1048")):
            df[col] = df[f"t_{t}"]
        return df
    if name == "cic17":
        return load_cic17()
    if name == "ctu13":
        return load_ctu13()
    raise ValueError(name)


def host_minutes(flows: pd.DataFrame, name: str) -> dict:
    """Common-feature host-minute rows with one boolean per attack family.

    X/fam/host/bucket: minutes with traffic only. Xd/Fd/gid/keep: the dense per-(host, day) grid (idle minutes
    included, rows contiguous per sequence) that the sequence model needs; X == Xd[keep]."""
    fam = list(FAMILIES[name])
    grid = build_grid(flows, 60, MONITORED[name], tuple(fam))
    keep = ((grid["in_n_flows"] + grid["out_n_flows"]) > 0).to_numpy()
    Xd = log_state(grid)[:, CF_IDX]
    Fd = grid[fam].to_numpy().astype(bool)
    key = grid["host"].astype(str) + "|" + grid["day"].astype(str)
    gid = key.ne(key.shift()).cumsum().to_numpy() - 1
    return {"X": Xd[keep], "fam": Fd[keep], "fam_names": fam, "host": grid["host"].to_numpy()[keep], "bucket": grid["bucket"].to_numpy()[keep],
            "Xd": Xd, "Fd": Fd, "gid": gid, "keep": keep, "host_d": grid["host"].to_numpy(), "bucket_d": grid["bucket"].to_numpy()}
