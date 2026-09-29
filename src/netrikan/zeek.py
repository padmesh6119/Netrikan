"""UWF-ZeekData24 loader: Zeek conn logs mapped onto the flow columns the feature pipeline uses.

Per-flow technique labels become five booleans. Rows labelled `Duplicate` are the same flows as
T1078 (labelled again under Initial Access) and are dropped; flows carrying several techniques keep
all of them. Zeek has no TCP flag counters, so they are derived from the `history` string
(SYN = S/s/H/h, ACK = A/a, FIN = F/f, RST = R/r). Zeek has no PSH flag either: PSH is proxied by
data-carrying segments (D/d) and is only a proxy.
"""
from __future__ import annotations

import glob
import os

import duckdb
import numpy as np
import pandas as pd

TECHS = ["T1595", "T1190", "T1078", "T1110", "T1048"]
TECH_LABEL = [f"t_{t}" for t in TECHS]
TECH_NAME = {
    "T1595": "Active Scanning",
    "T1190": "Exploit Public-Facing App",
    "T1078": "Valid Accounts",
    "T1110": "Brute Force",
    "T1048": "Exfil Over Alt. Protocol",
}
TECH_TACTIC = {
    "T1595": "Reconnaissance",
    "T1190": "Initial Access",
    "T1078": "Defense Evasion / Persistence / Priv. Esc.",
    "T1110": "Credential Access",
    "T1048": "Exfiltration",
}
MONITORED = r"^143\.88\."  # the UWF lab network; every host in the data lives here
PROTO = {"tcp": 6, "udp": 17, "icmp": 1}


def load_z24(root: str = "data/UWF_Datasets/ZeekData24/parquet", weeks: list[str] | None = None) -> pd.DataFrame:
    files = [f for f in sorted(glob.glob(os.path.join(root, "*.parquet")))
             if weeks is None or any(f"zeekdata24__{w}" in os.path.basename(f) for w in weeks)]
    if not files:
        raise FileNotFoundError(f"no ZeekData24 parquet files under {root}")
    con = duckdb.connect()
    con.execute("set TimeZone='UTC'")
    tech_sql = ",\n".join(f"bool_or(label_technique='{t}') AS t_{t}" for t in TECHS)
    df = con.execute(f"""
        select regexp_extract(filename,'zeekdata24__([0-9-]+)',1) as wk, uid,
               min(ts) as ts_epoch, any_value(src_ip_zeek) as "Src IP", any_value(dest_ip_zeek) as "Dst IP",
               any_value(dest_port_zeek) as dport, any_value(proto) as proto, any_value(duration) as dur,
               any_value(orig_pkts) as op, any_value(resp_pkts) as rp, any_value(orig_bytes) as ob, any_value(resp_bytes) as rb,
               any_value(orig_ip_bytes) as oib, any_value(resp_ip_bytes) as rib, any_value(history) as history,
               {tech_sql}
        from (select distinct * from read_parquet({files!r}, filename=true) where label_technique <> 'Duplicate')
        group by wk, uid
    """).fetchdf()
    return _to_flows(df)


def _to_flows(df: pd.DataFrame) -> pd.DataFrame:
    h = df["history"].fillna("")
    out = pd.DataFrame({
        "wk": df["wk"], "uid": df["uid"],
        "ts": pd.to_datetime(df["ts_epoch"], unit="s"),
        "Src IP": df["Src IP"], "Dst IP": df["Dst IP"],
        "Dst Port": df["dport"].fillna(0).astype(float), "Protocol": df["proto"].map(PROTO).fillna(0),
        "Flow Duration": df["dur"].fillna(0) * 1e6,
        "Total Fwd Packet": df["op"].fillna(0), "Total Bwd packets": df["rp"].fillna(0),
        "Total Length of Fwd Packet": df["ob"].fillna(0), "Total Length of Bwd Packet": df["rb"].fillna(0),
        "SYN Flag Count": h.str.count("[SsHh]"), "ACK Flag Count": h.str.count("[Aa]"),
        "FIN Flag Count": h.str.count("[Ff]"), "RST Flag Count": h.str.count("[Rr]"),
        "PSH Flag Count": h.str.count("[Dd]"),  # proxy: data-carrying segments
    })
    pk = (df["op"].fillna(0) + df["rp"].fillna(0)).to_numpy(dtype=float)
    by = (df["oib"].fillna(0) + df["rib"].fillna(0)).to_numpy(dtype=float)
    out["Packet Length Mean"] = np.divide(by, pk, out=np.zeros_like(by), where=pk > 0)
    for t in TECHS:
        out[f"t_{t}"] = df[f"t_{t}"].fillna(False).astype(bool)
    out["stage"] = 0
    out["labelled"] = True
    out["attack"] = out[TECH_LABEL].any(axis=1)
    return out.sort_values("ts", kind="stable").reset_index(drop=True)
