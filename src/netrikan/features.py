"""Entity-centric state representation: one row per (internal host, wall-clock bucket).

A bucket is a fixed span of absolute time (default 60 s). A flow contributes to the bucket of
its start time, once for the host that sent it ("out") and once for the host that received it
("in"), when that host is inside the monitored (private) address space. Nothing is windowed over
the network-wide flow stream: every sequence belongs to exactly one host on one day.

No identifier reaches the model: IPs, raw ports and timestamps are used only to group and order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIVATE = r"^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)"
EPOCH = pd.Timestamp("1970-01-01")

WEB, SSH, DB = {80, 443, 8080, 8000, 8443}, {22}, {3306, 5432, 1433, 1521, 27017}
BASE = ["n_flows", "n_peers", "n_ports", "pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd",
        "dur_mean", "syn", "rst", "ack", "psh", "fin", "pktlen_mean",
        "f_web", "f_ssh", "f_db", "f_low", "f_eph"]
FEATURES = [f"{d}_{b}" for d in ("in", "out") for b in BASE]
LOG_FEATURES = [f for f in FEATURES if not f.split("_", 1)[1].startswith("f_")]
LOG_IDX = np.array([FEATURES.index(f) for f in LOG_FEATURES])

_HUMAN = {
    "n_flows": "flows", "n_peers": "distinct peers", "n_ports": "distinct dst ports",
    "pkts_fwd": "fwd packets", "pkts_bwd": "bwd packets", "bytes_fwd": "fwd bytes",
    "bytes_bwd": "bwd bytes", "dur_mean": "mean flow duration", "syn": "SYN flags",
    "rst": "RST flags", "ack": "ACK flags", "psh": "PSH flags", "fin": "FIN flags",
    "pktlen_mean": "mean packet length", "f_web": "share of web-port flows",
    "f_ssh": "share of SSH-port flows", "f_db": "share of database-port flows",
    "f_low": "share of other <1024-port flows", "f_eph": "share of ephemeral-port flows",
}


def human(feature: str) -> str:
    d, b = feature.split("_", 1)
    return f"{'inbound' if d == 'in' else 'outbound'} {_HUMAN[b]}"


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _bucket(df: pd.DataFrame, bucket_s: int) -> pd.Series:
    secs = (df["ts"] - EPOCH).dt.total_seconds()
    return (secs // bucket_s).astype("int64")


def flow_events(df: pd.DataFrame, bucket_s: int = 60, monitored: str = PRIVATE) -> pd.DataFrame:
    """Duplicate every flow into an outbound row for its source and an inbound row for its
    destination, keeping only hosts inside the monitored address space."""
    b = df.assign(bucket=_bucket(df, bucket_s), dur=df["Flow Duration"] / 1e6)
    port = b["Dst Port"]
    b["f_web"] = port.isin(WEB).astype(float)
    b["f_ssh"] = port.isin(SSH).astype(float)
    b["f_db"] = port.isin(DB).astype(float)
    known = b["f_web"] + b["f_ssh"] + b["f_db"] > 0
    b["f_low"] = ((port < 1024) & ~known).astype(float)
    b["f_eph"] = ((port >= 1024) & ~known).astype(float)
    out = b[b["Src IP"].str.match(monitored)].assign(host=lambda d: d["Src IP"], peer=lambda d: d["Dst IP"], dir="out")
    inn = b[b["Dst IP"].str.match(monitored)].assign(host=lambda d: d["Dst IP"], peer=lambda d: d["Src IP"], dir="in")
    return pd.concat([out, inn], ignore_index=True)


def build_grid(df: pd.DataFrame, bucket_s: int = 60, monitored: str = PRIVATE, labels: tuple = ("stage",)) -> pd.DataFrame:
    """Dense per-(host, day) bucket grid of raw aggregates plus the worst stage seen in the bucket.
    Buckets with no traffic inside a host's active span are explicit zero rows, so sequence
    distance equals wall-clock distance."""
    ev = flow_events(df, bucket_s, monitored)
    labels = list(labels)
    agg = ev.groupby(["host", "dir", "bucket"]).agg(
        n_flows=("bucket", "size"), n_peers=("peer", "nunique"), n_ports=("Dst Port", "nunique"),
        pkts_fwd=("Total Fwd Packet", "sum"), pkts_bwd=("Total Bwd packets", "sum"),
        bytes_fwd=("Total Length of Fwd Packet", "sum"), bytes_bwd=("Total Length of Bwd Packet", "sum"),
        dur_mean=("dur", "mean"), syn=("SYN Flag Count", "sum"), rst=("RST Flag Count", "sum"),
        ack=("ACK Flag Count", "sum"), psh=("PSH Flag Count", "sum"), fin=("FIN Flag Count", "sum"),
        pktlen_mean=("Packet Length Mean", "mean"), f_web=("f_web", "mean"), f_ssh=("f_ssh", "mean"),
        f_db=("f_db", "mean"), f_low=("f_low", "mean"), f_eph=("f_eph", "mean"))
    wide = agg.unstack("dir", fill_value=0)
    wide.columns = [f"{d}_{b}" for b, d in wide.columns]
    for f in FEATURES:
        if f not in wide.columns:
            wide[f] = 0.0
    wide = wide[FEATURES].join(ev.groupby(["host", "bucket"])[labels].max().astype(int)).reset_index()
    wide["day"] = (wide["bucket"] * bucket_s) // 86400
    parts = []
    for (host, day), g in wide.groupby(["host", "day"], sort=True):
        full = pd.RangeIndex(g["bucket"].min(), g["bucket"].max() + 1, name="bucket")
        g = g.drop(columns=["host", "day"]).set_index("bucket").reindex(full, fill_value=0).reset_index()
        g["host"], g["day"] = host, day
        parts.append(g)
    grid = pd.concat(parts, ignore_index=True)
    grid[labels] = grid[labels].astype(int)
    return grid[["host", "day", "bucket"] + labels + FEATURES]


@dataclass
class Table:
    host: np.ndarray
    day: np.ndarray
    bucket: np.ndarray
    X: np.ndarray          # (N, F) signed-log1p features, unscaled
    stage: np.ndarray      # (N,) worst stage in the bucket, 0 = benign
    attack: np.ndarray     # (N,) bool
    fut: np.ndarray        # (N,) bool, any attack in the next K buckets
    fut_stage: np.ndarray  # (N,) worst stage in the next K buckets
    valid: np.ndarray      # (N,) bool, the full K-bucket horizon exists
    idx: np.ndarray        # (N, L) row indices of the L-bucket history, -1 = before sequence start
    start: np.ndarray      # (N,) first row of this row's (host, day) sequence
    end: np.ndarray        # (N,) one past the last row of the sequence
    K: int
    L: int
    bucket_s: int

    @property
    def n(self) -> int:
        return len(self.host)


def make_table(grid: pd.DataFrame, K: int = 5, L: int = 10, bucket_s: int = 60) -> Table:
    grid = grid.sort_values(["host", "day", "bucket"], kind="stable").reset_index(drop=True)
    n = len(grid)
    key = grid["host"].astype(str) + "|" + grid["day"].astype(str)
    gid = key.ne(key.shift()).cumsum().to_numpy() - 1
    glen = np.bincount(gid)
    gstart = np.concatenate([[0], np.cumsum(glen)[:-1]])
    start, end = gstart[gid], (gstart + glen)[gid]
    i = np.arange(n)
    stage = grid["stage"].to_numpy()
    attack = stage > 0
    cs = np.concatenate([[0], np.cumsum(attack)])
    hi = np.minimum(i + K, end - 1)
    fut = (cs[hi + 1] - cs[i + 1]) > 0
    valid = (i + K) <= (end - 1)
    fut_stage = np.zeros(n, dtype=int)
    for k in range(1, K + 1):
        j = i + k
        fut_stage = np.maximum(fut_stage, np.where(j < end, stage[np.minimum(j, n - 1)], 0))
    idx = i[:, None] - (L - 1 - np.arange(L))[None, :]
    idx = np.where(idx >= start[:, None], idx, -1).astype(np.int32)
    X = grid[FEATURES].to_numpy(dtype=np.float64)
    X[:, LOG_IDX] = signed_log1p(X[:, LOG_IDX])
    return Table(grid["host"].to_numpy(), grid["day"].to_numpy(), grid["bucket"].to_numpy(), X.astype(np.float32),
                 stage, attack, fut & valid, fut_stage * valid, valid, idx, start, end, K, L, bucket_s)
