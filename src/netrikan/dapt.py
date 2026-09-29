"""Loading CICFlowMeter-style flow CSVs (DAPT2020 layout) into one normalised frame.

Handles the defects found in docs/DATA_AUDIT.md: a file with no header row, two spellings of
the benign label, 12-hour timestamps without timezone, unsorted rows, and the same flow seen
from the public and private capture points.
"""
from __future__ import annotations

import glob
import gzip
import io
import os

import pandas as pd

# DAPT stage -> ordinal along the kill chain. Ordinal 0 is benign.
STAGE_ORDER = {
    "benign": 0,
    "normal": 0,
    "reconnaissance": 1,
    "establish foothold": 2,
    "lateral movement": 3,
    "data exfiltration": 4,
}
# Display names. DAPT's "Establish Foothold" is the closest of the brief's five phases to
# ATT&CK Initial Access; the brief's Command & Control has no DAPT label at all.
STAGE_NAMES = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "Exfiltration"]

ALIASES = {
    "Source IP": "Src IP",
    "Destination IP": "Dst IP",
    "Source Port": "Src Port",
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Total Fwd Packet",
    "Total Backward Packets": "Total Bwd packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Avg Packet Size": "Average Packet Size",
}
REQUIRED = [
    "Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration",
    "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "PSH Flag Count", "FIN Flag Count", "Packet Length Mean",
]
IDENTITY = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration"]
TS_FORMATS = ["%Y-%m-%d %H:%M:%S.%f", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]


def _parse_ts(s: pd.Series) -> pd.Series:
    for fmt in TS_FORMATS:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().mean() > 0.99:
            return out
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def read_flows(src, default_header: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV (path or file-like). Headerless files need `default_header`."""
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as fh:
            raw = fh.read()
    else:
        raw = src.read()
        raw = raw.encode() if isinstance(raw, str) else raw
    if raw[:2] == b"\x1f\x8b":  # gzip
        raw = gzip.decompress(raw)
    first = raw.split(b"\n", 1)[0].decode("utf-8", "ignore")
    headed = ("Src IP" in first) or ("Source IP" in first)
    if headed:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    else:
        if default_header is None:
            raise ValueError("CSV has no header row and no default header was supplied")
        df = pd.read_csv(io.BytesIO(raw), header=None, names=default_header, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=ALIASES)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Type-fix, timestamp-parse, label-harmonise, sort. Stage is optional (uploads)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.copy()
    df["ts"] = _parse_ts(df["Timestamp"])
    df = df[df["ts"].notna()]
    num = [c for c in REQUIRED if c not in ("Src IP", "Dst IP", "Timestamp")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num] = df[num].replace([float("inf"), float("-inf")], 0).fillna(0)
    if "Stage" in df.columns:
        df["stage"] = df["Stage"].astype(str).str.strip().str.lower().map(STAGE_ORDER)
        df["stage"] = df["stage"].fillna(0).astype(int)
        df["labelled"] = True
    else:
        df["stage"] = 0
        df["labelled"] = False
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def load_dapt_dir(path: str = "data/dapt2020/csv") -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    header = None
    for f in files:
        with open(f) as fh:
            line = fh.readline()
        if line.startswith("Flow ID"):
            header = [c.strip() for c in line.rstrip("\n").split(",")]
            break
    frames = []
    for f in files:
        d = read_flows(f, default_header=header)
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    n_raw = len(raw)
    # the same conversation is seen at the public and private capture points
    raw = raw.drop_duplicates(subset=[c for c in IDENTITY if c in raw.columns])
    df = normalise(raw)
    info = {"files": len(files), "rows_raw": n_raw, "rows_after_dedup": len(df)}
    return df, info
