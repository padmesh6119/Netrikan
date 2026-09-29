"""Inference: flows in, per-host risk timeline + explanations out. Used by the app and the CLI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models
from .dapt import STAGE_NAMES
from .features import FEATURES, LOG_IDX, EPOCH, Table, build_grid, make_table


def analyse(flows: pd.DataFrame, bundle: dict, bucket_s: int = 60):
    """Bucket the flows per monitored host and score every host-minute with the loaded models."""
    grid = build_grid(flows, bucket_s)
    if grid.empty:
        raise ValueError("no flow touches a private (monitored) address, so there is no host to model")
    tab = make_table(grid, bundle["K"], bundle["L"], bucket_s)
    pr = models.predict(bundle, tab, np.arange(tab.n), which=("lag", "world", "stage"), explain=True)
    return tab, pr


def times(tab: Table) -> pd.DatetimeIndex:
    return pd.to_datetime(tab.bucket.astype("int64") * tab.bucket_s, unit="s")


def top_drivers(bundle: dict, pr: dict, i: int, model: str = "lag", n: int = 8):
    """Exact TreeSHAP contributions (LightGBM pred_contrib), in log-odds, for one host-minute."""
    contrib = pr[f"{model}_contrib"][i][:-1]
    names = bundle["lag_names"] if model == "lag" else bundle["head_names"]
    order = np.argsort(-np.abs(contrib))[:n]
    return [(models.pretty(names[j]), float(contrib[j])) for j in order]


def _to_counts(z_log: np.ndarray) -> np.ndarray:
    x = z_log.astype(float).copy()
    x[..., LOG_IDX] = np.sign(x[..., LOG_IDX]) * np.expm1(np.abs(x[..., LOG_IDX]))
    return x


def rollout_counts(bundle: dict, tab: Table, pr: dict, i: int, features: list[str]):
    """Forecast (K steps) vs what was observed, in original units, for the chosen features."""
    sc = bundle["scaler"]
    fc = _to_counts(pr["world_rollout"][i] * sc.sd + sc.mu)
    K = tab.K
    obs = np.full((K, len(FEATURES)), np.nan)
    for k in range(1, K + 1):
        if i + k < tab.end[i]:
            obs[k - 1] = _to_counts(tab.X[i + k])
    cols = [FEATURES.index(f) for f in features]
    return fc[:, cols], obs[:, cols]


def flows_in_window(flows: pd.DataFrame, host: str, t0: pd.Timestamp, minutes: int) -> pd.DataFrame:
    m = ((flows["Src IP"] == host) | (flows["Dst IP"] == host)) & (flows["ts"] >= t0) & (flows["ts"] < t0 + pd.Timedelta(minutes=minutes))
    cols = ["ts", "Src IP", "Dst IP", "Dst Port", "Protocol", "SYN Flag Count", "Total Fwd Packet", "Flow Duration"]
    if flows["labelled"].any():
        cols.append("Stage")
    return flows.loc[m, cols]


def stage_name(k: int) -> str:
    return STAGE_NAMES[k]
