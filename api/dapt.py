"""DAPT2020 host forecasting: per-host P(attack in the next K minutes), stage, TreeSHAP drivers, world-model rollout.

A capture day is scored by the leave-one-day-out model that never saw it; an uploaded CSV by the all-days model.
"""
from __future__ import annotations

import io
import os
import uuid
from collections import OrderedDict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from netrikan import infer
from netrikan import metrics as M
from netrikan.dapt import STAGE_NAMES, load_dapt_dir, normalise, read_flows
from netrikan.features import human

from .common import Clean, cached, out, bucket_ms, ms, read_json

MODEL_DIR, METRICS = "models/demo", "results/demo/metrics.json"
ROLLOUT_FEATURES = ["in_n_flows", "in_n_ports", "in_syn", "out_n_flows"]
router = APIRouter(prefix="/api/dapt", default_response_class=Clean)
_uploads: OrderedDict[str, tuple[str, pd.DataFrame]] = OrderedDict()


def mx() -> dict:
    return read_json(METRICS)


@lru_cache(maxsize=8)
def bundle(name: str) -> dict:
    path = f"{MODEL_DIR}/{name}.joblib"
    if not os.path.exists(path):
        raise HTTPException(503, f"{path} not found. Run `make demo-train` first.")
    return joblib.load(path)


@cached(1)
def all_flows() -> pd.DataFrame:
    return load_dapt_dir()[0]


def source_flows(source: str) -> tuple[pd.DataFrame, str]:
    """(flows, model name) for a capture day or an uploaded file id."""
    if source.startswith("upload:"):
        if source not in _uploads:
            raise HTTPException(404, "Upload expired. Upload the file again.")
        return _uploads[source][1], "all"
    if source not in mx()["dataset"]["days"]:
        raise HTTPException(404, f"unknown day {source}")
    f = all_flows()
    return f[f["ts"].dt.strftime("%Y-%m-%d") == source].reset_index(drop=True), f"fold_{source}"


@cached(8)
def run(source: str):
    flows, model_name = source_flows(source)
    try:
        tab, pr = infer.analyse(flows, bundle(model_name))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return flows, model_name, tab, pr


def default_threshold(which: str) -> float:
    m = mx()
    t = m["alert_threshold_1_per_host_hour"] if which == m["primary_model"] else m["pooled"][which]["thr@1.0/h"]
    return float(min(max(t or 0.1, 0.01), 0.99))


@router.get("/meta")
@out
def meta():
    m = mx()
    ds = m["dataset"]
    return {"days": ds["days"], "onsets_by_day": ds["onsets_by_day"], "K": ds["horizon_K_buckets"], "L": ds["history_L_buckets"],
            "bucket_s": ds["bucket_seconds"], "primary": m["primary_model"], "stages": STAGE_NAMES,
            "thresholds": {w: default_threshold(w) for w in ("lag", "world")},
            "uploads": [{"source": k, "name": v[0], "n_flows": len(v[1])} for k, v in _uploads.items()]}


@router.get("/metrics")
@out
def metrics():
    return mx()


@router.post("/upload")
@out
async def upload(file: UploadFile):
    raw = await file.read()
    try:
        flows = normalise(read_flows(io.BytesIO(raw)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Could not read the file: {e}")
    key = f"upload:{uuid.uuid4().hex[:10]}"
    _uploads[key] = (file.filename or "upload.csv", flows)
    while len(_uploads) > 4:  # run()'s LRU cache bounds the scored copies
        _uploads.popitem(last=False)
    try:
        run(key)
    except HTTPException:
        _uploads.pop(key)
        raise
    return {"source": key, "name": file.filename, "n_flows": len(flows)}


@router.get("/scores")
@out
def scores(source: str):
    """Every host-minute of the source: both risk models, true stage (if labelled). Columnar."""
    flows, model_name, tab, pr = run(source)
    hosts = sorted(np.unique(tab.host))
    hidx = {h: i for i, h in enumerate(hosts)}
    return {"source": source, "model_name": model_name, "n_flows": len(flows), "labelled": bool(flows["labelled"].any()),
            "hosts": hosts, "bucket_s": tab.bucket_s, "K": tab.K,
            "rows": {"host": [hidx[h] for h in tab.host], "t": bucket_ms(tab.bucket, tab.bucket_s),
                     "lag": np.round(pr["lag"], 4), "world": np.round(pr["world"], 4),
                     "attack": tab.attack.astype(int), "stage": tab.stage.astype(int)}}


@router.get("/summary")
@out
def summary(source: str, model: str = "lag", thr: float = 0.2):
    """Headline numbers at an alert threshold: alerts, false alarms/host-hour, onsets warned and lead time."""
    flows, _, tab, pr = run(source)
    p = pr[model]
    alarm = p >= thr
    out = {"n_flows": len(flows), "n_hosts": len(np.unique(tab.host)), "host_minutes": tab.n, "alert_minutes": int(alarm.sum()),
           "labelled": bool(flows["labelled"].any())}
    if out["labelled"]:
        eps = M.onset_episodes(tab.attack, tab.start, tab.end, tab.K)
        lt = M.lead_times(p, eps, tab.K, thr) if len(eps) else np.array([])
        negs = tab.valid & ~tab.attack & ~tab.fut
        out.update({
            "false_alarms_per_host_hour": int((alarm & negs).sum()) / max(negs.sum() * tab.bucket_s / 3600, 1e-9),
            "onsets": len(eps), "onsets_warned": int((lt > 0).sum()),
            "median_lead_min": float(np.median(lt[lt > 0]) * tab.bucket_s / 60) if (lt > 0).any() else None,
            "onset_list": [{"host": tab.host[s], "t": ms(infer.times(tab)[s]), "stage": STAGE_NAMES[tab.stage[s]], "lead_min": int(l) * tab.bucket_s / 60}
                           for s, l in zip(eps, lt)],
        })
    return out


@router.get("/explain")
@out
def explain(source: str, i: int, model: str = "lag"):
    flows, model_name, tab, pr = run(source)
    if not 0 <= i < tab.n:
        raise HTTPException(404, "row out of range")
    b = bundle(model_name)
    T = infer.times(tab)
    labelled = bool(flows["labelled"].any())
    outcome = None
    if labelled:
        outcome = "attack_now" if tab.attack[i] else ("past_end" if not tab.valid[i] else ("attack_next" if tab.fut[i] else "quiet"))
    fc, obs = infer.rollout_counts(b, tab, pr, i, ROLLOUT_FEATURES)
    fl = infer.flows_in_window(flows, tab.host[i], T[i], tab.K + 1).head(200)
    fl = fl.assign(ts=fl["ts"].map(ms))
    m = mx()
    return {
        "i": i, "host": tab.host[i], "t": ms(T[i]), "p": float(pr[model][i]), "outcome": outcome,
        "true_stage": STAGE_NAMES[tab.stage[i]] if labelled else None,
        "stage_probs": [{"stage": s, "p": float(v)} for s, v in zip(STAGE_NAMES[1:], pr["stage"][i][1:])],
        "stage_accuracy": m["stage"]["accuracy"], "stage_majority": m["stage"]["majority_baseline_accuracy"],
        "drivers": [{"feature": n, "value": v} for n, v in infer.top_drivers(b, pr, i, model)],
        "rollout": [{"feature": human(f), "forecast": fc[:, j], "observed": obs[:, j]} for j, f in enumerate(ROLLOUT_FEATURES)],
        "flows": {"columns": list(fl.columns), "rows": fl.to_numpy().tolist()},
    }
