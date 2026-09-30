"""UWF-ZeekData24: ATT&CK technique recognition per host-minute + attacker burst forecasting.

Every week is scored by models trained WITHOUT that week. Attackers are chosen from the recognizer's
predicted flags, so the forecast path uses no ground truth.
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from netrikan import campaign as C
from netrikan import metrics as M
from netrikan.features import EPOCH, FEATURES, build_grid, human
from netrikan.zeek import MONITORED, TECH_LABEL, TECH_NAME, TECH_TACTIC, TECHS, load_z24

from .common import Clean, cached, out, bucket_ms, read_json, shap_top

MODEL_DIR, METRICS = "models/z24", "results/z24/metrics.json"
RUNGS = {"sched": "All-technique history (GBDT)", "world": "World model (GRU rollout + head)", "renewal": "Renewal hazard baseline"}
router = APIRouter(prefix="/api/z24", default_response_class=Clean)


def mx() -> dict:
    return read_json(METRICS)


@lru_cache(maxsize=8)
def bundle(name: str) -> dict:
    path = f"{MODEL_DIR}/{name}.joblib"
    if not os.path.exists(path):
        raise HTTPException(503, f"{path} not found. Run `make z24-train` first.")
    return joblib.load(path)


def model_for(week: str) -> tuple[str, bool]:
    ds = mx()["dataset"]
    if week in ds["attack_weeks"]:
        return f"fold_{week}", True
    if week in ds["benign_weeks"]:
        return f"fold_{ds['benign_held_out_by'][week]}", False
    raise HTTPException(404, f"unknown week {week}")


@cached(2)
def week_flows(week: str) -> pd.DataFrame:
    return load_z24(weeks=[week])


@cached(4)
def analyse(week: str, thr: float):
    model_name, _ = model_for(week)
    b = bundle(model_name)
    grid = build_grid(week_flows(week), 60, MONITORED, tuple(TECH_LABEL))
    X = C.log_state(grid)
    Y = grid[TECH_LABEL].to_numpy().astype(bool)
    traffic = ((grid["in_n_flows"] + grid["out_n_flows"]) > 0).to_numpy()
    P = C.predict_recognizer(b["recog"], X, traffic)
    Fp = P >= thr
    S = fut = valid = pr = None
    try:
        S = C.build_seqs(grid, X, Y, Fp, {d: week for d in grid["day"].unique()}, select=Fp)
        fut, valid = C.targets(S)
        pr = C.predict_forecasters(b["models"], S, S.Fp, world=b["world"])
    except ValueError:
        S = None  # no host sends flagged traffic this week
    return {"model_name": model_name, "host": grid["host"].to_numpy(), "t": bucket_ms(grid["bucket"].to_numpy()), "X": X, "Y": Y,
            "traffic": traffic, "P": P, "flagged": Fp, "S": S, "fut": fut, "valid": valid, "pr": pr}


def techs():
    return [{"id": t, "name": TECH_NAME[t], "tactic": TECH_TACTIC[t]} for t in TECHS]


@router.get("/meta")
@out
def meta():
    ds = mx()["dataset"]
    weeks = [{"week": w, "kind": "attack"} for w in ds["attack_weeks"]] + [{"week": w, "kind": "benign"} for w in ds["benign_weeks"]]
    return {"weeks": weeks, "K": ds["horizon_K_minutes"], "techniques": techs(), "rungs": RUNGS, "budgets": [0.5, 1.0, 2.0]}


@router.get("/metrics")
@out
def metrics():
    return mx()


@router.get("/week")
@out
def week(week: str, thr: float = 0.5):
    R = analyse(week, thr)
    _, is_attack = model_for(week)
    fl, Y, S = R["flagged"], R["Y"], R["S"]
    rows = []
    for j, t in enumerate(TECHS):
        row = {"id": t, "name": TECH_NAME[t], "tactic": TECH_TACTIC[t], "flagged": int(fl[:, j].sum())}
        if is_attack:
            tp = int((fl[:, j] & Y[:, j]).sum())
            row.update({"true": int(Y[:, j].sum()), "recall": tp / max(int(Y[:, j].sum()), 1), "precision": tp / max(int(fl[:, j].sum()), 1)})
        rows.append(row)
    fh = pd.Series(R["host"][fl.any(1)]).value_counts()
    attackers = []
    if S is not None:
        for h in sorted(set(S.host)):
            r = np.flatnonzero(S.host == h)
            days = sorted(set(pd.to_datetime(S.bucket[r].astype("int64") * 60, unit="s").strftime("%Y-%m-%d")))
            attackers.append({"host": h, "minutes": len(r), "days": days})
    return {"week": week, "is_attack": is_attack, "n_flows": len(week_flows(week)), "traffic_minutes": int(R["traffic"].sum()),
            "host_minutes": len(R["host"]), "flagged_any": int(fl.any(1).sum()),
            "benign_false_alarms_per_host_hour": None if is_attack else fl.any(1).sum() / max(len(R["host"]) / 60, 1e-9),
            "attacker_sequences": 0 if S is None else len(S.seq_bounds()), "techniques": rows,
            "flagged_hosts": [{"host": h, "minutes": int(n)} for h, n in fh.items()], "attackers": attackers}


@router.get("/recognize/host")
@out
def recognize_host(week: str, host: str, thr: float = 0.5):
    R = analyse(week, thr)
    _, is_attack = model_for(week)
    m = (R["host"] == host) & R["traffic"]
    fl, P = R["flagged"], R["P"]
    ticks = []
    for j, t in enumerate(TECHS):
        ticks.append({"id": t, "recognized": R["t"][m & fl[:, j]], "truth": R["t"][m & R["Y"][:, j]] if is_attack else []})
    idx = np.flatnonzero(m & fl.any(1))
    top = idx[np.argsort(-P[idx].max(1))][:20]
    return {"ticks": ticks, "top": [{"i": int(i), "t": int(R["t"][i]), "p": P[i]} for i in top]}


@router.get("/recognize/explain")
@out
def recognize_explain(week: str, i: int, thr: float = 0.5):
    R = analyse(week, thr)
    P = R["P"][i]
    j = int(np.argmax(P))
    contrib = bundle(R["model_name"])["recog"][j].predict(R["X"][i:i + 1], pred_contrib=True)[0][:-1]
    return {"i": i, "t": int(R["t"][i]), "probs": [{"id": t, "name": TECH_NAME[t], "p": float(P[k])} for k, t in enumerate(TECHS)],
            "explained": TECHS[j], "drivers": shap_top(contrib, [human(f) for f in FEATURES], 7)}


@router.get("/forecast")
@out
def forecast(week: str, host: str, tech: str, rung: str = "sched", budget: float = 1.0, day: str | None = None, thr: float = 0.5):
    R = analyse(week, thr)
    S, pr, fut, valid = R["S"], R["pr"], R["fut"], R["valid"]
    if S is None:
        raise HTTPException(404, "No host sends technique-flagged traffic in this week.")
    if rung not in RUNGS or tech not in TECHS:
        raise HTTPException(400, "unknown model or technique")
    tj = TECHS.index(tech)
    K = mx()["dataset"]["horizon_K_minutes"]
    thr_f = mx()["forecast"]["per_technique"][rung][tech][f"thr@{budget}/h"]
    pj, base = pr[rung][:, tj], pr["renewal"][:, tj]
    rows = np.flatnonzero(S.host == host)
    if not len(rows):
        raise HTTPException(404, f"{host} is not an attacker this week")
    dates = pd.to_datetime(S.bucket[rows].astype("int64") * 60, unit="s").strftime("%Y-%m-%d").to_numpy()
    days = sorted(set(dates))
    day = day if day in days else days[0]
    sel = rows[dates == day]
    t = bucket_ms(S.bucket[sel])
    burst = S.F[sel, tj]
    alert = (pj[sel] >= thr_f) & ~burst
    # week-level numbers for this technique across every attacker
    alarms = pj >= thr_f
    negs = valid & ~S.F[:, tj] & ~fut[:, tj]
    eps = M.onset_episodes(S.F[:, tj], S.start, S.end, K)
    lt = M.lead_times(pj, eps, K, thr_f) if len(eps) else np.array([])
    cand = sel[~S.F[sel, tj]]
    top = cand[np.argsort(-pj[cand])][:20]
    return {"days": days, "day": day, "K": K, "threshold": thr_f,
            "series": {"t": t, "i": sel, "model": np.round(pj[sel], 4), "baseline": np.round(base[sel], 4)},
            "bursts": t[burst], "alerts": {"t": t[alert], "p": pj[sel][alert]},
            "stats": {"onsets": len(eps), "warned": float((lt > 0).mean()) if len(eps) else None,
                      "false_alarms_per_attacker_hour": float((alarms & negs).sum() / max(negs.sum() / 60, 1e-9)),
                      "median_lead_min": float(np.median(lt[lt > 0])) if (lt > 0).any() else None},
            "top": [{"i": int(i), "t": int(bucket_ms(S.bucket[i:i + 1])[0]), "p": float(pj[i])} for i in top]}


@router.get("/forecast/explain")
@out
def forecast_explain(week: str, tech: str, i: int, thr: float = 0.5):
    """TreeSHAP for the all-technique-history model at one attacker-minute."""
    R = analyse(week, thr)
    S = R["S"]
    if S is None or not 0 <= i < S.n:
        raise HTTPException(404, "row out of range")
    tj = TECHS.index(tech)
    a, b = int(S.start[i]), int(S.end[i])
    sch = C.sched_feats(S.subset(np.arange(a, b)), S.Fp[a:b])
    Xr = np.c_[sch[i - a][None, :], S.X[i][None, :]]
    contrib = bundle(R["model_name"])["models"]["sched"][tj].predict(Xr, pred_contrib=True)[0][:-1]
    names = [n.replace("|", " · ") for n in C.SCHED_NAMES] + [human(f) for f in FEATURES]
    return {"i": i, "drivers": shap_top(contrib, names, 8)}


@router.get("/structure")
@out
def structure(week: str):
    """Minute-of-hour of each hourly burst and gaps between bursts (this week) + cross-technique lags (all weeks)."""
    _, is_attack = model_for(week)
    s = mx()["structure"]
    out = {"is_attack": is_attack, "uniform_std": s["uniform_minute_of_hour_std"],
           "minute_std": float(np.mean([v["std"] for v in s["minute_of_hour_of_first_flow"].values()])),
           "lags": {"techs": TECHS[:4], "matrix": [[s["cross_technique_lags"].get(a, {}).get(b_, {}).get("p_within_5") for a in TECHS[:4]] for b_ in TECHS[:4]]},
           "attackers": s["attackers"], "minute_hist": [], "gap_hist": []}
    if is_attack:
        fl = week_flows(week)
        for t in TECHS[:4]:
            x = fl[fl[f"t_{t}"]]
            mm = ((x["ts"] - EPOCH).dt.total_seconds() // 60).astype(int)
            first = (mm.groupby([x["Src IP"], mm // 60]).min() % 60).to_numpy()
            gaps = []
            for _, v in mm.groupby(x["Src IP"]):
                v = np.sort(v.unique())
                st_ = v[np.r_[True, np.diff(v) > 1]]
                gaps += list(np.diff(st_))
            gaps = np.array(gaps)
            out["minute_hist"].append({"id": t, "counts": np.histogram(first, bins=np.arange(0, 65, 5))[0]})
            out["gap_hist"].append({"id": t, "counts": np.histogram(gaps[gaps <= 180], bins=np.arange(0, 185, 5))[0]})
    return out
