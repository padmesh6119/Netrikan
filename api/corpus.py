"""Corpora the models never trained on (CIC-IDS2017 slices, CTU-13 s4), scored by leave-one-corpus-out detectors,
plus the zero-shot transfer evidence. Labels are used only to draw attack minutes and to score, never to alert."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics import roc_auc_score

from netrikan import extra

from .common import Clean, cached, out, bucket_ms, read_json

MET = "results/zero_shot/metrics.json"
DET = {"iforest": "IsolationForest (benign-only)", "surprise": "World-model surprise (GRU next-minute error)",
       "gbdt": "Supervised LightGBM (other labs' attacks)", "volume": "Heuristic: flow volume", "fanout": "Heuristic: fan-out (peers + ports)"}
SLICES = {"cic17": {"fri": "Friday: port scan + DDoS", "wed": "Wednesday: Heartbleed", "both": "Both days"}}
BUDGETS = [0.5, 1.0, 2.0, 5.0, 10.0]
router = APIRouter(prefix="/api", default_response_class=Clean)


def check(name: str):
    if name not in ("cic17", "ctu13"):
        raise HTTPException(404, f"unknown corpus {name}")
    if not os.path.exists(f"models/zero_shot/holdout_{name}.joblib"):
        raise HTTPException(503, "Zero-shot models not found. Run `make zero-shot` first.")


@cached(4)
def prepare(name: str, slice_key: str):
    check(name)
    flows = extra.load_corpus(name)
    if name == "cic17" and slice_key != "both":
        day = "2017-07-07" if slice_key == "fri" else "2017-07-05"
        flows = flows[flows["ts"].dt.strftime("%Y-%m-%d") == day].reset_index(drop=True)
    H = extra.host_minutes(flows, name)
    b = joblib.load(f"models/zero_shot/holdout_{name}.joblib")
    X = H["X"]
    I = {f: extra.CF.index(f) for f in extra.CF}
    vol = np.expm1(X[:, I["in_n_flows"]]) + np.expm1(X[:, I["out_n_flows"]])
    fan = sum(np.expm1(X[:, I[f]]) for f in ("in_n_peers", "out_n_peers", "in_n_ports", "out_n_ports"))
    scores = {"gbdt": b["gbdt"].predict_proba(X)[:, 1], "iforest": -b["iforest"].score_samples(X),
              "surprise": b["surprise"].score_grid(H["Xd"], H["gid"])[H["keep"]], "volume": vol, "fanout": fan}
    pct = {k: pd.Series(v).rank(pct=True).to_numpy() for k, v in scores.items()}
    return {"n_flows": len(flows), "H": H, "scores": scores, "pct": pct, "t": bucket_ms(H["bucket"]), "trained_on": b["trained_on"]}


def alarms(R, det: str, budget: float):
    """Strict top-k by raw score, k = budget alerts per host-hour of traffic (ties cannot exceed the budget)."""
    n = len(R["pct"][det])
    k = max(1, int(budget * n / 60))
    thr = float(np.sort(R["pct"][det])[::-1][min(k, n) - 1])
    a = np.zeros(n, dtype=bool)
    a[np.argsort(-R["scores"][det], kind="stable")[:k]] = True
    return a, thr


@router.get("/corpus/{name}/meta")
@out
def corpus_meta(name: str):
    check(name)
    tr = read_json(MET)["results"][name]["trained_on"]
    return {"name": name, "label": extra.LABEL[name], "trained_on": [extra.LABEL[t].split(" (")[0] for t in tr],
            "slices": SLICES.get(name, {"both": "All"}), "detectors": DET, "budgets": BUDGETS}


@router.get("/corpus/{name}")
@out
def corpus(name: str, det: str = "iforest", budget: float = 2.0, slice: str = "both"):
    if det not in DET:
        raise HTTPException(400, "unknown detector")
    R = prepare(name, slice)
    H = R["H"]
    fam, attack = H["fam"], H["fam"].any(1)
    alarm, thr = alarms(R, det, budget)
    pct = R["pct"][det]
    order = pd.Series(pct).groupby(H["host"]).max().sort_values(ascending=False)
    att = pd.Series(H["host"][attack]).value_counts()
    cand = [{"host": h, "attack_minutes": int(n)} for h, n in att.items()]
    cand += [{"host": h, "attack_minutes": 0} for h in order.index if h not in att.index][:25]
    # detector comparison on this slice (label-dependent evaluation only)
    comp = []
    for j, f in enumerate(H["fam_names"]):
        pos = fam[:, j]
        if not pos.sum():
            continue
        mm = pos | ~attack
        y = pos[mm].astype(int)
        comp.append({"family": extra.FAMILIES[name][f], "positives": int(pos.sum()), "base_rate": float(y.mean()),
                     "auc": {d: float(roc_auc_score(y, R["scores"][d][mm])) for d in DET}})
    top = list(order.index[:14])
    sel = np.isin(H["host"], top)
    hidx = {h: i for i, h in enumerate(top)}
    return {"n_flows": R["n_flows"], "n_hosts": len(set(H["host"])), "host_minutes": len(pct), "attack_minutes": int(attack.sum()),
            "alerts": int(alarm.sum()), "caught": (alarm & attack).sum() / max(attack.sum(), 1),
            "false_alerts_per_host_hour": (alarm & ~attack).sum() / max((~attack).sum() / 60, 1e-9), "threshold": thr,
            "families": [extra.FAMILIES[name][f] for f in H["fam_names"]], "hosts": cand, "comparison": comp,
            "heatmap": {"hosts": top, "host": [hidx[h] for h in H["host"][sel]], "t": R["t"][sel], "v": np.round(pct[sel], 3),
                        "attack": attack[sel].astype(int)}}


@router.get("/corpus/{name}/host")
@out
def corpus_host(name: str, host: str, det: str = "iforest", budget: float = 2.0, slice: str = "both"):
    R = prepare(name, slice)
    H = R["H"]
    alarm, thr = alarms(R, det, budget)
    m = H["host"] == host
    fam = H["fam"][m]
    return {"threshold": thr, "t": R["t"][m], "pct": np.round(R["pct"][det][m], 4), "alarm": alarm[m].astype(int),
            "family": np.where(fam.any(1), fam.argmax(1), -1)}


@router.get("/zeroshot")
@out
def zeroshot():
    m = read_json(MET)
    rows = []
    for h, r in m["results"].items():
        for f, x in r["families"].items():
            rows.append({"corpus": extra.LABEL[h].split(" (")[0], "family": x["name"], "positives": x["n_pos"], "base_rate": x["base_rate"],
                         "auc": {d: x[d]["roc_auc"] for d in DET if d in x}, "gbdt_ci90": x["gbdt"].get("roc_auc_ci90")})
    return {"protocol": m["protocol"], "note": m.get("note"), "corpora": m["corpora"], "detectors": DET, "rows": rows}
