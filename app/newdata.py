"""Shared renderer for the corpora the models never trained on (CIC-IDS2017 slices, CTU-13 s4).

Each corpus is scored by detectors trained WITHOUT it (results/zero_shot, models/zero_shot/holdout_<corpus>.joblib).
Four detectors need no labels from the corpus being scored; the supervised one was trained on the other labs' attacks.
Everything shown is computed live from the flows or read from results/zero_shot/metrics.json.
"""
import json
import os
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from netrikan import extra  # noqa: E402

MET = "results/zero_shot/metrics.json"
DET = {"iforest": "IsolationForest (benign-only)", "surprise": "World-model surprise (GRU next-minute error)",
       "gbdt": "Supervised LightGBM (other labs' attacks)", "volume": "Heuristic: flow volume", "fanout": "Heuristic: fan-out (peers + ports)"}
FAMCOL = ["#e4572e", "#1565c0", "#9c27b0", "#2e7d32", "#e6a100"]


@st.cache_resource(show_spinner="Loading flows, building per-host minutes and scoring with the held-out detectors…")
def prepare(name: str, slice_key: str):
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
    T = pd.to_datetime(H["bucket"].astype("int64") * 60, unit="s")
    return {"n_flows": len(flows), "H": H, "scores": scores, "pct": pct, "T": T, "trained_on": b["trained_on"]}


def render(name: str):
    if not (os.path.exists(MET) and os.path.exists(f"models/zero_shot/holdout_{name}.joblib")):
        st.error("Zero-shot models not found. Run `make zero-shot` first.")
        st.stop()
    mx = json.load(open(MET))
    with st.sidebar:
        st.title(extra.LABEL[name].split(" (")[0])
        st.caption("Scored by detectors that never saw this corpus (trained on: " + ", ".join(extra.LABEL[t].split(" (")[0] for t in mx["results"][name]["trained_on"]) + ").")
        slice_key = "both"
        if name == "cic17":
            slice_key = {"Friday: port scan + DDoS": "fri", "Wednesday: Heartbleed": "wed", "Both days": "both"}[st.radio("Slice", ["Friday: port scan + DDoS", "Wednesday: Heartbleed", "Both days"])]
        det = st.radio("Detector", list(DET), format_func=lambda d: DET[d])
        budget = st.select_slider("Alert budget (alerts per host-hour)", [0.5, 1.0, 2.0, 5.0, 10.0], value=2.0,
                                  help="Label-free: alert on the highest-scoring host-minutes, this many per host-hour of traffic.")
    R = prepare(name, slice_key)
    H, T = R["H"], R["T"]
    fam = H["fam"]
    attack = fam.any(1)
    pct = R["pct"][det]
    n = len(pct)
    k = max(1, int(budget * n / 60))
    thr = np.sort(pct)[::-1][min(k, n) - 1]
    alarm = np.zeros(n, dtype=bool)
    alarm[np.argsort(-R["scores"][det], kind="stable")[:k]] = True  # strict top-k: ties cannot exceed the budget

    st.title(extra.LABEL[name])
    st.caption("Zero-shot: nothing about this lab's traffic or attacks was used for training or threshold selection.")
    c = st.columns(5)
    c[0].metric("Flows", f"{R['n_flows']:,}")
    c[1].metric("Hosts / host-minutes", f"{len(set(H['host']))} / {n:,}")
    c[2].metric("Attack host-minutes", int(attack.sum()))
    c[3].metric("Alerts", int(alarm.sum()))
    c[4].metric("Attack minutes caught", f"{(alarm & attack).sum() / max(attack.sum(), 1):.0%}")
    st.caption(f"False alerts: {(alarm & ~attack).sum() / max((~attack).sum() / 60, 1e-9):.2f} per benign host-hour at this budget.")

    t1, t2, t3 = st.tabs(["Host timeline", "Detector comparison (this slice)", "Network heatmap"])
    with t1:
        order = pd.Series(pct).groupby(H["host"]).max().sort_values(ascending=False)
        att_hosts = list(pd.Series(H["host"][attack]).value_counts().index)
        cand = att_hosts + [h for h in order.index if h not in att_hosts][:25]
        host = st.selectbox("Host", cand, format_func=lambda h: f"{h}" + (f"   ({int(attack[H['host'] == h].sum())} attack minutes)" if h in att_hosts else ""))
        m = H["host"] == host
        d = pd.DataFrame({"time": T[m], "score percentile": pct[m]})
        d["end"] = d["time"] + pd.Timedelta(seconds=60)
        layers = [alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=12)).encode(x="time:T", y=alt.Y("score percentile:Q", scale=alt.Scale(domain=[0, 1])))]
        layers.append(alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q"))
        bands = []
        for j, f in enumerate(H["fam_names"]):
            idx = np.flatnonzero(m & fam[:, j])
            bands += [{"time": T[i], "end": T[i] + pd.Timedelta(seconds=60), "family": extra.FAMILIES[name][f]} for i in idx]
        if bands:
            layers.insert(0, alt.Chart(pd.DataFrame(bands)).mark_rect(opacity=0.3).encode(x="time:T", x2="end:T", color=alt.Color("family:N", scale=alt.Scale(range=FAMCOL))))
        al = d[alarm[m]]
        if len(al):
            layers.append(alt.Chart(al).mark_point(color="#d62728", filled=True, size=40).encode(x="time:T", y="score percentile:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
        st.caption(f"{DET[det]}: score shown as its percentile within this slice, so detectors are comparable. Shaded: true attack minutes by family (labels used only to draw, never to score). "
                   "Red dots: alerts at the chosen budget.")
    with t2:
        rows = []
        neg = ~attack
        for j, f in enumerate(H["fam_names"]):
            pos = fam[:, j]
            if pos.sum() == 0:
                continue
            mm = pos | neg
            y = pos[mm].astype(int)
            row = {"family": extra.FAMILIES[name][f], "positive minutes": int(pos.sum()), "base rate": float(y.mean())}
            for dname in DET:
                row[DET[dname]] = float(roc_auc_score(y, R["scores"][dname][mm]))
            rows.append(row)
        tb = pd.DataFrame(rows).set_index("family")
        st.markdown("**ROC-AUC on this slice** (0.5 = chance). Live computation.")
        st.dataframe(tb.style.format({c: "{:.2f}" for c in tb.columns if c not in ("positive minutes",)}).format({"positive minutes": "{:d}", "base rate": "{:.4f}"}), width="stretch")
        st.caption("Attack host-minutes are tens, and minutes within one host are not independent: differences of a few hundredths are noise. See the Zero-shot transfer page for intervals.")
        best = tb.drop(columns=["positive minutes", "base rate"]).idxmax(axis=1)
        st.markdown("Best detector per family: " + "; ".join(f"**{f}** → {b}" for f, b in best.items()))
    with t3:
        top = list(order.index[:14])
        sel = np.isin(H["host"], top)
        hm = pd.DataFrame({"host": H["host"][sel], "time": T[sel], "score": pct[sel]})
        st.altair_chart(alt.Chart(hm).mark_rect().encode(x="time:T", y=alt.Y("host:N", title=None), color=alt.Color("score:Q", scale=alt.Scale(scheme="orangered", domain=[0, 1]), title="percentile")).properties(height=60 + 24 * len(top)), width="stretch")
        st.caption("The 14 highest-scoring hosts. Hosts with attack minutes: " + (", ".join(att_hosts[:6]) if att_hosts else "none in this slice") + ".")
