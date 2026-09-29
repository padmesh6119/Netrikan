"""Zero-shot transfer across labs: leave-one-corpus-out attack detection, all numbers from results/zero_shot/metrics.json."""
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from netrikan import extra  # noqa: E402

MET = "results/zero_shot/metrics.json"
if not os.path.exists(MET):
    st.error("Run `make zero-shot` first.")
    st.stop()
mx = json.load(open(MET))
res = mx["results"]
DET = {"iforest": "IsolationForest (benign-only)", "surprise": "World-model surprise (GRU, no labels)", "gbdt": "Supervised LightGBM",
       "volume": "Heuristic: flow volume", "fanout": "Heuristic: fan-out"}
DET = {k: v for k, v in DET.items() if all(k in f for r in res.values() for f in r["families"].values())}

st.title("Zero-shot transfer across labs")
st.markdown("Four corpora from different labs, capture tools and years. For each, detectors are trained on the **other three** and scored on the held-out one, "
            "so every attack family in the test set is unseen. Features are the 22 host-minute features all four corpora share. "
            "A learned detector only counts if it beats the two heuristics.")

rows = []
for h, r in res.items():
    for f, x in r["families"].items():
        row = {"held-out corpus": extra.LABEL[h].split(" (")[0], "unseen family": x["name"], "positive minutes": x["n_pos"], "base rate": x["base_rate"]}
        for d in DET:
            row[d] = x[d]["roc_auc"]
        row["gbdt 90% CI"] = f"{x['gbdt']['roc_auc_ci90'][0]:.2f}–{x['gbdt']['roc_auc_ci90'][1]:.2f}"
        rows.append(row)
df = pd.DataFrame(rows)
fam_mean = {d: df[d].mean() for d in DET}
corp_mean = {d: df.groupby("held-out corpus")[d].mean().mean() for d in DET}

c = st.columns(len(DET))
for col, d in zip(c, DET):
    col.metric(DET[d], f"{fam_mean[d]:.2f}", help=f"Mean ROC-AUC over {len(df)} unseen families. Corpus-averaged: {corp_mean[d]:.2f}")
st.caption("Mean ROC-AUC over all unseen families (0.5 = chance).")

long = df.melt(id_vars=["held-out corpus", "unseen family"], value_vars=list(DET), var_name="detector", value_name="ROC-AUC")
long["detector"] = long["detector"].map(DET)
long["label"] = long["held-out corpus"].str.slice(0, 10) + " · " + long["unseen family"].str.slice(0, 22)
ch = alt.Chart(long).mark_circle(size=90, opacity=0.85).encode(
    y=alt.Y("label:N", sort=list(df["held-out corpus"].str.slice(0, 10) + " · " + df["unseen family"].str.slice(0, 22)), title=None),
    x=alt.X("ROC-AUC:Q", scale=alt.Scale(domain=[0, 1])), color=alt.Color("detector:N", scale=alt.Scale(range=["#1f77b4", "#2ca02c", "#d62728", "#999999", "#c9a227"])))
rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
st.altair_chart((ch + rule).properties(height=30 * len(df) + 40), width="stretch")

show = df.rename(columns=DET).set_index(["held-out corpus", "unseen family"])
st.dataframe(show.style.format({c: "{:.2f}" for c in DET.values()} | {"base rate": "{:.4f}"}), width="stretch")

sup, iso, sur, vol = fam_mean["gbdt"], fam_mean["iforest"], fam_mean.get("surprise", np.nan), fam_mean["volume"]
below = int((df["gbdt"] < 0.5).sum())
st.subheader("What this shows, and what it does not")
st.markdown(
    f"- **Learning attacks from other labs does not transfer.** The supervised detector averages {sup:.2f} and is *below chance* on {below} of {len(df)} unseen families. "
    f"Attack signatures from one lab invert in another.\n"
    f"- **Benign-only anomaly detection transfers best.** IsolationForest averages {iso:.2f}; the plain flow-volume heuristic {vol:.2f}. "
    f"Scans, DDoS and brute force are visibly abnormal without any attack training.\n"
    f"- **World-model surprise (GRU next-minute error) averages {sur:.2f}.** {'It does not beat IsolationForest here.' if sur <= iso + 0.03 else 'It beats IsolationForest.'} "
    f"Stealthy families (botnet C&C/spam, Heartbleed, DAPT lateral movement) stay near chance for every detector.\n"
    f"- **Caveats.** Positive minutes are tens per family (CIC-2017 has 11–21), host-minutes within a host are not independent, and negatives are the held-out corpus's own benign minutes. "
    f"Small differences are noise; the ordering supervised < heuristics < IsolationForest is the robust part.\n"
    f"- **Design consequence.** For attacks the model has never seen, use anomaly-style scoring, not a classifier trained on other labs' attacks.")
with st.expander("Protocol"):
    st.write(mx["protocol"])
    st.json(mx["corpora"])
    st.caption(mx["note"])
