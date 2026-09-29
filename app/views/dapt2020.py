"""Offline demo: flows in -> per-host attack-forecast timeline, stage, drivers, world-model rollout.

Run:  make demo        (needs `make demo-train` first)
Everything shown is computed live from the loaded flows or read from results/demo/metrics.json.
"""
import json
import os
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
from netrikan import infer  # noqa: E402
from netrikan import metrics as M  # noqa: E402
from netrikan.dapt import STAGE_NAMES, load_dapt_dir, normalise, read_flows  # noqa: E402
from netrikan.features import EPOCH, human  # noqa: E402

MODEL_DIR, METRICS = "models/demo", "results/demo/metrics.json"
STAGE_COLORS = {"Reconnaissance": "#e6a100", "Initial Access": "#e4572e", "Lateral Movement": "#9c27b0", "Exfiltration": "#b71c1c"}

if not (os.path.exists(METRICS) and os.path.isdir(MODEL_DIR)):
    st.error("No trained models found. Run `make demo-train` first.")
    st.stop()


@st.cache_data
def dapt_metrics():
    return json.load(open(METRICS))


@st.cache_resource
def dapt_bundle(name):
    return joblib.load(f"{MODEL_DIR}/{name}.joblib")


@st.cache_data(show_spinner="Loading DAPT2020 flows…")
def dapt_flows():
    return load_dapt_dir()[0]


@st.cache_data(show_spinner="Bucketing per host and scoring…")
def dapt_run(flows_key, flows, model_name):
    b = dapt_bundle(model_name)
    tab, pr = infer.analyse(flows, b)
    return tab, pr


mx = dapt_metrics()
ds = mx["dataset"]
K = ds["horizon_K_buckets"]
BUCKET = ds["bucket_seconds"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Netrikan")
    st.caption("Host-centric attack forecasting · offline · no cloud calls")
    source = st.radio("Traffic source", ["DAPT2020 day (held-out model)", "Upload flow CSV"])
    if source.startswith("DAPT"):
        day = st.selectbox("Capture day", ds["days"], index=1)
        st.caption("The model scoring this day was trained on the *other* four days, so nothing here is in-sample.")
        allf = dapt_flows()
        flows = allf[allf["ts"].dt.strftime("%Y-%m-%d") == day].reset_index(drop=True)
        key, model_name = f"dapt-{day}", f"fold_{day}"
    else:
        up = st.file_uploader("CICFlowMeter-style CSV (DAPT2020 columns)", type="csv")
        if up is None:
            st.info("Upload a CSV, or switch to a DAPT2020 day.")
            st.stop()
        try:
            flows = normalise(read_flows(up))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            st.stop()
        key, model_name = f"upload-{up.name}-{len(flows)}", "all"
        st.caption("Uploaded data is scored by the model trained on all five DAPT days, so a DAPT2020 file uploaded here is in-sample; use the day picker for honest held-out scoring.")
    primary = mx["primary_model"]
    which = st.radio("Risk model", ["lag", "world"], index=0 if primary == "lag" else 1,
                     format_func=lambda m: {"lag": "LightGBM + lags", "world": "World model (GRU rollout)"}[m],
                     help=f"The model with the higher pooled leave-one-day-out PR-AUC is '{primary}'.")
    default_thr = float(mx["alert_threshold_1_per_host_hour"] or 0.1) if which == primary else float(mx["pooled"][which]["thr@1.0/h"])
    thr = st.slider("Alert threshold", 0.01, 0.99, float(min(max(default_thr, 0.01), 0.99)), 0.01,
                    help="Default: the threshold that gave 1 false alarm per host-hour on the pooled held-out benign minutes.")

try:
    tab, pr = dapt_run(key, flows, model_name)
except ValueError as e:
    st.error(str(e))
    st.stop()
b = dapt_bundle(model_name)
p = pr[which]
T = infer.times(tab)
labelled = bool(flows["labelled"].any())
hosts = sorted(np.unique(tab.host))

st.title("Forecasting attacker progression before compromise")
st.caption(f"Per-host state in {BUCKET}s buckets → forecast the probability of attack traffic in the next {K} minutes → map to a kill-chain stage → explain.")

# ------------------------------------------------------------------ headline numbers
alarm = p >= thr
c = st.columns(4)
c[0].metric("Flows analysed", f"{len(flows):,}")
c[1].metric("Monitored hosts / host-minutes", f"{len(hosts)} / {tab.n:,}")
c[2].metric("Alert minutes", f"{int(alarm.sum()):,}")
if labelled:
    eps = M.onset_episodes(tab.attack, tab.start, tab.end, tab.K)
    lt = M.lead_times(p, eps, tab.K, thr) if len(eps) else np.array([])
    negs = tab.valid & ~tab.attack & ~tab.fut
    fa = int((alarm & negs).sum())
    c[3].metric("False alarms / host-hour", f"{fa / max(negs.sum() * BUCKET / 3600, 1e-9):.2f}")
    d = st.columns(3)
    d[0].metric("Attack onsets in this capture", len(eps))
    d[1].metric("Onsets warned ≥1 min ahead", f"{int((lt > 0).sum())} ({(lt > 0).mean():.0%})" if len(eps) else "–")
    d[2].metric("Median warning lead time", f"{np.median(lt[lt > 0]) * BUCKET / 60:.0f} min" if (lt > 0).any() else "–")
else:
    c[3].metric("Ground truth", "not in file")

tabs = st.tabs(["Risk timeline", "Explain a prediction", "Network heatmap", "Benchmark & evidence", "How it works"])

# ------------------------------------------------------------------ 1. timeline
with tabs[0]:
    rank = pd.DataFrame({"host": tab.host, "p": p, "alarm": alarm, "attack": tab.attack}).groupby("host").agg(
        peak_risk=("p", "max"), alert_minutes=("alarm", "sum"), attack_minutes=("attack", "sum")).sort_values("peak_risk", ascending=False)
    host = st.selectbox("Host", list(rank.index), format_func=lambda h: f"{h}   (peak risk {rank.loc[h, 'peak_risk']:.2f})")
    m = tab.host == host
    d = pd.DataFrame({"time": T[m], "p": p[m], "stage": [STAGE_NAMES[s] for s in tab.stage[m]], "attack": tab.attack[m]})
    d["end"] = d["time"] + pd.Timedelta(seconds=BUCKET)
    ymax = max(0.05, float(d["p"].max()) * 1.15)
    line = alt.Chart(d).mark_line(strokeWidth=1.8).encode(
        x=alt.X("time:T", title="time"), y=alt.Y("p:Q", scale=alt.Scale(domain=[0, ymax]), title=f"P(attack in next {K} min)"))
    layers = [line, alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q")]
    layers.append(alt.Chart(d[d["p"] >= thr]).mark_point(color="#d62728", size=45, filled=True).encode(x="time:T", y="p:Q"))
    if labelled and d["attack"].any():
        band = alt.Chart(d[d["attack"]]).mark_rect(opacity=0.28).encode(
            x="time:T", x2="end:T", color=alt.Color("stage:N", scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())), title="True stage"))
        layers.insert(0, band)
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    st.caption("Line: forecast risk. Dashed: alert threshold. Red dots: alerts. Shaded bands: true attack minutes (labelled data only). "
               "An alert *before* a band starts is early warning; an alert with no band within the next few minutes is a false alarm.")
    st.dataframe(rank.rename(columns={"peak_risk": "Peak risk", "alert_minutes": "Alert minutes", "attack_minutes": "True attack minutes"}).style.format({"Peak risk": "{:.2f}"}),
                 width="stretch")

# ------------------------------------------------------------------ 2. explain
with tabs[1]:
    hx = st.selectbox("Host ", list(rank.index), key="hx")
    idxs = np.flatnonzero(tab.host == hx)
    order = idxs[np.argsort(-p[idxs])][:25]
    pick = st.selectbox("Highest-risk minutes for this host", order,
                        format_func=lambda i: f"{T[i]:%d %b %H:%M}   risk {p[i]:.2f}" + ("   ● attack in progress" if labelled and tab.attack[i] else ""))
    i = int(pick)
    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"P(attack within {K} min) = {p[i]:.0%}")
        if labelled:
            fut = tab.fut[i] if tab.valid[i] else None
            st.write("Current minute is under attack." if tab.attack[i] else
                     ("**What happened next: an attack started within the horizon.**" if fut else
                      ("What happened next: no attack within the horizon." if fut is not None else "Horizon runs past the end of the capture.")))
        st.markdown("**Predicted stage of the forecast attack**")
        sp = pr["stage"][i]
        sdf = pd.DataFrame({"stage": STAGE_NAMES[1:], "prob": sp[1:]})
        st.altair_chart(alt.Chart(sdf).mark_bar().encode(x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="probability"), y=alt.Y("stage:N", sort=STAGE_NAMES[1:], title=None),
                                                       color=alt.Color("stage:N", legend=None, scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())))).properties(height=140), width="stretch")
        acc, maj = mx["stage"]["accuracy"], mx["stage"]["majority_baseline_accuracy"]
        st.caption(f"Read with care: across days this stage model scored {acc:.0%} (majority-class baseline {maj:.0%}); it only works when the attack type was seen in training. See Benchmark.")
    with right:
        st.subheader("Why (TreeSHAP, log-odds)")
        dr = infer.top_drivers(b, pr, i, which)
        ddf = pd.DataFrame(dr, columns=["feature", "contribution"])
        st.altair_chart(alt.Chart(ddf).mark_bar().encode(
            x=alt.X("contribution:Q", title="pushes risk up → / ← down"), y=alt.Y("feature:N", sort=None, title=None),
            color=alt.condition(alt.datum.contribution > 0, alt.value("#d62728"), alt.value("#1f77b4"))).properties(height=260), width="stretch")
    st.subheader("World-model forward simulation")
    feats = ["in_n_flows", "in_n_ports", "in_syn", "out_n_flows"]
    fcast, obs = infer.rollout_counts(b, tab, pr, i, feats)
    ks = np.arange(1, K + 1)
    rows = []
    for j, f in enumerate(feats):
        rows += [{"feature": human(f), "minutes ahead": int(k), "value": float(fcast[k - 1, j]), "series": "simulated by the model"} for k in ks]
        rows += [{"feature": human(f), "minutes ahead": int(k), "value": float(obs[k - 1, j]), "series": "actually observed"} for k in ks if not np.isnan(obs[k - 1, j])]
    rdf = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(rdf).mark_line(point=True).encode(x=alt.X("minutes ahead:Q", axis=alt.Axis(values=list(ks))), y=alt.Y("value:Q", title="per-minute value"),
                                                                 color="series:N", strokeDash="series:N").facet(facet=alt.Facet("feature:N", title=None), columns=2), width="stretch")
    st.caption("A GRU trained to predict the next host state is fed its own output for K steps. Compare the simulated and observed lines: this is the dynamics model. "
               "The Benchmark tab shows how much it improves the final forecast.")
    st.subheader("Flows around this minute")
    fl = infer.flows_in_window(flows, hx, T[i], K + 1)
    st.dataframe(fl.head(200), width="stretch", height=240)

# ------------------------------------------------------------------ 3. heatmap
with tabs[2]:
    hm = pd.DataFrame({"host": tab.host, "time": T, "risk": p})
    st.altair_chart(alt.Chart(hm).mark_rect().encode(x=alt.X("time:T", title="time"), y=alt.Y("host:N", title=None),
                                                     color=alt.Color("risk:Q", scale=alt.Scale(scheme="orangered", domain=[0, max(0.05, float(p.max()))]), title="risk")).properties(height=60 + 34 * len(hosts)), width="stretch")
    if labelled:
        at = pd.DataFrame({"host": tab.host[tab.attack], "time": T[tab.attack], "stage": [STAGE_NAMES[s] for s in tab.stage[tab.attack]]})
        st.altair_chart(alt.Chart(at).mark_tick(thickness=3, size=26).encode(x="time:T", y=alt.Y("host:N", title=None),
                                                                             color=alt.Color("stage:N", scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())))).properties(height=60 + 34 * len(hosts)), width="stretch")
        st.caption("Top: forecast risk per host. Bottom: true attack minutes by stage.")

# ------------------------------------------------------------------ 4. benchmark
with tabs[3]:
    NAMES = {"lr": "Logistic regression", "cur": "LightGBM (current minute)", "lag": "LightGBM (+ lags)", "world": "World model (GRU rollout + head)",
             "lag_shuf": "LightGBM (+ lags), history SHUFFLED", "world_shuf": "World model, history SHUFFLED"}
    pooled, boot = mx["pooled"], mx["bootstrap"]
    prev = pooled["lr"]["prevalence"]

    def table(pool, ci=None):
        rows = []
        for k, nm in NAMES.items():
            r = pool[k]
            row = {"Model": nm, "PR-AUC": r.get("pr_auc"), "ROC-AUC": r.get("roc_auc"), "Recall @1/host-h": r.get("recall@1.0/h"),
                   "Precision @1/host-h": r.get("precision@1.0/h"), "F1 @1/host-h": r.get("f1@1.0/h"), "FPR @1/host-h": r.get("fpr@1.0/h"),
                   "Onsets warned": r.get("onset_detected@1.0/h"), "Median lead (min)": r.get("median_lead_min@1.0/h"), "Brier": r.get("brier"), "ECE": r.get("ece")}
            if ci:
                lo, hi = ci[k]["pr_auc_ci90"]
                row["PR-AUC 90% CI"] = f"{lo:.2f}–{hi:.2f}"
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")

    st.subheader("Primary: leave-one-day-out (each day forecast by models that never saw it)")
    st.caption(f"Onset task: minutes where the host is currently benign; positive = attack traffic starts within {K} min. Base rate (a random scorer's PR-AUC) = {prev:.3f}. "
               f"{ds['onset_episodes']} onset episodes on {ds['host_days']} host-days across {ds['internal_hosts']} hosts, only 3 of which are ever attacked. Wide CIs are the honest consequence.")
    fmt = {c: "{:.3f}" for c in ["PR-AUC", "ROC-AUC", "Recall @1/host-h", "Precision @1/host-h", "F1 @1/host-h", "FPR @1/host-h", "Brier", "ECE"]}
    fmt.update({"Onsets warned": "{:.0%}", "Median lead (min)": "{:.1f}"})
    st.dataframe(table(pooled, boot).style.format(fmt, na_rep="–"), width="stretch")

    pdays = pd.DataFrame({NAMES[k]: {d: v.get("roc_auc") for d, v in mx["per_day"][k].items()} for k in ("lr", "cur", "lag", "world")}).T
    st.markdown("**Within-day ranking quality (ROC-AUC per held-out day).** Days with no attacks are blank.")
    st.dataframe(pdays.style.format("{:.2f}", na_rep="–"), width="stretch")

    st.subheader("Secondary: within-day blocked split (attack types shared between train and test)")
    sb = mx["secondary_blocked"]
    st.caption(sb["split"] + ". Easier by construction; never used to pick a model.")
    st.dataframe(table(sb["pooled"]).style.format(fmt, na_rep="–"), width="stretch")

    st.subheader("Stage forecast")
    s1, s2 = mx["stage"], sb["stage"]
    st.dataframe(pd.DataFrame({"Leave-one-day-out": [s1["accuracy"], s1["majority_baseline_accuracy"], s1["n"]],
                               "Within-day blocked": [s2["accuracy"], s2["majority_baseline_accuracy"], s2["n"]]},
                              index=["Accuracy", "Majority-class baseline", "Onset minutes evaluated"]).style.format("{:.2f}"), width="stretch")

    lag, wld = pooled["lag"], pooled["world"]
    shuf = lambda a, s: (pooled[s]["pr_auc"] - pooled[a]["pr_auc"]) / pooled[a]["pr_auc"]
    st.subheader("What this evidence does and does not support")
    lo, hi = boot["lag"]["pr_auc_ci90"]
    st.markdown(
        f"- **Ranking works, thresholds don't transfer.** Forecast PR-AUC {lag['pr_auc']:.2f} vs base rate {prev:.2f} pooled ({lag['pr_auc'] / prev:.1f}× lift); within a single held-out day ROC-AUC is 0.74–0.95. "
        f"Pooled alarm-budget recall is only {lag['recall@1.0/h']:.0%} because each day's attack type shifts the score scale.\n"
        f"- **World model vs GBDT with lags:** {wld['pr_auc']:.2f} vs {lag['pr_auc']:.2f} PR-AUC (lag 90% CI {lo:.2f}–{hi:.2f}). "
        f"{'No measurable improvement on this data.' if wld['pr_auc'] <= hi else 'Above the lag CI.'}\n"
        f"- **Shuffle control:** shuffling the order of the history minutes changes PR-AUC by {shuf('lag', 'lag_shuf'):+.0%} (lag) and {shuf('world', 'world_shuf'):+.0%} (world). "
        f"{'Neither model loses skill, so on this data the forecast comes from the current state, not from learned temporal dynamics.' if min(shuf('lag', 'lag_shuf'), shuf('world', 'world_shuf')) > -0.1 else 'At least one model relies on temporal order.'}\n"
        f"- **Stage forecasting** is at chance across days ({s1['accuracy']:.0%} vs {s1['majority_baseline_accuracy']:.0%} majority) because each capture day contains a different stage; "
        f"it reaches {s2['accuracy']:.0%} vs {s2['majority_baseline_accuracy']:.0%} majority when stages are shared.\n"
        f"- **Scope:** {ds['internal_hosts']} monitored hosts, {ds['attack_buckets']} attack minutes, one lab, one week. The pipeline is the deliverable; the numbers are what this small corpus supports.")

# ------------------------------------------------------------------ 5. how it works
with tabs[4]:
    st.markdown(f"""
**Pipeline (same code for training, benchmarking, this app and the CLI)**

1. **Ingest** CICFlowMeter-style flow CSVs; fix headerless files, unify labels, parse timestamps, drop flows duplicated between capture points.
2. **Entity-centric state.** Each *monitored* (private-address) host gets one feature vector per {BUCKET}s wall-clock bucket: inbound and outbound flow counts, distinct peers and ports, packets, bytes, SYN/RST/ACK/PSH/FIN counts, port-class shares (web, SSH, database, other-low, ephemeral). Signed log1p on heavy-tailed features; scalers fit on training rows only. IPs, raw ports and timestamps are never inputs.
3. **Dynamics.** A residual GRU learns P(next state | last {ds['history_L_buckets']} minutes), then is rolled forward {K} steps on its own predictions.
4. **Forecast.** A gradient-boosted head reads the current state, the simulated future (mean, max) and the latent state to output P(attack in the next {K} minutes). A separate model maps a forecast onset to a kill-chain stage.
5. **Explain.** Exact TreeSHAP attributions per prediction, plus the simulated-vs-observed trajectories.

**Stage mapping.** DAPT2020 labels → ATT&CK tactics: Reconnaissance → Reconnaissance, Establish Foothold → Initial Access, Lateral Movement → Lateral Movement, Data Exfiltration → Exfiltration. DAPT has no Command & Control label.

**Evaluation discipline.** Leave-one-day-out is primary; the blocked split is secondary. Alarm budgets are set on held-out benign minutes. Every experiment has a hypothesis and prediction in `results/registry.csv`, written before the run. Leakage tests: `make test`.

**Data used here:** {ds['source']}, {ds['flows_after_dedup']:,} flows, {ds['days'][0]} to {ds['days'][-1]}.
""")
