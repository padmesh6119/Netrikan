"""ZeekData24 page: technique recognition (ATT&CK mapping) + attacker-campaign forecasting.

Every week is scored by models trained WITHOUT that week. Numbers come from the loaded flows or
from results/z24/metrics.json.
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
from netrikan import campaign as C  # noqa: E402
from netrikan import metrics as M  # noqa: E402
from netrikan.features import EPOCH, FEATURES, build_grid, human  # noqa: E402
from netrikan.zeek import MONITORED, TECH_LABEL, TECH_NAME, TECH_TACTIC, TECHS, load_z24  # noqa: E402

MODEL_DIR, METRICS = "models/z24", "results/z24/metrics.json"
COL = {"T1595": "#e6a100", "T1190": "#e4572e", "T1078": "#9c27b0", "T1110": "#1565c0", "T1048": "#b71c1c"}
if not (os.path.exists(METRICS) and os.path.isdir(MODEL_DIR)):
    st.error("No ZeekData24 models found. Run `make z24-train` first.")
    st.stop()


@st.cache_data
def z24_metrics():
    return json.load(open(METRICS))


@st.cache_resource
def z24_bundle(name):
    return joblib.load(f"{MODEL_DIR}/{name}.joblib")


@st.cache_resource(show_spinner="Loading the week's flows, bucketing per host and scoring…")
def z24_analyse(week, model_name, thr_flag):
    b = z24_bundle(model_name)
    flows = load_z24(weeks=[week])
    grid = build_grid(flows, 60, MONITORED, tuple(TECH_LABEL))
    X = C.log_state(grid)
    Y = grid[TECH_LABEL].to_numpy().astype(bool)
    traffic = ((grid["in_n_flows"] + grid["out_n_flows"]) > 0).to_numpy()
    P = C.predict_recognizer(b["recog"], X, traffic)
    Fp = P >= thr_flag
    day2wk = {d: week for d in grid["day"].unique()}
    S = fut = valid = pr = None
    try:
        S = C.build_seqs(grid, X, Y, Fp, day2wk, select=Fp)  # attackers are chosen from PREDICTED flags: no ground truth used
        fut, valid = C.targets(S)
        pr = C.predict_forecasters(b["models"], S, S.Fp, world=b["world"])
    except ValueError:
        S = None  # no host sends flagged traffic in this week
    meta = grid[["host", "day", "bucket", "in_n_flows", "out_n_flows"]].copy()
    return {"flows": flows, "meta": meta, "X": X, "Y": Y, "traffic": traffic, "P": P, "S": S, "fut": fut, "valid": valid, "pr": pr}


mx = z24_metrics()
ds = mx["dataset"]
K = ds["horizon_K_minutes"]
weeks = ds["attack_weeks"] + ds["benign_weeks"]
RUNG = {"sched": "Schedule features (history GBDT)", "world": "World model (GRU rollout + head)", "renewal": "Renewal hazard baseline",
        "own": "Own-technique history", "state": "Current behaviour only", "clock": "Clock-aware ABLATION (not a valid model)"}

with st.sidebar:
    st.title("ZeekData24")
    st.caption("Zeek conn logs · ATT&CK technique recognition + attacker forecasting · offline")
    labels = [f"{w} ({'attack' if w in ds['attack_weeks'] else 'benign'})" for w in weeks]
    week = weeks[labels.index(st.selectbox("Capture week", labels))]
    is_attack_week = week in ds["attack_weeks"]
    model_name = f"fold_{week if is_attack_week else ds['benign_held_out_by'][week]}"
    st.caption("This week is scored by models trained on the *other* attack weeks and the other benign week.")
    thr_flag = st.slider("Recognizer threshold", 0.1, 0.9, 0.5, 0.05, help="Fixed at 0.5 in all reported results.")
    rung = st.radio("Forecast model", ["sched", "world", "renewal"], format_func=lambda r: RUNG[r])
    budget = st.select_slider("Alert budget (false alarms per attacker-hour)", [0.5, 1.0, 2.0], value=1.0,
                              help="Thresholds come from held-out negative minutes in the leave-one-week-out evaluation.")

R = z24_analyse(week, model_name, thr_flag)
meta, P, Y, S = R["meta"], R["P"], R["Y"], R["S"]
T = pd.to_datetime(meta["bucket"].to_numpy().astype("int64") * 60, unit="s")
flagged = P >= thr_flag

st.title("Attacker-campaign recognition and forecasting")
st.caption("Behaviour in each host-minute → which ATT&CK technique is running → when will this attacker's next burst of it start.")
c = st.columns(4)
c[0].metric("Flows", f"{len(R['flows']):,}")
c[1].metric("Host-minutes with traffic", f"{int(R['traffic'].sum()):,}")
c[2].metric("Minutes flagged (any technique)", f"{int(flagged.any(1).sum()):,}")
if is_attack_week:
    c[3].metric("Attacker sequences found", 0 if S is None else len(S.seq_bounds()))
else:
    ben_min = max(int(len(meta)), 1)
    c[3].metric("False alarms / host-hour (benign)", f"{flagged.any(1).sum() / (ben_min / 60):.2f}")

tabs = st.tabs(["Recognize techniques", "Forecast attacker bursts", "Campaign structure", "Benchmark & evidence"])

# ------------------------------------------------------------------ 1. recognize
with tabs[0]:
    rows = []
    for j, t in enumerate(TECHS):
        row = {"Technique": f"{t} {TECH_NAME[t]}", "ATT&CK tactic": TECH_TACTIC[t], "Flagged minutes": int(flagged[:, j].sum())}
        if is_attack_week:
            tp = int((flagged[:, j] & Y[:, j]).sum())
            row.update({"True minutes": int(Y[:, j].sum()), "Recall": tp / max(int(Y[:, j].sum()), 1), "Precision": tp / max(int(flagged[:, j].sum()), 1)})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Technique").style.format({"Recall": "{:.2f}", "Precision": "{:.2f}"}, na_rep="–"), width="stretch")
    if not is_attack_week:
        st.info("This is a benign week: every flagged minute is a false alarm.")
    hosts = pd.Series(meta["host"].to_numpy()[flagged.any(1)]).value_counts()
    if len(hosts):
        host = st.selectbox("Host", list(hosts.index), format_func=lambda h: f"{h}  ({hosts[h]} flagged minutes)")
        m = (meta["host"].to_numpy() == host) & R["traffic"]
        long = []
        for j, t in enumerate(TECHS):
            for i in np.flatnonzero(m & flagged[:, j]):
                long.append({"time": T[i], "technique": TECH_NAME[t], "kind": "recognized"})
            if is_attack_week:
                for i in np.flatnonzero(m & Y[:, j]):
                    long.append({"time": T[i], "technique": TECH_NAME[t], "kind": "ground truth"})
        if long:
            d = pd.DataFrame(long)
            if len(d) > 4000:
                d = d.sample(4000, random_state=0)
            order = [TECH_NAME[t] for t in TECHS]
            ch = alt.Chart(d).mark_tick(thickness=2, size=14).encode(
                x=alt.X("time:T"), y=alt.Y("technique:N", sort=order, title=None),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["recognized", "ground truth"], range=["#d62728", "#444444"])),
                yOffset="kind:N")
            st.altair_chart(ch.properties(height=230), width="stretch")
            st.caption("Each tick is one host-minute. Red: recognized by the model. Dark: ground truth (labelled weeks only).")
        # explanation for one flagged minute
        idx = np.flatnonzero(m & flagged.any(1))
        top = idx[np.argsort(-P[idx].max(1))][:20]
        pick = st.selectbox("Explain a flagged minute", top, format_func=lambda i: f"{T[i]:%d %b %H:%M}  " + ", ".join(f"{TECHS[k]} {P[i, k]:.2f}" for k in np.argsort(-P[i])[:2]))
        i = int(pick)
        ex = st.columns([1, 1])
        with ex[0]:
            pdf = pd.DataFrame({"technique": [f"{t} {TECH_NAME[t]}" for t in TECHS], "probability": P[i]})
            st.altair_chart(alt.Chart(pdf).mark_bar().encode(x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("technique:N", sort=None, title=None),
                                                           color=alt.Color("technique:N", legend=None, scale=alt.Scale(domain=list(pdf["technique"]), range=list(COL.values())))).properties(height=170), width="stretch")
        with ex[1]:
            jbest = int(np.argmax(P[i]))
            contrib = z24_bundle(model_name)["recog"][jbest].predict(R["X"][i:i + 1], pred_contrib=True)[0][:-1]
            o = np.argsort(-np.abs(contrib))[:7]
            cdf = pd.DataFrame({"feature": [human(FEATURES[q]) for q in o], "contribution": contrib[o]})
            st.altair_chart(alt.Chart(cdf).mark_bar().encode(x=alt.X("contribution:Q", title=f"TreeSHAP for {TECH_NAME[TECHS[jbest]]} (log-odds)"), y=alt.Y("feature:N", sort=None, title=None),
                                                           color=alt.condition(alt.datum.contribution > 0, alt.value("#d62728"), alt.value("#1f77b4"))).properties(height=200), width="stretch")
    else:
        st.write("No minute was flagged in this week.")

# ------------------------------------------------------------------ 2. forecast
with tabs[1]:
    if S is None:
        st.info("No host sends technique-flagged traffic in this week, so there is no attacker to forecast.")
    else:
        pr, fut, valid = R["pr"], R["fut"], R["valid"]
        hosts_s = sorted(set(S.host))
        c1, c2, c3 = st.columns(3)
        host = c1.selectbox("Attacker", hosts_s, key="atk")
        tj = c2.selectbox("Technique", list(range(len(TECHS))), format_func=lambda j: f"{TECHS[j]} {TECH_NAME[TECHS[j]]}")
        rows = np.flatnonzero(S.host == host)
        Tt = pd.to_datetime(S.bucket[rows].astype("int64") * 60, unit="s")
        days = sorted(set(Tt.date))
        day = c3.selectbox("Day", days)
        pj = pr[rung][:, tj]
        thr = mx["forecast"]["per_technique"][rung][TECHS[tj]][f"thr@{budget}/h"]
        base = pr["renewal"][:, tj]
        sel = rows[[d == day for d in Tt.date]]
        d = pd.DataFrame({"time": pd.to_datetime(S.bucket[sel].astype("int64") * 60, unit="s"), "model": pj[sel], "renewal baseline": base[sel]})
        long = d.melt("time", var_name="series", value_name="p")
        line = alt.Chart(long).mark_line().encode(x=alt.X("time:T"), y=alt.Y("p:Q", title=f"P({TECH_NAME[TECHS[tj]]} burst in next {K} min)"),
                                                  color=alt.Color("series:N", scale=alt.Scale(domain=["model", "renewal baseline"], range=["#1f77b4", "#aaaaaa"])))
        layers = [line, alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q")]
        bursts = pd.DataFrame({"time": pd.to_datetime(S.bucket[sel][S.F[sel, tj]].astype("int64") * 60, unit="s")})
        if len(bursts):
            layers.append(alt.Chart(bursts).mark_rule(color="#d62728", opacity=0.55).encode(x="time:T"))
        al = sel[(pj[sel] >= thr) & ~S.F[sel, tj]]
        if len(al):
            layers.append(alt.Chart(pd.DataFrame({"time": pd.to_datetime(S.bucket[al].astype("int64") * 60, unit="s"), "p": pj[al]})).mark_point(color="#d62728", filled=True, size=40).encode(x="time:T", y="p:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=300), width="stretch")
        st.caption("Blue: forecast from this attacker's recent history. Grey: renewal baseline. Red rules: actual bursts. Red dots: alerts at the chosen budget. "
                   "An alert followed by a rule within a few minutes is early warning.")
        # week-level summary for this technique
        allr = np.arange(S.n)
        popj = valid & ~S.F[:, tj]
        alarms = (pr[rung][:, tj] >= thr)
        negs = popj & ~fut[:, tj]
        far = (alarms & negs).sum() / max(negs.sum() / 60, 1e-9)
        eps = M.onset_episodes(S.F[:, tj], S.start, S.end, K)
        lt = M.lead_times(pr[rung][:, tj], eps, K, thr) if len(eps) else np.array([])
        s = st.columns(4)
        s[0].metric("Bursts (onsets) this week", len(eps))
        s[1].metric("Warned ≥1 min ahead", f"{(lt > 0).mean():.0%}" if len(eps) else "–")
        s[2].metric("False alarms / attacker-hour", f"{far:.2f}")
        s[3].metric("Median lead when warned", f"{np.median(lt[lt > 0]):.0f} min" if (lt > 0).any() else "–")
        # explanation
        st.subheader("Why this forecast (TreeSHAP)")
        if rung == "sched":
            cand = sel[~S.F[sel, tj]]
            top = cand[np.argsort(-pj[cand])][:20]
            pick = st.selectbox("Highest-risk minutes", top, format_func=lambda i: f"{pd.to_datetime(S.bucket[i] * 60, unit='s'):%H:%M}  risk {pj[i]:.2f}")
            i = int(pick)
            sch = C.sched_feats(S.subset(np.arange(S.start[i], S.end[i])), S.Fp[S.start[i]:S.end[i]])
            Xr = np.c_[sch[i - S.start[i]][None, :], S.X[i][None, :]]
            contrib = z24_bundle(model_name)["models"]["sched"][tj].predict(Xr, pred_contrib=True)[0][:-1]
            names = C.SCHED_NAMES + [human(f) for f in FEATURES]
            o = np.argsort(-np.abs(contrib))[:8]
            cdf = pd.DataFrame({"feature": [names[q].replace("|", " · ") for q in o], "contribution": contrib[o]})
            st.altair_chart(alt.Chart(cdf).mark_bar().encode(x=alt.X("contribution:Q", title="log-odds"), y=alt.Y("feature:N", sort=None, title=None),
                                                           color=alt.condition(alt.datum.contribution > 0, alt.value("#d62728"), alt.value("#1f77b4"))).properties(height=240), width="stretch")
            st.caption("'since' = minutes since the last recognized burst of that technique; 'n60'/'n150' = flagged minutes in the last 60/150 minutes.")
        else:
            st.caption("TreeSHAP is shown for the schedule-feature model. Switch the forecast model to 'Schedule features' in the sidebar.")

# ------------------------------------------------------------------ 3. structure
with tabs[2]:
    st.subheader("How the campaign is actually structured")
    st.markdown("Computed live from this week's flows (attack weeks) and from all attack weeks (table below).")
    fl = R["flows"]
    if is_attack_week:
        rows = []
        for t in TECHS[:4]:
            x = fl[fl[f"t_{t}"]]
            mm = ((x["ts"] - EPOCH).dt.total_seconds() // 60).astype(int)
            first = (mm.groupby([x["Src IP"], mm // 60]).min() % 60)
            rows += [{"technique": TECH_NAME[t], "minute of hour": int(v)} for v in first]
        h = pd.DataFrame(rows)
        st.altair_chart(alt.Chart(h).mark_bar().encode(x=alt.X("minute of hour:Q", bin=alt.Bin(step=5)), y="count():Q", color=alt.Color("technique:N")).properties(height=200), width="stretch")
        st.caption("First flow of each technique in each clock hour, per attacker. It is spread evenly over the hour: the bursts are not tied to a clock time.")
        g = []
        for t in TECHS[:4]:
            x = fl[fl[f"t_{t}"]]
            mm = ((x["ts"] - EPOCH).dt.total_seconds() // 60).astype(int)
            for _, v in mm.groupby(x["Src IP"]):
                v = np.sort(v.unique())
                st_ = v[np.r_[True, np.diff(v) > 1]]
                g += [{"technique": TECH_NAME[t], "gap": int(z)} for z in np.diff(st_)]
        gd = pd.DataFrame(g)
        st.altair_chart(alt.Chart(gd[gd.gap <= 180]).mark_bar(opacity=0.8).encode(x=alt.X("gap:Q", bin=alt.Bin(step=5), title="minutes between consecutive bursts"), y="count():Q", color="technique:N").properties(height=200), width="stretch")
        st.caption("Each technique repeats roughly hourly with large jitter, so the gap since the last burst is informative but far from decisive.")
    s = mx["structure"]
    lag = pd.DataFrame({b: {a: v[b]["p_within_5"] for a, v in s["cross_technique_lags"].items() if b in v} for b in TECHS[:4]}).T
    lag.index = [TECH_NAME[b] for b in lag.index]
    lag.columns = [TECH_NAME[a] for a in lag.columns]
    st.markdown("**Does one technique lead to another?** Probability that technique B (rows) starts within 5 minutes after a burst of technique A (columns), all attack weeks:")
    st.dataframe(lag.style.format("{:.2f}", na_rep="–"), width="stretch")
    st.caption(f"A random hourly schedule would give about 0.08. Values of 0.10–0.13 mean the techniques are only weakly coupled: there is no recon → exploit → credential-access chain to learn. "
               f"Minute-of-hour spread: std {np.mean([v['std'] for v in s['minute_of_hour_of_first_flow'].values()]):.1f} min vs {s['uniform_minute_of_hour_std']:.1f} for a perfectly uniform schedule.")
    st.dataframe(pd.DataFrame(s["attackers"]).set_index("host").sort_values("week"), width="stretch", height=260)

# ------------------------------------------------------------------ 4. benchmark
with tabs[3]:
    rec = mx["recognizer"]
    st.subheader("Task A: recognize the technique from one host-minute (leave-one-attack-week-out)")
    ta = pd.DataFrame({f"{t} {TECH_NAME[t]}": {
        "PR-AUC": v["pr_auc"], "Recall @0.5": v["at_0.5"]["recall"], "Precision @0.5": v["at_0.5"]["precision"], "F1 @0.5": v["at_0.5"]["f1"],
        "Benign alarms / host-hour @0.5": v["at_0.5"]["benign_alarms_per_host_hour"], "Fires on other techniques' minutes @0.5": v["at_0.5"]["other_technique_false_fire_rate"],
        "Positive minutes": v["n_pos"]} for t, v in rec["per_technique"].items()}).T
    st.dataframe(ta.style.format({c: "{:.3f}" for c in ta.columns if c != "Positive minutes"} | {"Positive minutes": "{:,.0f}"}), width="stretch")
    a = rec["any_attack_detector"]
    st.caption(f"Any-technique attack-vs-benign detector: PR-AUC {a['pr_auc']:.3f}; at 0.5 it flags {a['at_0.5_recall']:.0%} of attack minutes with {a['at_0.5_benign_alarms_per_host_hour']:.2f} benign alarms per host-hour.")
    conf = pd.DataFrame(rec["confusion@0.5"]).T
    conf.index = [TECH_NAME[t] for t in conf.index]
    conf.columns = [TECH_NAME[t] for t in conf.columns]
    st.markdown("**Confusion at 0.5:** among minutes that contain technique (row), the share flagged as technique (column). A minute can contain several techniques.")
    st.dataframe(conf.style.format("{:.2f}"), width="stretch")

    st.subheader("Task B: forecast the next burst of each technique (macro over 5 techniques)")
    fc = mx["forecast"]
    LAB = {"renewal": "Renewal hazard (time since last burst)", "state": "Current behaviour only", "own": "Own-technique history", "sched": "All-technique history (GBDT)",
           "sched_shuf": "  … history order SHUFFLED", "world": "World model (GRU rollout + head)", "world_shuf": "  … history order SHUFFLED (blocks)",
           "sched_oracle": "All-technique history, TRUE flags (upper bound)", "clock": "ABLATION: + wall-clock minute (violates no-timestamp rule)"}
    rows = {}
    for n, nm in LAB.items():
        v = fc["macro"][n]
        rows[nm] = {"PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Recall @1/h": v["recall@1.0/h"], "Precision @1/h": v["precision@1.0/h"], "F1 @1/h": v["f1@1.0/h"],
                    "Onsets warned": v["onset_detected@1.0/h"], "Brier": v["brier"], "ECE": v["ece"]}
    tb = pd.DataFrame(rows).T
    base = fc["macro"]["sched"]["prevalence"]
    st.caption(f"Population: minutes where the technique is not currently active; positive = a burst starts within {K} min. Base rate (random scorer's PR-AUC) = {base:.3f}. "
               f"{ds['attacker_hosts']} attacker hosts, {ds['attacker_sequences']} host-weeks. \"@1/h\" = threshold set for 1 false alarm per attacker-hour on held-out minutes.")
    st.dataframe(tb.style.format("{:.3f}"), width="stretch")
    pt = pd.DataFrame({LAB[n]: {TECH_NAME[t]: v.get("pr_auc") for t, v in fc["per_technique"][n].items()} for n in ("renewal", "state", "sched", "world", "sched_shuf")}).T
    st.markdown("**PR-AUC per technique**")
    st.dataframe(pt.style.format("{:.3f}"), width="stretch")

    m_ = fc["macro"]
    q = lambda n: m_[n]["pr_auc"]
    st.subheader("What this evidence does and does not support")
    st.markdown(
        f"- **Recognition is easy here, and that is a warning, not a win.** Techniques are separable at PR-AUC {min(v['pr_auc'] for v in rec['per_technique'].values()):.2f}–{max(v['pr_auc'] for v in rec['per_technique'].values()):.2f} "
        f"in a held-out week of the *same scripted campaign*; the attack scripts are highly stereotyped and attack minutes contain no benign traffic. It says little about real-world attacks.\n"
        f"- **Forecasting the next burst is genuinely temporal.** History models reach macro PR-AUC {q('sched'):.2f} against a base rate of {base:.3f} ({q('sched') / base:.1f}×); "
        f"current behaviour alone gives {q('state'):.2f} (no skill). **Shuffling the history order costs {(q('sched') - q('sched_shuf')) / q('sched'):.0%}** (LightGBM) and **{(q('world') - q('world_shuf')) / q('world'):.0%}** (world model), so the models use temporal order.\n"
        f"- **The world model matches, not beats, hand-built history features:** {q('world'):.3f} vs {q('sched'):.3f}. It learns the same signal from raw sequences; on this data there is no extra dynamics to find.\n"
        f"- **The signal is modest:** recall {m_['sched']['recall@1.0/h']:.0%} at one false alarm per attacker-hour; the schedule is roughly hourly with large jitter. Adding the wall clock (an ablation, not allowed in shipped models) reaches only {q('clock'):.2f}, so the jitter is not clock-locked.\n"
        f"- **Not a kill chain.** Other techniques' history adds {(q('sched') - q('own')) / q('own'):.0%} over own-technique history and cross-technique start lags are near random. "
        f"This corpus does not show recon → access → exfiltration progression; do not present it as such.\n"
        f"- **Scope:** one lab, one campaign replayed for 5 weeks, {ds['attacker_hosts']} attacker hosts. Held-out weeks are replays, so this measures repeatability of a schedule, not generalisation to unseen attacks.")
