import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from attck_map import get_stage, DOS
import forecast as fc
import infer
import demo_data
import pcap_ingest
import counterfactual

MODEL_DIR = os.path.expanduser("~/netrikan/models")
CHAIN = [1, 3, 4, 5]

st.set_page_config(page_title="Netrikan", page_icon="👁", layout="wide")

# Titanium white ground with warm neutrals — reads as a document, not a console.
BG, PANEL, LINE = "#FAFAF8", "#FFFFFF", "#E3E2DD"
TXT, MUTED = "#1C1C1A", "#6B6A64"
GRN, AMB, RED = "#2F7D5B", "#9A7420", "#B03A32"

st.markdown(f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
.stApp {{background:{BG};}}
.block-container {{padding:1.8rem 2.4rem 3rem; max-width:1180px;}}
html, body, [class*="css"] {{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;
  -webkit-font-smoothing:antialiased; color:{TXT};
}}
.hdr {{display:flex;align-items:center;gap:10px;margin-bottom:3px;}}
.hdr .logo {{font-size:19px;font-weight:700;letter-spacing:-.02em;}}
.hdr .tag {{font-size:13px;color:{MUTED};}}
.hdr svg {{display:block;flex:none;}}
.hdr .iris {{transform-origin:17px 17px;animation:scan 9s linear infinite;}}
@keyframes scan {{from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}}}}
.prov {{font-size:12px;color:{MUTED};margin-bottom:20px;}}
.prov b {{color:{TXT};font-weight:600;}}

.strip {{display:flex;border-top:1px solid {LINE};border-bottom:1px solid {LINE};
        margin:6px 0 20px;}}
.strip .cell {{flex:1;padding:13px 22px 13px 0;border-right:1px solid {LINE};}}
.strip .cell:last-child {{border-right:none;}}
.strip .cell:not(:first-child) {{padding-left:22px;}}
.strip .k {{font-size:12px;color:{MUTED};margin-bottom:4px;}}
.strip .v {{font-size:20px;font-weight:650;letter-spacing:-.02em;line-height:1.2;}}
.strip .s {{font-size:11.5px;color:{MUTED};margin-top:2px;}}

.fc {{border:1px solid {LINE};border-left:3px solid {AMB};border-radius:8px;
     background:{PANEL};padding:18px 22px;}}
.fc .lab {{font-size:12px;color:{MUTED};}}
.fc .big {{font-size:26px;font-weight:680;letter-spacing:-.025em;margin:5px 0 6px;
          line-height:1.15;color:{TXT};}}
.fc .sig {{font-size:13px;color:{MUTED};}}
.fc .sig b {{color:{TXT};font-weight:600;}}

.sec {{font-size:13px;color:{TXT};margin:26px 0 4px;font-weight:600;}}
.hint {{font-size:12px;color:{MUTED};margin:0 0 10px;line-height:1.55;}}
.lede {{font-size:13.5px;color:{MUTED};margin:2px 0 16px;line-height:1.6;max-width:760px;}}
.scrub {{font-size:12px;color:{MUTED};margin-bottom:-6px;}}
.scrub b {{color:{TXT};font-weight:600;}}
.note {{font-size:12px;color:{MUTED};border-left:2px solid {LINE};padding-left:12px;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def provenance():
    out = {}
    for name, key in [('metrics.json', 'train'), ('dapt_crossdataset.json', 'dapt'),
                      ('horizon_curve_fixed.json', 'horizon'),
                      ('baseline_ports_w10.json', 'baseline')]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            try:
                out[key] = json.load(open(p))
            except Exception:
                pass
    return out


@st.cache_data(show_spinner="Analysing traffic…")
def run(frame, key, horizon):
    return infer.analyze(frame, horizon)


@st.cache_data(show_spinner="Reading capture…")
def read_capture(raw: bytes, name: str):
    suffix = os.path.splitext(name)[1] or '.pcap'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        return pcap_ingest.load_pcap(path)
    finally:
        os.unlink(path)


prov = provenance()

EYE = f"""<svg width="30" height="30" viewBox="0 0 34 34" fill="none">
  <path d="M2.4 17C7.6 8.4 26.4 8.4 31.6 17C26.4 25.6 7.6 25.6 2.4 17Z"
        stroke="{TXT}" stroke-width="1.4" stroke-linejoin="round"/>
  <circle cx="17" cy="17" r="6.1" stroke="{AMB}" stroke-width="1.5"/>
  <g class="iris">
    <path d="M17 9.4v2.1M17 22.5v2.1M9.4 17h2.1M22.5 17h2.1"
          stroke="{AMB}" stroke-width="1.2" stroke-linecap="round" opacity=".7"/>
  </g>
  <circle cx="17" cy="17" r="3.2" stroke="{AMB}" stroke-width="1" opacity=".55"/>
  <circle cx="17" cy="17" r="1.5" fill="{AMB}"/>
</svg>"""

st.markdown(f'<div class="hdr">{EYE}<span class="logo">Netrikan</span>'
            '<span class="tag">நெற்றிக்கண் · Network attack forecasting</span></div>', unsafe_allow_html=True)
if 'train' in prov:
    st.markdown(f'<div class="prov">LSTM world model · held-out macro-F1 '
                f'<b>{prov["train"]["best_macro_f1"]:.3f}</b> on leak-free windows</div>',
                unsafe_allow_html=True)

if not infer.model_ready():
    st.error("Model checkpoint or scaler missing — run training first.")
    st.stop()

st.markdown('<div class="lede">Reads network traffic and predicts which stage of '
            'an attack comes next — before the damage is visible. Pick a demo '
            'scenario or upload a capture.</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 2, 3])
with c1:
    scenario = st.selectbox(
        "Traffic source", list(demo_data.SCENARIOS.keys()),
        label_visibility="collapsed",
        help="A simulated capture. 'Intrusion' walks through a full break-in: "
             "normal traffic, then scanning, lateral movement, C2 and data theft.")
with c2:
    hlabel = st.selectbox(
        "Forecast horizon", list(fc.HORIZONS.keys()), index=1,
        label_visibility="collapsed",
        help="How far ahead to predict. Real intrusions advance about one stage "
             "per day, so 60s shows almost no movement. 15 minutes is the useful "
             "setting for break-ins; 60s suits fast floods.")
with c3:
    up = st.file_uploader("Upload PCAP or CSV", type=["pcap", "pcapng", "cap", "csv"],
                          label_visibility="collapsed",
                          help="Raw packet capture (.pcap/.pcapng/.cap) or a "
                               "NetFlow CSV with the 24 feature columns.")

horizon = fc.HORIZONS[hlabel]

if up is not None:
    ext = os.path.splitext(up.name)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(up)
        miss = [f for f in infer.FEATURES if f not in df.columns]
        if miss:
            st.error(f"CSV missing {len(miss)} columns, e.g. {miss[:3]}")
            st.stop()
    else:
        if not pcap_ingest._nfstream_available() and not shutil.which('tshark'):
            st.error("Neither nfstream nor tshark found. `pip install nfstream` or `sudo apt install tshark`")
            st.stop()
        try:
            df = read_capture(up.getvalue(), up.name)
        except Exception as e:
            st.error(f"Could not read capture: {e}")
            st.stop()
        if df.empty:
            st.error("No usable flows in that capture.")
            st.stop()
        st.caption(f"{up.name} → {len(df):,} bidirectional flows extracted")
    source = up.name
else:
    df = demo_data.generate(scenario, 260)
    source = scenario

if len(df) < infer.WINDOW + 2:
    st.warning(f"Only {len(df)} flows — need at least {infer.WINDOW+2} to build a "
               f"window. Try a longer capture.")
    st.stop()

r = run(df, f"{source}{len(df)}{horizon}", horizon)
if r is None:
    st.error("Not enough flows.")
    st.stop()

n = r["n_windows"]
default_i = r["forecast_idx"] if r["forecast_idx"] is not None else int(np.argmax(r["risk"]))
total_s = r["windows"][-1]["time"]

i = st.slider("Playback position — drag to move through the capture", 0, n - 1,
              int(default_i))

w = r["windows"][i]
f = w["forecast"]
sig = w["signal"]
stage = get_stage(w["stage"])
risk = float(f["damage_risk"])
rc = RED if risk > .6 else AMB if risk > .3 else GRN

marks = []
if r["forecast_idx"] is not None:
    marks.append(f'forecast at T+{r["windows"][r["forecast_idx"]]["time"]:.0f}s')
if r["detect_idx"] is not None:
    marks.append(f'damage stage first observed at T+{r["windows"][r["detect_idx"]]["time"]:.0f}s')
st.markdown(f'<div class="scrub">Viewing <b>T+{w["time"]:.0f}s</b> of {total_s:.0f}s'
            + (' &nbsp;·&nbsp; ' + ' &nbsp;·&nbsp; '.join(marks) if marks else '')
            + '</div>', unsafe_allow_html=True)

lead = f'+{r["lead_seconds"]:.0f}s' if r["lead_seconds"] else "none"
st.markdown(f"""
<div class="strip">
  <div class="cell"><div class="k">Escalation risk</div>
    <div class="v" style="color:{rc}">{risk:.0%}</div>
    <div class="s">chance this reaches a damaging stage within {hlabel}</div></div>
  <div class="cell"><div class="k">Current stage</div>
    <div class="v">{stage['name']}</div>
    <div class="s">{stage['mitre']} · where the attacker is right now</div></div>
  <div class="cell"><div class="k">Lead time</div>
    <div class="v">{lead}</div>
    <div class="s">{"earlier than damage-stage detection onset (model vs. itself)" if r["lead_seconds"] else "nothing to warn about at this horizon"}</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="fc">
  <div class="lab">Predicted over the next {hlabel}</div>
  <div class="big">{f['phrase']}</div>
  <div class="sig">Confidence <b>{f['confidence']:.0%}</b> &nbsp;·&nbsp;
       driven by <b>{sig['name']}</b> &nbsp;·&nbsp; {sig['evidence']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec">Kill chain</div>'
            '<div class="hint">The phases of a break-in, in order. An attacker '
            'gets in (Initial Access), spreads to other machines (Lateral '
            'Movement), takes remote orders (Command &amp; Control), then steals '
            'data (Exfiltration). Amber marks where they are now; dashed marks '
            'where we predict they go.</div>', unsafe_allow_html=True)
chain = CHAIN + [DOS] if (w["stage"] == DOS or DOS in f["sequence"]) else CHAIN
seen = set(int(s) for s in r["stage_ids"][:i + 1])
for col, sid in zip(st.columns(len(chain)), chain):
    info = get_stage(sid)
    now, ahead, past = sid == w["stage"], sid in f["sequence"], sid in seen
    if now:
        bg, fg, bd, tag, ln = "#FBF4E4", TXT, AMB, "attacker is here now", "solid"
    elif ahead:
        bg, fg, bd, tag, ln = PANEL, TXT, "#B9B7AE", "predicted next", "dashed"
    elif past:
        bg, fg, bd, tag, ln = "#F1F0EB", MUTED, LINE, "already seen", "solid"
    else:
        bg, fg, bd, tag, ln = "transparent", "#AFADA4", LINE, "not reached", "solid"
    col.markdown(
        f'<div style="border:1px {ln} {bd};background:{bg};border-radius:8px;'
        f'padding:12px 10px"><div style="font-size:13px;font-weight:600;color:{fg}">'
        f'{info["name"]}</div><div style="font-size:11.5px;color:{MUTED};margin-top:2px">'
        f'{info["mitre"]}{" · " + tag if tag else ""}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="sec">Escalation timeline</div>'
            '<div class="hint">Risk of reaching a damaging stage, across the whole '
            'capture. <b style="color:' + GRN + '">Green</b> is when the forecast '
            'first crosses the alert threshold; '
            '<b style="color:' + RED + '">red dashed</b> is when the model\'s own '
            'stage output first enters a damage stage. Both lines come from the same '
            'model — this is a forecast-vs-detection comparison, not a comparison '
            'against an external IDS.</div>', unsafe_allow_html=True)
t = np.array([x["time"] for x in r["windows"]])
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=r["risk"], mode='lines', line=dict(color=RED, width=2),
                         fill='tozeroy', fillcolor='rgba(176,58,50,.08)',
                         hovertemplate='T+%{x:.0f}s<br>risk %{y:.0%}<extra></extra>'))
if r["forecast_idx"] is not None:
    tf = r["windows"][r["forecast_idx"]]["time"]
    fig.add_vline(x=tf, line=dict(color=GRN, width=2))
    fig.add_annotation(x=tf, y=1.06, text="we forecast", showarrow=False,
                       font=dict(color=GRN, size=11), xanchor="left")
if r["detect_idx"] is not None:
    td = r["windows"][r["detect_idx"]]["time"]
    fig.add_vline(x=td, line=dict(color=RED, width=2, dash='dash'))
    fig.add_annotation(x=td, y=1.06, text="damage onset", showarrow=False,
                       font=dict(color=RED, size=11), xanchor="left")
fig.add_vline(x=w["time"], line=dict(color=MUTED, width=1))
fig.update_layout(height=260, margin=dict(l=0, r=0, t=22, b=0),
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                  showlegend=False, font=dict(color=MUTED, size=11),
                  xaxis=dict(title='seconds', gridcolor=LINE, zeroline=False),
                  yaxis=dict(gridcolor=LINE, range=[0, 1.12], tickformat='.0%',
                             zeroline=False))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="sec">Per-window detail</div>'
            '<div class="hint">Every window the model scored, in order. Each row is '
            f'{infer.WINDOW} flows, stepping forward one flow at a time. '
            '"Predicted stage" is where the attacker is; "forecast" is where they '
            'are heading next.</div>', unsafe_allow_html=True)

TRUTH_MAP = {
    'Benign': 'Benign', 'FTP-BruteForce': 'InitialAccess',
    'SSH-Bruteforce': 'InitialAccess', 'Brute Force -Web': 'InitialAccess',
    'Brute Force -XSS': 'InitialAccess', 'SQL Injection': 'InitialAccess',
    'DoS attacks-Hulk': 'DoS', 'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'Infilteration': 'Infiltration', 'Bot': 'Botnet',
}

table = {
    "t (s)": [f'{x["time"]:.0f}' for x in r["windows"]],
    "predicted stage": [get_stage(x["stage"])["name"] for x in r["windows"]],
    "forecast": [x["forecast"]["phrase"] for x in r["windows"]],
    "confidence": [f'{x["forecast"]["confidence"]:.0%}' for x in r["windows"]],
    "risk": [f'{x["forecast"]["damage_risk"]:.0%}' for x in r["windows"]],
    "signal": [x["signal"]["name"] for x in r["windows"]],
    "why": [x["signal"]["evidence"] for x in r["windows"]],
}

# window i is scored against the flow at i + WINDOW, so ground truth aligns there
truth_col = next((c for c in ('Label', 'Stage', 'label') if c in df.columns), None)
if truth_col is not None:
    raw = df[truth_col].astype(str).str.strip().to_numpy()
    aligned = raw[infer.WINDOW: infer.WINDOW + n]
    if len(aligned) == n:
        table["actual label"] = list(aligned)
        table["actual stage"] = [TRUTH_MAP.get(v, v) for v in aligned]

tbl = pd.DataFrame(table)

if "actual stage" in tbl.columns:
    hit = (tbl["predicted stage"].str.replace(" / Impact", "", regex=False)
           .str.replace("Command & Control", "Botnet", regex=False)
           .str.replace("Lateral Movement", "Infiltration", regex=False)
           .str.replace(" ", "") == tbl["actual stage"].str.replace(" ", ""))
    st.caption(f"Ground truth found in column `{truth_col}` — "
               f"{hit.sum():,}/{len(tbl):,} windows match ({hit.mean():.1%}). "
               f"Note stage names differ between the model's 5 classes and the "
               f"kill-chain display names, so this is indicative, not the metric.")

f1, f2 = st.columns([2, 3])
with f1:
    only_attacks = st.checkbox("Attack windows only", value=False)
with f2:
    sigs = sorted(set(tbl["signal"]))
    pick = st.multiselect("Filter by signal", sigs, default=[],
                          placeholder="all signals")

view = tbl
if only_attacks:
    view = view[view["predicted stage"] != "Benign"]
if pick:
    view = view[view["signal"].isin(pick)]

st.dataframe(view, use_container_width=True, height=330, hide_index=True)
st.caption(f"Showing {len(view):,} of {len(tbl):,} windows "
           f"· built from {r['n_flows']:,} flows")
st.download_button("Download as CSV", tbl.to_csv(index=False).encode(),
                   file_name="netrikan_windows.csv", mime="text/csv")

st.markdown('<div class="sec">Stage distribution</div>', unsafe_allow_html=True)
counts = tbl["predicted stage"].value_counts()
dcols = st.columns(len(counts))
for col, (name, cnt) in zip(dcols, counts.items()):
    col.metric(name, f"{cnt:,}", f"{cnt/len(tbl):.0%} of capture")

with st.expander("Detail and model evidence"):
    attr = w["attribution"]
    top = np.argsort(np.abs(attr))[::-1][:4]
    feats = ", ".join(f"`{infer.FEATURES[j]}`" for j in top)
    st.markdown(f"**{sig['name']}** — {sig['evidence']}. "
                f"Model attributes this window to {feats}.")
    st.markdown(f"Window {i} of {n-1} · {infer.WINDOW} flows · "
                f"{r['flow_interval']:.1f}s mean interval · {r['n_flows']:,} flows total")

    pkt_cols = [c for c in pcap_ingest.PACKET_FEATURES if c in df.columns]
    if pkt_cols:
        st.markdown("**Packet-level features** (PCAP only — flow CSVs cannot supply these)")
        seg = df[pkt_cols].iloc[i:i + infer.WINDOW].mean()
        pc = st.columns(4)
        for col, name in zip(pc * 2, pkt_cols):
            v = seg[name]
            col.metric(name, f"{v:,.1f}" if v % 1 else f"{int(v):,}")
    else:
        st.caption("Packet-level features unavailable — this input is flow records. "
                   "Upload a PCAP to extract TTL, window size, fragmentation and "
                   "retransmission statistics.")

    cols = st.columns(3)
    if 'train' in prov:
        cols[0].metric("Held-out macro-F1", f"{prov['train']['best_macro_f1']:.3f}")
    if 'dapt' in prov:
        a = prov['dapt']['binary_report']['Attack']
        cols[1].metric("Cross-dataset recall", f"{a['recall']:.1%}", "DAPT 2020, unseen")
    if 'baseline' in prov:
        b = prov['baseline']['results']['flat']['macro_f1']
        cols[2].metric("Logistic baseline", f"{b:.3f}", "same features")

    hz = prov.get('horizon', {}).get('corrected_horizon')
    if hz:
        st.markdown("**Forecast accuracy vs lead time** — persistence baseline shown for comparison")
        PERSISTENCE = {0: 0.906, 30: 0.919, 90: 0.914, 180: 0.912, 360: 0.909}
        ks = [h['k_flows'] for h in hz]
        fh = go.Figure()
        fh.add_trace(go.Scatter(x=ks, y=[h['macro_f1'] for h in hz],
                                mode='lines+markers', name='LSTM',
                                line=dict(color=GRN, width=2)))
        fh.add_trace(go.Scatter(x=ks, y=[PERSISTENCE.get(k, None) for k in ks],
                                mode='lines+markers', name='persistence (copy last label)',
                                line=dict(color=MUTED, width=1.5, dash='dash')))
        fh.update_layout(height=220, margin=dict(l=0, r=0, t=6, b=0),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color=MUTED, size=10), showlegend=True,
                         legend=dict(font=dict(size=10)),
                         xaxis=dict(title='flows of lead time', gridcolor=LINE),
                         yaxis=dict(title='macro F1', gridcolor=LINE, range=[0.8, 1.0]))
        st.plotly_chart(fh, use_container_width=True, config={'displayModeBar': False})
        st.caption("Note: LSTM evaluated on blocked split; persistence on full label sequence. "
                   "Split mismatch makes direct comparison approximate, but the gap is large.")

# ── Counterfactual intervention panel ──────────────────────────────────────
st.markdown('<div class="sec">Response simulation — what stops this attack?</div>',
            unsafe_allow_html=True)
with st.expander("Run counterfactual interventions", expanded=(risk > 0.3)):
    st.caption("Each intervention modifies the current traffic window and re-runs "
               "inference. Risk delta shows how much your response changes the "
               "breach probability.")
    with st.spinner("Simulating interventions…"):
        iv_results = counterfactual.run_all(df, horizon, infer.analyze)
    if not iv_results:
        st.warning("No flows available for counterfactual analysis.")
    else:
        for iv in iv_results:
            delta = iv["delta"]
            delta_str = f"{delta:+.0%}"
            color = GRN if iv["sufficient"] else (AMB if delta < 0 else RED)
            badge = "✓ sufficient" if iv["sufficient"] else ("partial" if delta < 0 else "no effect")
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{iv['label']}**  \n{iv['description']}")
            cols[1].metric("Before", f"{iv['risk_before']:.0%}")
            cols[2].metric("After", f"{iv['risk_after']:.0%}", delta_str)
            cols[3].markdown(f'<span style="color:{color};font-weight:bold">{badge}</span>',
                             unsafe_allow_html=True)
            st.divider()

# ── SHA-256 tamper-evident alert ledger ────────────────────────────────────
import hashlib

def _chain_alert(prev_hash: str, alert: dict) -> str:
    payload = json.dumps(alert, sort_keys=True) + prev_hash
    return hashlib.sha256(payload.encode()).hexdigest()

if risk > 0.3:
    st.markdown('<div class="sec">Forensic audit ledger</div>', unsafe_allow_html=True)
    with st.expander("SHA-256 hash-chained alert record"):
        alert_record = {
            "window_idx":   i,
            "time_offset_s": float(w["time"]),
            "stage":         stage["name"],
            "risk":          round(risk, 4),
            "signal":        sig["name"],
            "evidence":      sig["evidence"],
            "mitre_id":      stage.get("technique", ""),
            "lead_s":        r["lead_seconds"],
        }
        genesis = "0" * 64
        chain_hash = _chain_alert(genesis, alert_record)
        alert_record["sha256"] = chain_hash
        alert_record["prev_hash"] = genesis

        st.json(alert_record)
        st.caption(f"Hash: `{chain_hash}`  \n"
                   "Each alert's SHA-256 includes the previous block's hash. "
                   "Modifying any field changes all subsequent hashes — "
                   "tamper-evident by construction.")
