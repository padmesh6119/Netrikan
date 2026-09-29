# NETRIKAN — SIH 2026 Idea PPT: full content

**For:** whoever builds the deck. **Template:** `SIH2026-IDEA-Presentation-Format.pptx`.
**PS:** SIH26153 · NTRO · Blockchain & Cybersecurity · Software

---

## 0. Read this first (5 rules)

1. **Max 6 slides including the title.** Delete template slide 7 (the instructions). Export as **PDF** — the portal takes nothing else.
2. **Do not rename the template's section headings** ("Proposed Solution", "Technical Approach", etc.). Put our punchline *under* them, never *instead of* them.
3. **Points, diagrams, numbers. No paragraphs.** If a bullet runs to two lines, cut it.
4. **Every number below is real and points to a file in `models/`.** Do not round it up. Do not add numbers that aren't here. Judges from NTRO will check.
5. Anything marked **(roadmap)** is not built yet. It must appear as roadmap on the slide, never as a feature.

Fill in: **Team ID ______ · Team Name ______**

---

## 1. The idea that makes us different

Everyone else in SIH26153 built a system that answers **"what attack is happening right now?"**
That is detection, and it's always too late: by the time you're sure, you've already been breached.

We answer three questions nobody else answers:

| Question | Everyone else | Netrikan |
|---|---|---|
| What's happening now? | ✅ | ✅ |
| **What happens next, and when?** | ❌ | ✅ warns **median 13.5 min before** the attack starts |
| **What do I do about it?** | ❌ | ✅ simulates each response and shows the risk drop |

The trap everyone falls into: 91–95% of network windows have the same label as the previous one. A model that copies the last label looks 90%+ accurate while forecasting nothing (this is the **persistence baseline**). Most teams never test against it. We did, and then built the model to beat it where it matters: the moment an attack *changes stage*. Persistence scores **exactly 0** there. We score **0.38–0.46**.

**The name.** *Netrikan* (நெற்றிக்கண்), the third eye, the one that sees what hasn't happened yet. Use it on the title slide.

---

## 2. The "standing ovation" moves

Build these into the deck. They're what separates us from the other 5 teams.

1. **Lead with a clock, not a model.** The first thing a judge sees on slide 2 is **"13.5 minutes"**, in huge type. Everything else explains how.
2. **Tie it to their law.** CERT-In Directions 20(3)/2022: a **6-hour mandatory incident report** and **180-day in-India log retention**. For NTRO, lead time isn't a feature, it's compliance margin. Air-gapped and offline isn't a nice-to-have either, it's a legal requirement. No other team will say this.
3. **"We tried to break our own model."** One strip on slide 4 showing the tests we ran *against ourselves*: held-out days, an unseen network, shuffled labels, 4 seeds. Judges have seen 100 decks claiming 99% accuracy. Nobody shows them the stress test. This is the credibility slam.
4. **Attackers transfer, normal doesn't.** On a network we never trained on (DAPT 2020, a real 4-day APT), **lateral movement is caught 94% of the time** but "normal" traffic doesn't transfer. So: one national attack-doctrine model, and a per-site baseline for normal traffic. That's exactly NTRO's shape: power, telecom, banking, railways.
5. **Fits on a USB stick.** A **0.89 MB** model, **1.4 ms** per decision, CPU-only. Substations and telecom cabinets don't have GPUs.
6. **Never reads content.** It uses metadata only (timing, sizes, flags), so it works on TLS 1.3 without decrypting and never touches citizen payloads.
7. **QR code to a 60-second demo video** in the corner of slide 3, if the video is ready. A live replay of a real attack capture where the forecast fires before the attack stage arrives.

---

## 3. Slide-by-slide content

### Slide 1 — Title page

- **Problem Statement ID:** SIH26153
- **Problem Statement Title:** Network Attack Forecasting from Network Traffic Data *(copy the exact title from the portal)*
- **Theme:** Blockchain & Cybersecurity
- **PS Category:** Software
- **Team ID / Team Name:** ______
- **Idea title (big):** **NETRIKAN, the third eye for national networks**
- **Tagline (one line under it):** *Forecasts the attacker's next move before it happens, and tells you how to stop it.*

---

### Slide 2 — Idea title / Proposed Solution

**Header block (huge):** `⏱ 13.5 min` — *median warning before an attack begins*
*(small caption: 2 of every 3 attack onsets warned, at a 5% false-alarm budget)*

**Proposed solution**
- An AI **world model** of network traffic. It learns how an intrusion *evolves* (scan → break-in → spread → steal), not just what one packet looks like.
- It watches live flow metadata and outputs, per host:
  - **Probability a new attack stage starts** in the next 30 s / 2.5 min / 7.5 min / 15 min
  - **Current kill-chain stage**, mapped to **MITRE ATT&CK + CAPEC**
  - **Why:** which traffic patterns drive the score (attention + gradient attribution)
  - **What to do:** simulates block-SMB / block-admin-ports / isolate-host and shows the risk after each

**How it addresses the problem**
- The PS asks for *forecasting*. Detection systems (Snort/Suricata-style) fire after the fact. Netrikan fires **before the stage transition**.
- Tested on the metric where copying the last label scores **0**. We score **0.38–0.46 at every horizon (5/5)**, on **32k–190k real transitions**.

**Innovation & uniqueness** *(3 icons in a row)*
- 🔮 **Onset forecasting**: trained to predict *when the attack advances*, not what stage it's in
- 🧪 **Response simulation**: "what stops this attack?" answered before you act
- 🔗 **Tamper-evident alert ledger**: SHA-256 hash-chained evidence, ready for a CERT-In report

---

### Slide 3 — Technical Approach

**Diagram (left-to-right flow, the centrepiece of the slide):**

```
 PCAP / live tap ──► Flow extractor ──► 30-flow sliding window ──► LSTM world model ──┬─► Onset forecast (4 horizons)
 (nfstream/tshark)    24 metadata          per host                (+ attention)       ├─► Stage → ATT&CK / CAPEC
                      features                                                          ├─► Next-state rollout (K steps)
                                          14 rule detectors ─────► Fusion ─────────────┘
                                                                     │
                                          Counterfactual engine ◄────┤  (block SMB / admin ports / isolate host)
                                          SHA-256 alert ledger  ◄────┘  → SOC dashboard (Streamlit)
```

**Technologies**
- **Python · PyTorch · ONNX Runtime**: model training and edge inference
- **nfstream / tshark**: PCAP → bidirectional flows
- **Streamlit + Plotly**: SOC dashboard
- **MITRE ATT&CK + CAPEC**: stage mapping · **SHA-256 hash chain**: evidence ledger
- **Data:** CIC-IDS-2018 (**6.1M windows, 7 capture days**), DAPT 2020 (real APT, **715 hosts**), CTU-13

**Methodology**
1. **Ingest:** PCAP → flows → 24 metadata features (no payload)
2. **Model:** one LSTM trunk, four heads: stage · breach · **next network state** · **onset within k windows**
3. **World-model rollout:** feed the predicted next state back in and simulate K steps ahead
4. **Calibrate:** temperature scaling so "70% risk" means 70% (ECE 0.005)
5. **Explain:** attention shows which flow in the window drove the forecast
6. **Respond:** counterfactual re-simulation per intervention → risk-after score
7. **Record:** every alert is hash-chained, so edits and deletions are detectable

**Bottom strip:** *Working prototype: PCAP upload → forecast → response simulation, live.* `[QR: 60-s demo]`

---

### Slide 4 — Feasibility and Viability

**Feasibility: already built and measured**

| | |
|---|---|
| Model size | **0.89 MB** |
| Decision latency | **1.4 ms** median (ONNX, 1 CPU thread, laptop) |
| Throughput | **~610 windows/s** on one thread |
| Hardware | CPU only, **no GPU**, runs offline/air-gapped |
| Traffic | Metadata only → works on **encrypted TLS 1.3** |

**"We tried to break our own model"** *(make this a visual strip of 4 tiles)*

| Stress test | Result |
|---|---|
| Hold out an entire capture day (7-fold) | onset AUC **0.64**, still well above chance |
| Never-seen network (DAPT 2020 APT) | onset AUC **0.70** (chance = 0.50) · lateral movement caught **94%** |
| Shuffle the time order | forecasting skill drops (PR-AUC **0.49 → 0.37**), so it's the *sequence* that carries signal |
| 4 training seeds | onset AUC **0.897 ± 0.007**, stable |

**Challenges → strategy**
- **False alarms on a new network** (normal traffic differs per site) → keep one national attack model and calibrate only the per-site baseline; set the alert threshold from an **operator false-alarm budget**
- **Unseen attack families** (zero-shot is hard for every model) → rule detectors run alongside the model, and disagreement between the two is flagged as "novel traffic"
- **Adversarial evasion** → metadata features are costly to fake at scale; the ledger preserves evidence either way
- **Deployment at scale** → ONNX export, per-host streaming, no cloud dependency

---

### Slide 5 — Impact and Benefits

**Target users:** NCIIPC / CERT-In SOCs, and critical-infrastructure operators (**power, telecom, banking, railways**).

**Impact** *(4 big-number tiles)*
- ⏱ **Minutes of warning instead of post-mortems.** Median **13.5 min** before an attack starts; 1 in 4 warnings arrives **36+ min** early
- 📋 **Compliance margin.** Lead time buys headroom inside CERT-In's **6-hour** reporting window, and the hash-chained ledger gives ready evidence
- 🇮🇳 **Sovereign and offline.** No data leaves the perimeter, matching the **180-day in-India** log-retention mandate
- 💸 **Near-zero hardware cost.** Runs on existing CPUs at the edge: no GPU, no cloud bill

**Benefits**
- **Security / national:** stops multi-stage APTs at lateral movement, before data exfiltration
- **Economic:** a breach prevented is cheaper than a breach investigated; analysts triage fewer, better-ranked alerts
- **Social / privacy:** no payload inspection, so citizen traffic content is never read
- **Scalable:** one attack-doctrine model shared across sectors, with local baselines per site

**Roadmap (label it as roadmap):** per-host sequential alerting (SPRT) · "time bought" per response in seconds · MITRE D3FEND countermeasure mapping · federated doctrine sharing across CII sites

---

### Slide 6 — Research and References

**Our evidence (all reproducible, one command: `scripts/reproduce.sh`)**
- Transition-window forecasting vs persistence, 5 horizons: `models/transition_f1.json`
- Lead-time curve at fixed false-alarm rates: `models/lead_time_cic_v2_w30.json`
- Leave-one-day-out, 7 folds: `models/lodo_cic_full_w30.json`
- Cross-network test on DAPT 2020: `models/dapt_cic_v2_w30.json`
- World-model rollout vs persistence: `models/rollout_eval_cic_v2_w30.json`

**Datasets**
- Sharafaldin et al., *CSE-CIC-IDS2018*, Canadian Institute for Cybersecurity / AWS
- Myneni et al., *DAPT 2020: Constructing a Benchmark Dataset for Advanced Persistent Threats*, 2020
- García et al., *CTU-13 botnet dataset*, CTU Prague, 2014

**Methods**
- Ha & Schmidhuber, *World Models*, 2018
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997
- Guo et al., *On Calibration of Modern Neural Networks* (temperature scaling), ICML 2017
- Ferro & Stephenson, *SEDI: extremal dependence indices for rare-event forecasts*, 2011
- MITRE ATT&CK® · MITRE CAPEC™ · MITRE D3FEND™
- CERT-In Directions No. 20(3)/2022-CERT-In, 28 April 2022

---

## 4. Design direction

- **Keep the template frame** (logos, footer). Inside it: a dark-navy content area, one accent colour (amber for "warning"), and white numbers.
- **One hero number per slide:** 13.5 min (S2) · the pipeline diagram (S3) · the 4-tile stress strip (S4) · the 4 impact tiles (S5).
- Icons over words. Kill-chain as a horizontal arrow: Recon → Initial Access → Lateral → C2 → Exfil, with the forecast marker *ahead* of the current stage.
- Font size ≥ 14 pt. If it doesn't fit, cut words, not font size.
- Put a **dashboard screenshot** on slide 3 or 5: run `streamlit run src/app.py`, upload `demo/real_intrusion_dapt_wednesday.csv`, and capture the risk curve rising at the attack. Don't use the built-in synthetic scenarios: their benign baseline false-alarms. See `DECKBOOK.md` §4.

---

## 5. Do NOT put on any slide

- ❌ "Beats persistence" **without** "on transition windows". On all-window F1, persistence wins; we publish that.
- ❌ `stage_macro_f1` from the LODO file. It's labelled DO_NOT_REPORT (see `LODO_ANALYSIS.md`).
- ❌ SPRT, "Time Bought", Suricata head-to-head, or D3FEND as **built**. They are roadmap.
- ❌ "99% accuracy" or any number not in this file.
- ❌ Competitor names. Say "12 public SIH26153 repos audited" if you must.
- ❌ "Zero-shot detection of unknown attack families." We tested it; it's weak. Frame it as the challenge on slide 4, where it already is.

---

## 6. Q&A cheat sheet (for the presenter)

| Likely question | Answer |
|---|---|
| "Your model loses to a trivial baseline?" | "On all-window F1, yes, and we published it. That metric rewards copying the last label. On the windows where the attack *changes stage*, the baseline scores 0 and we score 0.38–0.46 at every horizon." |
| "False-positive rate on a new network?" | "At the default threshold it's high, because 'normal' is local and doesn't transfer. The attack side does: 94% lateral-movement recall on an unseen APT. So we calibrate the benign baseline per site, and the operator sets the false-alarm budget." |
| "One LODO fold is below 0.5?" | "Fold 1: the held-out day has 12× more DoS than training ever saw. Day 4, also DoS, scores 0.72. It's a volume artefact, and we report it rather than hide it." |
| "Is it really a world model?" | "Its state head predicts the next network state 22% better than copying the current one, and it beats the window average too. That's measured in `rollout_eval_cic_v2_w30.json`." |
| "Can it run at a substation?" | "0.89 MB, 1.4 ms per decision on one CPU thread, fully offline." |
| "How do you know it's not leaking labels?" | "Shuffle the time order and forecasting skill drops. It learns from the sequence. Plus 7-fold held-out days and an entirely different network." |

---

## 7. Number → source (for anyone double-checking)

| Claim | Value | File |
|---|---|---|
| Median warning (k30, 5% FPR) | 810 s = 13.5 min · 66% of 15,898 onsets · p75 36 min | `lead_time_cic_v2_w30.json` |
| Onset AUC, in-dataset k5 | 0.901 (persistence 0.5) | `cic_v2_w30_metrics.json` |
| Onset AUC, 4 seeds | 0.897 ± 0.007 | `seed_variance_cic_full_w30.json` |
| Transition-window F1 | 0.380 / 0.459 / 0.464 / 0.455 / 0.438 vs 0.000 | `transition_f1.json` |
| LODO onset AUC k5 | 0.636 ± 0.093 (7 folds) | `lodo_cic_full_w30.json` |
| DAPT onset AUC k15 | 0.705 | `dapt_cic_v2_w30.json` |
| DAPT lateral-movement recall | 0.942 | `dapt_base_w30.json` (IDEAS.md §1) |
| Shuffled-order control, PR-AUC k5 | 0.494 vs 0.371 | `onset_pr_cic_v2_w30.json`, `onset_pr_shuffled_w30.json` |
| Rollout 1-step skill | +0.22 vs persistence, +0.06 vs window mean | `rollout_eval_cic_v2_w30.json` |
| Calibration ECE | 0.0088 → 0.0052 | `temperature.json` |
| Size / latency / throughput | 0.893 MB · 1.41 ms · 610/s | `latency.json` |
