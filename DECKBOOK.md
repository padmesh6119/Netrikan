# NETRIKAN — Deckbook

SIH 2026 · SIH26153 · NTRO · Blockchain & Cybersecurity · Software
Everything needed to build, present and defend the deck. Every number traces to a file in `models/`.

---

## 0. The one sentence

> **Every other system tells you what attack is happening. Netrikan tells you what happens next, how long you have, and what stops it.**

Say this sentence at the start and again at the end.

---

## 1. The story (5 beats, the spine of the whole deck)

1. **The problem is time.** Multi-stage attacks look harmless at every single step. Detection fires when the damage is done. Under CERT-In, that's also when a 6-hour reporting clock starts.
2. **The trap everyone falls into.** 91–95% of network windows have the same label as the previous one. A model that copies the last label looks 90%+ accurate and forecasts nothing. Most systems here are measured on exactly that.
3. **Our move.** We don't ask "what stage is this?" We ask **"will the attack advance in the next k minutes?"** On that question, copying the last label scores exactly **0**. We score **0.38–0.46 at every horizon.**
4. **Proof we didn't fool ourselves.** Held-out days. An unseen network. Shuffled time order. Four seeds. We attacked our own model before anyone else could.
5. **Built for NTRO's reality.** Runs offline on a CPU, 0.89 MB. Reads metadata only, so it works on encrypted traffic. Hash-chained evidence. Simulates the response before you act.

---

## 2. The wow moments (use all of them)

| # | Wow moment | The line | Source | Where it goes |
|---|---|---|---|---|
| W1 | **The clock** | "Median **13.5 minutes** of warning before an attack starts. Two of every three attacks warned." | `lead_time_cic_v2_w30.json` (k30, 5% FPR: 66% of 15,898 onsets, median 810 s) | Slide 2 hero |
| W2 | **The trap, exposed** | "Copying the last label scores 0 at the moment an attack changes stage. We score 0.46." | `transition_f1.json` | Slide 2 |
| W3 | **The law** | "CERT-In gives you 6 hours to report. We give you a head start before the clock even begins." | CERT-In Directions 20(3)/2022 | Slides 2 and 5 |
| W4 | **Self-attack** | "We tried to break our own model, four ways. Here's what survived." | LODO, DAPT, shuffled, seeds (see §5) | Slide 4 strip |
| W5 | **Attackers transfer, normal doesn't** | "On a network we'd never seen, lateral movement was caught 94% of the time. So: one national attack model, and local 'normal' per site." | `dapt_base_w30.json` | Slides 4 and 5 |
| W6 | **Time is the signal** | "Scramble the order of events inside each window and forecasting quality drops by a quarter. It's reading the sequence, not the snapshot." | `onset_pr_*` (PR-AUC 0.494 vs 0.371) | Slide 4 |
| W7 | **Fits on a USB stick** | "0.89 MB. 1.4 milliseconds per decision. No GPU. No cloud." | `latency.json` | Slide 4 |
| W8 | **Never reads content** | "Timing, sizes, flags. No payload. Works on TLS 1.3 without decrypting a single byte of citizen traffic." | 24 metadata features | Slides 3 and 5 |
| W9 | **What stops it** | "Before you pull the plug, Netrikan shows you what each response does to the risk: block SMB, block admin ports, isolate the host." | Counterfactual engine | Slide 3 + demo |
| W10 | **Evidence you can't edit** | "Every alert is hash-chained. Delete or change one and the chain breaks." | `src/ledger.py` | Slide 3 |
| W11 | **Honesty as a weapon** | "We publish where we lose. Every number here is a JSON file anyone can reproduce with one command." | `scripts/reproduce.sh` | Slide 6 + close |

The name: **Netrikan (நெற்றிக்கண்), the third eye, the one that sees what hasn't happened yet.** Use it on the title slide and in the first 10 seconds of speech.

---

## 3. Slide-by-slide

Rules from the template: **6 slides max including the title.** Keep the template headings. Points, not paragraphs. Export as **PDF**.

---

### SLIDE 1 — Title

**On slide**
- Problem Statement ID: **SIH26153**
- Problem Statement Title: *[exact title from portal]*
- Theme: **Blockchain & Cybersecurity** · PS Category: **Software**
- Team ID: ____ · Team Name: ____
- Big: **NETRIKAN — the third eye for national networks**
- Under it: *Forecasts the attacker's next move, and what stops it.*

**Visual:** the name large, the tagline, nothing else.

**Say (10 s):** "Netrikan means the third eye, the eye that sees what hasn't happened yet. That's what we built for India's networks."

---

### SLIDE 2 — Proposed Solution

**On slide**

Top, huge: **⏱ 13.5 min**
Caption: *median warning before an attack begins · 2 in 3 attacks warned · at a 5% false-alarm budget*

Proposed solution
- An AI **world model** of network traffic. It learns how intrusions *evolve*: scan → break-in → spread → steal.
- For each window it outputs:
  - **P(attack advances)** within 30 s · 2.5 min · 7.5 min · 15 min
  - **Kill-chain stage** mapped to MITRE ATT&CK + CAPEC
  - **Why**: the flows that drove the forecast
  - **What stops it**: the simulated risk after each response

How it addresses the problem
- The problem statement asks for forecasting. Detectors fire after the fact; **Netrikan fires before the stage changes.**
- Scored where copying the last label gets **0**, we get **0.38–0.46 at all 5 horizons**, over **32k–190k real transitions** per horizon.

Innovation
- **Onset forecasting**: trained on *when* the attack advances, not *what* it is
- **Response simulation**: "what stops this?" answered before you act
- **Tamper-evident ledger**: SHA-256 hash-chained alerts, ready for a CERT-In report

**Visual:** the clock top-left. Below it, an *illustrative* kill-chain arrow (Recon → Initial Access → Lateral → C2 → Exfil) with a "NOW" marker at Recon and a glowing "FORECAST" marker at Lateral.

**Say (45 s):** "Most security tools answer 'what's happening now'. By the time they're sure, you're breached. Here's the trap: in network data, 9 windows in 10 look like the one before. A model that just copies the last answer looks 90% accurate and predicts nothing. So we asked a different question: will this attack move to its next stage in the next few minutes? Copying scores zero on that. We score up to 0.46, and in practice that's a median of 13 and a half minutes of warning."

**Judge takeaway:** they understood something the other teams didn't.

---

### SLIDE 3 — Technical Approach

**On slide**

Pipeline (left → right):
```
PCAP / live tap → flow extractor (24 metadata features, no payload)
   → 30-flow sliding window → LSTM world model
        ├ onset forecast (4 horizons)
        ├ stage → ATT&CK / CAPEC
        ├ next-state rollout (simulates K steps ahead)
        └ attention: which flow drove it
   + 14 rule detectors → fusion
   → response simulator (block SMB · block admin ports · isolate host · rate-limit)
   → SHA-256 hash-chained alert ledger → SOC dashboard
```

Technologies
- Python · PyTorch · **ONNX Runtime** (edge inference)
- nfstream / tshark (PCAP → flows) · Streamlit + Plotly (SOC dashboard)
- MITRE ATT&CK + CAPEC mapping · SHA-256 hash chain
- Data: **CIC-IDS-2018** (6.1M windows, 7 days) · **DAPT 2020** (real APT, 715 hosts) · **CTU-13**

Methodology
1. Ingest: PCAP → bidirectional flows → 24 metadata features
2. One LSTM trunk, four heads: stage · breach · **next network state** · **onset within k**
3. World-model rollout: feed the predicted state back in and simulate ahead
4. Temperature calibration: "70% risk" means 70%
5. Response simulation: re-run the model on the traffic with each control applied
6. Every alert hash-chained

Corner: `[QR → 60-second demo]` · *Working prototype: upload a capture → forecast → simulate a response.*

**Visual:** the pipeline is the slide. A dashboard screenshot sits bottom-right, small.

**Say (40 s):** "Packets come in, and we keep only metadata: timing, sizes, flags. Thirty flows at a time go into one model with four jobs. The key two: predict the next network state, and predict whether the attack is about to advance. Because it predicts the next state, it can simulate forward. That's what makes it a world model, not a classifier. And because it can simulate, it can answer: if I block SMB right now, what happens to the risk?"

---

### SLIDE 4 — Feasibility and Viability

**On slide**

Already built and measured

| | |
|---|---|
| Model size | **0.89 MB** |
| Decision time | **1.4 ms** (ONNX, 1 CPU thread) |
| Throughput | **~610 windows/s** per thread |
| Hardware | CPU only · offline · air-gapped |
| Traffic | metadata only → works on **encrypted TLS 1.3** |

**We tried to break our own model**

| Attack on ourselves | What survived |
|---|---|
| Hide a whole day of traffic (7 folds) | onset AUC **0.64**, still above chance (0.50) |
| A network it had never seen (DAPT 2020 APT) | onset AUC **0.70** · lateral movement caught **94%** |
| Scramble the time order | forecasting quality drops (PR-AUC **0.49 → 0.37**), so the *sequence* matters |
| Retrain with 4 different seeds | onset AUC **0.897 ± 0.007**, stable |

Challenges → strategy
- **False alarms on a new network**: "normal" is local, attacks aren't. Keep one national attack model, calibrate normal per site, and let the operator set the alarm budget.
- **Brand-new attack families**: rule detectors run alongside the model; model/rules disagreement is flagged as "novel".
- **Evasion**: timing and volume patterns are expensive to fake at scale, and the ledger preserves evidence either way.
- **Scale**: ONNX export, streaming per window, no cloud dependency.

**Visual:** the 4-row stress table as 4 tiles, each with a ✓. The specs table as small icons.

**Say (45 s):** "Is it real? It's 0.89 megabytes and makes a decision in 1.4 milliseconds on a laptop CPU, so it runs at a substation. Is it overfitting? We tried to break it. We hid entire days: it still forecasts. We gave it a completely different network running a real APT: it caught lateral movement 94% of the time. We scrambled the time order and it got worse, which proves it's reading the sequence. And four retrains agree to within less than one point."

**Judge takeaway:** this team knows how models fail.

---

### SLIDE 5 — Impact and Benefits

**On slide**

Who: **NCIIPC / CERT-In SOCs · power, telecom, banking, railways**

Four tiles
- ⏱ **Minutes, not post-mortems.** Median **13.5 min** warning; 1 in 4 warnings arrives **36+ min** early
- 📋 **Compliance margin.** A head start inside CERT-In's **6-hour** reporting window, plus a tamper-evident evidence trail
- 🇮🇳 **Sovereign by design.** Fully offline; no data leaves the perimeter, matching the **180-day in-India log retention** mandate
- 💸 **Near-zero hardware.** Existing CPUs at the edge. No GPU, no cloud bill.

Benefits
- **National security:** stop multi-stage APTs at lateral movement, before exfiltration
- **Economic:** prevention is cheaper than incident response; analysts get fewer, ranked alerts with a recommended action
- **Privacy:** no payload inspection, so citizen content is never read
- **Scale:** one attack-doctrine model shared nationally, with local baselines per site

Roadmap *(label it as roadmap)*: per-host attacker tracking · per-host sequential alerting (SPRT) · "time bought" per response, in seconds · MITRE D3FEND countermeasure mapping · federated learning across CII sites

**Say (30 s):** "For NTRO this isn't a feature, it's compliance. The 6-hour reporting rule means every minute of warning matters. The 180-day in-India retention rule means nothing can go to a cloud API, and Netrikan never does. It runs on hardware that's already there."

---

### SLIDE 6 — Research and References

**On slide**

Our evidence: reproducible with one command, `scripts/reproduce.sh`
- Transition-window forecasting, 5 horizons: `transition_f1.json`
- Lead time at fixed false-alarm rates: `lead_time_cic_v2_w30.json`
- Leave-one-day-out, 7 folds: `lodo_cic_full_w30.json`
- Unseen network (DAPT 2020): `dapt_cic_v2_w30.json`
- World-model rollout vs persistence: `rollout_eval_cic_v2_w30.json`

Datasets
- Sharafaldin et al., *CSE-CIC-IDS2018*, Canadian Institute for Cybersecurity
- Myneni et al., *DAPT 2020: a benchmark dataset for advanced persistent threats*, 2020
- García et al., *CTU-13 botnet dataset*, CTU Prague, 2014

Methods
- Ha & Schmidhuber, *World Models*, 2018 · Hochreiter & Schmidhuber, *LSTM*, 1997
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017
- Ferro & Stephenson, *Extremal dependence indices for rare-event forecasts* (SEDI), 2011
- MITRE ATT&CK® · CAPEC™ · D3FEND™ · CERT-In Directions 20(3)/2022

**Say (10 s, closing):** "Every number you saw is a file in our repo, reproducible with one command. Netrikan: we don't tell you what happened. We tell you what happens next."

---

## 4. Demo script

### ⚠ Use real traffic, not the built-in synthetic scenarios

Checked on 2026-09-29 with the deployed model:

| Input | Max risk | Verdict |
|---|---|---|
| Built-in "Benign baseline" (synthetic) | **0.82** | ❌ false alarm, never show it |
| Built-in "Intrusion" (synthetic) | 0.999 from window 0, no forecast marker | ❌ alarms from the first second, so it shows nothing |
| `demo/real_benign_dapt_monday.csv`, real DAPT, all benign | **0.03** | ✅ quiet |
| `demo/real_intrusion_dapt_wednesday.csv`, real DAPT, foothold from flow 152 | 0.005 before the attack → **0.75** during it | ✅ clean rise |

The synthetic generator doesn't look like the traffic the model was trained on, so the demo uses the two real DAPT slices in `demo/`. Upload them through the file box; they're CSVs with all 24 features.

**Do not claim in the demo that the forecast fires before the attack.** On these replays the risk rises *as* the attack starts, not ahead of it. The 13.5-minute figure is a median over 15,898 real onsets (slide 2), not something a single replay reliably shows. Say "early" and point to the slide number; don't promise a lead on screen.

### 60-second video (QR on slide 3)

| t | On screen | Voice |
|---|---|---|
| 0–10 s | Upload `real_benign_dapt_monday.csv`. Flat risk, around 0.03 | "Real traffic from a real network, a normal Monday. Netrikan stays quiet." |
| 10–30 s | Upload `real_intrusion_dapt_wednesday.csv`. Drag playback across flow ~150. Risk goes from near 0 to 0.75 | "Wednesday: an attacker establishes a foothold. Watch the risk climb the moment the behaviour changes. This network was never in our training data." |
| 30–40 s | Stage label, ATT&CK/CAPEC mapping, attribution panel | "It names the stage, maps it to MITRE ATT&CK, and shows which flows drove it." |
| 40–52 s | Open the response panel, click **Simulate interventions**, show before → after risk per control | "What stops it? It simulates each response before you touch the firewall." |
| 52–60 s | Ledger panel, then title card | "Every alert is hash-chained evidence. Netrikan: the third eye for India's networks." |

Use the **15 minutes** horizon (the app default).

### Live demo (finale, 3 minutes)
1. Upload the Monday benign CSV: quiet. *"No false alarm on real normal traffic."*
2. Upload the Wednesday intrusion CSV and walk the slider across the onset.
3. Simulate interventions and read the before/after risk off the screen. Don't pre-script which control wins; say what it shows.
4. Pull the network cable. *"Still running. Fully offline."*

Before the finale:
- Run both CSVs once on the presenting laptop. The first analysis takes about 15 s to load the model.
- Rehearse the response simulation; it takes about 13 s on a CPU-only laptop.

---

## 5. Numbers bank (only these; every one verified)

| Claim | Value | File |
|---|---|---|
| Median warning (k30, 5% FPR) | 810 s = **13.5 min**, 66% of 15,898 onsets warned, p75 36 min | `lead_time_cic_v2_w30.json` |
| Warning at a strict 1% FPR (k15) | 27% of onsets, median 2.5 min | same |
| Transition-window F1, k=0/30/90/180/360 | 0.380 / 0.459 / 0.464 / 0.455 / 0.438 · persistence 0.000 | `transition_f1.json` |
| Transitions scored | 31,781 – 190,380 per horizon | same |
| Onset AUC in-dataset k5 | **0.901** (persistence 0.5) | `cic_v2_w30_metrics.json` |
| Onset AUC, 4 seeds | 0.897 ± 0.007 | `seed_variance_cic_full_w30.json` |
| Held-out days (7-fold) onset AUC k5 | 0.636 ± 0.093 | `lodo_cic_full_w30.json` |
| Held-out days, supported-class F1 | 0.538 ± 0.178 | same |
| Unseen network (DAPT) onset AUC k15 | **0.705** | `dapt_cic_v2_w30.json` |
| DAPT lateral-movement recall | **0.942** | `dapt_base_w30.json` |
| Shuffled time order, PR-AUC k5 | 0.494 → 0.371 | `onset_pr_cic_v2_w30.json`, `onset_pr_shuffled_w30.json` |
| World-model next-state skill | +22% vs persistence, +6% vs window mean; PASS | `rollout_eval_cic_v2_w30.json` |
| Calibration ECE | 0.0088 → 0.0052 | `temperature.json` |
| Model size / latency / throughput | 0.893 MB · 1.41 ms (ONNX) · 610/s per thread | `latency.json` |
| Attackers hold a phase | measured persistence 0.93–0.97 per stage on DAPT | `dapt_persistence.json` |

**Not in the bank, so never quote:** `stage_macro_f1_DO_NOT_REPORT`; per-window DAPT FPR 40% as a headline; the per-host 0.84 (one fold only); any "accuracy %".

---

## 6. Q&A bank

| Question | Answer (under 20 seconds each) |
|---|---|
| Your LSTM loses to a trivial baseline. | "On all-window F1, yes. We measured and published it. That metric rewards copying the last label. Where the attack changes stage, the baseline scores zero and we score up to 0.46, at every horizon." |
| What's your false-positive rate? | "It depends on the budget the operator sets. At a 5% budget we warn two in three attacks, a median 13.5 minutes early. At 1%, a quarter, 2.5 minutes early. We show the whole curve, not one cherry-picked point." |
| On a new network, FPR is high, isn't it? | "At the default threshold, yes, because 'normal' is local. But attacks transfer: 94% lateral-movement recall on a network we never trained on. So we ship one attack model and calibrate normal per site." |
| One of your held-out-day folds is below 0.5. | "Fold 1. That day has 12 times more DoS than training ever saw. Day 4, also DoS, scores 0.72. It's a volume artefact, and we report it rather than hide it." |
| Is it really a world model? | "Its state head predicts the next network state 22% better than copying the current one, and beats the window average too. We feed that prediction back in to simulate ahead." |
| How do you know it isn't memorising? | "Three tests: hide entire days, use a different network, scramble time order. It survives the first two and degrades on the third, which is exactly what a sequence model should do." |
| Why not a transformer? | "0.89 MB and 1.4 ms on a CPU. A substation doesn't have a GPU. We chose what deploys." |
| Encrypted traffic? | "Every feature is metadata: timing, size, flags. It never needs the payload, so TLS 1.3 is fine, and citizen content is never read." |
| Zero-day or unknown attacks? | "Honestly, unseen attack families are hard for every model; we tested it and it's weak. That's why the rule layer runs alongside, and disagreement between the two is surfaced as 'novel'." |
| Where's the blockchain? | "The alert ledger is a SHA-256 hash chain, the same integrity primitive. Alter or delete any alert and verification fails. It's evidence for a CERT-In report." |
| Does it track individual attackers? | "Our deployed model works at network level. Per-host tracking is our next step: early results on DAPT are strong but not yet confirmed at scale, so we don't claim it." |
| How would NTRO deploy it? | "A SPAN or tap port per site, flows in, ONNX model on existing hardware, fully offline. Alerts and ledger go to the SOC. One shared attack model, with local calibration." |
| What does the response simulation actually do? | "It edits the observed traffic as if the control were applied, for example dropping SMB, and re-runs the model. The change in risk is the answer." |
| Datasets are old. | "CIC-IDS-2018 is the standard benchmark, which is why we also test on DAPT 2020, a real multi-day APT on a network the model never saw." |
| What if the attacker is slow? | "Measured on DAPT, attackers hold each phase with 93–97% persistence, so they are slow. That's why we forecast the transition, not the state." |
| How is this different from Suricata or Snort? | "They match signatures on what already happened. We forecast the next stage from behaviour, with no signature needed. They're complementary: our rule layer plays the same role inside Netrikan." |

---

## 7. If we're shortlisted (finale extensions)

Add as appendix slides or talking points, **only once each is backed by a JSON:**
- **Per-host attacker tracking.** Currently a 1-fold hint (0.84 vs 0.65 onset AUC on DAPT). Needs ≥5 folds (Pranav, UWF data).
- **Time Bought.** "Blocking SMB buys N seconds before C2." It needs the counterfactual measured through the rollout.
- **Suricata head-to-head.** Same PCAP, first Suricata alert vs first Netrikan forecast, in seconds.
- **Per-host SPRT alerting.** Sets the false-alarm rate from a theorem instead of a tuned threshold.
- **D3FEND mapping.** Turns each forecast into a named MITRE countermeasure.
- **SEDI skill score.** Stays the same whether attacks are 1% or 30% of traffic, which ends the FPR argument.

---

## 8. Never say

- "Beats persistence" **without** "on transition windows"
- "99%", "near-perfect", or any accuracy percentage
- `stage_macro_f1` from the LODO file
- Per-host tracking, SPRT, Time Bought, Suricata comparison or D3FEND as **built**
- Competitor team names
- "Detects zero-days"
- "Blockchain-based" beyond the SHA-256 hash-chained ledger

---

## 9. Pre-submission checklist

- [ ] 6 slides including the title; template slide 7 deleted
- [ ] Template headings unchanged
- [ ] Team ID, team name and exact problem-statement title filled in
- [ ] Every number appears in §5 of this file
- [ ] Nothing from §8 appears on any slide
- [ ] Roadmap items labelled "roadmap"
- [ ] Dashboard screenshot is current
- [ ] QR code resolves to the demo video, or remove it
- [ ] Exported as PDF; opened once to check fonts and layout
- [ ] Presenter can deliver the §6 answers in under 20 s each
