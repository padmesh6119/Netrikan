You are helping with a competitive hackathon project called Netrikan — an AI-based
network attack forecasting system. The deadline is 2026-09-25 (6 days away).
Produce a single master document (markdown) called WINNING_PLAN.md.

---

## WHAT NETRIKAN IS

An LSTM-based network intrusion forecasting system that predicts which ATT&CK
kill-chain stage comes next before the damage is observable.
Stack: Python, PyTorch, Streamlit, tshark, Suricata. CPU-only deployment.
Trained on CIC-IDS-2018 (5.5M windows). Cross-dataset tested on DAPT 2020
(86,691 flows, real 4-day APT, never trained on).

Current real metrics (all verified from committed JSON files):
- macro-F1 on held-out: 0.885 (LSTM), 0.906 (persistence baseline) — LSTM loses
- Cross-dataset on DAPT 2020: attack recall 83.9%, precision 42.2%, FPR 41.6%, ROC-AUC 0.731
- Lateral movement recall on DAPT: 93.1% (per-host windowing raises it to 95.9%)
- Persistence baseline beats LSTM at every horizon k=0,30,90,180,360 (gap -0.017 to -0.050)

What's already built:
- LSTM world model with stage head + breach head + state head (rollout exists in model.py)
- Fusion layer: 30% model / 70% rules (14 rule detectors in signals.py)
- Markov kill-chain projection in forecast.py (DOCTRINE_SHAPE matrix, not learned)
- Counterfactual response engine (4 interventions, live risk delta in dashboard)
- SHA-256 hash-chained alert ledger in app.py
- CAPEC IDs mapped in attck_map.py
- Per-host entity eval done: lateral movement recall +0.028 (0.931 -> 0.959)
- Demo attack lab: real captured PCAP (nmap scan + SSH brute + SYN flood + C2 beacon)
- Suricata 8.0.6 + Emerging Threats Open ruleset (52,311 rules) installed locally

Known honest weaknesses we must own, not hide:
1. LSTM loses to persistence baseline -- stages persist 91% of the time
2. FPR 41.6% cross-dataset -- model over-fires on benign DAPT traffic
3. Lead-time claim is self-referential -- forecast threshold vs model's own detection, not vs real IDS
4. gradient x input mislabeled as SHAP in infer.py (already flagged in code, not yet fixed)
5. Transition matrix is ATT&CK doctrine, not learned probabilities

---

## JUDGING CONTEXT (verified)

The evaluating organisation is a national technical intelligence agency.
It oversees Critical Information Infrastructure protection.
CERT-In Directions No. 20(3)/2022-CERT-In (28 April 2022): 6-hour incident reporting requirement,
180-day log retention within Indian jurisdiction. Non-compliance: up to 1 year imprisonment.

Hackathon finale format: 36-hour continuous hackathon. Jury visits the same table 3-4 times
and scores VISIBLE DELTA between visits, not one polished demo. "Demo first, 60 seconds" is
the standard. Hardcoded/canned demos are explicitly named as a losing pattern by past juries.
Saying "dummy data" ends the conversation. Juries punish: fake metrics, no deployment,
template placeholder text in PPT, demo links that sleep/go blank.
Juries reward: naming real failures and mitigations, live demo on any input, deployment
feasibility with named integration points, a rough cost figure.

"Fully offline, air-gapped, no data leaves the perimeter" is a real selling point for the client organisation
because of the 180-day in-India log retention mandate.

---

## COMPETITOR LANDSCAPE (all verified from source code)

Best competitor: HowSuyash/AttackForecast -- real RSSM world model (DreamerV2-style),
per-host 60s buckets built and running, beats persistence at 9/10 horizons,
family-holdout FPR 0.002-0.019. This is the one repo that looks more finished than us today.

DurgeshLabs/What-the-hack -- real trained autoregressive LSTM rollout, JWT/RBAC/Docker/CI,
but network-wide windowing (no per-host) and gradient x input mislabeled as SHAP (same as us).

syednzaheer/DEFENDER -- real rollout, honest "our LSTM loses to logistic regression" admission,
SHA-256 source data hashing, but no per-host identity at all.

csxzor-devcs/sih -- best engineering process (108 tests, automated leakage CI gate,
claims-vs-code enforcement), zero empirical results, no committed weights.

siddhanthaditiyaa-beep/SIH-2026 -- real cross-day XGBoost result (87.7% precision on
genuinely unseen attack types), correctly-labeled tree-SHAP, but zero ATT&CK mapping.

soumyachk101/NetSentinel-AI -- fabricated. F1 0.94 claim with empty checkpoints/.

NOBODY has: counterfactual engine, calibration/conformal prediction, alert-budget controller,
TLS/JA3 fingerprinting, CAPEC mapping, adversarial robustness curve.

---

## WHAT TO PRODUCE

A single WINNING_PLAN.md with these sections:

### 1. THE THREE NARRATIVE PIVOTS
How to reframe our honest weaknesses as strengths the jury can check:
- Persistence baseline (we lose to it) -> how to turn this into a story
- 41.6% FPR cross-dataset -> what the base-rate arithmetic actually says at realistic
  attack base rates (use Axelsson's base-rate fallacy result if you know it)
- Self-referential lead time -> how to fix it with Suricata (installed) in one day

### 2. THE SIX SHIPPABLE UPGRADES (6 days, CPU-only)
For each, give: description, which file(s) to edit, estimated hours, and the
exact claim it lets us make to a jury. Ranked by (jury impact) / (hours to build).
Must include:
a. Transition-only / change-point evaluation -- re-score ONLY windows where the label
   changes (8-9% of data). This is the real forecasting metric. Likely much better than
   the all-windows score. ~50 lines in a new eval script.
b. Real Suricata lead-time: run Suricata offline on demo PCAP, extract first alert
   timestamp, compare to our forecast timestamp. Kills the circular-claim problem.
c. Temperature scaling calibration: fit T on a held-out calibration split, report ECE
   and reliability diagram. ~40 lines in src/calibration.py. Zero competitors have this.
d. z-score OOD flag: flag windows where 3 or more features are >4 sigma from training
   distribution as "insufficient evidence -- model may not apply". ~30 lines. Works on
   the frozen checkpoint, no retraining. Two competitors have weaker versions of this.
e. Brier skill score: normalise our Brier score against the persistence baseline.
   If BSS > 0 we beat persistence on probabilistic calibration even if macro-F1 doesn't.
   This is how meteorologists escape the persistence-baseline trap. ~10 lines.
f. SHA-256 source data provenance: hash the DAPT CSVs and embed digests in evaluation
   JSON. One competitor (DEFENDER) does this, nobody else. Directly reinforces our
   "checkable by jury" positioning.

### 3. THE DEMO SCRIPT (3 minutes, any input)
A step-by-step walkthrough that:
- Works on the captured demo PCAP (attack_small.pcap, 84KB)
- Shows the counterfactual panel live ("block SMB -> risk 0.71->0.12")
- Shows the SHA-256 ledger entry
- References the real Suricata alert timestamp vs our forecast timestamp
- Names our honest numbers when asked (never hides them)

### 4. THE JURY Q&A CHEAT SHEET
The 8 hardest questions a jury will ask, with the exact answer to give.
Include: "your LSTM loses to a simple baseline", "42% FPR is terrible", "show me
the source data", "why should the client trust this over a signature IDS", "what does CAPEC mean",
"where does the 6-hour CERT-In timeline fit", "can you run this air-gapped".

### 5. THE 5-SLIDE DECK OUTLINE
One sentence per slide, what the visual is, what number anchors it.
Slide 3 must use the lateral-movement recall result (0.931 -> 0.959 per-host).
Slide 5 must end with the counterfactual demo screenshot.

### 6. THE 2-MINUTE VIDEO STRUCTURE
Shot list with timestamps. The demo attack PCAP should feature in it.

---

Be concrete and direct. No AI disclaimers. Assume the reader is a
senior engineer with 6 days left and zero tolerance for filler.
