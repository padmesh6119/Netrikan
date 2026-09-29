# Netrikan: the whole project, explained

This is a study guide for presenting Netrikan. It covers what the product does, how every part works, what the numbers mean, what we can and cannot claim, and how to demo it. Every number here comes from a results file in the repo (`results/demo/metrics.json`, `results/z24/metrics.json`, `results/zero_shot/metrics.json`) or from the data audit (`docs/DATA_AUDIT.md`).

---

## 1. Netrikan in one minute

**One-line pitch:** Netrikan watches every host on a network minute by minute, forecasts whether an attack is about to start on that host, says which ATT&CK technique or stage it is, and explains why. It runs fully offline.

**What it actually does:**
1. Reads network flow records (CSV from CICFlowMeter, or Zeek / Argus logs).
2. Turns them into a **state vector per host per minute**: how many connections, to how many peers and ports, bytes, flags, which kinds of ports.
3. Runs a ladder of models, from logistic regression up to a **GRU world model** that simulates the host's next few minutes.
4. Outputs, for every host-minute:
   - the probability of an attack starting in the next 5 minutes;
   - the likely ATT&CK stage or technique;
   - the features driving that prediction (TreeSHAP).
5. Shows it all in a Streamlit dashboard with a risk timeline, explanations, a network heatmap and a benchmark page.

**What makes it different:** every claim is tested against a simpler baseline and a "shuffle control". Results that failed are reported alongside the ones that worked. Most competitors only report wins.

---

## 2. The problem (from `problem-statement.md`)

The SIH brief asks for a **world model** of network behaviour:

- Represent network state as feature vectors or graphs.
- Learn how that state changes over time, written P(S_t+1 | S_t): "given the network now, what does it look like next?"
- Roll the model forward K steps and estimate the probability of infiltration **before the attacker finishes**.
- Map the prediction to MITRE ATT&CK stages (Recon, Initial Access, Lateral Movement, C2, Exfiltration).
- Explain predictions (SHAP or attention).
- Provide an offline demo, and benchmark against logistic regression on F1, precision, recall and FPR.

**Why "world model" instead of a classifier:** a normal intrusion detector looks at one flow and says "bad" or "good". An attack is a process: scanning, then breaking in, then moving around, then stealing data. A world model tries to learn the process, so it can warn *before* the damaging step.

---

## 3. Concepts you need to be able to explain

| Term | Plain meaning |
|---|---|
| **Flow** | One connection between two machines (who, to whom, which port, how many bytes and packets, how long). Our raw input. |
| **Host-minute** | One monitored machine during one minute of wall-clock time. Our basic unit. Every flow is added to the minute it started in, for both the sender and the receiver. |
| **Entity-centric** | We model each host separately, not the whole network's jumbled stream of flows. That way "the last 10 minutes" means the last 10 minutes *of that host*. |
| **Horizon K** | How far ahead we forecast. K = 5 minutes. |
| **History L** | How much past the model looks at. L = 10 minutes (DAPT); 150 minutes (ZeekData24 attacker forecasting). |
| **Onset** | The first minute of an attack after at least 5 quiet minutes. Warning *before* an onset is what "forecasting" means. |
| **Onset task** | We score only minutes where the host is currently benign, and ask "will attack traffic start in the next 5 minutes?". This stops the model getting credit for noticing an attack that is already happening. |
| **Base rate / prevalence** | The share of positive examples. A random guesser's PR-AUC equals the base rate, so always compare against it. |
| **ROC-AUC** | The chance that a random attack minute scores higher than a random benign minute. 0.5 = coin flip, 1.0 = perfect. |
| **PR-AUC** | Average precision. Better than ROC-AUC when attacks are rare. Compare it to the base rate: 0.12 when the base rate is 0.045 is 2.6x better than random. |
| **Alarm budget** | "At most 1 false alarm per host per hour." We set the threshold to meet the budget on held-out benign data, then report recall at that threshold. This is how a SOC would actually run it. |
| **Lead time** | How many minutes before an onset the first alarm fired. |
| **Calibration (Brier, ECE)** | Whether a predicted "30%" really happens 30% of the time. Lower is better. |
| **Leave-one-day-out (LODO)** | Train on 4 days, test on the 5th, rotate. The test day is never seen in training, which is harder and more honest than a random split. |
| **Shuffle control** | Scramble the order of the history minutes and retrain. If the score barely drops, the model is not using time order, so it is not really forecasting dynamics. |
| **World model** | A neural network (GRU) trained to predict the next minute's state from the past. We feed its predictions back in to simulate K minutes ahead ("rollout"). |
| **GRU** | Gated recurrent unit, a small recurrent neural network (similar to an LSTM) that reads a sequence one step at a time and keeps a memory. |
| **LightGBM / GBDT** | Gradient-boosted decision trees. A strong, fast baseline on tabular data. |
| **Lag features** | The same features 1, 2 and 3 minutes ago, plus the 10-minute mean and max. This gives the tree model a view of the recent past. |
| **TreeSHAP** | An exact way to split a tree model's prediction into per-feature contributions ("inbound SYN count pushed risk up by +0.8"). |
| **Zero-shot** | Testing on a dataset, from a different lab and with different attacks, that the model never saw in training. |
| **IsolationForest** | An anomaly detector trained only on normal traffic. It flags things that are unusual; it needs no attack labels. |
| **MITRE ATT&CK** | The industry catalogue of attacker tactics (why) and techniques (how), e.g. T1595 Active Scanning, T1110 Brute Force. |

---

## 4. The data

We use four public datasets. Each one covers something the others don't.

### 4.1 DAPT2020, the main forecasting dataset
- **What:** a 5-day simulated APT (advanced persistent threat) campaign from 2019, as CICFlowMeter flows. 86,688 flows after removing duplicates.
- **Why it matters:** it is the **only** dataset we have where benign and attack traffic happen at the same time on the same network, with kill-chain stage labels. That is what forecasting needs.
- **Stages:** Reconnaissance, Establish Foothold (we map it to ATT&CK *Initial Access*), Lateral Movement, Data Exfiltration.
- **Size once modelled:** 9 monitored (private) hosts, 33 host-days, 16,490 host-minutes, **319 attack minutes**, **109 attack onsets**, and only 3 hosts that are ever attacked.
- **Defects we fixed** (in `src/netrikan/dapt.py`):
  - one file has no header row;
  - benign is spelled two ways ("BENIGN" and "Benign");
  - timestamps are 12-hour, day-first text;
  - no file is sorted by time;
  - the same flow appears at both the public and private capture points.
- **Catch:** each day contains basically one stage (Tue recon, Wed foothold, Thu lateral, Fri exfil). So "predict the stage on an unseen day" means predicting a stage never seen in training.

### 4.2 UWF-ZeekData24, a scripted attack campaign (University of West Florida)
- **What:** 1.9 million Zeek connection records from 2024, each attack flow labelled with an ATT&CK technique.
- **Techniques:** T1595 Active Scanning, T1190 Exploit Public-Facing App, T1078 Valid Accounts, T1110 Brute Force (91% of attack flows), T1048 Exfiltration.
- **Structure:** 5 attack-only weeks (Feb–Mar 2024) and 2 benign-only weeks (Oct–Nov 2024). The attackers are 15 lab hosts, which also appear as normal hosts in the benign weeks.
- **Key discovery from our analysis:** the attacks are not a kill chain. Every technique starts within the first hour and repeats roughly **hourly with heavy jitter**, all running in parallel for days. The 5 weeks are replays of the same scripted campaign.
- **Use:** technique recognition (behaviour → ATT&CK technique) and attacker forecasting ("when is this attacker's next burst of technique X?").

### 4.3 CIC-IDS2017 slices (Friday port scan + DDoS; Wednesday Heartbleed)
- 397,272 flows, 15 monitored hosts, 49 attack host-minutes (17 port scan, 21 DDoS, 11 Heartbleed). A corrected version with "failed attempt" flows removed.
- **Use:** zero-shot testing only. It is never trained on when it is the test set.

### 4.4 CTU-13 scenario 4 (botnet, 2011, Czech Technical University)
- 374,310 Argus flows, 1,091 hosts, 71 botnet host-minutes (spam, ICMP, C&C).
- **Use:** zero-shot testing. It is stealthy and low-volume, which makes it a hard case.

### 4.5 Data we looked at but don't use
- **ZeekData22 / ZeekDataFall22:**
  - rows duplicated up to 256 times, so "9.28M recon flows" is really 36,247;
  - attackers and benign hosts never overlap;
  - Fall22's benign traffic is 99.9% copied from ZeekData22.
  - We documented this in the audit and moved on.
- **CIC-2018 parquet:** it has no IP addresses or timestamps, so per-host modelling is impossible. The full 9.7 GB zip with IPs is on disk but is not used.

**Presenting tip:** the data audit is a strength. "We found that a popular dataset is duplicated 256x and its test set overlaps its training set" is exactly the kind of rigour judges remember.

---

## 5. The pipeline, step by step

```
flow CSV / Zeek / Argus
      │  src/netrikan/dapt.py, zeek.py, extra.py      (load, clean, normalise columns)
      ▼
per-host, per-minute state vectors
      │  src/netrikan/features.py                     (38 features, signed log1p, dense minute grid)
      ▼
model ladder
      │  src/netrikan/models.py      (DAPT: LR → LightGBM → +lags → GRU world model; stage model)
      │  src/netrikan/campaign.py    (Z24: technique recognizer; attacker forecaster ladder + GRU)
      │  src/netrikan/surprise.py    (GRU "surprise" anomaly score for zero-shot)
      ▼
metrics + registry
      │  src/netrikan/metrics.py     (PR/ROC-AUC, recall at alarm budget, lead time, ECE/Brier, bootstrap CIs)
      ▼
Streamlit app (app/) + CLI (scripts/infer.py)
```

### 5.1 State vector (features.py)
For each monitored host and each minute we compute the following, **twice**: once for traffic the host received (`in_`) and once for traffic it sent (`out_`):
- number of flows, distinct peers, distinct destination ports;
- forward and backward packets and bytes, mean flow duration;
- SYN / RST / ACK / PSH / FIN flag counts, mean packet length;
- share of flows to web ports, SSH, database ports, other ports below 1024, and high (ephemeral) ports.

That gives 19 × 2 = **38 features**.

**Hygiene rules:**
- Heavy-tailed counts go through signed log1p, so one huge DDoS minute doesn't dominate.
- Scalers are fitted on training rows only.
- **IP addresses, raw port numbers and timestamps are never model inputs.** They are used only to group and order the data, so the model cannot memorise "IP 192.168.3.29 is the victim".
- Minutes with no traffic inside a host's active period are kept as explicit zero rows, so "10 rows back" really means "10 minutes back".

### 5.2 The DAPT model ladder (models.py)
Each rung must beat the one below it, or it doesn't earn its complexity:

1. **Logistic regression** on the current minute. This is the brief's required baseline.
2. **LightGBM** on the current minute.
3. **LightGBM + lags:** the current minute, plus 1, 2 and 3 minutes ago, plus the 10-minute mean and max.
4. **World model:**
   - A residual GRU learns "next minute's state = this minute + a learned change" from the last 10 minutes.
   - At prediction time it is **rolled forward 5 steps**, feeding its own predictions back in.
   - A LightGBM head reads the current state, the simulated future (mean and max) and the GRU's internal memory, then outputs P(attack in the next 5 minutes).
5. **Stage model:** a separate LightGBM that, for a forecast attack, predicts which stage it will be.

### 5.3 The ZeekData24 models (campaign.py)
- **Task A, recognizer:** one LightGBM per technique, reading a single host-minute. It answers "which techniques are running right now?".
- **Task B, attacker forecaster.** For each attacker, it predicts "will technique T fire in the next 5 minutes?". Ladder:
  - **renewal hazard:** looks only at minutes since the last burst. The simplest schedule model.
  - **current behaviour only.**
  - **own-technique history:** minutes since the last burst, and bursts in the last 60 and 150 minutes, for that technique only.
  - **all-technique history ("sched"):** the same for all 5 techniques. This is our main model.
  - **GRU world model:** learns next-minute technique flags plus traffic volume, rolls 5 steps ahead, and a GBDT head reads the result.
- **Controls:**
  - shuffled-history versions of the history model and the world model;
  - a "true flags" upper bound (what if recognition were perfect);
  - a **clock ablation** that adds minute-of-hour. It breaks our no-timestamp rule, so it is labelled as an experiment, never shipped.
- **Deployment realism:** in the app, the forecaster's history comes from the recognizer's *predicted* flags, not ground truth.

### 5.4 Zero-shot detectors (scripts/zero_shot.py)
Four datasets. For each, we train on the other three and test on it. Features are the 22 host-minute features that all four datasets share; CTU-13's Argus logs have no flag counters or packet split. Detectors:
- **Supervised LightGBM** trained on the other labs' attacks.
- **IsolationForest** trained only on the other labs' benign minutes.
- **World-model surprise:** a GRU trained on benign minutes to predict the next minute; a big prediction error means "surprising".
- **Heuristics:** raw flow volume, and fan-out (distinct peers + ports).

---

## 6. How we keep ourselves honest

These rules come from a previous failed attempt that trained for weeks with no progress:

1. **Per-host, per-minute sequences.** Never windows over the mixed network stream.
2. **Hard splits.**
   - DAPT: leave-one-day-out.
   - ZeekData24: leave-one-attack-week-out.
   - Zero-shot: leave-one-dataset-out.
   - Easier "within-day" splits are shown only as secondary.
3. **Baselines first.** A neural model only counts if it beats LightGBM with lags.
4. **Shuffle control** on every sequence model.
5. **Pre-registered hypotheses.** Before each run, `results/registry.csv` records the hypothesis and our predicted outcome, then the actual outcome. Some predictions were confirmed and some refuted, and both are kept.
6. **Leakage tests** (`make test`, 15 tests). Examples:
   - corrupting all future flows must not change any past feature;
   - history windows never reach forward in time or across hosts;
   - the forecast target matches a brute-force recomputation;
   - scalers see only training rows.
7. **No tuning on test data.** Alarm thresholds come from held-out benign minutes.

---

## 7. Results, and what they mean

### 7.1 DAPT2020: forecasting attack onset (leave-one-day-out)
Onset task: host benign now, attack starts within 5 minutes. Base rate 0.045 (714 positive of 16,015 minutes).

| Model | PR-AUC | ROC-AUC | Recall @1 false alarm/host-hr | Precision | F1 | FPR | Onsets warned early | Median lead |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.119 | 0.68 | 0.083 | 0.188 | 0.115 | 0.017 | 23% | 4 min |
| LightGBM (current minute) | 0.120 | 0.72 | 0.088 | 0.198 | 0.122 | 0.017 | 27% | 4 min |
| LightGBM + lags | 0.117 | 0.71 | 0.090 | 0.201 | 0.124 | 0.017 | 17% | 3 min |
| GRU world model | 0.104 | 0.70 | 0.063 | 0.150 | 0.089 | 0.017 | 21% | 4 min |
| LightGBM + lags, history shuffled | 0.120 | 0.71 | 0.076 | | | | | |
| World model, history shuffled | 0.121 | 0.72 | 0.091 | | | | | |

The 90% confidence interval for PR-AUC is roughly 0.03–0.21 for every model.

**What this means, in words you can say:**
- **Ranking works:** every model is about **2.6x better than random** at putting "attack is coming" minutes above normal ones. Within a single held-out day, ROC-AUC is 0.74–0.95.
- **The world model does not beat the simple models,** and **shuffling the history does not hurt.** So on this data the warning comes from *what the host looks like right now* (early probing traffic), not from learned dynamics. We say this openly.
- **Stage prediction across days is 0%** (majority-class guess: 50%), because each test day contains a stage never seen in training. With a within-day split where stages are shared, it reaches 63% (majority: 58%).
- **Why:** 319 attack minutes, 3 attacked hosts, one lab, one week. The data is too small for dynamics learning to show a gain.

### 7.2 ZeekData24: technique recognition (leave-one-attack-week-out)
Threshold fixed at 0.5 (not tuned):

| Technique | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| T1595 Active Scanning | 0.998 | 0.99 | 0.996 | 0.99 |
| T1190 Exploit Public-Facing App | 1.000 | 1.00 | 1.00 | 1.00 |
| T1078 Valid Accounts | 0.998 | 0.97 | 0.995 | 0.99 |
| T1110 Brute Force | 1.000 | 1.00 | 1.00 | 1.00 |
| T1048 Exfiltration | 0.852 | 0.71 | 0.93 | 0.80 |

Benign false alarms: 0.006 per host-hour.

**What this means:** the behaviour-to-ATT&CK mapping works. But the near-perfect scores are a **warning sign, not a victory**:
- the attacks are identical scripts replayed each week;
- attack minutes contain no mixed-in benign traffic;
- so this measures repeatability, not real-world detection.

Say that before a judge does.

### 7.3 ZeekData24: forecasting the attacker's next burst
Base rate 0.067. Macro average over the 5 techniques:

| Model | PR-AUC | ROC-AUC | Recall @1/hr | Precision |
|---|---|---|---|---|
| Current behaviour only | 0.060 | 0.44 | 0.03 | 0.09 |
| Renewal hazard (time since last burst) | 0.189 | 0.79 | 0.16 | 0.28 |
| Own-technique history | 0.206 | 0.81 | 0.19 | 0.29 |
| **All-technique history (main)** | **0.221** | **0.82** | **0.20** | **0.31** |
| … history order shuffled | 0.159 | 0.76 | 0.15 | 0.24 |
| **GRU world model** | **0.212** | **0.82** | 0.19 | 0.27 |
| … history order shuffled | 0.158 | 0.77 | 0.15 | 0.24 |
| Upper bound (true flags as history) | 0.234 | 0.82 | 0.23 | 0.32 |
| Ablation: + clock minute (not allowed) | 0.275 | 0.85 | 0.23 | 0.31 |

**What this means:**
- **This is real temporal forecasting:** 3.3x better than chance, and **shuffling the history costs 28%** (LightGBM) and **26%** (world model). The models genuinely use *when* things happened.
- The **world model matches** the hand-built history features. It learns the same signal from raw sequences, but there is nothing extra to find.
- Other techniques' history adds only a little over a technique's own history, and cross-technique timing is near random. So **there is no kill-chain progression in this data**; it is parallel scheduled scripts.
- The recognizer's predicted flags are nearly as good as the truth (0.221 vs 0.234), so the forecast works in a deployable setup.

### 7.4 Zero-shot across labs (leave-one-dataset-out)
ROC-AUC per unseen attack family (0.5 = chance):

| Held-out dataset | Family (positive minutes) | Supervised | IsolationForest | GRU surprise | Volume | Fan-out |
|---|---|---|---|---|---|---|
| DAPT2020 | Reconnaissance (70) | 0.75 | 0.97 | 0.46 | 0.98 | 0.97 |
| DAPT2020 | Initial Access (93) | 0.42 | 0.93 | 0.50 | 0.92 | 0.94 |
| DAPT2020 | Lateral Movement (144) | 0.44 | 0.66 | 0.60 | 0.70 | 0.75 |
| DAPT2020 | Exfiltration (12) | 0.46 | 0.74 | 0.80 | 0.85 | 0.92 |
| ZeekData24 | Active Scanning | 0.74 | 0.81 | 0.96 | 0.36 | 0.52 |
| ZeekData24 | Exploit | 0.05 | 0.79 | 0.99 | 0.19 | 0.47 |
| ZeekData24 | Valid Accounts | 0.74 | 0.62 | 0.88 | 0.20 | 0.47 |
| ZeekData24 | Brute Force | 0.46 | 0.92 | 0.96 | 0.98 | 0.47 |
| ZeekData24 | Exfiltration | 0.44 | 0.50 | 0.94 | 0.25 | 0.52 |
| CIC-IDS2017 | Port scan (17) | 0.93 | 0.97 | 0.61 | 0.84 | 0.92 |
| CIC-IDS2017 | DDoS (21) | 0.88 | 1.00 | 0.46 | 1.00 | 0.93 |
| CIC-IDS2017 | Heartbleed (11) | 0.57 | 0.80 | 0.86 | 0.27 | 0.36 |
| CTU-13 | Botnet (71) | 0.34 | 0.58 | 0.18 | 0.90 | 0.85 |
| **Mean** | | **0.55** | **0.79** | **0.71** | **0.65** | **0.70** |

**What this means (a strong slide):**
- **A classifier trained on other labs' attacks does not transfer.** It averages 0.55 and is *below chance on 7 of 13 families*. Attack "signatures" from one lab can look like normal traffic in another.
- **Learning what normal looks like transfers best.** IsolationForest averages 0.79 with no attack labels at all.
- **GRU world-model surprise** is excellent on some families (ZeekData24 0.88–0.99, Heartbleed 0.86) and bad on others (CTU-13 botnet 0.18, DDoS 0.46). It averages 0.71.
- **Design conclusion:** for unseen attacks, use anomaly-style scoring on per-host state, not a classifier of known attacks.
- **Caveat:** positive minutes are small (11–144 outside ZeekData24), and minutes from the same host aren't independent, so small differences are noise.

---

## 8. The app, page by page (use this for the demo video)

Start with `make demo`, then open http://localhost:8501. The left sidebar is the **dataset chooser**.

### Page 1: DAPT2020 · pentest campaign (default)
The sidebar has:
- a capture day, scored by the model trained on the *other* days;
- an option to upload your own CSV;
- a risk model choice (LightGBM + lags, or world model);
- an alert threshold slider (default = 1 false alarm per host-hour).

The headline row shows flows, hosts, alert minutes, false alarms per host-hour, attack onsets, how many were warned early, and the median lead time.

Tabs:
1. **Risk timeline:**
   - a per-host line of P(attack in next 5 min);
   - the dashed threshold, red alert dots, and shaded true-attack bands coloured by stage;
   - a host ranking table.
2. **Explain a prediction:**
   - pick a high-risk minute to see the probability, what actually happened next, and the predicted stage bar chart;
   - a **TreeSHAP** bar chart of driving features;
   - the **world-model forward simulation** (simulated vs actually observed, 5 minutes ahead);
   - the raw flows around that minute.
3. **Network heatmap:** risk for every host over time, above the true attack minutes.
4. **Benchmark & evidence:**
   - the leave-one-day-out table including logistic regression, F1/P/R/FPR and the shuffle rows;
   - per-day ROC-AUC, the within-day secondary table and the stage table;
   - a written "what this does and doesn't support".
5. **How it works:** the pipeline and stage mapping.

**Best demo moment:** pick **2019-07-16**, host **192.168.3.29**. Show risk rising before the reconnaissance bands, then open "Explain a prediction" on the top minute.

### Page 2: UWF-ZeekData24 · scripted campaign
The sidebar has a week (attack or benign), the recognizer threshold, the forecast model and the alert budget.
1. **Recognize techniques:**
   - a per-technique table (flagged vs true minutes, recall, precision);
   - a timeline of recognized vs true technique minutes for one host;
   - TreeSHAP for a flagged minute.
2. **Forecast attacker bursts:**
   - pick an attacker, technique and day to see the model's forecast line against the renewal baseline;
   - actual bursts shown as red rules, alerts as dots;
   - onsets, % warned, false alarms per attacker-hour, lead time;
   - TreeSHAP for the history model ("minutes since last Brute Force burst").
3. **Campaign structure:**
   - burst timing is spread evenly across the hour;
   - gaps between bursts cluster around 60 minutes;
   - cross-technique coupling is weak, so there is no kill chain.
4. **Benchmark & evidence:** the recognizer, confusion and forecast ladder tables, with the written conclusions.

**Demo moment:** a benign week shows nearly zero alerts (0.01 per host-hour). Then switch to an attack week.

### Pages 3 and 4: CIC-IDS2017 and CTU-13
Both are **zero-shot**: they are scored by detectors that never saw them.
- **Sidebar:**
  - the CIC slice (Friday scan + DDoS / Wednesday Heartbleed / both);
  - the detector (IsolationForest, GRU surprise, supervised, volume, fan-out);
  - the alert budget.
- **Tabs:** the host timeline with attack bands, a live detector comparison on the slice, and a heatmap.
- **Demo moment:** on CIC Friday, IsolationForest catches the DDoS minutes. Then switch to the supervised model and watch recall drop. That shows why we use anomaly scoring for unseen attacks.

### Page 5: Zero-shot transfer across labs
The mean-ROC-AUC headline for each detector, a dot chart for all 13 unseen families, the full table and the conclusions.

**Also available:** a CLI, `make infer CSV=file.csv`, which writes per-host-minute risk, stage and the top 3 drivers.

---

## 9. How to run everything

```
make setup        # Python 3.12 venv, pinned packages, macOS LightGBM fix
make test         # 15 leakage/correctness tests
make demo-train   # DAPT2020 models (~2 min)
make z24-train    # ZeekData24 models (~8 min)
make zero-shot    # cross-lab experiment (~2 min)
make demo         # the app, fully offline
make audit        # regenerate the data audit numbers
```
Raw data lives in `data/` and is not in git.

---

## 10. What we claim and what we don't

**Claim confidently:**
- A working, offline, end-to-end pipeline:
  - flows → per-host state → forecast risk, stage or technique, and drivers → dashboard;
  - it works across three log formats (CICFlowMeter, Zeek, Argus).
- Forecast ranking is 2.6x (DAPT) and 3.3x (ZeekData24) better than chance on held-out days and weeks.
- On ZeekData24, the forecast provably uses temporal order: the shuffle control costs 26–28%.
- Technique recognition maps behaviour to ATT&CK techniques. It is near-perfect on a replayed campaign, with that caveat stated.
- For unseen attacks from other labs, anomaly scoring (0.79) transfers far better than supervised classification (0.55). We measured it across 4 datasets and 13 families.
- A data audit that found serious defects in popular datasets.

**Do not claim:**
- That the world model beats simpler models. It ties or loses; we say so.
- That stage prediction generalises to unseen days. It is 0% on DAPT leave-one-day-out.
- That ZeekData24 shows kill-chain progression. It shows parallel scheduled scripts.
- The numbers in the "differentiators" table of `COMPETITORS.md` (94% lateral recall, ONNX 1.4 ms, 25.4% host-hours). **They are not from this project.**

---

## 11. Questions judges will likely ask, with answers

**"Is this really a world model or just a classifier?"**
It includes a real world model: a GRU trained on P(next state | history), rolled forward K steps, and its simulated future is shown in the app. We also measured whether it helps, and on our data it matches but doesn't beat gradient-boosted trees. We'd rather show that than hide it.

**"How do you know the model uses time and not just the current snapshot?"**
The shuffle control. On ZeekData24, scrambling the history order drops performance by 26–28%, so it uses time. On DAPT it doesn't drop, and we report that as a limitation of the data.

**"Your recognition is ~100%. Isn't that overfitting?"**
Tested on held-out weeks, so it isn't memorised rows. But the weeks replay the same scripts, so it measures repeatability. That's why we also ran the zero-shot test on different labs.

**"How does it handle attacks it has never seen?"**
That is exactly the zero-shot experiment. Supervised detectors fail (0.55); benign-trained anomaly detection reaches 0.79. Netrikan's per-host state representation supports both.

**"Why so few attacks in DAPT?"**
Public multi-stage datasets are small. We audited every candidate. DAPT is the only one with benign and attack traffic concurrent on the same hosts, which forecasting requires.

**"Did you tune on the test set?"**
No. Thresholds come from held-out benign minutes. Every experiment's hypothesis and predicted outcome was written to `results/registry.csv` before it ran.

**"What about the logistic-regression baseline the brief requires?"**
It is the first row of every benchmark table, with F1, precision, recall and FPR at the same alarm budget.

**"Can it run offline?"**
Yes. No network calls; everything loads from local files.

---

## 12. Where Netrikan stands against the field (from `COMPETITORS.md`)

- **CRYONEX** claims 0.782 zero-shot with a causal Transformer on 60M flows. We ran our own zero-shot study. Our best label-free detector averages 0.79, but the metrics and data differ, so **don't compare the numbers directly**. Our edge is that we show where it fails.
- **foresight** is the most honest competitor and found similar negative results; we added the "world-model surprise" idea from it.
- **cyberpulse / sentinel-net** use GRU/LSTM on flow windows; most don't publish shuffle controls or held-out-day results.
- **Ideas we could still add:** MITRE D3FEND response guidance (map each detected technique to recommended defences), and a React/FastAPI front end.

---

## 13. Suggested 2-minute video script

1. **(0:00–0:15) Problem.** "Attacks are processes, not single packets. Netrikan forecasts, per host, whether an attack is about to start, and explains why."
2. **(0:15–0:40) DAPT page.**
   - Day 2019-07-16, host 192.168.3.29: risk rises before the reconnaissance bands.
   - Open "Explain a prediction": SHAP drivers plus the world-model forward simulation.
3. **(0:40–1:05) ZeekData24 page.**
   - Techniques recognized as ATT&CK IDs.
   - The attacker forecast line predicts the next brute-force burst.
   - "Shuffling time costs 28%, so it is genuinely using the sequence."
4. **(1:05–1:30) Zero-shot page.** "On attacks from labs it never saw, classifiers fail at 0.55; learning normal behaviour transfers at 0.79. So Netrikan uses anomaly scoring for the unknown."
5. **(1:30–1:50) Benchmark tab.** "Every model is compared with logistic regression and a shuffle control, and we report where our world model does not win."
6. **(1:50–2:00) Close.** "Offline, explainable, honestly evaluated. Netrikan."

## 14. Suggested 5 slides
1. **Problem and approach:** per-host world model, forecast, ATT&CK mapping, explanations.
2. **Architecture:** the pipeline diagram from section 5.
3. **Data and discipline:** 4 datasets, audit findings, leave-one-out splits, shuffle control, pre-registered hypotheses.
4. **Results:** DAPT forecasting (2.6x), Z24 forecasting (3.3x, −28% shuffled), the zero-shot table (0.55 vs 0.79).
5. **Limits and next steps:** world model ties the baselines, stage prediction doesn't generalise across days, and more concurrent multi-stage data is needed. Next: D3FEND guidance, the full CIC-2018 data.

---

## 15. File map

| Path | What it is |
|---|---|
| `problem-statement.md` | The SIH brief |
| `docs/DATA_AUDIT.md` | Dataset audit (defects, sizes, overlaps) |
| `src/netrikan/dapt.py` | CICFlowMeter loader and cleaning |
| `src/netrikan/zeek.py` | ZeekData24 loader (TCP flags derived from Zeek history strings) |
| `src/netrikan/extra.py` | CIC-2017 / CTU-13 loaders, common feature set |
| `src/netrikan/features.py` | Per-host, per-minute state vectors, targets, history windows |
| `src/netrikan/models.py` | DAPT ladder + GRU world model + stage model |
| `src/netrikan/campaign.py` | ZeekData24 recognizer, forecaster ladder, GRU world model |
| `src/netrikan/surprise.py` | GRU next-minute surprise detector |
| `src/netrikan/metrics.py` | All metrics, alarm budgets, lead time, bootstrap |
| `src/netrikan/infer.py` | Inference helpers shared by the app and CLI |
| `scripts/train_demo.py`, `train_z24.py`, `zero_shot.py` | Training + evaluation runs |
| `scripts/infer.py` | CLI scoring |
| `app/demo_app.py` | App entry point with dataset chooser |
| `app/views/*.py`, `app/newdata.py` | The five app pages |
| `results/registry.csv` | Every experiment's hypothesis, prediction, outcome |
| `results/*/metrics.json` | All reported numbers |
| `tests/` | 15 leakage and correctness tests |
