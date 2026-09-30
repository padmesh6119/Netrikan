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
5. Shows it all in a React console (FastAPI backend) with a network heatmap, click-to-explain risk timelines, a benchmark page and an incident-response dashboard that maps each alert to MITRE D3FEND countermeasures.

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
FastAPI (api/) + React console (web/) + CLI (scripts/infer.py)
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

## 8. The app, element by element (what each part shows and where it comes from)

Start with `make demo` (after `make web-install` once), then open http://localhost:8000. The React app (`web/`) never computes model output itself: every number, chart and list comes from the FastAPI backend (`api/`), which runs the same `src/netrikan` code and trained models used for training and benchmarking, or reads the evaluation files in `results/*/metrics.json`. Nothing leaves the machine.

### 8.0 How the app is wired

| Piece | What it does | Where |
|---|---|---|
| Top bar | Netrikan mark, two tabs (**Overview** = `/`, **Dashboard** = `/dashboard`), a theme selector (System / Light / Dark, saved in the browser) and a link to the API docs (`/api/docs`, auto-generated by FastAPI). | `web/src/components/Shell.tsx` |
| Dashboard section bar | Grouped like the Dashboard home, with a thin divider between groups: Home, then the forecast datasets (DAPT2020, ZeekData24, CIC-IDS2017, CTU-13), then Response, then Zero-shot. The active section is underlined. | `Shell.tsx` (`DashboardLayout`) |
| URL state | Every control (day, model, threshold, host, selected minute, tab, incident) is a query parameter, so any view can be bookmarked or shared. Old addresses such as `/response?ds=z24` redirect to `/dashboard/response?ds=z24`. | `web/src/lib/url.ts`, `web/src/main.tsx` |
| Data fetching | One `GET` per view through React Query; results are cached in the browser for the session. A thin bar at the top of the page shows while a view refetches. | `web/src/lib/api.ts` |
| Loading / error states | Grey skeleton blocks with a line saying what the server is computing; a red box with the server's error message if a request fails (for example, models not trained yet). | `web/src/components/ui.tsx` |
| Server caches | Each heavy computation (bucketing a day of flows, scoring a ZeekData24 week, building a lab's host-minutes) runs once and is kept in memory; concurrent requests for the same thing wait for the first instead of recomputing. At startup the server pre-computes the five DAPT days, ZeekData24 week 2024-03-03, CIC-IDS2017 Friday and CTU-13 (about 20 s), so first clicks are fast. | `api/common.py` (`cached`), `api/main.py` (`_warm`) |
| JSON | Numpy values are converted server-side; missing values (NaN) arrive as `null` and show as `-`. Times are the captures' own wall-clock minutes, shown as-is (no timezone shift). | `api/common.py` (`clean`, `bucket_ms`), `web/src/lib/format.ts` |

Charts are hand-built SVG (`web/src/components/charts/`). Common conventions: blue line = the model's score; red dots = alerts (score at or above the threshold); dashed horizontal line = alert threshold; shaded vertical bands = true attack minutes (ground truth, only drawn when the data has labels, never used to score); hovering shows the exact values; where it says so, clicking a point selects that minute.

---

### 8.1 Overview tab (`/`, `web/src/pages/Landing.tsx`)

A one-page summary for someone new to the project. Every figure on it is fetched live.

| Element | What it shows | Backend |
|---|---|---|
| Headline, subtext, "See a live forecast" button | Opens the DAPT2020 page on host 192.168.3.29, 16 Jul, Investigate tab. "Read the evidence" opens the Zero-shot page. | none (links) |
| Forecast chart (hero) | P(attack in the next 5 min) for host 192.168.3.29 on 16 Jul 2019, every minute, from the LightGBM + lags model trained on the other four days. Blue bands = true reconnaissance minutes. Red dots = minutes above the default threshold. | `GET /api/dapt/scores?source=2019-07-16` → `api/dapt.py: scores()` → `infer.analyse()` with `models/demo/fold_2019-07-16.joblib`; threshold from `GET /api/dapt/meta` |
| "One code path, from flow to action" | The six pipeline stages (ingest, model each host, learn dynamics, forecast, explain, respond). Descriptive text only. | none |
| Big "2.6x" tile | DAPT2020 pooled leave-one-day-out PR-AUC of LightGBM + lags divided by the base rate (the PR-AUC of a random scorer). The two bars show both values. | `GET /api/dapt/metrics` → `results/demo/metrics.json` (`pooled.lag.pr_auc`, `pooled.lr.prevalence`) |
| "3.3x" tile | ZeekData24 next-burst forecast, macro PR-AUC of the history GBDT over the base rate; the percentage is how much skill is lost when the history order is shuffled. | `GET /api/z24/metrics` → `results/z24/metrics.json` (`forecast.macro.sched`, `sched_shuf`) |
| "Attacks from labs it never saw" tile | Mean ROC-AUC over the 13 unseen attack families for the supervised GBDT and for IsolationForest, placed on a 0 to 1 line with 0.5 (chance) marked. | `GET /api/zeroshot` → `results/zero_shot/metrics.json` |
| "Where it stops" band | Stage-forecast accuracy on held-out days vs the majority-class baseline, and the fact that the world model does not beat the trees. | `GET /api/dapt/metrics` (`stage.accuracy`, `stage.majority_baseline_accuracy`) |
| Incident preview | The highest-scoring incident on DAPT2020, 17 Jul (threshold 0.48): priority, host, peak risk, stage (marked as a hypothesis), the first four D3FEND countermeasures and the first containment rules. | `GET /api/response/incidents` and `GET /api/response/incident` → `api/response.py` (see 8.5) |

---

### 8.2 Dashboard home (`/dashboard`, `web/src/pages/DashboardHome.tsx`)

An index of the views, grouped Forecast (DAPT2020, ZeekData24, CIC-IDS2017, CTU-13), Respond (Incident response) and Evidence (Zero-shot transfer). Each row links to its view and shows one live number:

| Row | Number | Backend |
|---|---|---|
| DAPT2020 | PR-AUC over base rate (as on the Overview tile) | `/api/dapt/metrics` |
| ZeekData24 | Next-burst PR-AUC over base rate | `/api/z24/metrics` |
| CIC-IDS2017 | Share of attack minutes caught on the Friday slice by IsolationForest at 2 alerts per host-hour | `/api/corpus/cic17?slice=fri&det=iforest&budget=2` |
| CTU-13 | Same, for CTU-13 (it is 0%: botnet traffic stays near chance for every detector) | `/api/corpus/ctu13?det=iforest&budget=2` |
| Incident response | Number of P1 incidents on DAPT2020, 17 Jul | `/api/response/incidents?dataset=dapt&source=2019-07-17` |
| Zero-shot transfer | Mean ROC-AUC of IsolationForest on unseen families | `/api/zeroshot` |

The footnote under the list (scope, stage accuracy) comes from `/api/dapt/metrics`.

---

### 8.3 DAPT2020 (`/dashboard/dapt`, `web/src/pages/Dapt.tsx`, backend `api/dapt.py`)

**Task:** for every monitored host and minute, P(attack traffic starts in the next K = 5 minutes).

**Toolbar**

| Control | Effect | Backend |
|---|---|---|
| Traffic source | A capture day (label shows the number of attack onsets that day) or an uploaded CSV. A day is scored by `models/demo/fold_<day>.joblib`, trained on the other four days, so nothing shown is in-sample. An upload is scored by `models/demo/all.joblib`. | `GET /api/dapt/meta` (day list, onsets per day); `run(source)` loads that day's flows and calls `infer.analyse()` |
| Upload CSV | Sends a CICFlowMeter-style CSV. The server parses it with `dapt.read_flows` + `normalise`, scores it and adds it to the source list (last four uploads are kept in memory). | `POST /api/dapt/upload` |
| Risk model | LightGBM + lags (the primary model, marked "best", chosen by pooled PR-AUC) or the World model (GRU rollout + GBDT head). Switching resets the threshold to that model's default. | both scores come from `scores()`; default thresholds from `meta()` |
| Alert threshold | Minutes at or above it are alerts. Default = the threshold that gave 1 false alarm per host-hour on held-out benign minutes; "reset" restores it. | `default_threshold()` reads `alert_threshold_1_per_host_hour` / `pooled.<model>.thr@1.0/h` |
| Respond to alerts | Opens the Response page on the same day, model and threshold. | link |

**Headline tiles** (`GET /api/dapt/summary?source&model&thr` → `summary()`)

| Tile | Meaning |
|---|---|
| Flows analysed | Flows in the capture (after de-duplication); hint = monitored (private-address) hosts and host-minutes. |
| Alert minutes | Host-minutes with risk ≥ threshold. |
| False alarms / host-hour | Alerts on minutes that are benign now and have no attack in the next 5 minutes, per hour of such minutes. |
| Onsets warned ≥1 min ahead | Attack onsets (first attack minute after at least 5 quiet minutes, `metrics.onset_episodes`) with at least one alert in the 5 minutes before (`metrics.lead_times`). |
| Median warning lead | Median of those lead times; at most 5 min by construction. |
| Ground truth | Shown instead of the last three when an uploaded file has no labels. |

**Network tab**

| Element | What it shows | Backend |
|---|---|---|
| Network risk heatmap | One row per host, one cell per minute, colour = forecast risk (stronger blue = higher; scale top right). Coloured strip under each row = true attack minutes by stage. Clicking a row opens that host in Investigate. | `GET /api/dapt/scores` (all rows: host, minute, `lag`, `world`, `attack`, `stage`) |
| Hosts table | Peak risk, alert minutes and true attack minutes per host, sortable; click to investigate. | computed in the browser from `/api/dapt/scores` |
| Attack onsets table | Each onset: time, host, stage, and "N min ahead" or "missed". Clicking opens the minute before the onset. | `summary().onset_list` |

**Investigate tab**

| Element | What it shows | Backend |
|---|---|---|
| Host selector + risk timeline | The host's risk every minute, threshold line, alert dots, stage bands. Clicking any minute selects it for the panels below. | `/api/dapt/scores` |
| Highest-risk minutes | The host's 25 highest-risk minutes; a red dot marks minutes that are already under attack. | computed in the browser |
| Forecast card | The selected minute's risk, what actually happened (attack already in progress / attack started within 5 min / no attack followed / horizon past the end of the capture), and the stage model's probabilities for the four stages, with its held-out accuracy printed underneath. | `GET /api/dapt/explain?source&i&model` → `explain()`: `tab.attack`, `tab.fut`, `tab.valid`, `pr["stage"]`, `metrics.json:stage` |
| Why this risk | The 8 largest TreeSHAP contributions for that minute, in log-odds; bars right raise the risk, left lower it. Feature names are human-readable versions of the model inputs (e.g. "outbound flows (10-min max)"). | `infer.top_drivers()` (LightGBM `pred_contrib`) |
| World-model forward simulation | Four small charts (inbound flows, inbound distinct ports, inbound SYN flags, outbound flows): the GRU's own 5-step rollout from that minute (solid) against what was actually observed (dashed), in original units. | `infer.rollout_counts()` |
| Flows around this minute | Raw flows touching the host from that minute to 6 minutes later (first 200). | `infer.flows_in_window()` |

**Benchmark & evidence tab** (all from `GET /api/dapt/metrics` = `results/demo/metrics.json`, written by `make demo-train`)

| Element | What it shows |
|---|---|
| Findings | Five statements computed from the metrics: ranking works but thresholds do not transfer; world model vs lags (with the lag model's 90% bootstrap CI); the history-shuffle control; stage forecasting at chance across days; data scope. |
| Leave-one-day-out table | For each model (logistic regression, LightGBM current minute, LightGBM + lags, world model, and the two shuffled-history controls): PR-AUC with 90% CI, ROC-AUC, recall, precision, onsets warned and median lead at 1 false alarm per host-hour, Brier score and ECE (calibration). |
| Within-day ranking | ROC-AUC per held-out day for each model. |
| Stage forecast accuracy | Accuracy vs majority-class baseline, leave-one-day-out and within-day. |
| Secondary blocked split | Same table for the easier within-day split; never used to pick a model. |

**How it works tab:** static description of the pipeline and the DAPT label → ATT&CK tactic mapping; step counts (bucket size, history length, horizon) come from `/api/dapt/meta`.

---

### 8.4 ZeekData24 (`/dashboard/zeek`, `web/src/pages/Zeek.tsx`, backend `api/z24.py`)

**Tasks:** (A) which of five ATT&CK techniques a host-minute contains; (B) for each attacker, P(its next burst of a technique starts within 5 minutes).

**Toolbar**

| Control | Effect | Backend |
|---|---|---|
| Capture week | Five attack weeks (one scripted campaign replayed) and two benign weeks. Each is scored by a model trained without it (`models/z24/fold_<week>.joblib`; benign weeks use the fold paired with them). | `GET /api/z24/meta`; `analyse(week, thr)` builds the per-host-minute grid, runs the recognizer, picks attackers from the *predicted* flags and runs every forecaster |
| Recognizer threshold | Probability above which a minute is flagged with a technique (0.5 in all reported results). | re-runs `analyse()` |
| Forecast model | History GBDT (all-technique history features), World model (GRU rollout + head) or the renewal baseline (time since last burst only). | selects `pr[rung]` |
| Alert budget | 0.5, 1 or 2 false alarms per attacker-hour; picks the matching held-out threshold. | `metrics.json: forecast.per_technique.<model>.<technique>.thr@<budget>/h` |
| Respond to alerts | Opens the Response page on the same week, model and budget. | link |

**Headline tiles** (`GET /api/z24/week?week&thr` → `week()`): flows in the week; host-minutes with traffic; minutes flagged with any technique; attacker sequences found (attack weeks) or false alarms per host-hour (benign weeks, where every flag is false); week type.

**Recognise techniques tab**

| Element | What it shows | Backend |
|---|---|---|
| Recognition table | Per technique (ID, name, ATT&CK tactic): flagged minutes, and on attack weeks true minutes, recall and precision this week. | `week().techniques` |
| Host activity lanes | For the chosen host, one row per technique: upper lane = minutes recognised by the model, lower lane = ground truth. | `GET /api/z24/recognize/host` |
| Most confident minutes | That host's 20 flagged minutes with the highest technique probability. | same |
| Technique probabilities | The recognizer's probability for each of the five techniques at the chosen minute. | `GET /api/z24/recognize/explain` |
| Why <technique> | TreeSHAP for the most likely technique's model at that minute. | same (`pred_contrib`) |

**Forecast attacker bursts tab**

| Element | What it shows | Backend |
|---|---|---|
| Attacker / Technique / Day | Which attacker sequence, technique and day to plot. | `GET /api/z24/forecast` |
| Tiles | Across all attackers this week, for this technique: bursts (onsets), share warned at least 1 minute ahead, false alarms per attacker-hour, median lead when warned, and the threshold in use. | `forecast().stats` via `metrics.onset_episodes` / `lead_times` |
| Burst forecast chart | Blue = chosen model's P(burst within 5 min); grey = renewal baseline; red vertical lines = actual bursts; red dots = alerts. Click a minute to explain it. | `forecast().series`, `bursts`, `alerts` |
| Highest-risk minutes / Why this forecast | TreeSHAP of the History GBDT at the chosen minute. "since" = minutes since the last recognised burst of that technique; "n60"/"n150" = flagged minutes in the last 60/150 minutes. Shown only for the History GBDT. | `GET /api/z24/forecast/explain` |

**Campaign structure tab** (`GET /api/z24/structure`): for the four techniques other than exfiltration, histograms of the minute-of-hour at which each hourly burst starts and of the gaps between consecutive bursts (both computed from this week's flows); a heat-table of P(technique B starts within 5 min after a burst of technique A) for those four, over all attack weeks (about 0.08 would be random); the attacker list across all weeks. The last two come from `metrics.json: structure`.

**Benchmark & evidence tab** (`GET /api/z24/metrics`): written findings; Task A per-technique recognizer table (PR-AUC, recall, precision, benign alarms per host-hour, firing on other techniques' minutes); Task B forecast ladder (renewal, current behaviour only, own-technique history, all-technique history, shuffled controls, world model, true-flag upper bound, clock ablation) with PR-AUC, ROC-AUC, recall, precision, onsets warned and Brier; PR-AUC per technique.

---

### 8.5 Response (`/dashboard/response`, `web/src/pages/Response.tsx`, backend `api/response.py`)

**What an incident is:** consecutive alert minutes on one host (on ZeekData24: one host and one technique), with gaps of up to 5 minutes merged. The same alerts as on the forecast pages; the Response page only groups and ranks them.

**Toolbar**

| Control | Effect |
|---|---|
| Telemetry | DAPT2020 or ZeekData24. |
| Day / model / threshold (DAPT) or week / forecast model / alert budget (ZeekData24) | Same meaning as on the forecast pages; the incidents are rebuilt from those alerts. |
| Show ground truth | Reveals, per incident, whether a real attack followed and how early it was warned. Off by default so triage is done as an analyst would, without labels. |

**Headline tiles** (`GET /api/response/incidents` → `incidents()`, plus your triage state)

| Tile | Meaning |
|---|---|
| Open incidents | Incidents not marked resolved; hint = all incidents raised. |
| Open P1 | Highest-priority incidents not yet resolved. |
| Contained / resolved | Your triage counts. Saved in this browser only (localStorage), not on the server. |
| Followed by a real attack | Share of incidents with an attack inside the alert run or within 5 minutes after it (ground truth). |
| Median warning (5-min horizon) | For incidents that ended in an attack: minutes between the earliest alert in the 5 minutes before the attack and the attack itself (`horizon_lead`, same definition as `metrics.lead_times`). Never above 5. Attacks with no alert never become incidents, so misses are counted on the forecast pages, not here. |

**Priority** (`priority()`): DAPT2020 uses only the risk (peak ÷ threshold) and the run length, because the stage model is at chance on held-out days. ZeekData24 weighs technique severity (exfiltration, exploitation and valid accounts highest, brute force next, scanning lowest) and forecast strength. Score ≥ 0.7 → P1, ≥ 0.4 → P2, else P3.

**Queue:** incidents sorted by priority then score. Each row: priority, host, technique and/or tactic, time window, peak risk, alert-run minutes, your status (New / Investigating / Contained / Resolved) and, with ground truth on, whether an attack followed. Filters: text (host, tactic, technique), priority, status.

**Selected incident** (`GET /api/response/incident?...&id` → `incident()`)

| Element | What it shows | Backend |
|---|---|---|
| Header | Priority, status, host, alerting window, number of minutes in the alert run, peak risk and how many times the threshold it reached. With ground truth on: whether an attack followed, its true stage (DAPT) and "warned N min ahead" or "no alert in the 5 min before it". Buttons: Copy report (Markdown incident report to the clipboard), Acknowledge → Mark contained → Resolve, Reopen. | incident record; status in localStorage |
| Threat → countermeasures | Left: the MITRE ATT&CK tactic (DAPT: the stage model's prediction and its probability; ZeekData24: technique ID and name from the recognizer, plus other techniques seen from this host in the last hour) and a confidence line. Right: the playbook. "First response (stage-agnostic)" always applies; the stage- or technique-specific group is marked "hypothesis" on DAPT2020. Each card is a MITRE D3FEND technique with its D3FEND tactic (Detect, Isolate, Deceive, Evict, Harden), a concrete action filled in with this host's real peers, ports and bytes, and why it applies. Cards are a checklist; ticking one moves a New incident to Investigating. | `FIRST_RESPONSE`, `BY_TACTIC`, `fill()`; peers/ports from `targets()` |
| Low-confidence note | DAPT only: the stage model's held-out accuracy vs majority baseline. ZeekData24 shows the recognizer's PR-AUC instead, with the caveat that it is in-corpus. | `metrics.json` |
| Risk around the incident | The model's score for this host (DAPT) or host and technique (ZeekData24) from 30 minutes before to 30 minutes after, with the incident window shaded; with ground truth on, true attack minutes in red. | `incident().series` |
| Why the model alerted | TreeSHAP drivers at the incident's peak minute (DAPT: LightGBM + lags or world-model head; ZeekData24: History GBDT). | `infer.top_drivers()` / `z24.forecast_explain()` |
| Who it talked to | From 5 minutes before the first alert to 5 minutes after the incident ends: top outbound and inbound peers and destination ports by flow count, total bytes out and in. | `targets()` over the flows |
| Containment rules | Example `iptables` rules: quarantine the host except an admin subnet you set; on lateral movement, block the internal peers it reached; on exfiltration, block the external destinations that received the most bytes; on scanning, brute force, exploitation or valid-account use, reject the host on the targeted ports. Copy button included. Templates for review: Netrikan never applies anything. | `rules()` |

The D3FEND mapping is hand-written in `api/response.py` (technique names, no IDs) and should be checked against d3fend.mitre.org before being presented as authoritative.

---

### 8.6 CIC-IDS2017 and CTU-13 (`/dashboard/labs/cic17`, `/dashboard/labs/ctu13`, `web/src/pages/Corpus.tsx`, backend `api/corpus.py`)

These two labs are scored by detectors trained on the other three corpora (`models/zero_shot/holdout_<lab>.joblib`, from `make zero-shot`); their labels are only used to draw attack minutes and to compute the scores shown, never to pick alerts. Each flow file is turned into per-host-minute rows with the 22 features all four corpora share (`extra.host_minutes`).

| Element | What it shows | Backend |
|---|---|---|
| Slice (CIC-IDS2017 only) | Friday (port scan + DDoS), Wednesday (Heartbleed) or both. | `prepare(name, slice)` |
| Detector | IsolationForest (trained on benign minutes only), world-model surprise (GRU next-minute prediction error), supervised LightGBM (trained on the other labs' attacks), flow-volume heuristic, fan-out heuristic (peers + ports). | all five scores computed in `prepare()` |
| Alert budget | Alerts = the top-scoring host-minutes, this many per host-hour of traffic. No labels involved. | `alarms()` |
| Tiles | Flows; hosts and host-minutes; attack host-minutes; share of attack minutes caught (and number of alerts); false alerts per benign host-hour. | `GET /api/corpus/<lab>` |
| Host timeline | The chosen detector's score as a percentile within the slice (so detectors are comparable), alert dots, the alert cut-off, and true attack minutes shaded by family. Gaps = minutes with no traffic from that host. Attacked hosts are listed first in the selector. | `GET /api/corpus/<lab>/host` |
| Detector comparison | ROC-AUC of every detector on this slice, per attack family, against that family's positives and the lab's benign minutes; 0.5 tick = chance; best per family highlighted. | `corpus().comparison` (live `roc_auc_score`) |
| Network heatmap | The 14 highest-scoring hosts over time (percentile), with true attack minutes marked. | `corpus().heatmap` |

---

### 8.7 Zero-shot transfer (`/dashboard/zero-shot`, `web/src/pages/ZeroShot.tsx`, backend `GET /api/zeroshot`)

All from `results/zero_shot/metrics.json` (leave-one-corpus-out across DAPT2020, ZeekData24, CIC-IDS2017 and CTU-13; written by `make zero-shot`).

| Element | What it shows |
|---|---|
| Five tiles | Mean ROC-AUC over the 13 unseen attack families for each detector; the best is marked, any below 0.5 is red. |
| Findings | Supervised transfer fails (and how many families it scores below chance); benign-only anomaly detection transfers best; world-model surprise vs IsolationForest; caveats on sample sizes; the design consequence. |
| ROC-AUC table | One row per held-out corpus and unseen family: positive minutes, each detector's ROC-AUC as a small bar with the 0.5 chance tick (best highlighted), and the supervised model's 90% bootstrap interval. |
| Protocol | The evaluation protocol text and each corpus's host-minutes and attack host-minutes. |

---

**Also available:** a CLI, `make infer CSV=file.csv`, which writes per-host-minute risk, stage and the top 3 drivers; and the raw API at `/api/docs`, where every endpoint above can be called directly.

---

## 9. How to run everything

```
make setup        # Python 3.12 venv, pinned packages, macOS LightGBM fix
make test         # 15 leakage/correctness tests
make demo-train   # DAPT2020 models (~2 min)
make z24-train    # ZeekData24 models (~8 min)
make zero-shot    # cross-lab experiment (~2 min)
make web-install  # once, needs Node 20+
make demo         # the console + API on http://localhost:8000, fully offline
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
   - In "Investigate a host", click a high-risk minute: SHAP drivers plus the world-model forward simulation.
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
| `src/netrikan/infer.py` | Inference helpers shared by the API and CLI |
| `scripts/train_demo.py`, `train_z24.py`, `zero_shot.py` | Training + evaluation runs |
| `scripts/infer.py` | CLI scoring |
| `api/main.py`, `api/dapt.py`, `api/z24.py`, `api/corpus.py` | FastAPI endpoints over the trained models |
| `api/response.py` | Incidents, ATT&CK → D3FEND playbook, containment templates |
| `web/src/pages/*.tsx` | The console pages (React + TypeScript) |
| `web/src/components/charts/*.tsx` | SVG charts: timeline, heatmap, SHAP bars, rollout, histograms |
| `results/registry.csv` | Every experiment's hypothesis, prediction, outcome |
| `results/*/metrics.json` | All reported numbers |
| `tests/` | 15 leakage and correctness tests |
