# Netrikan — Build Plan
## Network Attack Forecasting

---

## What We Are Building

A system that watches network traffic and predicts — before it completes — whether an attack is in progress, how far along the kill chain the attacker is, and exactly which traffic patterns are causing the alert.

Core innovation: instead of classifying individual packets, we model the *trajectory* of traffic over time using an LSTM that learns P(S_t+1 | S_t) — the probability of the next network state given the current one. This is a world model, not a classifier.

---

## Project Structure

```
netrikan/
├── data/
│   └── cic-ids2018/          # Downloaded CSVs (raw)
├── src/
│   ├── pipeline.py           # Feature extraction + windowing
│   ├── model.py              # LSTM world model (PyTorch)
│   ├── train.py              # Training script
│   ├── predict.py            # Inference + K-step rollout
│   ├── explain.py            # SHAP attribution
│   └── dashboard.py          # Streamlit UI
├── models/
│   └── lstm_world_model.pt   # Saved checkpoint
├── notebooks/
│   └── eda.ipynb             # Exploratory analysis
└── requirements.txt
```

---

## Phase 1 — Data & EDA

**Goal:** Understand the raw CSVs before touching model code.

**Files downloaded:**
- `Friday-16-02-2018` — Infiltration attacks (most relevant to our PS)
- `Wednesday-14-02-2018` — Brute force, XSS, SQL Injection
- `Thursday-15-02-2018` — DoS, DDoS
- `Friday-23-02-2018` — Botnet

**Tasks:**
1. Run EDA script — confirm column names, check label values, count nulls
2. Identify the `Label` column string values → map to MITRE ATT&CK stages
3. Check class imbalance (benign vs attack ratio)
4. Confirm no IP/timestamp leakage columns that would make the model cheat

---

## Phase 2 — Feature Pipeline (`pipeline.py`)

**Goal:** Raw CSV → clean time-windowed sequences ready for LSTM input.

**Feature selection (~20 features):**

Flow-level:
- `Dst Port`, `Protocol`
- `Flow Duration`
- `Tot Fwd Pkts`, `Tot Bwd Pkts`
- `TotLen Fwd Pkts`, `TotLen Bwd Pkts`
- `Fwd Pkt Len Max/Min/Mean`, `Bwd Pkt Len Max/Min/Mean`
- `Flow Bytes/s`, `Flow Pkts/s`
- `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`
- `Fwd IAT Mean`, `Bwd IAT Mean`
- `SYN Flag Cnt`, `ACK Flag Cnt`, `PSH Flag Cnt`, `RST Flag Cnt`
- `Pkt Len Variance`
- `Active Mean`, `Idle Mean`

Drop: source/destination IPs, raw timestamps, any label-derived columns.

**Windowing:**
- Sort by timestamp within each file
- Sliding window: size=10 flows, stride=1
- Each window → one input sample (shape: 10 × 20)
- Label: attack stage of the last flow in the window

**MITRE ATT&CK stage mapping:**
```
Benign           → 0 (No threat)
Brute Force      → 1 (Initial Access)
SQL Injection    → 1 (Initial Access)
XSS              → 1 (Initial Access)
DoS              → 2 (Impact/Disruption)
DDoS             → 2 (Impact/Disruption)
Infiltration     → 3 (Lateral Movement / C2)
Botnet           → 4 (C2 / Exfiltration)
```

**Output:** `X.npy` (N × 10 × 20), `y.npy` (N,) — stage labels

---

## Phase 3 — LSTM World Model (`model.py` + `train.py`)

**Architecture:**

```
Input: (batch, seq_len=10, features=20)
       ↓
LSTM(input=20, hidden=128, layers=2, dropout=0.3)
       ↓
Last hidden state (batch, 128)
       ↓
FC(128 → 64) + ReLU
       ↓
┌──────────────────┐
│ Stage head       │  FC(64 → 5) + Softmax  → kill-chain stage (0–4)
│ Breach head      │  FC(64 → 1) + Sigmoid  → breach probability (0–1)
└──────────────────┘
```

**Training:**
- Loss: CrossEntropy (stage) + BCELoss (breach) weighted sum
- Optimizer: Adam, lr=1e-3
- Epochs: 30, early stopping on val loss
- Train/val split: 80/20 within CIC-IDS-2018
- Test: CTU-13 (completely unseen — cross-dataset generalisation)

**Baseline for comparison:**
- Logistic regression on identical features (no temporal context)
- We report: how many stages earlier our model flags vs baseline

---

## Phase 4 — Inference + K-step Rollout (`predict.py`)

**At inference time:**
1. Take last 10 flow windows from live traffic
2. Feed through LSTM → get stage + breach probability
3. Roll K=5 steps forward: use predicted next state as input, repeat
4. Output: breach probability timeline (rising curve if attack progressing)

**Input accepted:**
- CSV file (pre-extracted NetFlow features)
- Future: live PCAP via CICFlowMeter → CSV → inference

---

## Phase 5 — SHAP Explainability (`explain.py`)

- Use `shap.DeepExplainer` on the PyTorch LSTM
- Per-alert output: which of the 20 features drove the prediction most
- Output: SHAP waterfall chart (feature name → contribution bar)
- Every alert shown on dashboard has a SHAP breakdown alongside it

---

## Phase 6 — Dashboard (`dashboard.py`)

**Streamlit UI — fully offline:**

Page layout:
1. Upload CSV → parse → run inference → show results
2. Breach probability timeline (line chart, rises as attack progresses)
3. Current kill-chain stage (coloured badge: green → yellow → orange → red)
4. Top flagged flows table (timestamp, stage, breach score)
5. SHAP waterfall for selected alert

No internet. No cloud. No API keys. Runs on localhost.

---

## Phase 7 — Validation

- Train on CIC-IDS-2018 (Wed + Thu + Fri-16 + Fri-23)
- Test on CTU-13 botnet dataset (never seen during training)
- Report: detection stage (ours vs logistic regression baseline)
- Report: false positive rate on benign-only traffic
- SHAP attribution verified against known attack signatures

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| ML | PyTorch |
| Feature engineering | Pandas, NumPy |
| Explainability | SHAP |
| ATT&CK mapping | mitreattack-python |
| Dashboard | Streamlit |
| Packaging | Docker |

---

## Dependencies

```
torch
pandas
numpy
scikit-learn
shap
streamlit
mitreattack-python
matplotlib
seaborn
```

---

## Build Order

1. EDA → confirm columns and labels
2. `pipeline.py` → produce X.npy, y.npy
3. `model.py` → define architecture
4. `train.py` → train, save checkpoint
5. `predict.py` → inference + rollout
6. `explain.py` → SHAP per alert
7. `dashboard.py` → full UI
8. Docker → package everything
