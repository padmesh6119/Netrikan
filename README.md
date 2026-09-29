> **See [STATUS.md](STATUS.md) for the authoritative current state.** This document
> was written earlier and is stale in places; where the two disagree, STATUS.md is right.

# Netrikan — நெற்றிக்கண்

**AI-based Network Attack Forecasting from Network Traffic Data**

Netrikan learns how network traffic evolves during an intrusion and forecasts
**which attack stage comes next**, before the damage is observable. Conventional
IDS asks *"is this flow malicious?"*. Netrikan asks *"where is this heading?"*.

```
Predicted over the next 15 minutes
  Lateral movement, then command & control
  confidence 59% · driven by SMB port scan · port 445 probing, 6 SYN/flow, 1 RST
  flagged before the damage stage became observable in the traffic record
```

---

## Quick start

```bash
git clone <repo-url> && cd netrikan
pip install -r requirements.txt
sudo apt install tshark          # only needed for PCAP input
./run_app.sh                     # -> http://localhost:8700
```

The dashboard accepts a `.pcap`, `.pcapng`, `.cap`, or NetFlow `.csv`, or runs a
built-in demo scenario. **Fully offline — no cloud APIs, no network calls.**

`run_app.sh` stops any previous instance, verifies every import, and prints a
repair command if the environment is broken.

---

## Results

Deployed checkpoint: **`models/cic_v2_w30.pt`** (window 30, 24 features), trained
on 6.16M windows across all 7 time-sorted CIC-IDS-2018 capture days. Every number
below comes from a committed JSON under `models/` — see `STATUS.md` for the full
record including the negative results.

### Detection, in-dataset

| Metric | Value | Source |
|---|---|---|
| macro-F1 | **0.837** | `cic_v2_run_summary.json` |
| accuracy | **92.7%** | ″ |
| per-class F1 | Benign 0.950 · InitialAccess 0.968 · DoS 0.979 · Botnet 0.879 · **Infiltration 0.407** | ″ |

**Does the sequence model earn its place?**

| Model | Sees | macro-F1 |
|---|---|---|
| Logistic regression | one flow | 0.420 |
| Logistic regression | flattened window (720 features) | 0.692 |
| **LSTM** | window as a sequence | **0.837** |

Sequence information is worth **+0.272**; the LSTM adds **+0.145** on top of the
flattened baseline.

### Forecasting — the actual objective

A persistence oracle ("next label = current label") scores 0.947 macro-F1 on all
windows and beats the model, because CIC attacks come in long contiguous runs.
That is label autocorrelation, not skill. Two metrics remove it:

| | model | persistence |
|---|---|---|
| macro-F1 on **transition windows** (n=31,781) | **0.343** | **0.000** |
| onset AUC (does a new stage begin within k?) k=5 | **0.901** | 0.500 |

Persistence scores 0.000 and 0.500 respectively **by construction** — it never
predicts a change. Onset PR-AUC is 0.254 / 0.494 / 0.609 / 0.624 at k=1/5/15/30
against no-skill baselines of 0.026 / 0.074 / 0.127 / 0.150.

**Lead time** (15,898 benign→attack onsets, 30 s/window):

| horizon | at 5% FPR | at 1% FPR | at 0.1% FPR |
|---|---|---|---|
| k=30 | **13.5 min**, 66% coverage | 3.5 min, 26% | 2.0 min, 2% |
| k=15 | 8.5 min, 61% coverage | 2.5 min, 27% | 1.5 min, 5% |

### Robustness — measured under harsher protocols than a single split

| protocol | result |
|---|---|
| **Leave-one-day-out** (7 folds) | onset AUC k5 **0.636 ± 0.093**, vs 0.902 on the single split |
| **Shuffled-history control** | onset PR-AUC k5 0.494 → **0.371** (−25%): the model does use temporal order |
| **Held-out family** (Infiltration, Botnet) | **fails** — both below chance, see limitations |
| Cross-dataset (DAPT 2020, unseen network) | attack recall 67.8%, precision 37.8% |

### Engineering

ONNX **1.41 ms**/window (610 windows·s⁻¹), PyTorch 3.01 ms; model **0.893 MB**,
opset 17, parity 1.9e-5. Calibration ECE 0.0088 → 0.0052. World-model rollout
beats persistence (MSE 0.612 vs 0.782) *and* the window mean (0.649) — PASS.

---

## Limitations — read these before quoting anything above

1. **No zero-shot generalization to unseen attack families.** Held out entirely,
   Infiltration scores ROC-AUC 0.29–0.37 and Botnet **0.198–0.332** — *below*
   chance, on 8k–40k real positives. A model that never saw a family reads its
   onsets as more benign than benign. Netrikan gives early warning for families
   represented in training; it does **not** detect novel attacks.
2. **The single-split onset AUC of 0.902 is optimistic.** Across capture days it
   is 0.636 ± 0.093, and one fold lands at 0.445 — below chance. Verified not to
   be undertraining: doubling epochs at a third the LR moved it only to 0.514.
3. **Flat performance across horizons is a single-split artifact.** In-split,
   onset AUC is ~0.89–0.90 for every k. Across days it *declines* with horizon
   (one fold: 0.715 at k=1 → 0.425 at k=30).
4. **Cross-dataset false positives are high.** Per-window FPR is 40.4%, and
   thresholding does not fix it (6.3% recall at 5% FPR). Rolling alerts up to
   (host, hour) cells gives the usable operating point: **25.4% of attack
   host-hours at a 1.85% false-alarm rate per benign host-hour**, against 49.9%
   for the shipped per-window decision.
5. **Infiltration is weak (F1 0.407)** and resists the obvious fix: full
   inverse-frequency rebalancing made it *worse* (0.357), as did every other
   axis. Its traffic genuinely overlaps benign.
6. **Single-seed.** Only the LODO result carries a spread; everything else is one
   run at seed 42.

---

## Evaluation integrity

Sliding windows overlap by `WINDOW-1` rows, so a **random** train/test split puts
near-identical windows on both sides. We measured **100% of validation windows
containing training rows** in our first attempt.

Every number above uses `train_v2.blocked_split()` — contiguous blocks assigned
whole to train or val, with `WINDOW-1` windows purged at every block edge.
Verified 0 overlap in 300,000 samples.

Reproduce the check:

```bash
cd src && python3 -c "
import numpy as np, sys; sys.path.insert(0,'.')
from sklearn.model_selection import train_test_split
from train_v2 import blocked_split
y = np.load('../data/processed/y.npy'); n = len(y)
def leak(tr, va, label):
    m = np.zeros(n, np.int8); m[tr] = 1
    s = va[np.random.default_rng(0).choice(len(va), 50000, replace=False)]
    print(f'{label:8} {sum(1 for i in s if m[max(0,i-9):min(n,i+10)].any())/500:.1f}% leaked')
idx = np.arange(n)
leak(*train_test_split(idx, test_size=.2, random_state=42, stratify=y), 'random')
leak(*blocked_split(y, purge=9), 'blocked')
"
```

---

## Architecture

```
PCAP ──tshark──┐
               ├─> 24 flow features ─> scale ─> windows of 30 ─┐
CSV ───────────┘                                               │
                                                               ▼
                                                    LSTM (2 layers, 128 hidden)
                                                    ├─ stage head   (5 classes)
                                                    ├─ breach head  (probability)
                                                    ├─ state head   (next state → rollout)
                                                    └─ onset head   (P(new stage ≤ k))
                                                               │
                              14 rule detectors ───> fusion ◄──┘
                                                               │
                        K-step rollout  (learned when world_w30.pt is present,
                        Markov DOCTRINE_SHAPE fallback otherwise)
                                                               ▼
                       7-stage trajectory + confidence + lead time + ledger
```

**Where the AI is:** one LSTM (~207K parameters, 862 KB) performs stage
classification, breach scoring, next-state prediction and onset forecasting. The
signal naming is 14 hand-written rules and the output text is string templates.
No language model is used anywhere. This is what lets the system run offline on
CPU with every alert traceable to a specific rule.

**Honest caveats on this diagram:** the state and onset heads exist in
`model.py` but the shipped `base_w30.pt` carries trained weights for neither, so
`analyze()` reports `rollout_source: "markov_fallback"` and `onset: None` until
they are trained. `infer.onset_trained()` and `infer.attention_trained()` report
this rather than letting an untrained head emit noise as a forecast. The 0.30/0.70
fusion weight is unmeasured. See `STATUS.md`.

---

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| **CIC-IDS-2018** | Training | 6 days, 5.5M windows after cleaning, 74.5% benign |
| **DAPT 2020** | Cross-dataset test | 86,691 flows, real 4-day APT campaign, never trained on |

DAPT is public at `https://gitlab.com/asu22/dapt2020` and carries a per-flow
`Stage` column, which is what let us measure real attacker dwell time (0.93
persistence per step) rather than assume it.

---

## Repository layout

```
src/
  model.py            LSTM, 4 heads: stage / breach / state / onset
  pipeline_v2.py      dataset builder, network-wide windows + onset labels
  pipeline_identity.py per-host windows WITH packet-level features
  train_v2.py         trainer  --data --tag [--attention] [--select-on]
  baseline.py         logistic-regression control
  eval_dapt.py        cross-dataset evaluation + onset AUC
  transition_eval.py  scoring on windows where the stage actually changes
  calibration.py      temperature scaling, ECE, reliability diagram
  dapt.py             DAPT 2020 loader
  pcap_ingest.py      PCAP -> flows (tshark preferred; it alone gives packet features)
  infer.py            inference, fusion, lead time
  signals.py          14 rule detectors
  forecast.py         K-step projection, confidence, ETA
  attck_map.py        7 ATT&CK stage definitions + CAPEC
  ledger.py           SHA-256 hash-chained alert ledger
  app.py              Streamlit dashboard

bench/
  sedi.py                   base-rate-independent skill score
  persistence_baseline.py   the baseline that beats us on all-window F1
  model_weight_ablation.py  sweep the model-vs-rules fusion weight
  suricata_comparison.py    external lead time vs Suricata

configs/
  train_v2.yaml                    drives pipeline_v2 and train_v2
  cic_attack_windows.example.json  label spec template for pipeline_identity

models/             checkpoints + metrics JSON for every run
data/
  cic-ids2018/      training CSVs        (download separately)
  dapt2020/         cross-dataset test   (download separately)
  processed/        built tensors
```

Further reading: **`STATUS.md` first** — it is the authoritative current state.
Then `PROJECT.md` (technical handbook) and `AI_HANDOFF.md` (context transfer for
automated agents); both are stale in places and defer to STATUS.md.

---

## Reproducing training

Window size comes from `configs/train_v2.yaml`, so the config reproduces the
shipped checkpoint rather than only describing it.

```bash
# dataset (also writes onset_k*.npy for the hazard head)
python3 src/pipeline_v2.py --out /storage/netrikan-base-w30

# classifier, and the rollout checkpoint the forecast needs
python3 src/train_v2.py --data /storage/netrikan-base-w30 --tag base_w30
python3 src/train_v2.py --data /storage/netrikan-base-w30 --tag world_w30

# evaluation
python3 src/eval_dapt.py       --data /storage/netrikan-base-w30 \
                               --model models/base_w30.pt --tag dapt_base_w30
python3 src/transition_eval.py --data /storage/netrikan-base-w30
python3 src/calibration.py     --data /storage/netrikan-base-w30
python3 src/baseline.py        --data /storage/netrikan-base-w30 --tag baseline_w30
python3 bench/model_weight_ablation.py --data /storage/netrikan-base-w30
```

Training runs ~25 min on an RTX 2050 (4 GB). Inference is CPU-only.

---

## Known limitations

- **`models/world_w30.pt` is not in this repository.** It is the trained
  `state_head` that makes the K-step rollout a learned simulation. Without it
  `forecast()` falls back to the hand-coded `DOCTRINE_SHAPE` matrix and
  `analyze()` reports `rollout_source: "markov_fallback"`. Until it is trained and
  committed, the forecast on a fresh clone is a Markov chain, not a world model.
- **Packet-level features never reach the model.** TTL, TCP window, fragmentation
  and retransmission counts are extracted by `pcap_ingest.py` and consumed only by
  4 of the 14 rules — the LSTM sees 24 flow features. CIC-IDS-2018's ML-ready CSVs
  cannot supply them; `pipeline_identity.py` fixes this but needs the raw PCAPs.
- **No training-data identity** — CIC-IDS-2018 ML-ready CSVs have no source/dest
  IP, so windows are time-local network slices, not per-host sequences.
  `pipeline_identity.py` builds per-host windows but has not been run.
- **Infiltration is the weakest class** (0.751) — rarest at 1.67% of windows and
  designed to resemble normal traffic.
- **Cross-dataset precision is ~38%, FPR is 41.6%** — more false alerts than real
  attack flows on DAPT 2020. Detection recall transfers to unseen networks;
  precision does not. Reported as SEDI (+0.583) because precision is not
  base-rate independent.
- **Persistence beats the LSTM on all-window macro-F1** — CIC-IDS-2018 attacks are
  long contiguous runs, so "repeat last label" scores 0.906. The fix is the onset
  target, not a better classifier; `transition_eval.py` and the onset head are the
  response and neither has been run on the real data yet.
- **70% of every decision comes from hand-written rules.** `MODEL_WEIGHT = 0.30`
  was set by hand, never swept. `bench/model_weight_ablation.py` measures it.
- **Kill-chain transition ordering follows ATT&CK doctrine**, not learned
  probabilities. Stage persistence is measured from DAPT; off-diagonal weights are
  hand-coded. Reconnaissance and Initial Access currently share DAPT's
  Reconnaissance dwell figure — see the note in `forecast.py`.
- **Confidence is uncalibrated** until `calibration.py` is run and
  `models/temperature.json` exists.
- **No DDoS coverage** — the two CIC-IDS-2018 DDoS days were not in the training
  set. Single-source DoS is covered.
- **Lead time is self-referential** in the dashboard — it compares the forecast
  threshold to the model's own damage-stage detection.
  `bench/suricata_comparison.py` replaces it with an external baseline and charges
  the model for the WINDOW flows it must observe first.
- **No test suite.** Self-checks exist in `ledger.py`, `calibration.py` and
  `bench/sedi.py`; there is no `tests/` directory.
- **PCAP feature extraction is our own CICFlowMeter-compatible implementation**,
  so minor definitional differences exist versus the original tool.

---

## License / attribution

Datasets are the property of their
respective publishers (Canadian Institute for Cybersecurity; Arizona State
University). MITRE ATT&CK® is a registered trademark of The MITRE Corporation.
