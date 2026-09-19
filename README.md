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

Measured on a **leak-free** split (see *Evaluation integrity* below).

| Metric | Value |
|---|---|
| macro-F1 (held-out, 1.1M windows) | **0.885** |
| persistence baseline (same data structure) | 0.906 — see note below |
| accuracy | 96.0% |
| cross-dataset attack recall (DAPT 2020, unseen) | **83.9%** |
| cross-dataset attack precision | **42.2%** |
| cross-dataset benign false-positive rate | **41.6%** (26,456 false alerts vs 19,279 true) |
| lateral-movement recall, cross-dataset | **93.1%** (n=2,451, one campaign) |
| logistic-regression baseline, same features | 0.741 |

Per class: Benign 0.976 · InitialAccess 0.969 · DoS 0.890 · Botnet 0.987 ·
Infiltration 0.604.

A longer-window variant (`ports_w30`) reaches **0.925 macro-F1** with Infiltration
at 0.755; it is trained but not yet deployed (requires reprocessing with port features).

### Does the sequence model earn its place?

| Model | Sees | macro-F1 |
|---|---|---|
| Logistic regression | one flow | 0.607 |
| Logistic regression | whole window | 0.742 |
| **LSTM** | whole window as a sequence | **0.885** |

Sequence information is worth +0.135 over the single-flow baseline.

### Horizon note — persistence baseline

**This result is critical context:** a persistence oracle (predict the same label as the
current window) scores **0.906** on the full label sequence — higher than the LSTM at
every horizon. The horizon models (below) were evaluated on a blocked split; the
persistence score is on the full data. The split mismatch means the comparison is
approximate, but a gap of 0.017–0.050 in persistence's favour across all horizons
cannot be explained by split differences alone.

The practical implication: CIC-IDS-2018 attacks come in long contiguous runs, so
the "future" label is nearly always the same as the present one. The horizon curve
measures label autocorrelation as much as forecasting skill. The transition-adjacent
metric (how the model does on windows where the label actually changes) is the
meaningful number; it is not yet computed. Run `bench/persistence_baseline.py` to
reproduce.

| lead (flows) | 0 | 30 | 90 | 180 | 360 |
|---|---|---|---|---|---|
| LSTM macro-F1 | 0.889 | 0.870 | 0.866 | 0.862 | 0.862 |
| **persistence** | **0.906** | **0.919** | **0.914** | **0.912** | **0.909** |

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
               ├─> 24 flow features ─> scale ─> windows of 10 ─┐
CSV ───────────┘                                               │
                                                               ▼
                                                    LSTM (2 layers, 128 hidden)
                                                    ├─ stage head   (5 classes)
                                                    └─ breach head  (probability)
                                                               │
                              14 rule detectors ───> fusion ◄──┘
                                                               │
                                        Markov kill-chain projection
                                                               ▼
                                     stage trajectory + confidence + lead time
```

**Where the AI is:** one LSTM (219,590 parameters, 862 KB) performs stage
classification and breach scoring. The forecast is a Markov chain, the signal
naming is 14 rules, the output text is string templates. No language model is
used anywhere. This is what lets the system run offline on CPU with every alert
traceable to a specific rule.

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
  model.py          LSTM, dual output heads
  pipeline_v2.py    dataset builder   --window N --out DIR [--with-ports]
  train_v2.py       trainer           --data --tag --horizon [--fixed-split]
  baseline.py       logistic-regression control
  eval_dapt.py      cross-dataset evaluation
  dapt.py           DAPT 2020 loader
  pcap_ingest.py    PCAP -> flows via tshark
  infer.py          inference, fusion, lead-time computation
  signals.py        14 rule detectors
  forecast.py       Markov projection, confidence, ETA
  attck_map.py      ATT&CK stage definitions
  app.py            Streamlit dashboard

models/             checkpoints + metrics JSON for every run
data/
  cic-ids2018/      training CSVs
  dapt2020/         cross-dataset test CSVs
  processed/        built tensors (symlink)
```

Further reading: `PROJECT.md` (full technical handbook), `AI_HANDOFF.md`
(context transfer for automated agents).

---

## Reproducing training

```bash
cd src
python3 pipeline_v2.py --window 30 --out /storage/netrikan-base-w30
python3 train_v2.py    --data /storage/netrikan-base-w30 --tag base_w30 --epochs 25
python3 eval_dapt.py   --data /storage/netrikan-base-w30 --model ../models/base_w30.pt \
                       --tag dapt_base_w30
python3 baseline.py    --data /storage/netrikan-base-w30 --tag baseline_w30
```

Training runs ~25 min on an RTX 2050 (4 GB). Inference is CPU-only.

---

## Known limitations

- **Infiltration is the weakest class** (0.604 live, 0.755 in `ports_w30`) — it is
  the rarest at 1.67% and is designed to resemble normal traffic.
- **Cross-dataset precision is ~42%, FPR is 41.6%** — the model flags 26,456 benign
  windows as attacks on DAPT 2020 (more false alerts than real attack flows).
  Detection recall transfers better than precision to unseen networks.
- **Persistence beats LSTM on this data structure** — CIC-IDS-2018 attacks are long
  contiguous runs; a trivial "repeat last label" baseline outscores the LSTM at every
  horizon. The model learns attack-period density, not attacker trajectories.
  True per-host forecasting requires identity-bearing data (DAPT path; in progress).
- **No training-data identity** — CIC-IDS-2018 ML-ready CSVs have no source/dest IP.
  Windows are time-local network slices, not per-host sequences.
- **Kill-chain transition ordering follows ATT&CK doctrine**, not learned
  probabilities. Stage *persistence* is measured from DAPT; off-diagonal transition
  weights are hand-coded.
- **No DDoS coverage** — the two CIC-IDS-2018 DDoS days were not in our training
  set. Single-source DoS is covered.
- **Lead time is self-referential** — the dashboard compares the forecast threshold
  against the model's own damage-stage detection, not against an external IDS.
- **PCAP feature extraction is our own CICFlowMeter-compatible implementation**,
  so minor definitional differences exist versus the original tool.

---

## License / attribution

Datasets are the property of their
respective publishers (Canadian Institute for Cybersecurity; Arizona State
University). MITRE ATT&CK® is a registered trademark of The MITRE Corporation.
