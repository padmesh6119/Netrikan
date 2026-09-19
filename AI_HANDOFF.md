# AI_HANDOFF.md — Netrikan

Context transfer for an AI agent taking over this repository. Written to be
self-contained: assume no access to prior conversation. Dense by design.

Human-facing version: `PROJECT.md`. This file is the operational one.

---

## 0. PRIME DIRECTIVES — read before editing anything

1. **Never split sliding windows randomly.** Windows overlap by `WINDOW-1` rows.
   A random train/test split leaks 100% of validation data into training. Always
   use `train_v2.blocked_split()`. This is the single most important invariant in
   the project. Details in §5.
2. **Never change the label offset in the pipeline.** `y[i] = labels[i + WINDOW]`
   — the label is the flow *after* the window. That offset is what makes this
   forecasting rather than classification. It is the entire thesis.
3. **Do not overwrite `models/lstm_world_model.pt`** without being asked. It is
   the checkpoint the dashboard loads. A better model (`base_w30.pt`) exists but
   is deliberately not promoted — see §11.
4. **Do not re-add port features.** Tested, measured, rejected. See §10.
5. **Do not claim SHAP.** The code uses gradient × input. See §12.4.
6. **Verify before asserting.** Numbers in this file came from
   `models/*.json`. Re-read those files rather than trusting memory.

---

## 1. What this is

A network-attack **forecasting** system for network intrusion detection
Netrikan, issued by the client organisation.

Conventional IDS asks *"is this traffic malicious?"* — classification of the
present. This asks *"which attack stage comes next?"* — prediction of the future.

Output shape:

```
Predicted over the next 15 minutes
  Lateral movement, then command & control
  confidence 59% · driven by SMB port scan · port 445 probing, 6 SYN/flow, 1 RST
```

Name: Netrikan (நெற்றிக்கண்), Tamil for "third eye".

---

## 2. Architecture — data flow with exact shapes

```
PCAP file ──tshark──┐
                    ├─> DataFrame[n_flows, 24 feat + Dst Port + Protocol + Timestamp]
NetFlow CSV ────────┘                    │
                                         ├─> StandardScaler.transform()
                                         ├─> sliding windows, stride 1
                                         ▼
                              ndarray (n_windows, WINDOW, 24) float32
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                   LSTM stage_head            LSTM breach_head
                   (batch, 5) logits          (batch,) sigmoid
                            │
                            ├─> softmax → model_probs[5]
                            │
                            │   signals.detect(raw_window, ports) → {name, score,
                            │                                        stage_hint, evidence}
                            ▼
                   infer._fuse(): 0.30*model + 0.70*signal_evidence → probs[6]
                            │
                            ▼
                   forecast.forecast(probs[6], interval, WINDOW, horizon_s)
                            │   Markov projection: p = p @ TRANSITION, repeated
                            ▼
                   {phrase, confidence, damage_risk, eta_seconds, sequence}
```

**Note the dimensionality change:** the model emits **5** classes; the forecast
operates over **6** chain stages. `attck_map.MODEL_TO_CHAIN` maps them.
EXFIL (5) is forecast-only — it is never a training label and the model cannot
predict it directly.

---

## 3. Repo map

### Inference path (what the dashboard uses)

| File | Role | Key API |
|---|---|---|
| `src/model.py` | LSTM definition | `WorldModel(input_size=24)` → `(stage_logits, breach_prob)` |
| `src/infer.py` | Orchestrator | `analyze(df, horizon_seconds) -> dict` |
| `src/signals.py` | 14 rule detectors | `detect(raw_window, ports) -> dict` |
| `src/forecast.py` | Markov projection | `forecast(probs, interval, window, horizon) -> dict` |
| `src/attck_map.py` | Stage constants | `STAGE_INFO`, `get_stage()`, `short()` |
| `src/pcap_ingest.py` | PCAP → flows | `load_pcap(path) -> DataFrame` |
| `src/demo_data.py` | Synthetic scenarios | `generate(scenario, n, seed) -> DataFrame` |
| `src/app.py` | Streamlit UI | entry point |

### Training / evaluation path

| File | Role |
|---|---|
| `src/pipeline.py` | Original dataset builder (24 feat, WINDOW=10). Superseded. |
| `src/pipeline_v2.py` | Parameterised builder. `--window N --out DIR [--with-ports]` |
| `src/train.py` | Original trainer. Contains the first `blocked_split`. |
| `src/train_v2.py` | Current trainer. `--data --tag --horizon --fixed-split` |
| `src/train_horizon.py` | Horizon sweep driver (superseded by `run_phase3.sh`) |
| `src/baseline.py` | Logistic-regression control |
| `src/eval_dapt.py` | Cross-dataset eval |
| `src/dapt.py` | DAPT 2020 loader |

### Orchestration

`run_app.sh` (launch dashboard, kills stale servers, verifies imports),
`run_overnight.sh`, `run_phase2.sh`, `run_phase3.sh` (training campaigns, all
resumable — stages skip if output exists).

---

## 4. Data contracts

### The 24 features — exact names, order matters

Defined identically in `infer.FEATURES`, `pipeline.FEATURES`,
`pipeline_v2.BASE_FEATURES`, `pcap_ingest.FEATURES`. **If you change one, change
all four.** `signals.F` maps name → index and must match the same order.

```python
['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
 'Flow Byts/s', 'Flow Pkts/s',
 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
 'Fwd IAT Mean', 'Bwd IAT Mean',
 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
 'PSH Flag Cnt', 'ACK Flag Cnt',
 'Pkt Len Var', 'Active Mean', 'Idle Mean']
```

Metadata carried but **not** model input: `Dst Port`, `Protocol`, `Timestamp`,
and (PCAP only) `Src IP`, `Dst IP`. Used by `signals.py` and for flow identity.

### Class labels

```python
0 Benign · 1 InitialAccess · 2 DoS · 3 Infiltration · 4 Botnet
```

Raw → class mapping in `pipeline_v2.LABEL_MAP`. Note the dataset misspells
`Infilteration`; the map handles it.

Distribution (5,549,436 windows): Benign 74.51%, DoS 11.79%, InitialAccess
6.87%, Botnet 5.16%, **Infiltration 1.67%**. Imbalance ratio 45:1.

### Chain stages (forecast space, 6 values)

```python
BENIGN=0 · INITIAL_ACCESS=1 · DOS=2 · LATERAL=3 · C2=4 · EXFIL=5
```

`DAMAGE_STAGES = {DOS, C2, EXFIL}`. `CHAIN_POS` in `forecast.py` gives on-chain
ordering; DoS is off-chain (position 99).

### On-disk artifacts

```
data/processed/         symlink -> /storage/sih-processed
  X.npy                 (5549436, 10, 24) float32, 5.3 GB
  y.npy                 (5549436,) int64
  scaler.pkl            sklearn StandardScaler, fit on all rows
/storage/sih-base-w30/  (5549416, 30, 24) — best model's dataset
/storage/sih-ports-w10/ (5549436, 10, 36) — rejected experiment
/storage/sih-ports-w30/ (5549416, 30, 36) — rejected experiment
```

`pipeline_v2.py` also writes `features.txt` (one name per line). `eval_dapt.py`
reads it to reconstruct the feature list — **always write it** if you add a
dataset builder.

---

## 5. THE LEAK — the most important thing in this repo

### Mechanism

Windows are built stride-1, so consecutive windows share `WINDOW-1` rows:

```
window 5000: flows 5000..5009
window 5001: flows 5001..5010   ← 9 of 10 rows identical
```

A random split (`train_test_split`) places these on opposite sides. Measured:

```
val windows sampled  : 200,000
sharing rows w/ train: 200,000  (100.00%)
```

### The fix — `train_v2.blocked_split(y, purge, n_blocks=500, val_frac=0.2)`

1. Cut the index range into 500 contiguous blocks
2. Assign whole blocks to train or val
3. **Drop `purge` windows at every block edge**, where `purge = WINDOW - 1`
4. Stratify block assignment by each block's dominant class

Verified: 0 overlap / 300,000 sampled. Cost: ~9,000 purged windows (0.16%).

### Consequences you must respect

- **`purge` must equal `WINDOW - 1`.** Window 10 → purge 9. Window 30 → purge 29.
  Passing the wrong value silently reintroduces leakage.
- Published CIC-IDS-2018 results of 0.98–0.99 are largely inflated by this. Our
  0.92 is not worse; it is honest. Do not "fix" the lower number.

### Second, subtler version of the same bug (already fixed)

In the horizon sweep, `blocked_split` was stratified on the **shifted** labels,
so each horizon `k` got a different validation set. Result: k=360 scored *better*
than k=30, which is impossible. Fixed with `train_v2.py --fixed-split`, which
pins blocks to unshifted labels.

**Use `models/fx_k*_metrics.json` for horizon claims. The `ports_w10_k*` files
are invalid.**

---

## 6. Module APIs

### `infer.analyze(df, horizon_seconds) -> dict`

Returns:

```python
{
  "windows": [ { "idx", "signal", "stage_probs"(6,), "stage",
                 "forecast", "breach", "attribution"(24,), "time" }, ... ],
  "stage_ids": ndarray, "breach": ndarray, "risk": ndarray,
  "n_windows": int, "flow_interval": float,
  "forecast_idx": int|None,   # first sustained escalation warning
  "detect_idx":   int|None,   # first sustained damage stage (IDS equivalent)
  "lead_seconds": float|None, # (detect_idx - forecast_idx) * flow_interval
  "n_flows": int,
}
```

`flow_interval` derives from the `Timestamp` column (median positive diff,
capped at 300s) or falls back to `DEFAULT_FLOW_INTERVAL = 1.2`.

### `signals.detect(raw_window, ports) -> dict`

`raw_window` is **unscaled** `(WINDOW, 24)`. Returns
`{name, score, stage_hint, evidence, alternatives}`. Highest-scoring rule wins.
Fallback is `{"name": "Normal traffic", "score": 0.0, "stage_hint": BENIGN}`.

### `forecast.forecast(stage_probs, flow_interval, window, horizon_seconds) -> dict`

Returns `{phrase, sequence, current_stage, confidence, damage_risk,
eta_seconds, steps, horizon_seconds, projections}`.

`HORIZONS = {"60 seconds":60, "15 minutes":900, "1 hour":3600, "6 hours":21600,
"24 hours":86400}`. Default UI selection is **15 minutes** (index 1).

---

## 7. Constants — where behaviour is tuned

| Constant | File:line | Value | Notes |
|---|---|---|---|
| `WINDOW` | `infer.py:27` | 10 | Must match the checkpoint's training window |
| `MODEL_WEIGHT` | `infer.py:33` | 0.30 | Model vs rules in fusion. **Untuned** — set when the model was mid-training. Worth revisiting. |
| `HORIZON_SECONDS` | `forecast.py:55` | 60 | Default only; UI overrides |
| `MAX_STEPS` | `forecast.py:56` | 400 | Caps projection length. Chain converges ~360 steps, so 6h and 24h give identical output. |
| `MEASURED_PERSISTENCE` | `forecast.py` | 0.90–0.98 | Diagonal, measured from DAPT |
| `DOCTRINE_SHAPE` | `forecast.py` | — | Off-diagonal, ATT&CK convention, **not learned** |
| `IDLE_GAP` | `pcap_ingest.py` | 1.0 s | CICFlowMeter's default is 5 s — known skew |

---

## 8. The forecast engine — non-obvious behaviour

### Transition matrix construction

`TRANSITION` is **built**, not hardcoded: `_build_transition()` takes the
measured diagonal from `MEASURED_PERSISTENCE`, takes relative off-diagonal
weights from `DOCTRINE_SHAPE`, and scales each row so it sums to
`1 - persistence`. Editing either dict regenerates the matrix at import.

### Why persistence is 0.93, not 0.43

Measured from DAPT 2020's 86,691 labelled flows. An earlier hand-written guess of
0.43 made the forecast escalate ~10× too eagerly. **Do not revert to intuition.**

### Markov convergence trap (fixed — do not reintroduce)

A Markov chain converges to its stationary distribution regardless of start
state. The stationary distribution here is:

```
[Benign 0.334, IA 0.075, DoS 0.030, Lateral 0.107, C2 0.341, Exfil 0.113]
```

C2 has the largest mass (highest persistence, 0.968). Over a long horizon, **any**
input drifted to "Command & control" — including pure benign traffic, which
produced "Command & control, 79% confidence" on clean captures.

Two guards now exist in `forecast.py`:

1. `_trajectory()` returns `[]` immediately if
   `current_stage == BENIGN and start[BENIGN] >= 0.5`
2. Confidence for the no-escalation case reads the **current** distribution `p`,
   not the horizon-end `end`, which was measuring convergence rather than evidence

If you touch the projection logic, re-verify on the "Benign baseline" scenario:
all windows must read "No escalation expected".

### Lead time

`forecast_idx`: first window with `damage_risk >= 0.45`, sustained 3 windows,
while `current_stage not in DAMAGE_STAGES`.
`detect_idx`: first window in a damage stage, sustained 3 windows.
The 3-window sustain prevents a single noisy window from faking a lead time.

At the 60-second horizon, `forecast_idx` is legitimately `None` — real intrusions
do not escalate that fast. This is correct behaviour, not a bug.

---

## 9. Results — exact, from `models/*.json`

### In-dataset (leak-free blocked split, 1.1M val windows)

| tag | feat | window | macro-F1 | Infiltration |
|---|---|---|---|---|
| `ports_w30` | 36 | 30 | 0.9249 | 0.755 |
| **`base_w30`** | 24 | 30 | **0.9221** | **0.751** |
| `ports_w20` | 36 | 20 | 0.9184 | 0.738 |
| `ports_w10` | 36 | 10 | 0.8937 | 0.654 |
| original (**live**) | 24 | 10 | 0.8851 | 0.604 |

Seed variance: `ports_w10` 0.8937 vs `ports_w10_seed2` 0.8949 → **±0.001**.
Differences above ~0.01 are real.

### Baseline control (`baseline_ports_w10.json`)

| model | macro-F1 | Infiltration |
|---|---|---|
| logreg, single flow | 0.607 | 0.077 |
| logreg, flattened window | 0.742 | 0.140 |
| LSTM | 0.922 | 0.751 |

Sequence is worth +0.135; the LSTM on top is worth another +0.180.

### Cross-dataset, DAPT 2020 (never trained on)

| model | attack recall | precision | lateral movement |
|---|---|---|---|
| original (24, w10) | **0.839** | 0.422 | 93.1% |
| `base_w30` (24, w30) | 0.797 | 0.381 | 94.2% |
| `ports_w30` (36, w30) | 0.757 | 0.303 | 96.6% |

Known failure mode: nearly everything collapses into the Infiltration class, and
`InitialAccess` was predicted **zero times** across 86,591 windows. Detection
generalises; classification does not.

### Corrected horizon curve (`horizon_curve_fixed.json`)

| k flows | macro-F1 | Infiltration | DoS |
|---|---|---|---|
| 0 | 0.8889 | 0.627 | 0.887 |
| 30 | 0.8704 | 0.623 | 0.817 |
| 90 | 0.8661 | 0.631 | 0.791 |
| 180 | 0.8621 | 0.616 | 0.791 |
| 360 | 0.8622 | 0.629 | 0.791 |

360 flows of lead costs 0.027 macro-F1. Infiltration is flat; DoS absorbs the
loss (floods are bursty, intrusions persist).

**Caveat to state proactively:** the curve is flat partly *because* attack state
persists, so the model may be reading "an attack is ongoing" rather than
predicting a transition.

---

## 10. Failed experiments — DO NOT REPEAT

### Port features (rejected)

Expanded `Dst Port` into 12 binary service-role features (`svc_web`, `svc_smb`,
`svc_rdp`, …). 24 → 36 features. Code still exists in
`pipeline_v2.PORT_GROUPS` / `PORT_FEATURES`, reachable via `--with-ports`.

Sanity check passed (`svc_remote` fired on 99.76% of the FTP-BruteForce day), and
in-dataset score rose 0.885 → 0.925. **But two variables changed at once**
(features *and* window 10 → 30). The control run isolated it:

| | in-dataset | DAPT recall |
|---|---|---|
| `ports_w30` (36 feat) | 0.9249 | 0.757 |
| `base_w30` (24 feat) | 0.9221 | **0.797** |

Port features were worth **0.003** and cost **0.04 cross-dataset recall**. All
the real gain came from the longer window. Ports are environment-specific — the
model memorised which services this network runs. **Dropped.**

### Learning the transition matrix from DAPT (not possible)

DAPT runs one attack phase per day, so consecutive flows never cross a stage
boundary: **0 cross-stage transitions**. Time-bucketing recovers the ordering
(at 6h granularity: Recon → Foothold → Lateral → Exfiltration) but yields only
**7 transitions** — not enough to estimate probabilities. Would need many
independent campaigns. DAPT gave persistence and timescale only.

---

## 11. Current state and open decisions

1. **`base_w30` is not deployed.** `models/lstm_world_model.pt` is still the
   original 24-feature WINDOW=10 model. Promoting `base_w30` requires
   `infer.WINDOW = 30` **and** repointing `SCALER_PATH` to
   `/storage/sih-base-w30/scaler.pkl`. Copying the `.pt` alone will silently
   produce garbage — shape mismatch on the window axis is not always an error.
   Trade-off: +0.037 macro-F1 and +0.147 Infiltration, but −0.042 DAPT recall.
2. `MODEL_WEIGHT = 0.30` is untuned (§7).
3. `models/BEST.txt` says `ports_w30`, chosen purely on in-dataset macro-F1 by an
   automated script. **It does not account for cross-dataset degradation.** The
   considered recommendation is `base_w30`.
4. `Netrikan_idea.md` still claims SHAP and a 60-second horizon. Both are
   inaccurate (§12).

---

## 12. Known weaknesses — state these before anyone else finds them

1. **Infiltration 0.751** vs 0.90+ elsewhere. Rarest class (1.67%) and designed
   to mimic normal traffic.
2. **Cross-dataset precision ~40%.** Many false positives on unfamiliar networks.
3. **Transition ordering is doctrine, not learned** (§10).
4. **Explainability is gradient × input, not SHAP.** One backward pass instead of
   thousands of feature-subset evaluations. Similar ranking, far cheaper, not the
   same algorithm. The idea doc overstates this.
5. **"Next 60 seconds" does not hold for APT.** DAPT's campaign advanced ~1 stage
   per day. 60s suits volumetric floods only. UI default is 15 minutes.
6. **PCAP extraction is a reimplementation of CICFlowMeter**, not the tool itself.
   Train/serve skew risk; `IDLE_GAP` differs (1s vs 5s). Java 21 is installed if
   someone wants the real jar.
7. **DoS precision 0.857** — it over-predicts DoS on benign traffic (18,766
   benign windows misread as DoS in `base_w30`).

---

## 13. Environment — fragile, read this

Python 3.12 with **three layered package trees**: `/usr/lib/python3/dist-packages`
(Debian), `~/.local/lib/python3.12/site-packages` (pip `--break-system-packages`,
takes precedence), and `/usr/local/lib/python3.12/dist-packages`.

**This broke once already.** numpy was upgraded to 2.5.3 in `~/.local`, shadowing
system numpy 1.26.4, while system `pandas`/`sklearn`/`matplotlib` were compiled
against numpy 1.x. Symptom:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

Fixed by upgrading the numpy-1 builds. **The last offender was `bottleneck`** — an
optional accelerator pandas imports silently at startup. Everything else was
fixed and the error persisted until that was found eight frames deep in the
import chain.

Current working set: numpy 2.5.3, pandas 3.0.5, sklearn 1.9.0, matplotlib 3.11.1,
torch 2.11.0+cu130, plotly 7.0.0, streamlit 1.63.0, pyarrow 25.0.1.

Repair command if it breaks again:

```bash
pip install --break-system-packages --upgrade \
    numpy pandas scikit-learn matplotlib bottleneck numexpr pyarrow
```

`run_app.sh` now verifies all six imports before launching and prints this
command on failure.

**pandas 3.0.5 is a major version** (breaking changes vs the 2.1.4 the code was
written against). Full pipeline re-verified after upgrade: 250 windows, 111s lead
time, identical forecasts. If odd behaviour appears in dataframe handling, pin
`pandas>=2.3,<3` — it also supports numpy 2.

Other environment notes: GPU is an RTX 2050, **4 GB VRAM** — batch 1024 fits,
much larger will not. `/home` runs ~90% full, which is why `data/processed` is a
symlink to `/storage`. `sudo` requires a password; `/storage` is user-writable
and `/mnt/extra` is not. **Do not touch `nvme1n1` — it is the Windows install,
including EFI and recovery partitions.**

---

## 14. Commands

```bash
# dashboard (kills stale servers, verifies imports)
cd ~/netrikan && ./run_app.sh            # -> http://localhost:8700

# rebuild dataset
python3 src/pipeline_v2.py --window 30 --out /storage/sih-base-w30

# train
python3 src/train_v2.py --data /storage/sih-base-w30 --tag base_w30 --epochs 25

# horizon point (ALWAYS use --fixed-split for comparability)
python3 src/train_v2.py --data /storage/sih-ports-w10 --tag fx_k90 \
        --horizon 90 --fixed-split --epochs 10

# cross-dataset
python3 src/eval_dapt.py --data /storage/sih-base-w30 \
        --model ~/netrikan/models/base_w30.pt --tag dapt_base_w30

# baseline control
python3 src/baseline.py --data /storage/sih-base-w30 --tag baseline_w30

# verify the leak fix yourself
python3 -c "
import numpy as np, sys; sys.path.insert(0,'src')
from sklearn.model_selection import train_test_split
from train_v2 import blocked_split
y = np.load('data/processed/y.npy'); n = len(y)
def leak(tr, va, label):
    m = np.zeros(n, np.int8); m[tr] = 1
    s = va[np.random.default_rng(0).choice(len(va), 50000, replace=False)]
    print(f'{label:8} {sum(1 for i in s if m[max(0,i-9):min(n,i+10)].any())/500:.1f}% leaked')
idx = np.arange(n)
leak(*train_test_split(idx, test_size=.2, random_state=42, stratify=y), 'random')
leak(*blocked_split(y, purge=9), 'blocked')
"
```

Training campaigns (`run_overnight.sh`, `run_phase2.sh`, `run_phase3.sh`) are
resumable — each stage skips if its output file exists. Phase 2 and 3 accept a
PID to wait on for chaining.

---

## 15. Common tasks

**Add a signal rule** → append to `cands` in `signals.detect()`. Tuple is
`(name, score, stage_hint, evidence_string)`. Test against the "Benign baseline"
scenario to confirm it does not fire on normal traffic — this has bitten twice
(C2 beaconing and DNS tunneling both originally fired on benign DNS).

**Change window size** → rebuild the dataset with `pipeline_v2.py --window N`,
retrain, then update `infer.WINDOW` *and* `infer.SCALER_PATH`. Three places.

**Add a chain stage** → `attck_map.py` (`STAGE_INFO`, `N_STAGES`), then
`forecast.py` (`MEASURED_PERSISTENCE`, `DOCTRINE_SHAPE`, `CHAIN_POS` — all must
grow to match), then `app.py`'s `CHAIN` list.

**Debug the dashboard** → check for stale servers first (`pgrep -af streamlit`).
A stale process holds old modules *and* an old theme, which looks exactly like a
code bug. `run_app.sh` handles this.

**Regenerate demo data** → `demo_data.generate(scenario, n, seed)` reseeds its
RNG per call, so it is deterministic. An earlier module-level RNG advanced on
every call and produced different traffic each rerun — fatal for a live demo.

---

## 16. Pitch positioning — the three defensible claims

Use these in that order; they are the differentiators.

1. **"We found the leak."** Sliding windows overlap 90%; random splitting puts
   near-identical windows on both sides. Measured 100% contamination, fixed it
   with a purged blocked split. Our numbers are post-fix; most published ones are
   not.
2. **"We measured lead time, not just accuracy."** 360 flows of lead costs 0.027
   macro-F1. Nobody publishes an accuracy-vs-lead-time curve.
3. **"We tested on a network we had never seen."** Trained 2018, tested on an
   unseen 2019 APT campaign: 80% attack recall, 94% lateral movement.

Supporting: logistic regression gets 0.61 where the LSTM gets 0.92 — the answer
to "why not something simpler."

Everything else (the dashboard, signal naming, ATT&CK output) is good packaging
but not differentiating.

**Where the AI actually is:** one LSTM for stage classification and breach
probability. The forecast is a Markov chain, the signals are 14 hand-written
rules, the text is f-string templates. **No language model anywhere.** Say this
plainly — anyone who opens `signals.py` sees fourteen `if` statements
immediately, and claiming otherwise is the only way it costs anything.
