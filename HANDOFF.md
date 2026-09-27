# HANDOFF — read this first

You are picking up **Netrikan** (நெற்றிக்கண், "the third eye"): AI-based network
attack *forecasting* from flow data. It asks "where is this heading?", not "is this
flow malicious?".

Written 2026-09-27 for a new person on a different Claude account. Everything below
is either a measured number from a committed JSON under `models/`, or a warning.

---

## 0. The 60-second version

- **Deployed model:** `models/cic_v2_w30.pt`, loaded by `src/infer.py`. Trained on
  6.16M windows over all 7 CIC-IDS-2018 capture days, window 30, 24 flow features,
  5 classes, four heads (stage / breach / state / onset) + temporal attention.
- **It works well in-dataset and degrades off-distribution.** That is the whole
  story. In-dataset macro-F1 0.837 and onset AUC 0.901; across capture days onset
  AUC drops to 0.636 ± 0.093; for an attack family it never trained on, it scores
  *below chance*.
- **The project's value is its honesty.** There are more measured negative results
  here than positive ones, and that is deliberate — see §6.

### Which document to trust

1. **`STATUS.md` — the authority.** Long, dated, every claim sourced to a JSON.
   When anything disagrees with it, it wins.
2. **This file** — orientation and where we stopped.
3. `README.md` — accurate as of 2026-09-27 (results + limitations rewritten).
4. `MASTER.md`, `AI_HANDOFF.md`, `PROJECT.md`, `IDEAS.md`, `ARSENAL.md`,
   `WINNING_PLAN.md`, `BUILD.md`, `IMPLEMENTATION_GUIDE.md`, `RESUME.md` —
   **historical, stale in places.** `MASTER.md` carries a correction block at the
   top listing five claims later measurement falsified. Do not quote these files
   without checking `STATUS.md`.

---

## 1. Environment — do this before anything

```bash
cd /path/to/netrikan
.venv/bin/python -c "import torch; print(torch.__version__)"   # venv already exists

# The built dataset (8.9 GB) lives in /tmp and is WIPED ON REBOOT:
ls /tmp/netrikan-cic-full-w30/X.npy || {
  .venv/bin/python src/pipeline_v2.py --out /tmp/netrikan-cic-full-w30   # ~3 min
  .venv/bin/python bench/_make_file_id.py --data /tmp/netrikan-cic-full-w30
}
```

Training env vars — **all three matter**:

```bash
export NETRIKAN_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 NETRIKAN_WORKERS=0
```

`NETRIKAN_WORKERS=0` is not a preference. At the old hardcoded `4`, this machine
drove swap to 5.1 GB, epoch times climbed 170s → 672s, and a 7-fold sweep died with
`RuntimeError: Shared memory manager connection has timed out`. Worker count never
changes results (batch order comes from the sampler; verified bit-identical at 2
and 0).

Reproduce anything: `bash scripts/reproduce.sh` (29 stages, dependency-ordered).
`DRY=1` prints the plan without running. `STAGES="evals"` runs a subset.

---

## 2. The metrics, all of them

### In-dataset (CIC-IDS-2018, purged blocked split)

| metric | value | error bar (n=4 seeds) |
|---|---|---|
| macro-F1 | **0.837** | ± 0.0085 |
| accuracy | 92.7% | — |
| onset AUC k=1/5/15/30 | 0.889 / **0.901** / 0.902 / 0.893 | k5 ± 0.0068 |
| onset PR-AUC k=1/5/15/30 | 0.254 / 0.494 / 0.609 / 0.624 | — |
| transition-window macro-F1 | **0.343** (persistence **0.000**) | — |
| per-class F1 | Benign 0.950 · InitialAccess 0.968 · DoS 0.979 · Botnet 0.879 · **Infiltration 0.407** | Infil ± 0.027 |

**Baseline ladder** — this is the "does the LSTM earn its place" answer:
logreg on one flow **0.420** → logreg on the flattened window **0.692** → LSTM
**0.837**. Sequence info worth +0.272; the LSTM worth **+0.145** on top.

**Why transition-only matters:** a persistence oracle ("next label = current
label") scores **0.947** on all windows and beats the model. CIC attacks come in
long runs, so that is autocorrelation, not skill. On the 31,781 windows where the
label actually changes, persistence is **0.000** by construction and the model is
0.343 (Brier skill +0.488). Onset AUC is the same idea: persistence scores exactly
0.500 because it never predicts a change.

**Lead time** (15,898 benign→attack onsets, 30 s/window):

| horizon | @ 5% FPR | @ 1% FPR | @ 0.1% FPR |
|---|---|---|---|
| k=30 | **13.5 min**, 66% coverage | 3.5 min, 26% | 2.0 min, 2% |
| k=15 | 8.5 min, 61% | 2.5 min, 27% | 1.5 min, 5% |
| k=5 | 3.5 min, 50% | 2.0 min, 16% | 1.5 min, 2% |

### Robustness

| protocol | result |
|---|---|
| **Leave-one-day-out**, 7 folds | onset AUC k5 **0.636 ± 0.093** (single split: 0.902); worst fold 0.445, *below chance* |
| **Shuffled-history control** | onset PR-AUC k5 0.494 → **0.371** (−25%): the model genuinely uses temporal order |
| **Held-out Infiltration** | ROC-AUC 0.29–0.37 — **below chance** |
| **Held-out Botnet** | ROC-AUC **0.198–0.332**, PR-AUC below no-skill at every horizon, on 8k–40k positives |
| World-model rollout | **PASS** — MSE 0.612 vs persistence 0.782 *and* window-mean 0.649, 10/10 steps |
| Calibration | ECE 0.0088 → 0.0052 at T=1.076 |

### Cross-dataset (DAPT 2020, different network, zero retraining)

| metric | value |
|---|---|
| attack recall / precision | 67.8% / 37.8% |
| per-window FPR | **40.4%** |
| breach ROC-AUC | 0.694 |
| onset transfer k=15 | 0.705 |
| **entity roll-up (host, hour), persist=3** | **25.4% of attack host-hours at a 1.85% false-alarm rate per benign host-hour** |

**Quote the entity number, not the per-window FPR.** The shipped per-window
decision false-alarms on **49.9%** of benign host-hours; rolling up to host-hour
cells cuts that 27× to 1.85%. An operator reads alerts about hosts, not windows.

### Engineering

ONNX **1.41 ms**/window (610 windows/s), PyTorch 3.01 ms; **0.893 MB**, opset 17,
parity 1.9e-5, fully offline.

---

## 3. THE most important caveat

**Cross-dataset metrics carry ~8× the seed noise of in-dataset ones.**

| | mean ± std over 4 seeds |
|---|---|
| in-dataset macro-F1 | 0.8337 ± **0.0085** |
| onset AUC k5 | 0.8968 ± **0.0068** |
| **DAPT breach ROC-AUC** | 0.808 ± **0.0561** |

Two runs with the *same* config, data, seed 42 and selected epoch 8 differ by
**0.045** in DAPT AUC while being identical in-dataset (0.8369 vs 0.8367) — MPS
training is not bit-deterministic and transfer amplifies what in-dataset absorbs.

Consequences you must respect:
1. In-dataset numbers are safe as point values. **No single-run cross-dataset
   number is** — including the 67.8% recall and 40.4% FPR above. They are one draw.
2. The deployed checkpoint is an ordinary draw in-dataset (within 1σ) but sits
   **2.0σ below** the cross-dataset mean. It drew badly; it is not broken.
3. In-dataset checkpoint selection is nearly **uninformative** about transfer.

---

## 4. Where we stopped, and what to do next

Branch **`eval/task4b-task7-lodo`**, not pushed. All work is committed.

### Needs a human decision
**Promote a fresh checkpoint for cross-dataset use?** A fresh epoch-4/5 checkpoint
transfers better than the deployed one (20 of 20 tried beat it, DAPT AUC 0.707–0.864
vs 0.694) at a cost of ~0.006 in-dataset macro-F1 and ~0.006 onset AUC. "Retrain and
stop at epoch 4–5" is a *recipe*, so adopting it involves no test-set peeking.
Deliberately **not done** — it is a real tradeoff and was left to the owner.

### Ranked next steps
1. **Re-capture `demo_attack_lab/attack_small.pcap` with SMB traffic.** It has none,
   so 3 of 4 counterfactuals show zero delta and the live demo looks flat. Highest
   value of anything remaining — a flat demo costs more than any metric gains.
   Needs traffic generation and probably sudo.
2. **Suricata external baseline.** `bench/suricata_comparison.py` is written and
   waiting; needs `brew install suricata`. An external reference ("Suricata alerted
   at T, we warned at T−8.5 min") is worth more to a reviewer than any internal
   ablation, because it is the one number you cannot be accused of self-grading.
3. **Finish Task 8a properly** — a 5th seed at a constant epoch budget, plus
   per-seed transition macro-F1 and onset PR-AUC (currently blank, not guessed).
4. **Loss-side reweighting for Infiltration (0.407).** Sampling-side is ruled out
   with numbers (see §6).

### Do NOT bother with
- **More ensembling.** Closed with measurements — both probability and rank
  averaging fail to convert a +0.118 AUC gain into anything usable at a working FPR.
- **Chasing the LODO number with architecture changes.** The measured causes are
  class-prior mismatch and single-day family confinement; a bigger model fixes
  neither.
- **Raising the sampler rebalance exponent.** Measured worse on every axis.

---

## 5. Hard rules — breaking these invalidates results

1. **Never split sliding windows randomly.** Windows overlap by `WINDOW-1` rows, so
   a random split leaks ~100% of validation into training (measured, not feared).
   Always `train_v2.blocked_split()` with `purge = WINDOW - 1`.
2. **Never change the label offset.** `y[i] = labels[i + WINDOW]` — the label is the
   flow *after* the window. That offset is what makes this forecasting.
3. **Time-sort before anything temporal.** The CIC CSVs are not stored in time
   order. `pipeline_v2.load_all()` sorts; do not add a path that bypasses it. Before
   this fix, one day yielded 3,926 transitions instead of 23,948, and **every prior
   CIC run was affected**.
4. **Respect capture-file boundaries.** No window, onset label or evaluation pair
   may span two capture days.
5. **Do not re-add port features to the model input.** Measured: +0.003 in-dataset,
   −0.04 cross-dataset recall. They stay for the rule layer only.
6. **Never set `attention=True` on a checkpoint trained without it** — it changes
   how the trunk pools, silently. `infer._load_model()` auto-detects; keep it so.
7. **Do not present an untrained head's output.** Use `infer.onset_trained()` /
   `attention_trained()`. A checkpoint predating a head loads fine under
   `strict=False` and emits noise.
8. **Label the explanation method honestly.** Real SHAP (`shap.GradientExplainer`)
   on demand; gradient×input always-on. Every output carries `attribution_method`.
9. **Never invent a number.** Every figure in a doc must come from a committed JSON
   a reader can regenerate. A blank is acceptable; a plausible fabrication is not.
10. **Record negative results with equal prominence.** This is the project's whole
    credibility model.

---

## 6. Every negative result — do not quietly drop these

A reviewer who finds one of these unstated will discount everything else.

| finding | evidence |
|---|---|
| **No zero-shot generalization to unseen attack families.** Trained without a family, the model rates its onsets *more benign than benign*. | Infiltration ROC-AUC 0.29–0.37; Botnet **0.198–0.332**, PR-AUC below no-skill at every horizon, 8k–40k positives |
| **The single-split onset AUC of 0.902 is optimistic.** | LODO 0.636 ± 0.093; one fold 0.445. Not undertraining: 2× epochs at ⅓ LR moved it only to 0.514, and longer training made that fold *worse* |
| **Flat AUC across horizons is a single-split artifact.** | Across days it declines — one fold 0.715 at k=1 → 0.425 at k=30 |
| **InitialAccess does not transfer across days at all.** | F1 **0.00** both directions (trained 381k→tested 566; trained 566→tested 381k) |
| **Thresholding cannot fix the cross-dataset FPR.** | 6.3% recall at 5% FPR. The argmax is not better *ranked*, just looser: matched at FPR 0.404, breach 0.681 vs argmax 0.678 |
| **Ensembling fails.** Both combiners. | prob AUC 0.812 / rank 0.800 vs baseline 0.694 — but recall@5% 0.061 / 0.055 vs baseline **0.070**. The AUC gain sits where no operating point uses it |
| **Full inverse-frequency rebalancing is worse on every axis, including its target.** | p=1.0: macro-F1 0.798 vs 0.837, **Infiltration 0.357 vs 0.407**, Botnet 0.805 vs 0.879. Ran 18 epochs vs 13, so not undertrained. `p=0.5` is a measured good default |
| **"Less fitting transfers better" was refuted by seeds.** | Believed on one seed; across 4, epochs 3–5 transfer *better* than 1–2 and the peak epoch moved 1→4→4→5 |
| **The hand-written rules hurt generalization.** | DAPT SEDI: pure rules **−0.256**, pure model **+0.416**. `MODEL_WEIGHT` is now 0.90, so the 14 named detectors are largely decoration cross-network |
| **CTU-13 is near-degenerate — never headline its 0.9989.** | LSTM 0.9989 vs flattened logreg 0.9940, i.e. +0.005. It is a control showing the task is easy |
| **Two LODO folds are family holdouts in disguise.** | Botnet lives only on day 0; InitialAccess effectively only on day 6 (day 2 has 566 windows). Read `train_support_of_present_classes`, not just the flag |
| **Per-host windowing is not a free win.** | +2.8pp lateral recall for +8.3pp FPR, net SEDI loss 0.5825 → 0.4983. Present as a lateral-movement toggle with the trade stated, or not at all |

### Bugs found and fixed — worth knowing the class of mistake
- **`eval_dapt.py` reported the wrong metric.** The onset loop reused `auc` as its
  loop variable, so the JSON field `roc_auc` was silently the **k=30 onset AUC**.
  Proof: `dapt_cic_v2_w30.json` had `roc_auc == onset k30 == 0.6791673686497901`
  exactly. True breach AUC is **0.6936**. Fixed; `breach_roc_auc` is now explicit.
- **Hardcoded `num_workers=4`** killed long sweeps (see §1).
- **`pipeline_v2.load_all()` never time-sorted** — affected every prior CIC run.
- **Four files had `~/netrikan/...` hardcoded**; `eval_dapt.py` computed every DAPT
  result then crashed on save. All now repo-relative.
- **`classification_report` without `labels=`** crashed on any corpus missing a class.

---

## 7. Map of the code

| path | what |
|---|---|
| `src/infer.py` | the deployed entry point: `analyze()`, fusion, `explain()`. Check `MODEL_PATH` / `MODEL_WEIGHT` here, not in the docs |
| `src/model.py` | `WorldModel`: LSTM trunk + stage/breach/state/onset heads, optional attention, `rollout()` |
| `src/train_v2.py` | the trainer. `--select-on combined` is the deployed criterion. Env: `NETRIKAN_WORKERS`, `NETRIKAN_DEVICE` |
| `src/pipeline_v2.py` | dataset build: time-sorts, windows, onset labels, respects file boundaries |
| `src/eval_dapt.py` | cross-dataset eval |
| `src/forecast.py`, `src/attck_map.py` | kill-chain projection (7 stages; RECON is id **6**, appended not inserted) |
| `src/signals.py` | 14 rule detectors (4 need packet input) |
| `src/explain_shap.py` | real SHAP via `GradientExplainer` (chosen over DeepExplainer: DeepLIFT is unreliable through LSTM gates) |
| `bench/lodo.py` | leave-one-day-out. **Quote `supported_class_f1`, never `stage_macro_f1_DO_NOT_REPORT`** |
| `bench/operating_points.py` | threshold sweep + `min_persist` + entity roll-up |
| `bench/transfer_vs_epoch.py` | transfer vs training epoch |
| `bench/lodo_ensemble.py` | ensemble study (negative) |
| `bench/heldout_family.py`, `bench/lead_time.py`, `bench/onset_pr.py`, `bench/latency.py`, `bench/rollout_eval.py`, `bench/seed_variance.py` | the rest of the evidence |
| `scripts/reproduce.sh` | the whole chain, 29 stages |

`models/` holds every result JSON. Checkpoints are gitignored except a whitelist in
`.gitignore` — the per-fold and per-epoch `.pt` files are intentionally untracked;
their metrics JSONs carry the results.

---

## 8. If you are an AI agent picking this up

Read `STATUS.md` top to bottom first — it is long but it is the only file guaranteed
current. Then:

- Verify claims against **code and JSONs**, not prose. The docs have drifted before;
  that is why `STATUS.md` exists.
- Check the dataset exists in `/tmp` before planning any training (§1).
- Before quoting a cross-dataset number anywhere, re-read §3. Most of a day was
  spent discovering that single-run cross-dataset comparisons are near-worthless
  here.
- When an experiment comes back negative, write it down with the same prominence as
  a win, in `STATUS.md` and in the commit message. Several findings in §6 were
  believed to be wins for an hour before a control refuted them; one
  (`transfer_vs_epoch`) was refuted by the *next* experiment in the same session and
  is retracted in the commit that follows it. That trail is a feature.
- Do not cherry-pick a checkpoint by its DAPT score. That is test-set selection and
  the project explicitly refuses it — say so if asked to.
