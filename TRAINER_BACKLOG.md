# Netrikan — trainer backlog

Everything the model side lacks and everything we want next. Written for whoever
runs the overnight harness. `~/netrikan` == this repo. Outputs land in `models/`;
every phase stage skips itself if its output JSON already exists, so re-running is
safe. Follow the same `stage "NAME" "$guard.json" $PY script ...` pattern as
`run_phase2.sh`.

Current best in-dist: **ports_w30 macro-F1 0.9249, Infiltration 0.755**.
Current cross-dataset (base_w30 / 24-feat, DAPT): recall 0.839, precision 0.422,
FPR 0.416, ROC-AUC 0.731.

---

## Competitor delta — audited 2026-09-27

Straight answer to "do we beat them on every aspect": **no.** Five *shipped
features* now exist in rival repos that we don't ship. **But not one of the five
survives inspection** — every implementation is fit on leaked, synthetic, or
toy-scale data. So the instruction to the trainer is not "match the checkbox," it
is **ship each of these at DAPT scale (86,591 real flows) with held-out-test
verification — be the first to do it correctly.** That is a winnable bar; the
competitor versions below show exactly where they cut the corner.

Only repos that pushed **since 2026-09-18** are covered (the delta). Pre-Sep-18
repos were graded last session and did not move — do not re-chase them.

**Repos that moved this week:** SAKETH070706/SIH_2K26 (09-27),
Aditya-Coder477/NEXUS-Forecast (09-27), overhyped-pratham/cybersentinel-ai (09-27),
yash-yb/What-the-hack (09-20, moved from DurgeshLabs).

### C1 — Transition-dynamics analysis  → we still win if we ship at scale
cybersentinel published `docs/transition_dynamics_analysis.md` ("Phase 8C
scientific report") with LR/GRU/WorldModel + 3 ablations. **Their holdout is 44
sequences with 6 true transitions.** Ours (P0 §1) runs on the DAPT / val set where
change-rate is 0.079–0.093 → thousands of transitions. Ship P0 §1 with the
`n_transitions` field populated and we beat their report on scale and honesty. This
was our unique angle; it no longer is, so P0 §1 is now also a **race**.

### C2 — Calibration is shipped by 4 teams, correctly by 0  → P1 §2, reframed
- **NEXUS-Forecast**: per-horizon Platt scaling, `docs/PROBABILITY_CALIBRATION.md`.
  Fit on val (n=3483), extreme shrinkage (slope 0.195). **Val ECE 0.289→0.062 but
  test Brier 0.124→0.219 — calibration makes test ~77% worse.** Cautionary example.
- **cybersentinel-ai**: `artifacts/calibration/temperature.json` T=1.568, but
  "fitted_split" is two named synthetic traces, not a dataset split.
- **SAKETH070706**: re-fit T 1.3808 → 1.1985 in the *same commit* that resolved
  host-fingerprint leakage — the earlier temperature was fit on leaked data.
- **What-the-hack**: `ai/evaluation/calibration.py` + tests, but no published numbers.
Our win condition is in P1 §2: report ECE+Brier on a held-out **test** partition
and ship only if test improves. Nobody has cleared that.

### C3 — GRU dynamics head (L2)  → NEW backlog item, low urgency
NEXUS ships `models/world_model/gru/best_model.pt` (2-layer, hidden 128, horizons
1/3/6) and csxzor has a GRU latent-dynamics model. But NEXUS's `metadata.json`:
`best_epoch 5, epochs_trained 12` on a 35-epoch budget, val_loss 2.11 — it
converged instantly and stopped. That's a checkpoint, not a demonstrated dynamics
model. **Do:** build the encoder+GRU dynamics head (was tier-3 [H]); its only
success criterion is beating persistence on transition windows (C1/P0). Don't rush
it to match a checkpoint that doesn't work.

### C4 — Adaptive / online learning with rollback  → NEW backlog item
cybersentinel ships `artifacts/adaptation/{before,after}_adaptation_metrics.json` +
`rollback_verification.json` + `threat_memory.json` (v2.0.0→v2.1.0). **But it's
n=48 adapted / n=60 eval, with auroc 1.0 and fpr 0.0 — degenerate synthetic.**
**Do:** a small retrain-on-recent-slice path that (a) evaluates before/after on a
real held-out DAPT slice, (b) auto-rolls-back if test macro-F1 regresses. Output
`models/adaptation_report.json`. Ship it honest at real scale and it beats theirs.

### C5 — Per-horizon calibrated baselines  → folds into §3/§4
NEXUS ships logistic-regression baselines per horizon (`logistic_regression_k1/k3/k6`).
We publish persistence + one logreg. **Do:** add a logreg baseline at each k in the
horizon sweep so every horizon claim has a per-k naive floor, not just persistence.

### Not the trainer's job (app/product) — logged so nothing is lost
- **nDPI deep-packet inspection** (mdayaan11/prism-ai `dpi/ndpi_adapter.py`) — we use
  tshark; nDPI does app-layer protocol ID.
- **LLM analyst copilot** (prism-ai Gemini; cybersentinel Ollama/Gemini tiering).
- **Live Zeek ingestion adapter** (What-the-hack `zeek_live_adapter.py`; ENCLIVRA).
- **Integrity manifest w/ cross-platform hashes** (NEXUS) — relates to our SHA
  ledger (P3/B1).
- **Kill-chain reconstruction graph engine** (SabarishR08/cyber-killchain-...).
- **Large test suites** (cybersentinel 328, prism 265) — we have partial coverage.

### Our moat still holds — do NOT trade these away chasing the above
- DAPT cross-dataset at **86,591 real flows** vs their 44–60 synthetic samples.
- **Published FPR at scale** (41.6%) vs competitor degenerate AUROC=1.0 / FPR=0.0.
- Real **4-day APT** dataset (DAPT 2020), not botnet snapshots.
- **Counterfactual intervention panel wired live** — still rare in the field.

---

## P0 — decides the whole forecasting thesis

### 1. Transition-window F1  (NEW script: `src/eval_transition.py`)
**The single most important number we do not have.** Persistence beats the LSTM
on all-window macro-F1 at every horizon (gap −0.017 to −0.050) *only because
~91% of windows have no state change* (`change_rate` 0.08–0.09 in
`persistence_baseline.json`). Persistence is trivially right on the static 91%.
The honest question: on the **windows where the stage actually changes**, does the
LSTM beat persistence? If yes → the forecasting claim stands and this becomes the
headline metric. If no → we stop calling it forecasting and reframe as calibrated
multi-stage detection.

Do:
- Reuse the exact fixed-split val set from the `fx_k*` runs (same blocks).
- Mark a window as a *transition* when `y[t+k] != y[t]` (label changes over the
  horizon). Also emit the change-onset variant (benign→attack) separately.
- Compute macro-F1 on transition-only windows for **both** the LSTM and the
  persistence baseline, per k ∈ {0,30,90,180,360}.
- Output `models/transition_f1.json`:
  `{k, n_transitions, lstm_transition_f1, persistence_transition_f1, gap,
    lstm_onset_recall, persistence_onset_recall}` per k. **`n_transitions` is a
  required field** — it is the number that makes this beat the competition (see
  Competitor delta §C1: cybersentinel published a transition report on *6*
  transitions; our change-rate 0.079–0.093 over the val set is orders of magnitude
  more). State the count.

Done when: `transition_f1.json` exists with all 5 horizons, per-k `n_transitions`,
and a one-line verdict (`lstm_wins_transition: true/false`) at the top.

---

## P1 — table stakes we're losing points on

### 2. Temperature calibration  (NEW script: `src/calibrate.py`)
We ship raw softmax. Four competitors now commit calibration (SAKETH070706,
NEXUS-Forecast, cybersentinel-ai, What-the-hack) — but **none is verified on a
held-out test partition, and the one that shows test numbers gets worse on test**
(NEXUS: test Brier 0.124 → 0.219 calibrated; see Competitor delta §C2). So the
open goal is not "catch up" — it is **be the first to calibrate correctly**.

Do:
- Take the promoted checkpoint's val logits, fit a single scalar T by minimizing
  val NLL (LBFGS, ~50 iters). Do **not** refit on test/DAPT.
- Report Expected Calibration Error (15-bin) **and Brier** on val AND on a held-out
  test partition. This test-partition check is the bar no competitor has cleared.
- Save T as a sidecar `models/<tag>_calib.json` (`{"temperature": T, "ece_val_pre":,
  "ece_val_post":, "ece_test_pre":, "ece_test_post":, "brier_test_pre":,
  "brier_test_post":, "bins":[...]}`). Inference divides logits by T before softmax.

Done when: `<tag>_calib.json` exists, **`ece_test_post < ece_test_pre` AND
`brier_test_post <= brier_test_pre`** (ship only if test genuinely improves — if it
doesn't, report that and do not deploy the temperature), and `infer.py`/`app.py`
read the sidecar (app-side wiring tracked in §9/P3).

### 3. Cross-dataset FPR is 0.416 — bring it down at a fixed operating point
Recall 0.84 is fine; precision 0.42 means ~1 false alarm per true one. We report
FPR at the default 0.5 threshold, which is not how a SOC runs.
`eval_dapt.py` currently emits only the fixed-threshold report.

Do (extend `eval_dapt.py`):
- Sweep the decision threshold; report recall + precision at fixed FPR targets
  {0.05, 0.10, 0.20} and the threshold that hits each.
- Add these to every `dapt_*.json` under a `operating_points` key.

Done when: `dapt_crossdataset.json` (and the ports variants) carry
`operating_points` and we can quote "recall X% at 10% FPR".

### 4. Lock the model-selection decision (in-dist vs transfer)
`RESUME.md` already found port features help in-dist (+3pp) but wreck transfer
(ports_w30 DAPT macro 0.471 vs base_w30 0.589). This is our best technical story —
but no model is *promoted* yet (`lstm_world_model.pt` is still the old 24-feat).

Do: decide and write it down in `BEST.txt` / `NETRIKAN_SOLUTION.md`:
- Promote **base_w30** (no ports) as the deployed/cross-dataset model — it
  generalizes. Keep **ports_w30** as the in-dist benchmark number only.
- Or ship both and label each by regime. Either way, stop leaving it implicit.

Done when: `lstm_world_model.pt` points at a deliberate choice and the doc says
which model owns which claim.

---

## P2 — strengthen, once P0/P1 land

### 5. `--state-weight` sweep to attack the persistence gap
`train_v2.py --state-weight` already exists. Persistence wins by nailing the static
windows; upweighting transition/change rows in the loss is the direct lever to make
the LSTM competitive *there* without wrecking overall F1.

Do: sweep `--state-weight {2,4,8}` on `sih-ports-w10` at k=30, re-run
`eval_transition.py` on each. Output `models/stateweight_sweep.json`
(`{weight, macro_f1, transition_f1, gap}`).

Done when: the sweep JSON exists and we know the weight that maximizes transition-F1
without dropping all-window macro-F1 more than ~1pp.

### 6. Seed variance / confidence intervals
Only `ports_w10_seed2` exists (0.8949 vs 0.8937 — good, tight). We quote point
estimates with no band.

Do: add seeds 3–5 on the promoted config (cheap, 20 epochs each). Emit
`models/variance.json` (`mean, std, n_seeds, per_class_std`).

Done when: we can write "macro-F1 0.92 ± σ (5 seeds)".

### 7. Known ceilings — document, don't chase
- **Data Exfiltration detect rate 0.20** on DAPT is off n=15 support. Not fixable
  by training; note as a stated limitation, not a bug.
- **Infiltration F1 0.60–0.76** is the weakest class and the hardest — any gain
  here is real signal, so watch it in every run's `report`.

---

## P3 — not the trainer's job, but part of "everything we lack" (owner: app)

Listed so nothing is dropped. These are app/pitch, not training runs.

- **B1 — SHA-256 ledger chain resets every alert** (`src/app.py:468-471`): genesis
  is recomputed per alert, so the tamper-evident chain never actually chains. Fix:
  carry prev-hash forward.
- **B8 — remove the "LSTM beats persistence" claim** from slides/README until §1
  proves it on transition windows. Right now it's false on all-window F1.
- **B9 — per-host claim in pitch sentence 1** overstates the entity eval; soften to
  what `a1_entity_compare.json` actually shows (+lateral-movement recall on
  per-host segmentation).
- **Architecture doc (2 pages)** — top-tier competitors have one; we don't.
- **Demo video ≤2 min.**

---

## Quick reference — what maps to what

| Want | Script | Output artifact | New? |
|---|---|---|---|
| Transition-window F1 (§1, C1) | `eval_transition.py` | `transition_f1.json` (`n_transitions` req.) | **new** |
| Calibration, test-verified (§2, C2) | `calibrate.py` | `<tag>_calib.json` | **new** |
| Fixed-FPR operating points (§3) | `eval_dapt.py` (extend) | `dapt_*.json` `operating_points` | edit |
| Per-horizon logreg baseline (C5) | `baseline.py` per k | `baseline_k*.json` | edit |
| State-weight sweep (§5) | `train_v2.py --state-weight` | `stateweight_sweep.json` | exists |
| GRU dynamics head (C3) | new dynamics trainer | `world_model_gru.pt` + transition eval | **new**, low urgency |
| Adaptive learning + rollback (C4) | new retrain path | `adaptation_report.json` | **new** |
| Seed variance (§6) | `train_v2.py` (seeds 3–5) | `variance.json` | exists |
| Model promotion (§4) | manual | `BEST.txt`, `lstm_world_model.pt` | decision |

Order of attack: **§1 first** — it either saves the forecasting pitch or tells us to
pivot it. Everything else is downstream of that answer.
