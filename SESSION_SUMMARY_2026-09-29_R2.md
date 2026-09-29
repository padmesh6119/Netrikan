# Session summary — 2026-09-29, Round 2

Branch `eval/task4b-task7-lodo`. This follows `SESSION_HANDOFF_2026-09-29.md`
(the transition-F1 session). Read `TRAINER_BACKLOG.md` → "Round 2" for the full
context. This file covers what this session did and where to start next.

---

## What we were actually doing

The backlog's Round 2 was a list of six open questions (R1–R6) that decide
what the pitch is allowed to claim. It was not new features. Each item is a
measurement with a stated pass/fail rule, run against the deployed checkpoint
`cic_v2_w30.pt` on `/tmp/netrikan-cic-full-w30` (6.16M windows, all 7 CIC-IDS-2018
days) and on DAPT 2020 (86,391 windows) for cross-dataset checks. We also did C5
(per-horizon naive baselines) because it was cheap and strengthens the
transition-F1 claim.

The rule throughout: report the number as it comes out, even when it hurts
the pitch. Two results below do hurt (R1 at 30 steps, R3).

---

## Results

| Item | Output | Result |
|---|---|---|
| R1 rollout at 30 steps | `models/rollout_eval_cic_v2_w30_k30.json` | State space beats persistence 30/30 steps. **Stage level:** the rollout beats the held step-1 classifier on transition windows at steps 2–24 and loses at 25–30. Pitch: *"the classifier forecasts; the rollout illustrates"* for the 15-min horizon. |
| R2 calibration, held out | `models/cic_v2_w30_calib.json` | Block-level fit/test split. Test ECE-15 0.0074 → 0.0073, Brier 0.0959 → 0.0958, T=1.094. Ship gate passes, and it also passes a harder chronological split. The gain is tiny: the model was already calibrated. |
| R3 DAPT fixed-FPR | `models/dapt_{cic_v2,cic_full,base}_w30.json` → `operating_points` | **cic_v2 recall 10.7% at 10% FPR** (6.3% @ 5%, 31.7% @ 20%). This reproduces the committed numbers exactly. Transfer is weak at SOC-usable FPRs. Not a slide strength. |
| R4 LODO fold-1 DoS | `models/lodo_f1_dos_diag.json`, `bench/lodo_f1_dos_diag.py` | **Hypothesis confirmed at short horizons.** The failure is in windows currently labelled DoS (AUC 0.20). Class-weighted retrain (`--balance-power 1.0`): onset k1 0.425 → 0.614, k5 0.445 → 0.610. k15/k30 barely move. Past epoch 3, Benign F1 collapses to ~0.05. |
| R5 LODO without fold 0 | `models/lodo_cic_full_w30.json` → `aggregate_excl_family_holdout` | Folds 1–6: onset k5 0.631 ± 0.099, supported-class F1 0.559 ± 0.184. |
| R6 promote model | `models/BEST.txt` | `cic_v2_w30.pt` deployed; per-claim ownership table; `lstm_world_model.pt` retired. |
| C5 per-k logreg | `models/baseline_k{0,30,90,180,360}.json` | The LSTM beats the flat-window logreg on transition-F1 at every k: +0.05 / +0.08 / +0.09 / +0.06 / **+0.02**. The margin over a linear model is modest and shrinks at long horizons. |

## Code changes

- `bench/rollout_eval.py`: stage-level scoring against persistence, the held
  classifier, and the Markov `TRANSITION` matrix. Adds a transition-enriched
  sample and drops windows that cross a capture day.
- `src/calibration.py`: `--heldout` mode (block-interleaved and chronological
  splits, 15-bin ECE, Brier, ship gate) writes `models/<tag>_calib.json`.
- `src/infer.py`: reads `<ckpt>_calib.json` first (T=1 if `deploy` is false) and
  falls back to `temperature.json`.
- `src/eval_dapt.py`: `operating_points` at FPR {5, 10, 20}% for the breach and
  stage-attack-mass scores, plus a `--scaler` flag.
- `bench/lodo.py`: `aggregate_excl_family_holdout`.
- `src/baseline.py`: `--horizons` mode for per-k logreg.
- `bench/lodo_f1_dos_diag.py`: new.

## Environment gotchas (read before running anything)

- **`/tmp/netrikan-cic-full-w30` is being eaten by macOS /tmp cleanup.** It
  had lost `scaler.pkl`, `features.txt` and `onset_k*.npy`. This session rebuilt
  `onset_k{1,5,15,30}.npy` from `y.npy` + `file_id.npy` (same logic as
  `pipeline_v2.py`). The scaler lives in `models/scaler_cic_full.pkl`: pass
  `--scaler models/scaler_cic_full.pkl` to `eval_dapt.py`. If `X.npy` goes
  too, rebuild with `scripts/reproduce.sh` (stage 1). Consider moving the data
  somewhere outside `/tmp`.
- Use `.venv/bin/python`, with `NETRIKAN_WORKERS=0` for training (shared-memory
  timeouts otherwise) and `NETRIKAN_DEVICE=mps`.
- `models/lodo_f1_cw.pt` exists locally but is gitignored like other experiment
  checkpoints. Only its metrics JSON is committed.
- `src/dapt_entity_eval.py` still defaults to `lstm_world_model.pt`. It was left
  alone on purpose: it loads the scaler from `--data`, so switching only the
  model would pair it with the wrong scaler. Fix both together if you touch it.
- `dapt_crossdataset.json` and `dapt_ports_w*.json` were **not** regenerated:
  the first records no model path, and the ports checkpoints aren't on this machine.

---

## Where to take off from

In rough priority order:

1. **Fix the pitch text** to match R1/R3/C5 (app/pitch owner). Say "the classifier
   forecasts; the rollout illustrates" at 15 min. Drop any "recall at 10% FPR"
   strength claim. Quote the LSTM-vs-logreg margin honestly (small at k=360).
   Check `MASTER.md:414` ("beats persistence at 9/10 horizons"), flagged last session.
2. **R4 follow-up, if pursued:** class weighting helps fold 1 but is fragile
   (Benign collapse after epoch 3). Before applying it to all 7 LODO folds, try
   a milder `--balance-power` (0.75) or a class-balanced *onset* loss, and select
   on a criterion that penalises Benign collapse.
3. **R3 follow-up:** cross-dataset transfer at low FPR is the weakest number we
   have. `STATUS.md` already records that thresholding, ensembling and full
   rebalancing did not fix it. The next lever is probably features/domain
   adaptation, not tuning.
4. **Remaining backlog (multi-hour training jobs, not started):**
   - §5 state-weight sweep. The spec'd `sih-ports-w10` data is gone, so re-spec
     it on `cic-full-w30` with `train_v2.py --state-weight {2,4,8} --horizon 30 --fixed-split`.
   - §6 clean 5-seed variance on the `cic_v2_w30` recipe → `models/variance.json`.
   - C3 GRU dynamics head (only success criterion: beat persistence on transition windows).
   - C4 adaptive retrain with rollback → `models/adaptation_report.json`.
5. P3 app items (architecture doc, demo video) are still open.
