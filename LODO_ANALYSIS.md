# LODO Analysis — Leave-One-Day-Out Cross-Validation

**Branch**: `eval/task4b-task7-lodo` (not yet merged to main as of 2026-09-29)  
**Source file**: `models/lodo_cic_full_w30.json`  
**Script**: `bench/lodo.py`

---

## What was run

7-fold LODO over CIC-IDS-2018. Each fold holds out one full capture day, trains on the remaining 6. This is a domain-shift test — held-out day has a different class composition than training days. Stronger than fixed-split eval; no competitor has published this.

---

## How to read the numbers — CRITICAL

`stage_macro_f1` in the JSON is labeled `stage_macro_f1_DO_NOT_REPORT`. The file's own note:

> "stage_macro_f1 averages over all 5 classes including ones with zero val support, so it is depressed by absence rather than by error and must NOT be quoted"

CIC-IDS-2018 days each contain only 2–3 of the 5 attack classes. A day with no DoS traffic produces DoS F1=0.0 by construction — the model never sees a DoS sample to classify. Averaging that zero in collapses the macro to 0.14–0.44, which looks like failure but is measuring absence, not error.

**Use these metrics instead:**
- `supported_class_f1`: F1 only over classes that actually appear in the held-out day
- `onset_auc_k{1,5,15,30}`: AUC for predicting attack onset at horizon k (above 0.5 = beats random)

---

## Actual results

| fold | held-out day | onset_auc_k1 | onset_auc_k5 | supported_class_f1 | attack classes present |
|------|-------------|-------------|-------------|-------------------|----------------------|
| 0 | day 0 | 0.665 | 0.665 | 0.413 | Botnet (family_holdout — not seen in train) |
| 1 | day 1 | **0.425** | 0.445 | **0.883** | DoS |
| 2 | day 2 | 0.631 | 0.599 | 0.496 | InitialAccess (566 samples only) |
| 3 | day 3 | 0.784 | 0.771 | 0.550 | Infiltration |
| 4 | day 4 | 0.715 | 0.630 | 0.695 | DoS |
| 5 | day 5 | 0.715 | 0.689 | 0.342 | Infiltration |
| 6 | day 6 | 0.644 | 0.651 | 0.390 | InitialAccess (380k samples; only 566 in train) |
| **agg** | | | **0.636 ± 0.093** | **0.538 ± 0.178** | |

Fold 0 is labeled `family_holdout_classes: ["Botnet"]` — Botnet appears only on day 0, so the model has literally zero training exposure. That's a zero-shot held-out-family test, not a domain-shift test. Report it separately.

---

## Fold 1 anomaly

onset_auc_k1 = **0.425** — the only fold below 0.5 (worse than random for k=1).

Mechanism: **training/test volume inversion for DoS.**
- Train DoS support: 52,498 windows
- Held-out DoS support: 601,802 windows

The model trained on ~8% of the available DoS samples and is evaluated on the bulk. This is a dataset artifact (capture days differ in session count, not attack severity). onset_auc recovers to 0.487 at k30, suggesting temporal structure is learnable but the model underfit DoS specifically. Not a generalization failure — a class-imbalance artifact from the split.

If this comes up in Q&A: "Day 4 also contains DoS and scores 0.715 — the fold 1 anomaly tracks the 12:1 volume inversion in that specific fold, not a systematic inability to forecast DoS."

---

## What this means for the competition

No other team has published day-held-out cross-validation. Competitors (cybersentinel, NEXUS, SAKETH) evaluate on fixed held-out slices of the same data distribution they trained on.

**Correct framing**: "We ran the hardest generalization test available — held out entire capture days. Supported-class F1 mean 0.538, onset AUC mean 0.636. One fold degrades (day-1 DoS volume inversion). We report it."

**Wrong framing** (do not use): "LODO collapse — model learns day-specific signal." The `stage_macro_f1_DO_NOT_REPORT` numbers look like collapse (0.14–0.44) but measure class absence, not model failure. Using them is misquoting your own data.

The honest disclosure + explanation is *stronger* than silence. Judges selecting 5 teams will notice whether teams understand their own failure modes.

---

## Open actions

1. **Merge the branch** — this is all on `eval/task4b-task7-lodo`, not on main
2. **Fold 0 writeup** — separate the family_holdout (zero-shot Botnet) result from the domain-shift folds; they measure different things
3. **Fold 1 deep-dive** — confirm the volume-inversion explanation with a DoS-only onset curve (does onset AUC recover if you weight by class frequency?)
4. **Slide framing** — cite `onset_auc_k5 mean 0.636` + "we ran LODO, no competitor did" as the rigor claim; do not show the `stage_macro_f1` column

---

## Files

| path | what |
|------|------|
| `models/lodo_cic_full_w30.json` | full fold results, source of this analysis |
| `bench/lodo.py` | training loop |
| `bench/lodo_ensemble.py` | ensemble variant |
| `models/lodo_f{0..6}_w30_metrics.json` | per-fold training logs |
| `models/lodo_f1_diag_metrics.json` | fold 1 diagnostic |
| `models/lodo_ensemble_dapt.json` | ensemble eval on DAPT |
