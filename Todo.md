# Todo.md — competitive-gap work order

**This file is a prompt.** Hand it to a coding agent (or work it yourself) top to
bottom. It specifies eight tasks that close measured gaps against the SIH26153
field. Each task states what to build, where, how to verify it, and what counts
as done.

Read `STATUS.md` first. It is the authoritative description of the current state
and it overrides `README.md`, `MASTER.md`, `AI_HANDOFF.md`, `PROJECT.md`,
`IDEAS.md`, `ARSENAL.md`, `WINNING_PLAN.md`, `BUILD.md` and
`IMPLEMENTATION_GUIDE.md` wherever they disagree.

---

## Ground rules

1. **Do not invent numbers.** Every figure that lands in a doc must come from a
   committed JSON under `models/` that a reader can regenerate with the command
   printed next to it. If a task cannot be run because the data or a checkpoint
   is missing, say so explicitly in the task's output file and leave the number
   blank. A blank is acceptable; a plausible-looking fabrication is not.
2. **No hardcoded absolute paths.** `STATUS.md` records that `eval_dapt.py`,
   `dapt_entity_eval.py`, `pipeline.py` and `train.py` all shipped with
   `~/netrikan/...` baked in and one of them crashed on save after computing
   every result. Use `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`,
   as `transition_eval.py` and `calibration.py` already do.
3. **Respect capture-file boundaries.** `pipeline_v2.load_all()` tags each row
   with `_file`. No window, onset label, or evaluation event may span two
   capture days. Any new code that windows or pairs rows must honour this.
4. **Time-sort before anything temporal.** The CIC CSVs are not stored in time
   order. `load_all()` now sorts; keep it that way and do not add a path that
   bypasses it.
5. **Record negative results.** If an ablation shows the model does worse, that
   goes in the output JSON and in `STATUS.md` with the same prominence as a
   win. This project's credibility rests on that habit — see the CTU-13 world
   model verdict (`PARTIAL`) and the DAPT recall regression already documented.
6. **One task, one commit.** Commit message names the task number and the JSON
   it produced.

### Environment

```bash
# Locate the built dataset directory (contains X.npy, y.npy, onset_k*.npy).
# The CIC full run used a --data directory built by src/pipeline_v2.py.
# Set it once and reuse:
export DATA=/path/to/netrikan-cic-full-w30     # FIND THIS FIRST; do not guess
export CKPT=models/cic_full_w30.pt
```

If `$DATA` no longer exists on this machine, rebuild it before starting:

```bash
python3 src/pipeline_v2.py --out "$DATA"        # reads configs/train_v2.yaml (window: 30)
```

Reference numbers to beat or reproduce, all from
`models/cic_full_run_summary.json`: onset AUC k5 **0.9018**, stage macro-F1
**0.8357**, transition-window macro-F1 **0.3913** vs persistence **0.0000**,
rollout MSE **0.605** vs persistence 0.782 and window-mean 0.649, ECE
**0.0045 → 0.0028** at T=1.032.

---

## Task 1 — Lead time in seconds

**Why.** We have no lead-time number at all. Four competitors lead with one
(NetraVerse reports median lead seconds per event; HORIZON makes lead-time-vs-FPR
its headline chart; What-the-hack formalises Mean Lead Time; Drishti claims 6.9
min vs 1.3 min for logreg). The problem statement is about early warning, and
onset AUC does not answer "how much warning". This is the single largest gap.

**Build** `bench/lead_time.py`.

Definitions, to be stated in the file's docstring and kept fixed:

- An **onset event** is an index `t` where the stage label changes from Benign
  to any attack class, within one `_file`. Non-benign→non-benign transitions are
  a separate event class; report them separately, do not merge.
- A **warning** for event at `t` is the earliest index `s < t` such that the
  onset-head probability at the chosen horizon exceeds threshold `θ` and stays
  above it for `min_persist` consecutive windows (default 2, to suppress
  single-window flicker). Lead time is `(t - s) * seconds_per_window`.
- Derive `seconds_per_window` from the actual median inter-flow time in the
  window, not from an assumed constant. Record the value used in the output.
- An event with no qualifying warning before `t` contributes to **coverage**,
  not to the lead-time distribution. Report both; a high median lead over 3 of
  40 events is not a result.

**Sweep θ over the full range and report, for each of FPR ∈ {0.001, 0.01, 0.05}:**
threshold, coverage (fraction of events warned), median lead seconds, p25/p75
lead, and mean lead. FPR is measured on benign windows that precede no event
within the horizon.

**Baselines in the same table, same thresholds:** persistence (by construction
0 coverage, since it never predicts a change — state this, do not omit the row)
and the flattened-window logistic regression from `src/baseline.py`.

**Run:**
```bash
python3 bench/lead_time.py --data "$DATA" --model "$CKPT" \
    --horizons 1 5 15 30 --out models/lead_time_cic_full_w30.json
```

**Done when** `models/lead_time_cic_full_w30.json` exists with a lead-time
distribution per horizon per operating point, event counts are stated, and
`STATUS.md` carries the table. If coverage is low, say so in the finding field.

---

## Task 2 — PR-AUC and recall at fixed FPR

**Why.** We report ROC-AUC on the onset head. At our positive rate ROC-AUC is
flattering, and a judge who knows this will ask. NetraVerse reports PR-AUC plus
`recall_at_fpr_0.001 / 0.01 / 0.05` with the achieved FPR next to each — the
correct presentation for a rare-event forecaster. We will still win on PR-AUC
because we have real positives in volume; we just have to show it.

**Extend** the onset evaluation (wherever `onset_auc_k*` is currently computed —
check `src/train_v2.py` and `src/eval_dapt.py`) and add a standalone
`bench/onset_pr.py` so it can be re-run without retraining.

For each horizon k ∈ {1, 5, 15, 30} emit:

- positive rate and absolute positive count (the denominator matters)
- ROC-AUC (keep it, for continuity with existing numbers)
- **PR-AUC** (average precision)
- **recall at FPR 0.001, 0.01, 0.05**, each with the threshold and the
  *achieved* FPR, since the exact target is rarely attainable
- the no-skill PR baseline (= positive rate), so PR-AUC is readable

**Run:**
```bash
python3 bench/onset_pr.py --data "$DATA" --model "$CKPT" \
    --out models/onset_pr_cic_full_w30.json
```

**Done when** the JSON exists, and `STATUS.md` §0 reports PR-AUC alongside the
existing onset AUC table rather than replacing it.

---

## Task 3 — Shuffled-history control

**Why.** The strongest available critique of this project is "the LSTM is
reading a static host or environment fingerprint, not temporal dynamics." One
ablation rebuts it. NetraVerse ran exactly this and their PR-AUC collapsed
0.077 → 0.006, which is the most persuasive single row in their results file.
It is cheap: no new model, no new data.

**Build.** Add `--shuffle-history` to `src/train_v2.py` (and whatever dataset
class it uses). When set, permute the timestep axis **within each window**,
independently per sample, with a fixed seed. Labels, onset labels and the split
are untouched. The window keeps exactly the same rows; only their order is
destroyed.

Train a full run with the flag and compare against the deployed checkpoint's
numbers on identical splits:

```bash
python3 src/train_v2.py --data "$DATA" --tag shuffled_w30 --shuffle-history
python3 bench/onset_pr.py --data "$DATA" --model models/shuffled_w30.pt \
    --out models/onset_pr_shuffled_w30.json
```

**Interpretation, to be written down whichever way it goes:**

- Large collapse → the model reads temporal order. This is the result we want
  and it belongs in the headline.
- Little or no collapse → **the LSTM is not using sequence information**, our
  +0.144 "value of LSTM on top of flattened logreg" needs re-examination, and
  that must be reported. Do not bury it.

**Done when** `models/shuffled_w30.pt` and its PR JSON are committed and
`STATUS.md` has an "ordered vs shuffled" row with both numbers.

---

## Task 4 — Held-out attack family

**Why.** Generalization to an unseen attack family is the PS's implicit ask and
we have never tested it. We are also sitting on a ready-made story: Infiltration
is our worst class on all windows (F1 **0.415**) but our *best relative* class on
transitions (**0.562**). If the model can forecast Infiltration onsets having
never trained on Infiltration, that is a strong claim. NetraVerse ran this
(held out Infiltration, PR-AUC 0.049/0.067/0.101) but on 3/5/7 test positives and
had to label it "indicative, not precise". We have the volume to do it properly.

**Build** `bench/heldout_family.py`.

- Take one stage id as `--holdout` (start with Infiltration = 3, then repeat for
  Botnet = 4).
- Remove from the **training split** every window whose label is the held-out
  class *and* every window whose onset label at any horizon is triggered by a
  transition into it. Leaving the onset labels in is a leak — the model would
  learn to predict the class it supposedly never saw.
- Retrain with the same config and seed, then evaluate **only** on held-out-class
  onset events in the validation split.
- Report the same metric set as Task 2, plus the absolute positive count. If
  positives are few, print the count next to every number and say the estimate
  is noisy.

**Run:**
```bash
python3 bench/heldout_family.py --data "$DATA" --holdout 3 \
    --out models/heldout_infiltration_w30.json
python3 bench/heldout_family.py --data "$DATA" --holdout 4 \
    --out models/heldout_botnet_w30.json
```

**Done when** both JSONs exist, each states its positive count, and `STATUS.md`
has a held-out-family section.

---

## Task 5 — Train and commit `world_w30.pt`, and prove the rollout is honest

**Why.** `STATUS.md` §2 says this is the one thing that is actually broken:
`models/world_w30.pt` does not exist, so a fresh clone falls back to
`DOCTRINE_SHAPE` (a hand-written ATT&CK constant matrix) and reports
`rollout_source: "markov_fallback"`. `MASTER.md` §13 currently instructs the
reader to describe the forecast as a neural rollout, which on a clone is false.

There is a second half to this. HowSuyash/AttackForecast ships
`tests/smoke_model.py`, which asserts that an imagination-only loss produces
**zero gradient in the encoder** — a mechanical proof the K-step rollout is not
secretly reading observations. We have no equivalent, and our CTU-13 rollout
came back `PARTIAL` (beat persistence, lost to the window mean, reached a fixed
point with stage volatility 0.0001). That is exactly the failure such a test
catches.

**Do both:**

```bash
# a) train and commit the checkpoint (.gitignore has already been fixed to allow it)
python3 src/train_v2.py --data "$DATA" --tag world_w30 --state-weight 0.3
git add -f models/world_w30.pt

# b) re-run the rollout benchmark against it
python3 bench/rollout_eval.py --data "$DATA" --model models/world_w30.pt \
    --steps 10 --out models/rollout_eval_world_w30.json
```

**c) Add `tests/test_rollout_isolation.py`:** build a loss from
`WorldModel.rollout()` outputs only, backpropagate, and assert that
`model.lstm` parameter gradients are `None` or all-zero for the steps past the
observed window — i.e. no gradient path from an imagined state back to an
observation the model should not be using at that step. If the current
`rollout()` cannot satisfy this, that is a finding: write it down, do not weaken
the test to pass.

**Then correct the docs.** After the checkpoint is committed, update
`MASTER.md` §13 so the neural-rollout claim is conditioned on
`result['rollout_source'] == "learned"`, and update `STATUS.md` §2 to reflect
that it is fixed. If the rollout benchmark again returns `PARTIAL` or worse
against the window-mean baseline, the claim stays qualified in both files.

**Done when** `models/world_w30.pt` is tracked, the rollout JSON is committed,
the isolation test passes or its failure is documented, and §2 of `STATUS.md`
no longer describes a broken repo.

---

## Task 6 — Raise `MODEL_WEIGHT`, defended by the ablation we already ran

**Why.** `src/infer.py:59` sets `MODEL_WEIGHT = 0.30` — 30% model, 70%
hand-written rules — and `STATUS.md` marks it **unmeasured**. It is no longer
unmeasured. `bench/model_weight_ablation.py` on DAPT 2020 found binary-detection
SEDI rising monotonically with model weight: pure rules (0.0) scores **−0.256**,
pure model (1.0) scores **+0.050**. The 14 port-based detectors in `signals.py`
encode CIC's environment and do not transfer.

Meanwhile Divyansh1982006/Threatora makes "strict zero leakage — `srcip`,
`dstip`, `sport`, `dsport` and timestamps stripped into 16 canonical slots" a
headline design claim. Against that, a 70% hand-rule weight built on port groups
is the weakest point in our stack, and we have the measurement that proves it.

**Do:**

1. Re-run `bench/model_weight_ablation.py` and confirm the monotonic curve is
   reproducible on the current checkpoint. Commit the JSON.
2. Change `MODEL_WEIGHT` to the value the curve supports. Do not silently pick
   1.0 — report SEDI at each weight and choose defensibly. Put the chosen value
   and its justification in a comment at `infer.py:59` citing the JSON.
3. **Make it configurable**, not just re-hardcoded: `NETRIKAN_MODEL_WEIGHT` env
   var with the measured value as default, so a deployment on a network unlike
   CIC can raise it further without a code edit.
4. Re-run whatever evaluations depend on fusion output so the committed numbers
   match the new default. Do not leave `STATUS.md` quoting results from the old
   weight.

**Note the tradeoff honestly.** `STATUS.md` already records that the
onset-selected `cic_full_w30` checkpoint regressed DAPT attack recall to
**29.5%** from `base_w30`'s 79.7%. Changing the fusion weight interacts with
that. Report both axes; do not present a single improved number.

**Done when** `infer.py` reads the env var with a measured default, the comment
cites `models/model_weight_ablation_cic_full_w30.json`, and `STATUS.md`'s "Fusion
weight — unmeasured" row is replaced with the measured value.

---

## Task 7 — Leave-one-day-out cross-validation

**Why.** We report one purged blocked split (500 blocks, val_frac 0.2, seed 42).
That is correct but it is a single number with no variance and no domain shift.
manansheth296-tech/sentinel-net commits both leave-file-out and leave-day-out
results — including an honest fold where the held-out file had zero positives and
F1 came out 0.0. We have **seven CIC-IDS-2018 capture days** already tagged by
`_file` in the pipeline. The protocol costs us nothing but compute.

**Build** `bench/lodo.py`.

- For each of the 7 days: train on the other 6, evaluate on the held-out day.
- `pipeline_v2.load_all()` already writes `_file`; use it as the fold key rather
  than re-deriving day boundaries.
- Report per fold: onset PR-AUC per horizon, stage macro-F1, transition-window
  macro-F1, and the **class support of the held-out day** — several CIC days
  contain only one attack family, so a fold with zero positives for a class is
  expected and must be reported as such, not averaged away.
- Report mean ± std across folds, and name the worst fold explicitly.

**Run:**
```bash
python3 bench/lodo.py --data "$DATA" --out models/lodo_cic_full_w30.json
```

**Done when** the JSON has all 7 folds with support counts, `STATUS.md` reports
mean ± std next to the existing single-split numbers, and the worst fold is
named in prose.

---

## Task 8 — Cheap wins: multi-seed, ONNX, and the latency number

Three small items, one commit each.

**8a — Five-seed training.** All headline numbers are single-seed (`seed: 42` in
`configs/train_v2.yaml`). csxzor-devcs/sih runs 5 seeds and ships
`scripts/reproduce.sh`. Train seeds 42, 43, 44, 45, 46 and report mean ± std for
onset AUC k5, onset PR-AUC k5, stage macro-F1 and transition macro-F1 into
`models/seed_variance_cic_full_w30.json`. If the spread is wide, that is the
finding and every headline number needs an error bar. Add `scripts/reproduce.sh`
running the full chain end to end.

**8b — ONNX export.** Threatora ships ONNX opset 17 at **0.35 MB** and **<1.5 ms
per window**, and makes air-gapped CPU deployment a headline. We ship a `.pt` and
a Streamlit app and have no latency number at all. Add `src/export_onnx.py`
exporting `WorldModel` (all four heads; note `attention=False` must match the
checkpoint — `model.py` warns that a checkpoint trained without attention must
not be loaded with it). Verify output parity against PyTorch to within 1e-4 and
fail the export if it is not.

**8c — Latency and footprint benchmark.** `bench/latency.py`: median, p95 and p99
per-window inference latency on a single CPU thread, throughput in windows/sec,
and model file size, for both the PyTorch and ONNX paths. Write
`models/latency.json`. Report the hardware. For calibration against the field:
cybersentinel-ai reports 17.6 ms and 56.8 events/sec, Threatora claims <1.5 ms
and >5,000 windows/sec.

**Done when** the three JSONs and `scripts/reproduce.sh` are committed and
`STATUS.md` has a performance section.

---

## Final step — reconcile the docs

After all eight tasks, update `STATUS.md`:

- Refresh the "Last reconciled against the source" date.
- Add the new numbers to §0.
- Clear the fixed items from §2 and §3.
- Add a short section listing anything that came out negative, in the style of
  the existing "Honest tradeoffs found this run".

Then re-read `MASTER.md`, `README.md` and `WINNING_PLAN.md` for claims the new
measurements have falsified, and fix them. `STATUS.md` exists precisely because
those files have drifted from the code before.

---

## Ordering

Tasks 1–4 are independent and can run in parallel. Task 5 should start early
because it needs a training run. Task 6 depends on nothing but should land after
5 so the fusion change is evaluated against the committed world model. Tasks 7
and 8a are compute-bound — queue them. 8b and 8c are quick and can be done any
time.

Priority if time is short: **1, 3, 5, 2** — lead time, the shuffled control, the
missing checkpoint, and PR-AUC. Those four close the gaps that a knowledgeable
reviewer will find first.
