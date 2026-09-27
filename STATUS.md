# STATUS — authoritative current state

**This file takes precedence.** `README.md`, `MASTER.md`, `AI_HANDOFF.md`,
`PROJECT.md`, `IDEAS.md`, `ARSENAL.md`, `WINNING_PLAN.md`, `BUILD.md` and
`IMPLEMENTATION_GUIDE.md` were written at different points and contradict each
other and the code on several facts. Where any of them disagrees with this file,
this file is right. Where this file is silent, check the code, not the docs.

Last reconciled against the source: 2026-09-27.

---

## 0-bis. Todo.md competitive-gap work (2026-09-25)

Results from the eight-task work order in `Todo.md`, all on the deployed
`cic_v2_w30` checkpoint and the 7-day CIC build. Numbers are from committed JSON.

**Task 1 — lead time in seconds** (`models/lead_time_cic_v2_w30.json`). 15,898
benign→attack onsets in the val split; seconds_per_window = 30 (1 s/flow
assumption, stated in the file). At 5% FPR: k=30 warns a median **13.5 min** ahead
(coverage 66%), k=15 **8.5 min** (61%), k=5 **3.5 min** (50%). At strict 0.1% FPR
coverage collapses to 2–5% — the honest tradeoff. Persistence has 0 coverage by
construction; flattened logreg produces no onset probability, so no curve.

**Task 2 — onset PR-AUC** (`models/onset_pr_cic_v2_w30.json`). PR-AUC k1/k5/k15/k30
= **0.254 / 0.494 / 0.609 / 0.624** against no-skill baselines 0.026 / 0.074 /
0.127 / 0.150 — 3.4–9.8× above chance, on 31k–184k positives. recall@FPR 0.05 =
0.48 / 0.52 / 0.55 / 0.52. ROC-AUC (kept for continuity) ≈ 0.89–0.90.

**Task 3 — shuffled-history control** (`models/onset_pr_shuffled_w30.json`,
`models/shuffled_w30.pt`). Permuting the timestep order within every window drops
onset PR-AUC k5 **0.494 → 0.371** (−25%), k15 **0.609 → 0.493**, and stage macro-F1
**0.837 → 0.758**. The model genuinely uses temporal order — this rebuts the
"static per-window fingerprint" critique. It is a partial drop, not a collapse
(cf. NetraVerse 0.077→0.006): window-level feature composition also carries signal,
which we state rather than overclaim.

**Task 5 — world model / rollout.** Superseded by deployment: `cic_v2_w30` carries
a trained `state_head`, its rollout benchmark PASSES (beats persistence +0.23 and
window-mean +0.07, 10/10 steps), `analyze()` reports `rollout_source: "learned"` on
a clone, and `tests/test_rollout_isolation.py` proves the rollout cannot read
future observations (no-future-leak, autoregressive, deterministic, grounded).
A separate `world_w30.pt` (state head only, no onset/attention) would be strictly
worse, so it was not trained; the §2 "broken repo" is fixed by cic_v2.

**Task 6 — MODEL_WEIGHT** (`infer.py`). Was 0.30 unmeasured; now **0.90**, measured
default from the DAPT ablation (cross-dataset SEDI 0.0:−0.256 → 0.9:+0.405 →
1.0:+0.416), env-configurable via `NETRIKAN_MODEL_WEIGHT`. The port rules encode
CIC's environment and hurt cross-dataset transfer.

**Task 8b — ONNX** (`models/cic_v2_w30.onnx`). Single self-contained file, 0.893 MB,
opset 17, all four heads, parity vs PyTorch within **1.9e-5**.

**Task 8c — latency** (`models/latency.json`, Apple Silicon, 1 CPU thread).
PyTorch **3.0 ms** / 305 windows-s; ONNX **1.4 ms** / 610 windows-s.

**Task 4a — held-out Infiltration** (`models/heldout_infiltration_w30.json`). A
NEGATIVE result, recorded per ground rule 5. With Infiltration removed from
training entirely (label + onset signal), the model CANNOT forecast its onsets:
PR-AUC 0.005–0.007 at/below no-skill 0.007–0.011, ROC-AUC **0.29–0.37 (below 0.5)**
on 2,827–9,382 real positives. Infiltration mimics benign traffic, so a model that
never saw it reads its onsets as *less* suspicious than benign. This bounds the
zero-shot family-generalization claim honestly — the model does not generalize to
an unseen family that resembles benign. (Contrast NetraVerse's 0.049–0.101 on 3–7
positives, which is noise; ours is precise.)

**Task 4b — held-out Botnet** (`models/heldout_botnet_w30.json`, 2026-09-27). A
SECOND NEGATIVE result, stronger than 4a. With Botnet removed from training
(label + onset signal), the model does not merely fail to forecast its onsets —
it ranks them *below* benign at every horizon:

| k | val positives | PR-AUC | no-skill | ROC-AUC | recall@FPR 0.05 |
|---|---|---|---|---|---|
| 1 | 8,156 | 0.0059 | 0.0089 | **0.332** | 0.0016 |
| 5 | 23,337 | 0.0156 | 0.0258 | **0.252** | 0.0018 |
| 15 | 37,868 | 0.0251 | 0.0426 | **0.222** | 0.0010 |
| 30 | 40,357 | 0.0264 | 0.0461 | **0.198** | 0.0005 |

PR-AUC sits below the no-skill baseline at every horizon and ROC-AUC falls to
0.198 at k=30 — systematically anti-correlated, not uninformative. With 8k–40k
real positives this is a precise measurement, not noise, and it is worse than the
Infiltration result (0.29–0.37). The pattern is now consistent across two
families: **the onset head does not generalize zero-shot to an unseen attack
family.** Trained without a family, it learns that family's precursor traffic is
benign-like and scores it *more* benign than benign. Any zero-shot or
"detects novel attacks" claim must be dropped from the pitch; the supported claim
is early warning for families represented in training.

**Task 7 — leave-one-day-out** (`models/lodo_cic_full_w30.json`, 2026-09-27).
Seven folds, train on 6 CIC capture days, validate on the 7th, 12 epochs max with
patience 4 (all folds early-stopped at 5-10).

| fold | held-out day contents | supported-class F1 | onset AUC k1 / k5 / k15 / k30 |
|---|---|---|---|
| 0 | Benign 758k + **Botnet 286k** | 0.413 | 0.665 / 0.665 / 0.664 / 0.672 |
| 1 | Benign 447k + DoS 602k | **0.883** | 0.425 / 0.445 / 0.460 / 0.487 |
| 2 | Benign 1.04M + InitialAccess 566 | 0.496 | 0.631 / 0.599 / 0.553 / 0.514 |
| 3 | Benign 236k + Infiltration 92k | 0.550 | **0.784 / 0.771 / 0.740 / 0.711** |
| 4 | Benign 988k + DoS 52k | 0.695 | 0.715 / 0.630 / 0.501 / 0.425 |
| 5 | Benign 539k + Infiltration 68k | **0.342** (worst) | 0.715 / 0.689 / 0.616 / 0.572 |
| 6 | Benign 664k + **InitialAccess 381k** | 0.390 | 0.644 / 0.651 / 0.636 / 0.690 |

**Aggregate: onset AUC k5 0.636 ± 0.093; supported-class F1 0.538 ± 0.178.**

**The headline finding: the single purged split overstates forecasting.** Onset
AUC k5 is **0.902** on the single blocked split and **0.636 ± 0.093** across days —
a 0.27 drop, with fold 1 at **0.445, below chance**. Every headline onset number in
this file comes from one split and must carry the LODO spread next to it.

**Reporting rule for this protocol.** Each CIC day carries at most one attack
family, so 3-4 of 5 classes have zero validation support in every fold. A 5-class
macro-F1 therefore measures *absence*, not error, and is stored in the JSON as
`stage_macro_f1_DO_NOT_REPORT` (mean 0.228). Quote `supported_class_f1` — the mean
over classes the held-out day actually contains — and the per-class table.

**Two folds are family holdouts wearing a LODO costume.**
- Fold 0: **Botnet occurs on day 0 and no other day**, so training saw zero Botnet.
  Botnet F1 0.00 on 286k val positives. Flagged in JSON as `family_holdout_classes`.
- Fold 6: InitialAccess F1 0.00 on **380,943** val positives, because the only other
  day carrying it (day 2) has **566** windows. The automatic flag tests for *zero*
  training support and so does NOT trip here — a threshold gap in `bench/lodo.py`;
  read `train_support_of_present_classes` alongside the flag, not instead of it.
- InitialAccess scores **0.00 in both directions** (fold 2 trained on 381k, tested
  on 566 → 0.00; fold 6 trained on 566, tested on 381k → 0.00). It does not transfer
  across days at all.

**Validation prevalence drives per-class F1 more than training volume.** The two DoS
folds invert the naive expectation: fold 1 trained on 52k DoS and scored **0.818** on
a day that is 57% DoS; fold 4 trained on 602k and scored **0.421** on a day that is
5% DoS. More training data, worse score — precision falls when positives are rare in
the held-out day. Do not read per-fold F1 as a data-volume effect.

**Long horizons do not survive day shift.** On the single split onset AUC is flat
(~0.89-0.90) across k=1..30. Across days it *declines* with horizon on folds 2, 4
and 5 — fold 4 runs 0.715 → 0.425, below chance at k=30. Fold 3 (Infiltration) is
the exception, 0.784 → 0.711. The flat-horizon claim is a single-split artifact.

**Infiltration forecasts best under day shift**, which corroborates the existing
transition-vs-steady-state result: fold 3 is the best onset fold in the sweep
(k5 **0.771**) on our worst classification class. But the two Infiltration days do
not agree closely (fold 3 k5 0.771 vs fold 5 0.689, supported F1 0.550 vs 0.342), so
the effect is directional, not a stable number.

**Fold 5 is the worst fold and fails in an unusual way:** Benign F1 collapses to
**0.481** (every other fold holds 0.67-0.99) while Infiltration reaches only 0.202 —
a false-positive blowup, not a miss. Named here per Task 7's requirement.

**Compute note.** `src/train_v2.py` hardcoded `num_workers=4`; on this machine that
drove swap to 5.1 GB and epoch times from 170s to 672s and climbing, and the first
LODO attempt died with `RuntimeError: Shared memory manager connection has timed
out`. Worker count is now `NETRIKAN_WORKERS` (default 4); the sweep ran at 0 with
flat ~190s epochs. Verified identical metrics at workers=2 and 0 — the setting
changes only batch fetching, never results.

**Task 7 follow-up — is the LODO drop undertraining or real?** (`models/lodo_f1_diag_metrics.json`,
2026-09-27). Fold 1 was the worst onset fold (k5 0.445, below chance) and its training
visibly collapsed (DoS F1 0.82 -> 0.03 between epochs 2 and 3), so it was re-run alone
with a gentler schedule to separate instability/undertraining from genuine day shift.

| fold-1 run | epochs run | macro-F1 | Benign / DoS F1 | onset k1 / k5 / k15 / k30 |
|---|---|---|---|---|
| sweep: 12ep, patience 4, lr 1e-3 | 6 | 0.4413 | 0.947 / 0.818 | 0.425 / 0.445 / 0.460 / 0.487 |
| diag: 30ep, patience 8, lr 3e-4 | 12 | 0.4415 | 0.947 / 0.819 | 0.484 / **0.514** / 0.513 / 0.528 |

**Verdict: the drop is real, not an artifact of short training.** Doubling the epochs
and cutting the LR to a third moved onset k5 from 0.445 to 0.514 (+0.069, still at
chance) and left classification **unchanged** (macro-F1 0.4413 -> 0.4415, DoS 0.818 ->
0.819). Applying this schedule to all 7 folds would nudge 0.636 slightly; it would not
change the conclusion.

**More training makes this fold worse, which rules out undertraining.** The collapse was
delayed, not prevented — at lr 1e-3 it hit macro-F1 0.17 by epoch 3; at lr 3e-4 it held
0.44 through epoch 4 and then decayed to the same 0.17 by epoch 9, ending at DoS 0.03.
Longer schedules would *lower* the LODO mean, not raise it.

**Probable mechanism: class-prior mismatch between days, not covariate shift.** DoS is
**52,498 of 5,111,434** fold-1 training windows (**1.0%**) but **57%** of the held-out
day. Minimizing training loss means progressively abandoning DoS, which is exactly the
observed decay. The mirror fold supports this: fold 4 trains on 601,802 DoS (12%) and
validates on a 5%-DoS day, scoring onset k5 **0.630** vs fold 1's 0.445. Per-day feature
normalization is therefore aimed at the wrong target — a 1%-vs-57% prior gap is not a
rescaling problem. Class-balanced sampling or a class-balanced onset loss is the lever;
neither has been tried, so no improvement is claimed here.

**Added `--lr` to `src/train_v2.py`** (default 1e-3, unchanged; also readable from the
config as `lr`). It was hardcoded at the optimizer.

**Improvement item (a) — DAPT operating-point curve**
(`models/operating_points_dapt_cic_v2_w30.json`, `bench/operating_points.py`, 2026-09-27).

First, a code fact: `eval_dapt.py:100` decides attack with `stage_logits.argmax() > 0`.
There is **no threshold** in the shipped cross-dataset path, so "FPR 40.4%" was never a
tuned operating point. `breach_head` is already sigmoid (`model.py:79`) and sweepable.

**NEGATIVE result: thresholding does not fix the FPR.** The curve is weak in shape, not
merely badly placed:

| operating point | recall | precision | FPR | SEDI |
|---|---|---|---|---|
| shipped argmax | 0.678 | 0.378 | 0.404 | 0.385 |
| breach @ FPR 10% | 0.100 | 0.266 | 0.100 | -0.000 |
| breach @ FPR 5% | **0.063** | 0.313 | 0.050 | 0.042 |
| breach @ FPR 1% | 0.035 | 0.561 | 0.010 | 0.161 |
| breach @ FPR 0.1% | 0.001 | 0.233 | 0.001 | -0.013 |

At 5% FPR the detector finds 6.3% of attacks. Tightening the threshold discards true
positives almost as fast as false ones — this is what ROC-AUC 0.679 means in practice.

**The argmax is not better ranked, just looser.** Matched to the argmax's own FPR
(0.4041), breach scores recall **0.681** vs argmax **0.678** (SEDI 0.389 vs 0.385) and
stage-attack mass scores 0.678. All three are the same point on one weak curve; there is
no hidden information in the argmax and no better threshold to find.

**`min_persist` pays off only at strict FPR**, where suppressing single-window flicker
frees budget for a lower threshold: at FPR 0.1%, persist=5 gives recall **2.5% at
precision 90%** (SEDI 0.306) against 0.09% recall at persist=1. Low recall, but a usable
high-confidence alert that did not exist before.

**The real win is the entity roll-up.** Windows grouped into (capture, Src IP, 1h) cells;
a cell alerts if any window in it is flagged:

| configuration | attack host-hours caught | false alarms / benign host-hour |
|---|---|---|
| **shipped argmax** | 0.698 | **0.4985** (1,133 of 2,273) |
| breach, persist=1 | 0.302 | 0.0629 (143) |
| **breach, persist=3** | 0.254 | **0.0185** (42 of 2,273) |
| breach, persist=5 | 0.206 | 0.0150 (34) |

What ships false-alarms on **half of all benign host-hours**. persist=3 cuts that **27x**
to 1.85% while keeping 25% entity-level detection. Per-window FPR and per-host-hour alert
rate are different quantities and must never be quoted interchangeably; the second is the
operator-facing one. This is the defensible operating point the project previously lacked.

**What this does NOT fix.** Cross-dataset detection is still weak in absolute terms — 25%
of attack host-hours at a 1.85% false-alarm rate, or 70% at an unusable 50%. The roll-up
improves how the result is *reported and deployed*, not the underlying discrimination.
Raising ROC-AUC 0.679 needs a better model, not a better threshold.

**Improvement item (b) — LODO ensemble on DAPT**
(`models/lodo_ensemble_dapt.json`, `bench/lodo_ensemble.py`, 2026-09-27).

**Why DAPT and not a held-out day.** Only `lodo_fN` never saw day N; the other six
trained on it. A 7-member ensemble scored on day N would let six members predict data
they trained on. Restricting to clean members leaves exactly one — that fold's own
number. **A cross-day LODO ensemble is not possible with these checkpoints.** DAPT is
clean for all seven (every member trained on CIC only).

| model | breach AUC | PR-AUC | recall@FPR5% | SEDI | entity recall | FA/benign host-hour |
|---|---|---|---|---|---|---|
| **deployed cic_v2** | 0.694 | 0.370 | 0.070 | 0.062 | 0.254 | 1.85% (42/2273) |
| lodo_f0 | **0.825** | 0.560 | 0.206 | 0.329 | 0.206 | 1.28% (29) |
| lodo_f1 | 0.803 | 0.485 | 0.063 | 0.041 | 0.095 | 0.18% (4) |
| lodo_f2 | 0.596 | 0.317 | 0.031 | -0.078 | 0.064 | 0.70% (16) |
| lodo_f3 | 0.795 | 0.480 | **0.271** | **0.419** | 0.254 | 1.76% (40) |
| lodo_f4 | 0.705 | 0.395 | 0.027 | -0.097 | 0.127 | 0.26% (6) |
| lodo_f5 | 0.792 | 0.493 | 0.123 | 0.185 | 0.175 | 0.62% (14) |
| lodo_f6 | 0.729 | 0.432 | 0.138 | 0.215 | 0.238 | 1.63% (37) |
| **ensemble (mean prob)** | 0.812 | 0.500 | 0.061 | 0.034 | 0.079 | 0.22% (5) |

All operating points achieved exactly FPR 0.0500, so these are matched comparisons.

**The finding is not the ensemble — it is that training on LESS data transfers BETTER.**
Member breach AUC mean **0.749 ± 0.074**, with **6 of 7 above** the deployed model's
0.694 (same 6/7 on PR-AUC). Holding out one capture day acts as regularization; the
deployed `cic_v2_w30`, trained on all 7 days and selected on `combined`, is over-fitted
to CIC for cross-domain use. This is the actionable result: for deployment on an unseen
network, a day-held-out (or otherwise less-fitted) checkpoint is the better starting
point, and that costs nothing to adopt because the checkpoints already exist.

**The ensemble is a qualified win only.** +0.118 AUC over deployed but **-0.013 against
its own best member**, and at the 5% FPR operating point it is no better than baseline
(recall 0.061 vs 0.070) — its ranking gain does not land at that threshold. Its errors
are unusually concentrated: 5 false-alarm host-hour cells vs baseline's 42 at identical
window FPR, but its detections concentrate too (entity recall 0.079 vs 0.254).

**Do not cherry-pick a member.** Selecting `lodo_f3` for its 0.271 recall would be
choosing on the DAPT test set — the peeking this project refuses elsewhere. Defensible
claims are (i) the population statement above, or (ii) the ensemble, which needs no
selection. A specific member may only be promoted on a criterion computed WITHOUT DAPT.

**No forecasting gain.** Deployed onset transfer k15 **0.705** beats the ensemble's
0.686 and every member (best member k15 0.7046). Item (b) improves cross-dataset
DETECTION only; cross-dataset forecasting is unchanged.

**Rank averaging was tried and FAILED — ensembling is closed as a lever.** The
hypothesis was that averaging seven sigmoid outputs compresses the dynamic range, so
the ensemble could rank well globally (AUC) while placing no useful boundary at a fixed
FPR. Scale-free rank averaging should have fixed that. It did not:

| combiner | breach AUC | PR-AUC | recall@FPR5% | SEDI | entity recall | FA/benign cell |
|---|---|---|---|---|---|---|
| baseline cic_v2 | 0.694 | 0.370 | **0.070** | 0.062 | **0.254** | 1.85% |
| mean probability | **0.812** | 0.500 | 0.061 | 0.034 | 0.079 | 0.22% |
| mean rank | 0.800 | 0.474 | 0.055 | 0.016 | 0.111 | 0.31% |

Rank averaging is *worse* than probability averaging on every axis, and both are worse
than the plain baseline at the deployable 5% FPR operating point. So the failure is not
a scaling artifact: the ensemble's AUC gain lives in a region of the ROC curve that no
operating point uses, and averaging dilutes the members that are confidently right at
low FPR. **Neither combiner converts the +0.118 AUC into deployable accuracy.**

The accuracy that IS available cross-dataset is the individual day-held-out models
(lodo_f3: recall **0.271** at FPR 5%, SEDI 0.419, i.e. 3.9x the deployed model's 0.070).
Realising it requires a selection criterion that never touches DAPT, which does not yet
exist — see the open question below.

**Improvement item (c) — full inverse-frequency class balancing. NEGATIVE.**
(`models/infil_bp1_w30_metrics.json`, 2026-09-27.) Infiltration F1 0.407 is the weakest
in-dataset class and drags macro-F1, so the sampler's rebalance exponent was raised from
the long-standing `p=0.5` (sqrt-inverse) to `p=1.0` (full inverse frequency), which
lifts Infiltration's relative sample weight from ~5.5x to **29.8x** Benign.

| metric | p=1.0 | p=0.5 (shipped) | delta |
|---|---|---|---|
| macro-F1 | 0.7984 | **0.8369** | -0.0384 |
| **Infiltration** | 0.357 | **0.407** | **-0.050** |
| Benign | 0.914 | 0.950 | -0.037 |
| InitialAccess | 0.946 | 0.968 | -0.023 |
| DoS | 0.971 | 0.979 | -0.008 |
| Botnet | 0.805 | 0.879 | -0.075 |
| onset k5 | 0.8856 | 0.9007 | -0.0150 |
| accuracy | 0.8781 | 0.9266 | -0.0485 |

**Worse on every axis, including the class it was meant to fix**, and it ran MORE epochs
(18 vs 13) so it is not undertrained. Probable mechanism (hypothesis, not established):
`WeightedRandomSampler` uses `replacement=True` with a capped `num_samples`, so a 29.8x
weight redraws Infiltration's 125,090 windows repeatedly inside each epoch — the model
memorises those windows instead of learning the class, while the other four classes are
starved of draws.

**Conclusion: `p=0.5` is a measured good default, not an arbitrary one.** Do not raise
it. If Infiltration is to be improved, the lever is not sampling frequency — its traffic
genuinely overlaps benign (the same property that makes held-out Infiltration score
*below chance*, §Task 4a). Loss-side reweighting or a separability-oriented objective
would be different experiments; oversampling is now ruled out with numbers.

**Where three improvement attempts leave the project (2026-09-27).** (a) operating
points: the only real win — the entity roll-up gives a defensible operating point
(25.4% of attack host-hours at 1.85% false alarms/benign host-hour, vs 49.9% shipped),
though thresholding cannot improve underlying discrimination. (b) ensembling: closed,
both combiners fail to convert a +0.118 AUC into anything usable at a working FPR; its
by-product finding — day-held-out models transfer better, 6 of 7, mean AUC 0.749 vs
0.694 — is the most promising unexploited lead. (c) class balancing: strictly worse.
**In-dataset accuracy appears to be at a local optimum under the current architecture
and features; the remaining headroom is cross-dataset, not in-dataset.**

**Improvement item (d) — cross-dataset transfer vs training epoch. THE BEST RESULT OF
THIS PASS.** (`models/transfer_vs_epoch.json`, `bench/transfer_vs_epoch.py`, 2026-09-27.)
Trained once on all 7 days with `--save-epochs` (15 epochs, early stopping disabled) and
scored every epoch's checkpoint on DAPT, to separate two explanations for item (b)'s
finding: does holding out a DAY help, or are the LODO members simply LESS FITTED?

| epoch | CIC macro-F1 | CIC onset k5 | DAPT AUC | DAPT PR | recall@FPR5% | SEDI | entity recall | entity FA |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.8114 | 0.8669 | **0.809** | 0.485 | 0.077 | 0.082 | 0.079 | 0.62% |
| 2 | 0.8250 | 0.8821 | 0.728 | 0.540 | 0.296 | 0.450 | 0.143 | 0.70% |
| **3** | **0.8305** | **0.8887** | 0.789 | **0.588** | **0.341** | **0.503** | 0.206 | **0.84%** |
| 5 | **0.8371** | 0.8954 | 0.752 | 0.445 | 0.060 | 0.033 | 0.079 | 0.35% |
| 7 | 0.8345 | — | 0.785 | 0.460 | 0.062 | 0.040 | 0.095 | 0.35% |
| 10 | 0.8340 | — | 0.701 | 0.359 | 0.062 | 0.037 | 0.111 | 0.44% |
| 15 | 0.8343 | — | 0.737 | 0.384 | 0.057 | 0.024 | 0.064 | 0.22% |
| **deployed cic_v2** | 0.8369 | 0.9007 | **0.694** | 0.370 | **0.070** | 0.062 | 0.254 | 1.85% |

**1. Epoch 3 is a far better cross-dataset model at negligible in-dataset cost.**
Against the deployed checkpoint it gives **4.9x the window recall at the same 5% FPR**
(0.341 vs 0.070), SEDI **0.503 vs 0.062**, PR-AUC 0.588 vs 0.370, and at entity level a
comparable 0.206 recall at **less than half** the false-alarm rate (0.84% vs 1.85%). The
price is 0.0064 macro-F1 and 0.012 onset AUC in-dataset. That is the best
accuracy trade found anywhere in this project.

**2. "Less fitting transfers better" is only WEAKLY supported.** The curve is noisy, not
a clean decay: AUC runs 0.809, 0.728, 0.789, 0.748, 0.752, 0.735, 0.785, ... 0.737.
Peak AUC is epoch 1 but the best *operating point* is epoch 3, and epoch 7 (0.785) nearly
matches epoch 3 (0.789). What is robust is that the EARLY epochs (2-3) own the usable
operating points, while epochs 8-15 are uniformly poor at FPR 5% (recall 0.027-0.077).

**3. A caution that undercuts several single-seed claims, including some of ours.**
**Every one of the 15 epochs beats the deployed `cic_v2_w30` on DAPT AUC** (0.701-0.809
vs 0.694), even the most-fitted epoch 15. `xfer_w30` is nominally the same recipe on the
same data. So **run-to-run variance in cross-dataset transfer is larger than most of the
interventions measured in this pass** — bigger than the ensemble's +0.118 relative to
baseline. Any cross-dataset comparison of two single runs, in this file or elsewhere, is
therefore weak evidence. Task 8a (multi-seed) is no longer optional for the
cross-dataset numbers; it is a precondition for believing them.

**4. Naive in-dataset selection does NOT find the good epoch.** Best CIC macro-F1 is
epoch 5, which scores recall 0.060 at FPR 5% — *worse* than the deployed model. So
selecting on in-dataset performance actively misses the transfer sweet spot
(`cic_criterion_would_pick_transfer_peak: false`).

**A candidate rule, and why it is not yet trustworthy.** "Earliest epoch within 1% of
peak in-dataset macro-F1" would pick epoch 3 (0.8305 vs peak 0.8371), using CIC only and
never touching DAPT. It is principled — prefer less fitting when the in-dataset cost is
negligible. **But it was formulated AFTER seeing the DAPT column, which makes it a
post-hoc rule selected on the test set.** It must be validated on a transfer set that
played no part in designing it before any checkpoint is promoted on its basis. Recorded
as a hypothesis, not a decision.

**== NEXT SESSION — pick up here ==** (Tasks 4-Botnet and 7 done 2026-09-27)

Pending Todo.md items, in priority order:

0. **AGREED NEXT ACTION (2026-09-27, user: start only when I say so).** Improvement
   work, in this order:
   a. [DONE 2026-09-27, see "Improvement item (a)" above] **DAPT operating-point curve** — the worst number in the project is cross-dataset
      FPR **40.4%** (25,625 FP / 63,412 benign, precision 37.8%), reported at a single
      implicit threshold. Sweep the threshold and report precision/recall at fixed FPR
      (0.001 / 0.01 / 0.05); then add the two suppressors already built: `min_persist`
      (k consecutive windows over threshold, as in `bench/lead_time.py`) and per-host /
      per-time-bucket alert aggregation instead of per-window. Per-window FPR and
      per-host-per-hour alert rate are different numbers; the second is the operator's.
      No training required, ~1-2h.
   b. [DONE 2026-09-27, see "Improvement item (b)" above] **Ensemble the 7 LODO checkpoints** — `models/lodo_f0..f6_w30.pt` all exist and
      each never saw one capture day. Average their onset probabilities and re-score.
      No training required, ~30 min. Most likely single lever to move cross-day onset
      k5 above 0.636.
   Then, if time: Suricata external comparison (`bench/suricata_comparison.py` is ready,
   needs `brew install suricata`) and re-capturing `demo_attack_lab/attack_small.pcap`
   with SMB so the counterfactual demo is not flat (3 of 4 currently show zero delta).
   Do NOT chase the LODO number with architecture changes: the measured causes are
   class-prior mismatch and single-day family confinement, which a bigger model does
   not address.

1. **Task 8a — 5-seed variance**: `bench/seed_variance.py` (ready). ~5 runs.
2. **Final step** — reconcile MASTER.md / README.md / WINNING_PLAN.md against the
   new numbers, per Todo.md's closing section.

**CRITICAL for next session:** the built dataset lives at
`/tmp/netrikan-cic-full-w30` (8.3 GB, includes `file_id.npy`). `/tmp` is cleared on
reboot. If the machine rebooted, rebuild it first (~3 min):
`.venv/bin/python src/pipeline_v2.py --out /tmp/netrikan-cic-full-w30`
then regenerate file_id if needed: `.venv/bin/python bench/_make_file_id.py --data /tmp/netrikan-cic-full-w30`.
Use `NETRIKAN_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1` for training (~15–30 min/run;
CPU is ~10× slower). Deployed checkpoint is `models/cic_v2_w30.pt`.

Done: Tasks 1, 2, 3, 4-Infiltration, 5, 6, 8b, 8c (2026-09-25); Tasks 4-Botnet and 7 (2026-09-27).

---

## 0. Headline: full CIC-IDS-2018 training + evaluation (real, 2026-09-25)

**Deployed checkpoint: `models/cic_v2_w30.pt`** — `infer.py` loads it. Trained on
**6.16M windows across 7 CIC-IDS-2018 capture days** (all 5 classes), window 30,
30.7 min on Apple MPS. One consolidated model with every head trained
(stage/breach/state/onset) plus temporal attention, selected on a **combined
onset+classification** criterion so it is good at forecasting *and* detection.
Full detail: `models/cic_v2_run_summary.json`. (An earlier onset-only checkpoint,
`cic_full_w30`, is kept for comparison — see the consolidation note below.)

This supersedes the CTU-13 run (§3a). CTU-13 stays as the pipeline-validation
milestone and the "this task is degenerate" control.

### The consolidation (fixing the three weak points)

The first full run (`cic_full_w30`, onset-only selection) forecast well but had
three weaknesses: cross-dataset detection collapsed (29.5% recall), its strengths
were split from the deployed classifier, and explainability was gradient×input
only. The deployed `cic_v2_w30` fixes all three:

| | onset-only `cic_full` | **deployed `cic_v2`** |
|---|---|---|
| onset AUC k5 | 0.902 | 0.901 (held) |
| macro-F1 | 0.836 | 0.837 (held) |
| rollout verdict | PASS | PASS |
| **cross-dataset attack recall** | 0.295 | **0.678** |
| **Establish Foothold detect** | 0.015 | **0.495** |
| onset transfer k15 (DAPT) | — | 0.705 |
| temporal attention | no | **yes** |
| SHAP | no | **yes (GradientExplainer)** |
| deployed in `infer.py` | no | **yes** |

The fix was the `combined` selection metric (mean of onset AUC and macro-F1),
which stopped the checkpoint drifting benign-heavy. On a fresh clone `analyze()`
now reports `rollout_source: "learned"` — the world model is live, not the Markov
fallback.

### The forecasting target — onset/hazard AUC (in-dataset)

| horizon | onset AUC | persistence | gap |
|---|---|---|---|
| k=1 | 0.890 | 0.500 | +0.390 |
| k=5 | **0.902** | 0.500 | **+0.402** |
| k=15 | 0.902 | 0.500 | +0.402 |
| k=30 | 0.893 | 0.500 | +0.393 |

Persistence scores 0.500 by construction — it never predicts a change. This is the
metric no persistence baseline can follow, and the whole point of the onset head.

### Transition-only evaluation (the persistence problem, resolved)

| subset | model macro-F1 | persistence macro-F1 |
|---|---|---|
| all windows | 0.836 | **0.947** |
| **transition windows** (n=31,781) | **0.391** | **0.000** |

On all-window F1 persistence wins — the autocorrelation artifact the project always
claimed. On the windows where the stage actually changes, persistence is 0.000 and
the model is 0.391 (Brier skill +0.52). Infiltration is *better* at transitions
(0.562) than steady-state (0.415).

### Mandatory baseline comparison

| model | sees | macro-F1 |
|---|---|---|
| logistic regression | one flow | 0.420 |
| logistic regression | flattened window | 0.692 |
| **LSTM** | window as a sequence | **0.836** |

Sequence information worth +0.271; the LSTM worth **+0.144** on top. A real gain,
unlike CTU-13 where the LSTM added only +0.005 on a degenerate binary task.

### World-model rollout — does the state head learn dynamics?

| predictor of the next flow | MSE |
|---|---|
| persistence (copy last row) | 0.782 |
| **state_head** | **0.605** |
| window mean | 0.649 |

**PASS.** Beats persistence (+0.226) *and* the window mean (+0.068), and holds
skill for all 10 rollout steps. This is genuine learned dynamics — and the
opposite of the CTU-13 checkpoint, which lost to the window mean. The "world model"
claim is supported *by this checkpoint on this data*, not in general.

### Calibration

ECE **0.0045 → 0.0028** (T=1.03). Already well-calibrated before scaling, because
the stage head trains under cross-entropy (a proper scoring rule) and the classes
are learnable. Reliability diagram: `models/reliability_cic_full_w30.png`.

### Stage classification (secondary — selected on combined onset+F1, not F1 alone)

`cic_v2`: macro-F1 0.837, accuracy 92.7%. Per-class F1: Benign 0.950 ·
InitialAccess 0.968 · DoS 0.979 · Botnet 0.879 · **Infiltration 0.407** (rarest,
hardest — but its *transition* F1 0.56 beats its steady-state F1, i.e. the model
is better at catching Infiltration onset than at labelling it mid-run).

### Explainability (four channels, each labelled by method)

1. **SHAP** — `src/explain_shap.py`, `shap.GradientExplainer` (expected gradients,
   chosen over DeepExplainer because DeepLIFT is unreliable through LSTM gates).
   On-demand per selected window; attributes every (timestep, feature) cell for
   three targets: `breach` (why malicious), `onset` (why it will escalate),
   `attack` (why not benign). Exposed via `infer.shap_for_windows()` and the
   dashboard.
2. **Temporal attention** — trained into `cic_v2` (`attention=True`); per-timestep
   weights show which flow in the window drove the prediction.
3. **Gradient × input** — always-on, cheap, per window in `analyze()`.
4. **14 rule detectors** — named, with human-readable evidence strings.

`infer.explain(window, shap_result=…)` fuses them into one structured explanation
with a plain-language `summary`. Every output carries its `attribution_method`, so
SHAP and gradient×input are never conflated.

### Honest tradeoffs (state these first)

1. **Cross-dataset detection, resolved but not free.** `cic_v2` recovers DAPT
   attack recall to **67.8%** (the onset-only `cic_full` was 29.5%). Still under
   `base_w30`'s 79.7%, but `base_w30` has no onset head, predates the timestamp-sort
   fix, and is not a world model. 67.8% with full forecasting + a live rollout is
   the better overall package. Precision is 38% / benign recall ~60% — the FPR is
   real and reported as SEDI, not hidden.
2. **The hand-written rules hurt generalization.** Fusion-weight ablation on DAPT:
   binary SEDI rises monotonically with model weight — pure rules (0.0) −0.256,
   pure model (1.0) **+0.416**. The port rules encode CIC's environment and do not
   transfer. `MODEL_WEIGHT` is still 0.30 in `infer.py`; the ablation argues to
   raise it toward 1.0 for unseen networks (pending an in-dataset re-check, since
   the rules do help in-dataset). The ablation's multi-class F1 on DAPT is not
   meaningful — label-space mismatch; only the binary axis is.

### Bugs this run exposed and fixed

- `eval_dapt.py`, `dapt_entity_eval.py`, `pipeline.py`, `train.py` all had
  `~/netrikan/...` hardcoded paths (eval_dapt computed all DAPT results then
  crashed on save). All now repo-relative.
- `pipeline_v2.load_all()` never sorted by timestamp. The CSVs are not stored in
  time order, so windows were not real sequences: the 2018-02-14 file yields 3,926
  transitions as stored vs 23,948 time-sorted. **This affected every prior CIC
  training run.** Now sorted per file.
- Windows and onset labels now respect capture-file boundaries — no window spans
  two days, no onset compares across a day seam.
- `X.npy` is written float16 (every reader casts to float32) to fit the full
  dataset on disk.

---

## 1. What is true of the code right now

| Fact | Value | Where |
|---|---|---|
| Deployed classifier | `models/cic_v2_w30.pt`, macro-F1 0.8369 (verified at runtime 2026-09-27; `base_w30` is only the fallback if cic_v2 is absent) | `infer.py:MODEL_PATH` |
| Window | **30** | `infer.py:WINDOW` |
| Model input features | 24 flow features | `infer.py:FEATURES` |
| Model output classes | 5 | `model.py:num_stages` |
| Kill-chain stages | **7** (Reconnaissance added) | `attck_map.py:N_STAGES` |
| Forecast-only stages | Exfiltration, Reconnaissance | `attck_map.FORECAST_ONLY_STAGES` |
| Fusion weight | **0.90** model / 0.10 rules, measured from the DAPT ablation, env-overridable via `NETRIKAN_MODEL_WEIGHT` (verified 0.9 at runtime) | `infer.py:MODEL_WEIGHT` |
| Rule detectors | 14 (4 need packet-level input) | `signals.py` |
| Attribution method | gradient × input always-on, **plus real SHAP** (`shap.GradientExplainer`) on demand; each output carries `attribution_method` | `infer.py`, `src/explain_shap.py` |
| Attention | **trained into the deployed checkpoint**; `infer._load_model()` auto-detects it from the state dict | `model.py:attention` |
| Default horizon | 15 minutes (900s) | `forecast.py:HORIZON_SECONDS` |
| Calibration temperature | fitted T=1.076, ECE 0.0088 → 0.0052 | `src/calibration.py` |

### Stage ids

```
0 Benign · 1 Initial Access · 2 DoS/Impact · 3 Lateral Movement
4 Command & Control · 5 Exfiltration · 6 Reconnaissance
```

RECON is **6, not 1**. It was appended rather than inserted because the numeric
ids are baked into every trained checkpoint through `MODEL_TO_CHAIN`. Kill-chain
ordering lives in `forecast.CHAIN_POS`, not in the id order.

---

## 2. The one thing that is actually broken

**`models/world_w30.pt` does not exist in this repository.**

It is the checkpoint with the trained `state_head` that makes K-step rollout a
learned simulation rather than a hand-written transition table. Without it
`forecast()` falls back to `DOCTRINE_SHAPE`, a matrix of ATT&CK-doctrine
constants. `analyze()` now reports which path ran, in `result['rollout_source']`:
`"learned"` or `"markov_fallback"`. A fresh clone gets `markov_fallback` and a
warning on stderr.

`.gitignore` has been fixed to allow the file to be tracked. **It still has to be
trained and committed.** Until then, do not claim the forecast is a neural
rollout — `MASTER.md` §13 currently tells you to, and on a clone that claim is
false.

Train it with `--state-weight 0.3` and commit it:

```bash
python3 src/train_v2.py --data /storage/netrikan-base-w30 --tag world_w30
git add -f models/world_w30.pt
```

---

## 3. Built this pass — code complete, **not yet trained**

Everything here is implemented and unit-tested. None of it has seen the real
dataset, because the training corpus is not on this machine. The code paths are
correct; the numbers do not exist yet.

| Capability | File | Status |
|---|---|---|
| Onset/hazard head, k = 1, 5, 15, 30 | `model.py`, `pipeline_v2.py`, `train_v2.py`, `eval_dapt.py` | code done, **needs training** |
| Onset labels, boundary-safe | `pipeline_v2.make_onset_labels()` | tested |
| Per-host identity pipeline + packet features | `pipeline_identity.py` | code done, **needs raw PCAPs** |
| Temporal attention pooling | `model.py` (`attention=True`) | code done, **needs training** |
| Transition-only evaluation | `transition_eval.py` | tested on synthetic |
| Temperature scaling, ECE, reliability diagram | `calibration.py` | self-check passes |
| Fusion-weight ablation | `bench/model_weight_ablation.py` | code done, **needs DAPT** |
| Suricata external lead time | `bench/suricata_comparison.py` | parser tested, **needs suricata** |
| Hash-chained alert ledger | `ledger.py` | tested, tamper-detecting |
| Reconnaissance stage | `attck_map.py`, `forecast.py`, `signals.py`, `app.py` | live |

### Run order once the data is in place

```bash
# 1. rebuild the dataset (writes onset labels; window comes from configs/train_v2.yaml)
python3 src/pipeline_v2.py --out /storage/netrikan-base-w30

# 2. train. --select-on defaults to onset_auc_k5 when onset labels exist
python3 src/train_v2.py --data /storage/netrikan-base-w30 --tag onset_w30
python3 src/train_v2.py --data /storage/netrikan-base-w30 --tag world_w30   # for rollout

# 3. the numbers that matter
python3 src/transition_eval.py --data /storage/netrikan-base-w30 --model models/onset_w30.pt
python3 src/eval_dapt.py      --data /storage/netrikan-base-w30 --model models/onset_w30.pt
python3 src/calibration.py    --data /storage/netrikan-base-w30 --model models/onset_w30.pt
python3 bench/model_weight_ablation.py --data /storage/netrikan-base-w30
python3 bench/suricata_comparison.py demo_attack_lab/attack_small.pcap
```

---

## 3a. First real training run — CTU-13

CTU-13 was named in the problem statement and both CSVs were committed, but no
code read them. `src/ctu13.py` now builds them into the standard dataset layout,
and a full training run completed. Everything below is a real measured number
from committed JSON, not a projection.

`models/ctu13_run_summary.json` consolidates it. 92,152 windows, window 30,
50,521 train / 12,631 val on a purged blocked split, 15 epochs, 2.9 min on CPU.

### The mandatory baseline comparison

| Model | Sees | macro-F1 |
|---|---|---|
| Logistic regression | one flow | 0.8353 |
| Logistic regression | whole window, flattened | 0.9940 |
| **LSTM** | whole window as a sequence | **0.9989** |

Sequence information is worth **+0.159**. The LSTM on top is worth **+0.005**.

On CIC-IDS-2018 that second number was **+0.180**. The collapse is the finding:
CTU-13 binary botnet detection is near-degenerate. It has no pre-compromise
baseline, so every infected-host window is labelled attack and a linear model on
the raw flattened window already solves it. `MASTER.md` §8 asserted this about a
competitor's CTU-13 results; this run is the evidence for it, measured here.

**Do not quote the 0.9989 as a headline.** It is a number from an easy task. Its
value is as the control that shows the task is easy.

### The world model, measured for the first time

`bench/rollout_eval.py` tests whether `state_head` learned dynamics or just
learned to copy. Consecutive flows are highly correlated, so a low raw MSE proves
nothing — it has to beat the trivial baselines.

| Predictor of the next flow's feature vector | MSE |
|---|---|
| copy the last observed row (persistence) | 1.693 |
| **state_head** | **0.974** |
| mean of the window | 0.893 |

Skill vs persistence **+0.425**; skill vs window mean **−0.090**.

**Verdict: PARTIAL.** The state head is not echoing its input, but it loses to
averaging the window. It has learned a smoothed central estimate, not a
trajectory. K-step drift is nearly flat (0.974 → 1.019 over 10 steps) and
predicted-stage volatility across the rollout is **0.0001** — the free-running
simulation reaches a fixed point almost immediately instead of evolving.

The rollout is real code running real trained weights. On this corpus it does not
simulate a trajectory, and the "world model" claim should not be made from this
checkpoint. Re-run `bench/rollout_eval.py` after training on CIC-IDS-2018 before
making it at all.

### What CTU-13 could not do

| Target | Outcome |
|---|---|
| Onset head | Trained, **unscoreable**. 1 label transition in 92,152 windows, so onset labels are degenerate (positive rate 1.1e-05 at k=1) and AUC is undefined. `train_v2` detected this and fell back to macro-F1 selection. |
| Transition-only eval | **Not computable.** 0 transition windows in the val split; `transition_eval.py` refuses rather than reporting a number. |
| Calibration | ECE **0.0008 before** scaling — already calibrated, because 99.9% accuracy at 99.98% confidence *is* calibrated. Nothing to fix. The machinery is verified separately by `calibration.py --self-check` (0.280 → 0.010). |
| Rule layer | Inert. CTU-13 has no `Dst Port`, so most of the 14 detectors cannot fire. |
| Per-host windowing | Impossible. No IP columns. |

CTU-13 also lacks `PSH Flag Cnt`; `Bwd PSH Flags` is substituted and recorded in
`feature_notes.json` so the substitution is not invisible.

### Bugs this run exposed and fixed

- `train_v2.py` and `baseline.py` called `classification_report` without
  `labels=`, so both **crashed** on any corpus not containing all 5 classes.
- `--select-on auto` would have selected on an undefined onset AUC and **never
  saved a checkpoint**. Now detects degenerate onset labels and falls back.
- `baseline.py` had `MODEL_DIR` hardcoded to `~/netrikan/models`.
- `train_v2.py` now selects MPS when available (`NETRIKAN_DEVICE` overrides).

---

## 4. Still not done

Done this session (§0): onset head trained (AUC 0.90), transition eval computed,
fusion ablation run, calibration run, rollout skill measured (PASS), DAPT
cross-dataset eval run. `MEASURED_PERSISTENCE` was measured against DAPT
(`bench/dapt_persistence.py` → `models/dapt_persistence.json`) — the declared
values match DAPT per-capture dwell; per-host dwell is longer (0.958–0.999), noted
below.

**Resolved this session:** deployment checkpoint chosen and wired (`cic_v2_w30`,
§0); world model live on a clone (rollout_source "learned"); cross-dataset
detection fixed (0.30→0.68); explainability upgraded to real SHAP + attention;
`MEASURED_PERSISTENCE` measured (`bench/dapt_persistence.py`).

| Item | Why it matters |
|---|---|
| Raise `MODEL_WEIGHT` above 0.30 | The ablation shows the rules hurt cross-dataset generalization (pure model SEDI +0.416 vs pure rules −0.256). Still 0.30; needs an in-dataset re-check before changing the default, since rules help in-dataset. |
| Apply the per-host persistence re-derivation, or don't | `models/dapt_persistence.json` has it: declared values match DAPT per-capture dwell, but per-host dwell is longer (0.958–0.999). Editing `forecast.MEASURED_PERSISTENCE` changes every forecast number — a deliberate call, not auto-applied. |
| Push Infiltration above 0.41 | Rarest class (1.67%). Its transition F1 (0.56) already beats steady-state; a focal loss or higher class weight might lift the steady-state number. |
| Packet-level features into training | Needs raw PCAPs (450 GB — not feasible on this disk; the user confirmed). `pipeline_identity.py` is ready if they ever land. |
| Any test suite | `_check()` self-tests in `ledger.py`, `calibration.py`, `explain_shap.py`, `bench/sedi.py`, plus the unit tests run this session, but no `tests/` directory. |
| Suricata external lead-time | `bench/suricata_comparison.py` is ready; needs `suricata` installed (`brew install suricata`). |
| Re-capture the demo PCAP with SMB | `attack_small.pcap` has no SMB, so 3 of 4 counterfactuals correctly show zero delta and the demo looks flat. |

---

## 5. Corrections to specific documents

**`README.md`** — the results table describes `lstm_world_model.pt` (macro-F1
0.885, Infiltration 0.604). That checkpoint is retired. The deployed model is
`base_w30.pt` (0.9221 / 0.751). README also says the transition-adjacent metric
"is not yet computed"; `src/transition_eval.py` now computes it, though it has
not been run on the real dataset.

**`AI_HANDOFF.md`** — stale on: `WINDOW` (says 10, is 30), `MODEL_PATH` (says
`lstm_world_model.pt`), chain size (says 6 stages, is 7), and the claim that
`MODEL_WEIGHT` was "measured on validation set" (it was not; the code comment has
been corrected). Its prime directives 1 (never split windows randomly), 2 (never
change the label offset) and 4 (do not re-add port features) still hold and are
still the most important rules in the project. Directive 5 ("do not claim SHAP")
is **superseded**: `src/explain_shap.py` now computes real SHAP values via
`shap.GradientExplainer` (expected gradients, chosen over DeepExplainer because
DeepLIFT is unreliable through LSTM gates). SHAP is the on-demand explanation for
a selected window; gradient x input remains the always-on per-window attribution.
Both are labelled by method in the output, so neither is misrepresented.

**`MASTER.md`** — §13 claims the neural rollout is wired and live. The wiring
exists; the checkpoint does not ship. §10 lists the SHA-256 ledger as BUILT; it
was a single unchained block until this pass. Its "Remaining" table is partly
stale: SEDI is committed, the ledger is fixed, transition eval exists.

**`IDEAS.md` / `ARSENAL.md`** — these two disagree with `WINNING_PLAN.md` on
per-host windowing. `IDEAS.md` §"Resolve before any slide" states the resolution
correctly and should be followed: per-host is **not** a default improvement. It
buys +2.8pp lateral recall for +8.3pp FPR and is a net SEDI loss
(0.5825 → 0.4983). Present it as a lateral-movement-specific operator toggle with
the trade stated, or not at all.

**`BUILD.md`** — an early plan. It describes `predict.py`, `explain.py`,
`dashboard.py` and `notebooks/`, none of which exist under those names, and
specifies SHAP, which is not what the code does. Historical only.

**`configs/train_v2.yaml`** — previously said `window: 10` while the shipped model
is window 30, and `seed`/`n_blocks`/`val_frac` were never read by the trainer.
Both fixed; the config now genuinely drives the run.

---

## 6. Invariants — do not break these

1. **Never split sliding windows randomly.** Windows overlap by `WINDOW-1` rows;
   a random split leaks 100% of validation into training. Always
   `train_v2.blocked_split()`, always with `purge = WINDOW - 1`.
2. **Never change the label offset.** `y[i] = labels[i + WINDOW]` — the label is
   the flow *after* the window. That offset is what makes this forecasting.
3. **Label the explanation method honestly.** `src/explain_shap.py` computes real
   SHAP (`shap.GradientExplainer`) on demand; `analyze()` computes gradient × input
   always. Each output carries its `attribution_method`. Do not relabel one as the
   other — the point is that both are named for what they are.
4. **Do not re-add port features to the model input.** Measured: worth +0.003
   in-dataset, costs −0.04 cross-dataset recall. The service-role features still
   exist for `pipeline_identity --with-ports` and for the rule layer.
5. **Never enable `attention=True` on a checkpoint trained without it.** It
   changes how the trunk pools the window, so every prediction changes silently.
   `infer._load_model()` auto-detects it from the state dict; keep it that way.
6. **Do not present an untrained head's output.** `infer.onset_trained()` and
   `infer.attention_trained()` exist for this. A checkpoint predating a head
   loads fine under `strict=False` and emits noise from it.
7. **Re-check the benign baseline after touching `signals.py` or the transition
   matrix.** A Markov chain converges to its stationary distribution regardless
   of start state, and C2 holds the most stationary mass, so benign traffic has
   twice drifted into "Command & control". Every window of
   `demo_data.generate('Benign baseline', 260)` must read "No escalation
   expected".
