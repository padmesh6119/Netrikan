# Session handoff — 2026-09-29

Branch `eval/task4b-task7-lodo`. Written for a friend picking this up next — read
`TRAINER_BACKLOG.md` first for the full backlog and competitor context; this file
is just "what changed this session" plus an updated status of everything in it.

---

## What got done this session

### P0 §1 — Transition-window F1 sweep: DONE

This was the backlog's single highest-priority item ("the number that decides the
whole forecasting thesis"). Result:

**The LSTM beats persistence on transition-only macro-F1 at every horizon tested
(5/5).** The forecasting claim stands.

| k | n_transitions | LSTM transition-F1 | persistence | gap | LSTM onset recall |
|---|---|---|---|---|---|
| 0   | 31,781  | 0.380 | 0.000 | +0.380 | 0.474 |
| 30  | 187,573 | 0.459 | 0.000 | +0.459 | 0.691 |
| 90  | 168,677 | 0.464 | 0.000 | +0.464 | 0.697 |
| 180 | 190,380 | 0.455 | 0.000 | +0.455 | 0.692 |
| 360 | 185,045 | 0.438 | 0.000 | +0.438 | 0.686 |

Persistence is exactly 0 at every horizon **by construction** — "transition
window" is defined as the subset where persistence's prediction (last known
label) differs from the truth, so it cannot score above 0 there. That's a sanity
check baked into the metric, not a result. The result is the LSTM column: solidly
positive everywhere, peaking around k=90.

`n_transitions` (32K–190K per horizon) is also the number that wins the C1
competitor-delta race: cybersentinel's published transition report used 6
transitions total; ours is 4–5 orders of magnitude larger.

**How it was produced (methodology note — read before trusting the number):**

- The original `fx_k*` sweep referenced in the backlog (window=10, port features,
  `/storage/sih-ports-w10`) no longer exists on this machine — only its metrics
  JSONs survived. Rather than fake a match to it, we retrained the sweep from
  scratch on the data that *does* exist locally: `/tmp/netrikan-cic-full-w30`
  (6.16M windows, window=30, 24 features, all 7 CIC-IDS-2018 capture days).
- **5 new checkpoints** were trained specifically for this experiment:
  `models/htz_k{0,30,90,180,360}.pt`, via
  `train_v2.py --horizon {k} --fixed-split`, standard recipe (25 epochs,
  patience 5, batch 1024, 1.2M samples/epoch, select-on `combined`).
  `--fixed-split` pins the blocked train/val split to the *unshifted* labels, so
  all 5 horizons share the exact same held-out row indices — that's what makes
  the k values comparable to each other.
- **These are NOT the deployed checkpoints.** `base_w30.pt` / `cic_v2_w30.pt` /
  `cic_full_w30.pt` are untouched. `htz_k*` are dedicated, throwaway-if-you-want
  experiment checkpoints that exist only to answer this one question honestly.
  Total training time: ~2h52m (5 runs × ~30-38 min on Apple MPS).
- New script: `src/eval_transition.py` — generalizes the existing
  `src/transition_eval.py` (which only scored one checkpoint, no `n_transitions`
  field, no per-horizon sweep) to the full spec: per-k `n_transitions`, onset
  recall variant (benign→attack transitions specifically), and a top-level
  `lstm_wins_transition` verdict.
- Output: `models/transition_f1.json` (the file itself, plus the 5
  `models/htz_k*_metrics.json` training logs, plus `models/htz_k*.pt`
  checkpoints — the `.pt` files are gitignored like all non-whitelisted
  checkpoints; only the JSONs are meant to be committed).

**What this does NOT resolve:** the model-promotion decision (P1 §4,
`models/BEST.txt`) is still open. This experiment used its own checkpoints for a
narrow purpose; it doesn't tell you which of `base_w30` / `cic_v2_w30` /
`cic_full_w30` should actually ship.

---

## Everything else — status as of this session

(Surveyed but not built this session, except where noted. Order follows the
backlog's own priority.)

### P1 — table stakes
- **§2 Calibration — NOT DONE to spec.** `src/calibration.py` +
  `models/temperature.json` / `temperature_ctu13.json` exist and are self-tested,
  but only report **val**-split ECE (fit/score halves of val). The backlog's
  actual ask — verify on a genuine held-out **test** partition, ship only if test
  improves — is not implemented. This is the next-highest-value item: per the
  competitor delta (C2), zero competitors have cleared this bar either.
- **§3 Fixed-FPR operating points — PARTIALLY DONE.**
  `models/operating_points_dapt_cic_v2_w30.json` has full threshold-sweep curves
  already. But none of the six `dapt_*.json` files carry an `operating_points`
  key inline, as the backlog specifies — the data exists in a separate file, not
  merged in.
- **§4 Model promotion — NOT DECIDED.** `models/BEST.txt` still only states the
  in-dist number (`ports_w30 macroF1=0.9249`). No written decision on
  base_w30-for-cross-dataset vs ports_w30-for-in-dist, despite `RESUME.md`
  already having the evidence.

### P2
- **§5 State-weight sweep — infra exists, sweep not run.**
  `train_v2.py --state-weight` works; no `models/stateweight_sweep.json`.
- **§6 Seed variance — PARTIAL.** `models/seed_variance_cic_full_w30.json` (4
  seeds) exists but is explicitly self-documented as not a clean 5-seed
  replication (mixed epoch budgets), and it does not include per-seed
  transition-F1 or onset-PR (blank, not guessed). No generic `models/variance.json`.
- **§7 Known ceilings —** just documentation; not verified this session whether
  README/pitch state these as limitations correctly.

### Competitor-delta items
- **C1 (transition dynamics)** → same as P0 §1, **now done** (see above).
- **C2 (calibration)** → same as P1 §2, not done.
- **C3 (GRU dynamics head)** → not built at all. No `world_model_gru.pt`.
- **C4 (adaptive/rollback learning)** → not built. No `adaptation_report.json`.
- **C5 (per-horizon logreg baseline)** → `src/baseline.py` has `LogisticRegression`
  but current baseline JSONs only report `"last"`/`"flat"` modes, not one logreg
  per horizon k.

### P3 (app-side, owner: app — not the trainer's job)
- **B1 (ledger genesis bug)** — checked `src/ledger.py` in full this session:
  it looks **already fixed**, contradicting the backlog's description. `build()`
  uses one `GENESIS` per chain call and threads `prev` correctly through the
  loop; its self-check explicitly asserts chaining/tamper/deletion detection all
  work. The line numbers the backlog cites (`app.py:468-471`) now point at
  unrelated counterfactual-UI code, not the ledger call (`app.py:481`). Confirm
  with whoever wrote the backlog entry before redoing this — it may have been
  fixed since, or the description may be stale.
- **B8 ("LSTM beats persistence" claim)** — found one likely instance still
  standing: `MASTER.md:414` — *"Stage F1: 0.537 vs persistence 0.478 — beats
  persistence at 9/10 horizons."* This contradicts `persistence_baseline.json`
  (persistence wins all-window macro-F1 at every k). Very likely the exact claim
  B8 wants pulled or reframed around the new transition-only result instead —
  worth confirming what "Stage F1" metric that line was quoting before editing.
- **B9 (per-host claim)** — not found overstated in README/MASTER as checked;
  may already be handled correctly (they already present it as a tradeoff).
- **Architecture doc, demo video** — neither exists.

---

## Suggested next step

Per the backlog's own stated order of attack, **P1 §2 (test-partition-verified
calibration)** is next — it's the other item where "be first to do it correctly"
is still a fully open goal (every competitor's calibration either wasn't
test-verified or made test worse). `src/calibration.py` is a real starting point,
not a from-scratch build: it needs a genuine held-out test split added alongside
the existing val fit/score split, and the ship/no-ship gate the backlog specifies
(`ece_test_post < ece_test_pre` AND `brier_test_post <= brier_test_pre`).
