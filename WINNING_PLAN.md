# Netrikan V2 — The Annihilation Plan

> No time constraint. Build what's actually correct.

---

## The Core Diagnosis

Every team in SIH26153 — including AttackForecast — is training a model to
answer **"what stage is the traffic in right now?"** That question is dominated
by persistence (91–95% of windows have the same label as the previous one).
The model learns to copy the last label. This is not a failure — it is the
target doing exactly what it was asked. The correct question is:

> **"Does a new attack stage begin within the next k windows?"**

That is the onset/hazard formulation. Persistence scores exactly 0.000 on it
by definition — it always predicts "no onset." Your current model, trained on
the wrong target, already scores 0.591 on transition windows. Retrain with
onset as the primary head and that number climbs on a metric where no
competitor can follow.

That single change, plus the four below, produces a system that is
differentially better on every axis that can be verified from the repo.

---

## The Five Changes

### 1. Onset head — new primary training target

**What**: Add `onset_head: nn.Linear(64, K)` to WorldModel alongside the
existing stage/breach/state heads. K = 4 horizons: k=1, k=5, k=15, k=30
windows ahead. For each horizon k, the binary label is:
```
onset_k[t] = 1  if  stage[t+k] != stage[t]  else 0
```
Train with BCE loss on all K heads simultaneously. Stage head stays as a
secondary head — it still matters, but it is no longer the optimisation target.

**Why it annihilates**: Persistence = 0.000 on onset. Your current model
(wrong target) = 0.591 on transition windows (measured, in
`models/persistence_baseline.json`). csxzor formulated this target and never
trained. ShadowCat built a hazard head and shipped no weights. AttackForecast
never considered it. That slot is open.

**Code changes**:
- `src/model.py` — add `onset_head = nn.Linear(64, K)`, return onset logits
  from `forward()`
- `src/pipeline_v2.py` — generate onset label array alongside `y`: shift y by
  k, compare, write `onset_k.npy` for each k
- `src/train_v2.py` — add onset BCE term to loss; save best checkpoint on
  onset AUC at k=5 (the operationally meaningful horizon), not macro-F1
- `src/eval_dapt.py` — add onset eval section: AUC, recall, precision at
  k=1,5,15,30. Report persistence baseline = 0 on all k.

**Expected result**: onset AUC 0.80–0.90 vs persistence 0.000. The entire
persistence narrative collapses. This is the number on slide 3.

---

### 2. Temperature scaling — prerequisite for rollout being meaningful

**What**: After training, fit a single scalar T on a held-out calibration
split (not used in any evaluation) by minimising NLL:
```
T* = argmin_T  -Σ log softmax(logits / T)[true_class]
```
Apply T to stage_head logits at inference time. Measure ECE (Expected
Calibration Error) before and after.

**Why it matters**: Without calibration, `model.rollout()` compounds
overconfidence — the 0.99+ top-probability at every step is unphysical. A
miscalibrated rollout is confident nonsense over 5 steps. A calibrated rollout
is a defensible probabilistic forecast.

**Why it annihilates**: Zero competitors have calibrated confidence. The line
"when we say 70%, we mean 70% — ECE before: X, after: Y, reliability diagram
in the repo" is unchallengeable. It also directly answers the jury question
"how do I trust these probabilities?"

**Code changes**:
- New file `src/calibration.py` (~50 lines):
  - `fit_temperature(logits, labels)` → scalar T via scipy minimize_scalar
  - `compute_ece(probs, labels, n_bins=10)` → float
  - `plot_reliability(probs, labels)` → saves PNG to models/
  - Run once after training, save T to `models/temperature.json`
- `src/infer.py:_load_model()` — load T from temperature.json, apply in
  `forward()` override or post-hoc

---

### 3. Time Bought (τ_b) — the unique output

**What**: For each counterfactual intervention (block SMB, isolate host, etc.),
compute not just risk_before/risk_after but:
```
τ_b = (first_damage_step_with_intervention - first_damage_step_baseline) × step_seconds
```
Output: "Blocking SMB buys 847 seconds before C2 onset." This is wired through
`model.rollout()` — the world model literally simulates "what happens next if
we apply this control."

**Why it annihilates**: No competitor has any answer to "what should I do
about this alert?" Netrikan answers with a measured time delta. The
counterfactual engine already exists and works. This is the last 50 lines that
make it genuinely useful.

**Code changes**:
- `src/counterfactual.py` — add `time_bought(df, intervention, model, steps,
  step_seconds)` that runs rollout on baseline and intervention DataFrames,
  finds first damage step in each, returns delta in seconds
- `src/app.py` — show τ_b next to risk delta in the counterfactual panel

---

### 4. SPRT per-host alert controller

**What**: Sequential Probability Ratio Test (Wald 1945) per host. Accumulate
log-likelihood ratio across windows:
```
Λ_t = Λ_{t-1} + log(P(window | attack) / P(window | benign))
```
Alert when Λ_t > h (upper threshold). Reset when Λ_t < -h (lower). Thresholds
set by operator-specified FPR budget:
```
h ≈ log((1 - β) / α)
```
where α = desired FPR and β = desired miss rate.

**Why it annihilates**: SPRT is optimal in the Neyman-Pearson sense — no test
of the same sample size has a lower miss rate at the same FPR. It handles the
per-host streaming case correctly (each host has its own Λ accumulator, resets
after alert). Zero competitors have this. Jury asks "how do you control false
alarms?" — you answer with a 1945 theorem.

**Code changes**:
- New file `src/sprt.py` (~60 lines): `SPRTMonitor` class with `update(window_probs)` → returns `(alert: bool, ratio: float)`
- `src/infer.py` — add per-host SPRT accumulation in `analyze()`
- `src/app.py` — show per-host SPRT state in the entity panel

---

### 5. Suricata as the real external baseline (not a feature)

**What**: Run `suricata -r <pcap> -l /tmp/sout` on every demo PCAP. Extract
first alert timestamp from `eve.json`. Compare to Netrikan's
`first_forecast_index × flow_interval`. The delta is the real lead-time claim.

Also: pipe Suricata EVE JSON alerts into the fusion layer as a structured
evidence channel (replaces the 14 hand-coded rules in `signals.py`). This
moves MODEL_WEIGHT from 0.30/0.70 to something that can be justified by
ablation rather than guesswork.

**Why it matters**: The current lead-time claim compares the model's forecast
threshold to the model's detection threshold — a circular self-comparison.
Suricata as external baseline makes it a real number. "Netrikan alerted 111
seconds before Suricata's first rule fired on the same PCAP" is verifiable by
anyone with the PCAP.

**Code changes**:
- `src/suricata_feed.py` (~80 lines): parse eve.json, extract alert timestamps,
  convert to feature signals compatible with `signals.py` interface
- `src/infer.py` — add `suricata_eve_path` optional param to `analyze()`
- One shell script: `bench/run_suricata_comparison.sh` — runs suricata,
  runs infer, outputs `models/suricata_comparison.json`

---

## The New WorldModel Architecture

```python
class WorldModel(nn.Module):
    # heads share one LSTM trunk
    #   stage_head:  64 → 5    (current stage classification — secondary)
    #   breach_head: 64 → 1    (P(malicious) — kept)
    #   state_head:  64 → 24   (next state for rollout — kept)
    #   onset_head:  64 → K    (NEW: P(new stage in next k windows) for K horizons)

    def forward(self, x):
        ...
        onset_logits = self.onset_head(shared)   # (batch, K)
        return stage_logits, breach_prob, next_state, onset_logits
```

Loss function:
```python
loss = (
    stage_loss(logits, stage_labels)           # CrossEntropy, secondary
    + 0.5 * breach_loss(breach, binary_labels) # BCE, kept
    + state_weight * state_loss(nxt, nxt_true) # MSE, for rollout quality
    + 2.0 * onset_loss(onset_logits, onset_labels)  # BCE per k, PRIMARY
)
```

The 2.0 weight on onset is intentional — this is the target the model optimises
for. Stage classification is a bonus.

---

## Build Order

| Phase | What | Depends on | Time estimate |
|---|---|---|---|
| 0 | Temperature scaling | existing base_w30.pt | 2–3 hours |
| 1 | Onset label generation | pipeline_v2.py | 3–4 hours |
| 2 | Onset head + training | Phase 1 complete | 1 day (training) |
| 3 | Onset eval on DAPT | Phase 2 checkpoint | 2–3 hours |
| 4 | Time Bought (τ_b) | calibrated rollout (Phase 0) | 3–4 hours |
| 5 | SPRT alert controller | Phase 2 (has breach probs) | 4–5 hours |
| 6 | Suricata lead-time bench | existing pcap | 2–3 hours |
| 7 | Suricata EVE feed | Phase 6 | 4–5 hours |

Start with Phase 0 (temperature scaling) — it is fast, standalone, and a
prerequisite for rollout being trustworthy. The current rollout outputs 0.99+
confidence at every step; calibrated rollout is the difference between
demonstrating a concept and demonstrating a tool.

Phase 2 (onset training) is the single most important thing. Everything else
is multiplier. Without onset, you are entering a World Models problem with a
system that predicts autocorrelation.

---

## What the Competition Table Looks Like After V2

| Capability | **Netrikan V2** | AttackForecast | What-the-Hack | ShadowCat | csxzor |
|---|---|---|---|---|---|
| Onset/hazard target | **YES** | NO | NO | head built, no weights | formulated, no training |
| Beats persistence | **YES (onset AUC >> 0)** | 9/10 horizons | unknown | unknown | N/A |
| Calibrated confidence (ECE) | **YES** | NO | NO | NO | NO |
| Time Bought (τ_b) | **YES** | NO | NO | NO | NO |
| SPRT per-host alert | **YES** | NO | NO | NO | NO |
| Cross-dataset eval | **YES** | NO | NO | NO | NO |
| CAPEC mapping | **YES** | partial | NO | NO | NO |
| Counterfactual engine | **YES** | NO | NO | NO | NO |
| Real lead-time (vs Suricata) | **YES** | NO | NO | NO | NO |
| Honest FPR published | **YES** | NO | NO | NO | NO |

AttackForecast is the only team that comes close. After V2, they have one thing
Netrikan doesn't: per-host windowing in production. You have everything else,
and onset/hazard plus calibration are things they cannot quickly add (it
requires retraining their RSSM from scratch).

---

## Jury Answers After V2

**"Your LSTM loses to a simple baseline"**
> "On the stage-classification metric, yes. We measured it and published it —
> `models/persistence_baseline.json`. But that metric is wrong for a forecasting
> system. The correct metric is onset recall: does a new attack stage appear in
> the next k windows? Persistence scores 0.000 on that by definition — it
> always predicts 'no change.' Our onset head scores [AUC]. That is the
> forecasting signal. The stage classification head is secondary output we kept
> for analyst readability."

**"How do you trust these probabilities?"**
> "Temperature scaling. We fit a calibration scalar on a held-out split and
> report ECE before and after. The reliability diagram is in models/. When we
> say 70%, we're right ~70% of the time — that's what calibration means. ECE
> after scaling: [number]. Zero other teams in this room have measured this."

**"So what do I do when the forecast fires?"**
> "The counterfactual panel shows you exactly what to do and what it's worth.
> Blocking SMB ports delays C2 onset by τ_b = [N] seconds — measured by running
> the world model forward through the intervention. You get a ranked list of
> controls with a quantified time delta. No other team has this."

**"How do you handle per-host tracking at scale?"**
> "SPRT — Sequential Probability Ratio Test, Wald 1945. Each host gets its own
> likelihood ratio accumulator. Alert threshold set by operator-specified FPR
> budget. Optimal in the Neyman-Pearson sense — no fixed-sample-size test beats
> it at the same FPR. We implemented it as SPRTMonitor in src/sprt.py."

**"How do you know you alert before a signature IDS?"**
> "We ran Suricata 8.0.6 with 52,311 Emerging Threats rules offline on the same
> PCAP. First Suricata alert: [timestamp]. Netrikan onset forecast crossed at:
> [timestamp]. Delta: [N] seconds. That is an external comparison, not us
> comparing our forecast threshold to our detection threshold. The result is in
> models/suricata_comparison.json."

---

## Files to Create / Modify

| File | Status | What changes |
|---|---|---|
| `src/model.py` | MODIFY | add `onset_head`, return onset logits from forward() |
| `src/pipeline_v2.py` | MODIFY | generate onset_k.npy for k=1,5,15,30 |
| `src/train_v2.py` | MODIFY | add onset BCE loss (weight 2.0), save on onset AUC |
| `src/calibration.py` | CREATE | temperature fit, ECE, reliability diagram |
| `src/sprt.py` | CREATE | SPRTMonitor class |
| `src/suricata_feed.py` | CREATE | parse eve.json → feature signals |
| `src/counterfactual.py` | MODIFY | add time_bought() via rollout |
| `src/infer.py` | MODIFY | load T, apply calibration, per-host SPRT accumulation |
| `src/eval_dapt.py` | MODIFY | add onset AUC section, persistence = 0.000 row |
| `bench/run_suricata_comparison.sh` | CREATE | automates the lead-time comparison |
| `src/app.py` | MODIFY | show τ_b, onset probability, SPRT ratio per host |

---

## The 3-Sentence Pitch (V2)

> "Netrikan is the only system trained to forecast when an attack advances —
> not what stage it's in. Every other approach here optimises for persistence,
> which means copying the last label: that's not forecasting, it's
> autocorrelation. We publish our calibrated error rates, counterfactual
> response times, and every metric is in a JSON file in the repo."
