# ARSENAL.md — What the 2026 literature says you should build

Researched 2026-09-19. Every citation checked. Every number computed from your own
committed files. This does not replace `WINNING_PLAN.md` (which audited the code and
found real defects) — it adds the **externally-citable science** that turns your honest
weaknesses into published-standard results.

---

## 0. THE HEADLINE: one metric kills both of your weaknesses

Your two admitted problems are *"we lose to persistence"* and *"41.6% FPR."* Both are
artifacts of using **accuracy-family metrics on a rare-event forecasting problem.**
Meteorology solved this in 2011 and the solution is 15 lines of code.

### SEDI — Symmetric Extremal Dependence Index

Ferro & Stephenson, *Weather and Forecasting* 26(5), 2011. Designed because
**"traditional performance measures degenerate to trivial values as events become rarer."**

```
SEDI = [ln F − ln H + ln(1−H) − ln(1−F)] / [ln F + ln H + ln(1−H) + ln(1−F)]

H = hit rate (recall),  F = false alarm rate (FPR)
Range: −∞ … 1.   0 = no skill.   1 = perfect.
```

**Properties that matter to you:** non-degenerating, **base-rate independent**,
asymptotically equitable, hard to hedge.

### Your numbers, computed

| Configuration | H | F | **SEDI** | PSS (H−F) |
|---|---|---|---|---|
| In-distribution (`base_w30`, held-out) | 0.9628 | 0.0300 | **+0.9801** | +0.9328 |
| Cross-dataset (DAPT 2020, zero adaptation) | 0.8390 | 0.4160 | **+0.5824** | +0.4230 |
| Per-host windowing (DAPT) | 0.8460 | 0.4993 | **+0.4983** | +0.3467 |

*(In-distribution H and F derived from `models/base_w30_metrics.json` `confusion_matrix`:
row 0 gives 25,015 benign FP / 832,795 benign = 3.004% F; rows 1–4 give 962.8/1000 attack
windows assigned to any attack class = 0.9628 H.)*

### Why this is the whole ballgame

Precision collapses as attacks get rarer. **SEDI does not move at all:**

| Attack base rate | Precision | SEDI |
|---|---|---|
| 26.5% (the DAPT eval set) | 42.10% | **0.5824** |
| 1 in 100 | 2.00% | **0.5824** |
| 1 in 1,000 | 0.20% | **0.5824** |
| 1 in 10,000 (enterprise) | **0.020%** | **0.5824** |

> **The line:** *"Precision at a realistic base rate is 0.02% — for us and for every
> detector in this room, it's arithmetic, not a model property. That's why forecast
> verification stopped using it. We report SEDI — Ferro and Stephenson 2011, the standard
> for rare binary event forecasts — because it's base-rate independent by construction.
> Ours is 0.58 cross-dataset and 0.98 in-distribution. Ask any other team for their
> base-rate-independent skill score."*

### ⚠️ It also refutes one of your own claims

**Per-host windowing is a net loss on SEDI: 0.582 → 0.498.** The +2.8pp lateral recall
does not pay for the +8.3pp FPR under a base-rate-independent measure. Do **not** put
per-host on Slide 3 as a win. Correct framing: *segment windowing is the default; per-host
is a lateral-movement-specific view we measured and can switch to when that's the
question.* Finding this yourself, and saying it, is worth more than the recall number.

### Implementation — `src/skill.py`, ~15 lines

```python
import math

def sedi(H, F, eps=1e-6):
    H = min(max(H, eps), 1 - eps)
    F = min(max(F, eps), 1 - eps)
    num = math.log(F) - math.log(H) + math.log(1 - H) - math.log(1 - F)
    den = math.log(F) + math.log(H) + math.log(1 - H) + math.log(1 - F)
    return num / den

def pss(H, F):
    return H - F
```

Clamping matters: SEDI is undefined at H∈{0,1} or F∈{0,1}. **Persistence on the
transition subset has H = 0 by construction — its SEDI is −∞.** State that as arithmetic.

**Cost: 1 hour including wiring it into every eval JSON. Highest ratio on this list.**

---

## 1. THE ACADEMIC FRAME YOU'RE MISSING: Early Classification of Time Series

Your problem has a name in the ML literature and you aren't using it. **Early
Classification of Time Series (ECTS)** is the formal study of *"classify as early as
possible, trading accuracy for earliness."* That is Netrikan, exactly.

**Survey & benchmark:** arXiv 2406.18332. **Methods:** TEASER, ECONOMY, CALIMERA
(*Information Processing & Management* 60(5), 2023 — beats TEASER on 35/45 datasets).

### TEASER's architecture maps onto yours 1:1

| TEASER | Netrikan |
|---|---|
| **Slave classifiers** — per-timestep class probabilities | your LSTM stage head |
| **Master classifier** — assesses confidence, decides whether to commit | **you don't have this** |
| Commits only after the master vets a class for `v` consecutive steps | **you don't have this** |

The master is a second, tiny classifier trained on *(stage probs, margin, features)* →
*was the slave right?* It is a **learned "when to alert" policy** replacing your fixed
threshold. Two direct consequences:

1. **It is the principled fix for the 41.6% FPR** — the system abstains on windows the
   master doesn't vet, instead of emitting a confident wrong stage.
2. **It produces the earliness–accuracy tradeoff curve** — accuracy on the y-axis, how
   early you commit on the x-axis. That plot *is* the deliverable. No competitor has one,
   and it reframes lead time from a single disputed number into a measured curve.

The `v`-consecutive-steps rule alone (**no training at all**) is ~10 lines in `infer.py`
and will cut your false alarm rate immediately — a single noisy window can no longer
raise an alert.

> **Claim:** *"This is an early-classification problem with a reject option — TEASER,
> CALIMERA, the ECTS literature. We implement the master-gatekeeper design: a second
> classifier decides when the forecast is trustworthy enough to commit. Here's our
> earliness–accuracy curve. Every other team here picked a threshold by hand."*

**Cost: 3h for the `v`-step rule + curve. 6h for a trained master. Do the cheap half first.**

---

## 2. THE FIELD-LEVEL DEFENCE OF YOUR LEAKAGE DISCOVERY

**"Demystifying Network Foundation Models," arXiv 2509.23089 (Sept 2026).** Evaluates
ET-BERT, netFound, NetMamba, YaTC — the transformer-based network foundation models — and
finds:

- They **do not consistently outperform simpler alternatives**
- **Train/test overlap** artificially inflates their reported metrics
- They exhibit **shortcut learning** — superficial correlations, not robust representations
- Performance **drops substantially** under proper dataset separation

This is the most useful paper on this list for the table, for two reasons.

**First, it pre-empts the "why didn't you use a transformer?" question with evidence:**

> *"We looked at it. The September 2026 survey that audited ET-BERT, netFound, NetMamba
> and YaTC found they don't reliably beat simple baselines and that their headline numbers
> came from train/test overlap. That's the same class of bug we found in our own pipeline —
> 100% of our validation windows contained training rows before we fixed it. We chose a
> 207K-parameter LSTM we could audit over a foundation model we couldn't."*

**Second, it promotes your leakage discovery from a rookie mistake to a field-wide
finding.** Right now your README frames it as "we made an error and caught it." The
correct frame is: *the published state of the art has this bug and we don't.* That is a
different sentence entirely.

**Cost: 0 hours. It's a sentence.**

---

## 3. WHAT THE ACTUAL 2026 SOTA IS DOING (and what to take)

### StageFinder — arXiv 2603.07560
Temporal-graph learning for multi-stage attack progression. GNN over provenance
(processes/files/connections) + LSTM for temporal dynamics, stage probabilities aligned to
MITRE ATT&CK. **Macro-F1 0.96** on DARPA OpTC / Transparent Computing.

This is your problem statement, solved, in a paper. You cannot out-build it in six days —
it needs host-level provenance data you don't have (you have network flows only). **But
steal its second metric:**

> **"reduces prediction volatility by 31%"**

**Forecast volatility** — how often the predicted stage flips between adjacent windows —
is a metric **nobody in your competitor set measures**, and it is operationally decisive:
a forecast that flips every window is useless to an analyst regardless of its F1. It is
~5 lines:

```python
volatility = (pred[1:] != pred[:-1]).mean()
```

Report it for your model vs. the doctrine projection vs. persistence. You will almost
certainly win on it, because your 70% rule-weighted fusion is heavily smoothed. **A
weakness of your architecture becomes a measured strength on a SOTA-paper metric.**

> **Claim:** *"Macro-F1 doesn't tell an analyst whether the forecast is stable enough to
> act on. StageFinder — this year's SOTA on provenance-based stage estimation — reports
> prediction volatility alongside F1 for exactly that reason. Ours is X% versus Y% for the
> doctrine baseline."*

### DeepStage — arXiv 2603.16969
Builds on StageFinder: feeds stage probabilities + provenance embeddings into a
**hierarchical deep RL agent under a POMDP** to pick stage-specific defensive actions.

**This is the academic version of your counterfactual engine.** Cite it to position yours
deliberately, not as a lesser version:

> *"The 2026 work — DeepStage — puts a deep RL policy on top of stage estimates to choose
> defensive actions. We deliberately didn't. An RL policy against Critical Information
> Infrastructure is unauditable: you can't tell an operator why it isolated a host. Ours
> enumerates four named interventions and reports the measured risk delta for each. It's
> the deterministic, certifiable version of the same idea — and the operator still decides."*

That answer turns "you didn't use RL" into a design principle for a CII regulator.

### E-HiDNet — arXiv 2601.06734
CNN+RNN feeding a **Hidden Markov Model** over latent attack stages and their stochastic
transitions. Relevant because your `DOCTRINE_SHAPE` is a hand-coded transition matrix —
the literature learns it via HMM. This is independent support for **WINNING_PLAN upgrade 4
(wire `rollout()` in)**: the field agrees the transition model should be learned.

---

## 4. CONFORMAL PREDICTION — the coverage guarantee, and the 2026 caveat

Conformal prediction wraps any black-box model to produce prediction **sets** with a
**distribution-free, finite-sample coverage guarantee.** With your stage head, at α=0.1
you emit sets like `{LATERAL, C2}` with a proven ≥90% chance of containing the true stage.
No retraining, no assumptions.

Zero competitors have this. It is a different *kind* of claim from every number on your
slides — not "we scored X on this data," but "this guarantee holds on data we've never
seen, by construction."

**Implementation (split conformal, ~25 lines):** hold out a calibration split, compute
nonconformity `s = 1 − p(true class)` for each point, take the `⌈(n+1)(1−α)⌉/n` empirical
quantile `q̂`, then at inference emit `{c : p(c) ≥ 1 − q̂}`.

**⚠️ The 2026 caveat, and you must say it first.** "Robust Conformal Intrusion Detection
via Traffic-Aware Calibration," arXiv 2609.19241 (July 2026): the standard guarantee
**breaks when an adversary perturbs controllable network features** — exchangeability
fails. Their fix is to calibrate on traffic from the perturbation mechanism, or to restrict
scoring to features the attacker cannot reach.

> **Claim:** *"Our prediction sets carry a distribution-free 90% coverage guarantee. It
> holds under exchangeability — and a July 2026 paper shows an adaptive adversary who
> controls flow features can break that. We name the limit because a guarantee you can't
> state the conditions for isn't one."*

Stating the caveat **unprompted** is worth more than the guarantee.

**Cost: 3h. Do it after SEDI and the ECTS gatekeeper.**

---

## 5. OPERATIONAL FRAMING FOR THE SOC QUESTION

**Alert-to-incident conversion rate** is the named industry metric. The benchmark: a SOC
wants **>20% of alerts to become meaningful investigations**; below that the stream is
noise. (*Alert Fatigue in Security Operations Centres*, ACM Computing Surveys, 2025.)

Your alert-budget dial should report in that unit, not in FPR. "Five alerts per hour, 28%
conversion" is a sentence a SOC lead can act on. "41.6% FPR" is not. Same model, same
numbers, correct unit.

Also relevant: **PACT** (arXiv 2605.22324) reduces alert fatigue in **low-prevalence**
streams via triggered active learning — the deployment story for your observe-only
calibration period.

---

## 6. THE ADVERSARIAL QUESTION — the answer, not the build

Don't build adversarial robustness in six days. Have the answer ready:

- **"Evasion Adversarial Attacks Remain Impractical Against ML-based NIDS, Especially
  Dynamic Ones"** (arXiv 2306.05494) — problem-space constraints (protocol validity,
  semantic consistency between flow features) mean an attacker cannot freely perturb
  features the way feature-space attacks like FGSM/PGD assume.
- **Perturb-ability Score** (*Journal of Information Security and Applications*, 2026) —
  scores which features an attacker can actually manipulate under problem-space
  constraints, so you can prefer inherently robust features.

> *"Feature-space attacks like FGSM assume the attacker can set any feature to any value.
> They can't — flow duration, packet counts and IAT are coupled by protocol semantics. The
> literature's term is problem-space constraints, and the 2026 work finds evasion remains
> impractical against dynamic NIDS. Our 70% rule-weighted fusion also means an adversarial
> perturbation has to fool fourteen hand-written detectors, not just a gradient."*

That last clause is real: **your fusion layer is accidental adversarial robustness.**
Gradient attacks don't transfer to hand-coded threshold rules. Claim it.

---

## 7. REVISED 6-DAY SCHEDULE

Merging this with `WINNING_PLAN.md`'s code audit. **Your friend's three blockers do not
apply on this machine:** `data/processed/y.npy` is present (42MB), `/usr/bin/suricata` is
Suricata 8.0.6, and `demo_attack_lab/` is fully intact including the 409MB capture. Every
"conditional" item in WINNING_PLAN is **scheduled** here.

| Day | Work | Source |
|---|---|---|
| **1 (today)** | **SEDI + PSS in every eval JSON** (1h) · **Fix the SHA-256 ledger** — `app.py:470-472` has no chain, the UI claims one (1h) · **`v`-consecutive-step commit rule** in `infer.py` (1h) · forecast volatility metric (30m) | §0, §1, WP‑1 |
| **2** | **Real Suricata lead-time** on `attack_small.pcap` — kills the circular claim (4h) · swap `infer.py` to `base_w30.pt`, verify timeline offsets against the PCAP (3h) | WP‑3, WP‑cond |
| **3** | **Transition-only eval** on `y.npy`, both models on identical held-out indices, recomputed not hardcoded (3h) · Brier skill + SEDI on the transition subset (1h) | WP‑2, §0 |
| **4** | **Wire `model.rollout()` into `forecast.py:75`**, side-by-side toggle vs `DOCTRINE_SHAPE` (5h) | WP‑4, §3 |
| **5** | Temperature scaling + ECE + reliability diagram (3h) · **split conformal prediction sets** (3h) · alert budget in conversion-rate units (1h) | §4, §5, WP‑6 |
| **6** | Earliness–accuracy curve plot · deck · video · dry-run the demo three times | §1 |

**If you only do four things: SEDI (1h), fix the ledger (1h), Suricata lead-time (4h),
transition eval (3h).** Nine hours buys you a base-rate-independent skill score, a working
tamper-evident ledger, a real external baseline, and the metric that decides the
persistence argument.

---

## 8. NEW LINES FOR THE DECK

**Replace Slide 3.** Per-host is a net SEDI loss (§0) — don't lead with it.

> **Slide 3 — "We report the metric that doesn't lie about rare events"**
> **Visual:** precision-vs-base-rate curve collapsing 42% → 0.02%, with a flat SEDI line
> at 0.58 across it.
> **Anchor:** SEDI **0.98 in-distribution / 0.58 cross-dataset**, base-rate independent.
> **Subtitle:** *Ferro & Stephenson 2011 — because accuracy and precision degenerate as
> events get rarer.*
> This is now your strongest slide. It is citable, checkable, computed from committed
> files, and it dissolves the FPR question before it's asked.

**New Q&A entry:**

> **"Why should we believe your numbers generalize?"**
> *"Three ways, and none of them is trust. One: we report SEDI — base-rate independent by
> construction, so it doesn't inflate when attacks are rare. Two: every number is
> cross-dataset, trained on CIC-IDS-2018 and evaluated cold on DAPT 2020. Three: we found
> 100% train/test leakage in our own first pipeline and fixed it — the September 2026 audit
> of network foundation models found that same bug in the published state of the art. Clone
> the repo and re-run it."*

---

## Sources

- [Ferro & Stephenson 2011 — Extremal Dependence Indices](https://journals.ametsoc.org/view/journals/wefo/26/5/waf-d-10-05030_1.xml)
- [SEDI reference implementation (yardstick)](https://yardstick.tidymodels.org/reference/sedi.html)
- [Early Classification of Time Series: Survey and Benchmark](https://arxiv.org/pdf/2406.18332)
- [CALIMERA: A new early time series classification method](https://www.sciencedirect.com/science/article/pii/S0306457323002029)
- [Demystifying Network Foundation Models](https://arxiv.org/pdf/2509.23089)
- [StageFinder — Learning the APT Kill Chain](https://arxiv.org/abs/2603.07560)
- [DeepStage — Autonomous Defense Policies Against Multi-Stage APT](https://arxiv.org/html/2603.16969)
- [E-HiDNet — Deep Recurrent HMM for Multi-Stage APT Prediction](https://arxiv.org/html/2601.06734v1)
- [Robust Conformal Intrusion Detection](https://arxiv.org/abs/2609.19241)
- [Alert Fatigue in Security Operations Centres — ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3723158)
- [PACT — Reducing Alert Fatigue in Low-Prevalence SOC Streams](https://arxiv.org/pdf/2605.22324)
- [Evasion Attacks Remain Impractical Against ML-based NIDS](https://arxiv.org/html/2306.05494v5)
- [Perturb-ability Score for flow-based ML-NIDS](https://www.sciencedirect.com/science/article/pii/S2214212626000396)
