# Netrikan — Differentiating Ideas

**SIH26153 · NTRO · drafted 2026-09-21**

31 ideas that separate Netrikan from the field. Ranked by strength within
sections. Every entry tags **[proven]** (evidence on disk today), **[cheap]**
(buildable in hours from existing code), or **[hypothesis]** (data exists to
test it; test not yet run).

State that tag in the deck. It is what makes a claim credible to an intelligence
panel, and it costs nothing.

Competitor basis: 12 public SIH26153 repos scanned 2026-09-21. Closest rival is
AttackForecast (RSSM on CTU-13) — better rollout discipline, materially worse
classifier (stage macro-F1 0.537 vs 0.9221), zero predictions for 2 of 5 stages,
no cross-dataset evaluation.

---

# I. The spine

## 1. The Transfer Thesis — attack doctrine is universal, benign traffic is local
**[proven]** `models/dapt_base_w30.json`

On DAPT 2020, a real 4-day APT on a network never seen in training:

| Class | Cross-dataset recall |
|---|---|
| Lateral Movement | **0.942** |
| Reconnaissance | 0.797 |
| Establish Foothold | 0.756 |
| **Benign** | **0.531** |

The field reads this as "good recall, bad FPR." Wrong reading. It shows the
learned representation of *attack* transferred to a foreign network and the
learned representation of *normal* did not.

**Consequence:** never retrain the attack model per site — calibrate the benign
baseline per site. One national doctrine model, local normality. Exactly what
NTRO needs across power, telecom, banking, railways.

The 47% FPR stops being the number to apologise for and becomes the evidence for
the thesis. No competitor can contest it; none of them left their training set.

## 2. Predict the next *target*, not the next stage
**[hypothesis]** — data in hand, test not run

Everyone predicts "what stage comes next." Nobody predicts **which machine gets
hit next.** "Lateral Movement, p=0.64" is not actionable. "192.168.3.29 is the
next hop, p=0.71" lets a defender pre-position on a named host.

`models/a1_entity_compare.json` shows DAPT lateral movement concentrating on one
host — 2,292 of 2,451 windows. No rival's dataset has the columns to try this;
AttackForecast says so about CTU-13 directly.

## 3. Predictability is the signature — a rival published the proof and discarded it
**[proven]** by arithmetic on a competitor's published number

AttackForecast reported model surprise on benign vs malicious at **ROC-AUC
0.210** (mean surprise 0.44 benign, 0.27 malicious) and wrote it off as an
anomaly.

Invert the decision rule: 0.210 flipped is **0.790**. That is a working detector
they threw away.

**Automated attack tooling is more regular than human behaviour. A host that has
become *too predictable* is being driven by a machine.** Orthogonal to the
supervised head — fires on regularity, not signature.

## 4. Attack archaeology — run the world model backwards
**[hypothesis]** — novel across the whole scan

Every system here runs now → future. Run it in reverse: given the observed
state, **what must have happened before?** Search for state sequences whose
forward rollout lands on what is actually seen.

Answers the question every SOC asks and never answers: *how did they get in, and
when?* Initial Access is the stage that gets missed — quiet, early, and rolled
out of the window by the time anything fires.

Forecasting finds the future. This finds patient zero.

## 5. The shape of the silence
**[hypothesis]** — DAPT's 4-day structure supports a direct test

Everyone models attack traffic. Nobody models its absence. But an APT's defining
property is dwell. Netrikan's own DAPT-measured persistence: Foothold 0.968,
Lateral 0.931, Recon 0.929 — attackers hold a phase ~10x longer than the
hand-written doctrine estimate that was corrected against it.

**Does benign traffic on a compromised host differ from benign traffic on a
clean one?** A dormant implant still beacons, resolves, idles — differently.

If it holds, compromise is detectable **during the quiet** — the one window
where every other system in this field is silent by construction.

---

# II. Measurement and proof

## 6. Suricata as an honest external clock
**[cheap]** — verified present: Suricata **8.0.6 RELEASE** (`/usr/bin/suricata`)
and Snort **2.9.20** (`/usr/sbin/snort`). Two independent clocks, both installed.

Every team's lead-time claim is self-referential, Netrikan's included:
`first_detection_index` uses **the model's own output** as the detection
baseline. The docstring admits it.

Run Suricata offline on the same PCAP. Lead time = Netrikan's first warning
minus Suricata's first rule hit. An external, adversarial, industry-standard
clock that the model cannot influence.

Not one of the 12 competitors does this. It is the single most credible metric
available in this problem statement, and it is sitting unused on the machine.

Note: `pacman -Qs suricata` returns nothing — it was not installed via pacman —
and `/etc/suricata` is root-only, so a non-root or sandboxed session will
conclude it is absent. The binary is world-executable and `suricata -V` works.

## 7. Transition-only evaluation
**[cheap]**

Persistence wins because stage labels are autocorrelated — most windows have the
same label as the last one. Evaluate **only on windows where the label changes**.
Persistence scores ~0 there by construction.

That is the honest arena for a forecaster, and nobody publishes it. It converts
the baseline that currently beats you into the baseline that cannot compete.

## 8. SEDI — a skill score that does not move with base rate
**[proven]** — `bench/sedi.py` → `models/sedi.json`, committed 2026-09-22

Symmetric Extremal Dependence Index (Ferro & Stephenson, 2011). Meteorology
solved rare-event forecast scoring 50 years ago; this field never imported it.

SEDI is invariant to event prevalence, so the entire FPR argument dissolves —
the score is the same whether attacks are 1% or 30% of traffic.

| Setting | H (recall) | F (FPR) | SEDI |
|---|---|---|---|
| In-distribution (`base_w30` held-out) | 0.9628 | 0.0300 | **+0.9801** |
| DAPT 2020 cross-dataset | 0.8390 | 0.4159 | **+0.5825** |
| DAPT 2020 per-host windowing | 0.8460 | 0.4993 | **+0.4983** |

Hit rate and false-alarm rate are derived from `base_w30_metrics.json`'s
confusion matrix and `a1_entity_compare.json` — no dataset needed to reproduce.

## 9. Proper scoring rules, not accuracy
**[cheap]**

A forecast is scored by a proper scoring rule — Brier score, Brier Skill Score
against climatology. Weather has done this for 70 years. Every team here reports
F1 on a forecasting task, which is a category error.

Adopting forecast-verification methodology *because this is a forecasting
problem* is a framing no competitor has reached for.

## 10. The earliness–accuracy curve
**[cheap]**

Frame the problem as Early Classification of Time Series (TEASER, CALIMERA —
established literature). Publish the **curve**, not a point: accuracy as a
function of how early you commit.

One curve replaces a dozen arguments about thresholds, and it is the natural
shape of the answer to "how early can you warn?"

## 11. Calibration and a reliability diagram
**[cheap]** — temperature scaling is ~40 lines

When the system says 0.71, does it happen 71% of the time? Nobody in this field
checks. Every competitor's confidence is uncalibrated and none of them know it.

A reliability diagram is a devastating slide precisely because it is so easy and
so absent. It is also the prerequisite for alert budgets, SPRT and conformal
prediction below.

## 12. Prediction volatility as a quality metric
**[cheap]**

`(pred[1:] != pred[:-1]).mean()` — from StageFinder (2026). A forecast that
flip-flops between stages is useless to an operator even when it is accurate on
average. Nobody measures stability, so nobody knows theirs is bad.

## 13. Out-of-distribution flag — "I have not seen a network like this"
**[cheap]**

The model should say when it is outside its training distribution instead of
confidently guessing. This speaks directly to the cross-dataset result: flag
DAPT as OOD and the benign errors become a *declared* known-unknown rather than
a silent failure.

An honest "I don't know" is a feature no competitor offers.

## 14. Publish the Infiltration label artifact
**[proven]** — `models/persistence_baseline.json`

Persistence scores **0.99998** on Infiltration because there are 2 label
transitions in 1,115,125 windows. That is a dataset defect, not a model result.

Publishing it is a contribution to the field — it tells every other CIC-IDS-2018
team that their Infiltration numbers are meaningless — *and* it disarms the
persistence attack, because that one artifact class is the entire source of
persistence's apparent advantage.

On DoS, Netrikan beats persistence by **+0.281** (`base_w30`, DoS F1 0.9024 vs
persistence 0.6210) or **+0.269** (`metrics.json` final report, 0.8904). Always
name the checkpoint — the two differ and an unqualified number invites the
mismatch challenge.

---

# III. Adversary economics

## 15. Evasion budget — price the attack, don't just catch it
**[hypothesis]**

Measure how much an attacker must slow down or perturb to evade detection.
*"Evading Netrikan costs the attacker 4x dwell time."*

This reframes success from "we caught it" to **"we made it expensive"** — which
is the actual goal of defence, and the language an intelligence organisation
already thinks in. Nobody in this field frames results as cost imposed.

## 16. Attacker-side counterfactual
**[cheap]** — mirrors the defender engine

Perturb the recorded attack to be slower and quieter, re-run, and measure
detection degradation. Produces the evasion budget empirically rather than by
assertion, and it stress-tests your own claims before a judge does.

## 17. Chain-completion probability
**[hypothesis]**

Not "what is the next stage" but **"will this campaign reach exfiltration at
all?"** Most intrusions die before completing. Forecasting *completion* is a
different and more decision-relevant target than forecasting the next label —
it is what determines whether anyone gets out of bed.

## 18. Poisoning resistance for per-site baselines
**[design]**

Idea #1 implies learning local normality. An attacker already inside can poison
that baseline slowly. Acknowledge it, defend with robust statistics and a
held-out clean reference period.

Raising the attack on your own architecture before the panel does is a maturity
signal, and no competitor's threat model includes themselves.

---

# IV. Operator reality

## 19. Alert budget instead of a threshold
**[cheap]**

Operators do not have thresholds. They have capacity — *"I can handle 20 alerts
a shift."* Invert the problem: given a budget of N alerts, maximise attacks
caught.

That is the real operating point, it reframes FPR completely, and it is how the
customer actually thinks.

## 20. SPRT — make "when to alert" a theorem, not a hyperparameter
**[cheap]** — needs #11 first

Wald's Sequential Probability Ratio Test (1945). Accumulate log-likelihood ratio
per host; fire when it crosses a bound set by the operator's error budget.
Provably minimises expected time-to-decision for that budget.

Every competitor picks a threshold by hand. This derives it.

## 21. Conformal prediction — distribution-free coverage
**[cheap]** — needs #11 first

Guarantee: *"the true stage is in this set 95% of the time"*, with no
distributional assumptions. Caveat honestly — adversarial traffic violates the
exchangeability assumption, and saying so is worth more than the guarantee.

## 22. Map to D3FEND, not just ATT&CK
**[cheap]**

Everyone maps to ATT&CK — the *offensive* taxonomy. Almost nobody maps to
**D3FEND**, MITRE's defensive counterpart. Mapping each forecast to a named
D3FEND countermeasure turns a prediction into a prescription, in an official
MITRE framework.

This is the cleanest available answer to "what does prevention actually mean
here."

## 23. Lead time as the headline, not F1
**[cheap]** — needs #6

Operators do not buy macro-F1. Publish **median seconds of warning** with a
distribution and a floor, measured against Suricata. ThreatCast lists this as
their roadmap item #1 and has not built it.

## 24. Human-in-the-loop authorization, and the ledger that means something
**[design]**

The system proposes; the operator authorizes; the hash chain records who decided
what, when, under what model state, and why the alternatives were rejected.

Autonomous blocking on critical infrastructure is a non-starter for NTRO.
Proposal-only is not a limitation — it is the answer to the FPR problem, because
a false positive then costs ten seconds of attention rather than a downed grid.

It is also what finally makes a tamper-evident ledger worth having: it should
log **decisions**, not predictions. Over time the accept/reject history becomes
a preference model — the site's own **revealed doctrine**.

---

# V. Deployment reality

## 25. Graceful degradation under partial instrumentation
**[cheap]**

CII sites are air-gapped and unevenly instrumented. Measure performance under
feature dropout: what survives when only NetFlow is available and packet
features are gone?

Nobody tests degraded mode. NTRO lives in degraded mode.

## 26. Works on encrypted traffic, by construction
**[proven]** — every feature is already metadata

Flags, timing, volume, inter-arrival statistics. No payload inspection, so it
works on TLS 1.3, needs no decryption, and does not read citizen traffic
content.

That is a legal and privacy argument for deployability on national
infrastructure, and no competitor states it explicitly even though several
qualify.

## 27. Footprint as a feature
**[proven]**

862 KB model, CPU-only inference, runs on a laptop. Substations and telecom
cabinets do not have GPUs. Quantify latency, memory and model size and put them
on a slide — every competitor demands more hardware than the deployment site
has.

## 28. Federated attack doctrine across CII sites
**[design]** — follows from #1

If attack representations transfer and benign ones do not, sites can share
**attack models without sharing traffic**. A national defensive architecture
that is privacy-preserving by construction, derived from your own measured
result rather than asserted.

This is the slide that scales from a hackathon project to NTRO's actual mandate.

---

# VI. System design

## 29. Disagreement between the two channels is itself a signal
**[cheap]** — the two channels already exist

Netrikan fuses a rule layer and the LSTM at 0.70/0.30. Today they are silently
blended. **Surface the disagreement instead:** agreement means high confidence;
disagreement means novel or evasive traffic — the rules see nothing and the
model is alarmed, or the reverse.

Netrikan uniquely has two independent channels already built. Turning a fusion
weight into a detection channel costs almost nothing.

## 30. Multi-resolution time
**[cheap]** — the horizon sweep already exists

DoS lives in seconds, APTs live in days. One window size cannot serve both. Run
parallel models at several timescales and fuse. The k=0…1440 horizon sweep in
`models/` is already the evidence base for choosing them.

## 31. Explain the *forecast*, not the classification
**[cheap]**

Everyone runs IG or SHAP on "why is this an attack." Nobody explains **"why do
you think it will escalate."** Attribute across the rollout trajectory: which
feature in the current state is driving the predicted future transition.

Explaining a prediction about the future is a different and harder question, and
it is the one this problem statement is actually asking.

---

# Honesty discipline

What makes AttackForecast's README read as trustworthy is not its numbers — its
stage macro-F1 is 0.537 against Netrikan's 0.9221, and it produces zero
predictions for two of five stages. It is that they:

1. ran the baseline that could kill them and published the table, including
   where they lose;
2. found their own data leak and documented it with before/after numbers;
3. stated their dataset's fatal limitation in their own README.

Two of those three are writing, not engineering.

NTRO is an intelligence organisation; its panels grade sources for reliability. A
deck full of 99% numbers reads as an unreliable source. A **"Where we fail"**
slide reads as a credible one.

**Netrikan's honest failure list:** Infiltration F1 0.751 · DoS precision 0.857 ·
DAPT benign recall 0.531 · rollout drift at depth (state MSE 0.5747) ·
counterfactuals measure delay against a *non-adaptive* adversary · per-host
windowing was a net SEDI loss (0.582 → 0.498) and was not adopted.

That last one is a published negative result. Almost nobody has one.

## Resolve before any slide: the per-host framing

`WINNING_PLAN.md` headlines per-host windowing as a win (lateral recall
0.931 → 0.959). This document calls it a net loss. Both cannot be the story, and
a panel reading two of your own documents will find the mismatch.

The measured trade, from `a1_entity_compare.json`:

| | Segment | Per-host |
|---|---|---|
| Lateral recall | 0.9310 | **0.9587** |
| Attack precision | **0.4215** | 0.3940 |
| FPR | **0.4159** | 0.4993 |
| SEDI | **+0.5825** | +0.4983 |

Per-host buys +2.8pp lateral recall and pays +8.3pp FPR. On a base-rate-independent
score it is a net loss. **Framing to use:** not a default improvement — a
lateral-movement-specific view, offered as an operator toggle, with the trade
stated. Say it that way in both documents or drop it from one.
