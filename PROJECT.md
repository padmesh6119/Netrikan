# Netrikan — நெற்றிக்கண்

**AI-based Network Attack Forecasting from Network Traffic Data**

Complete technical handbook. Written so someone who has never seen this project —
and does not know security or machine learning — can read it top to bottom and
understand every part.

The name is Tamil for *the third eye*: the one that sees what the other two can't.

---

# Table of contents

1. [What we built and why](#1-what-we-built-and-why)
2. [Background: packets, flows, features](#2-background-packets-flows-features)
3. [Background: the kill chain](#3-background-the-kill-chain)
4. [Background: windows and why they make this forecasting](#4-background-windows)
5. [The dataset](#5-the-dataset)
6. [The bug we found in our own code](#6-the-bug-we-found)
7. [The model](#7-the-model)
8. [How training works](#8-how-training-works)
9. [How to read the metrics](#9-how-to-read-the-metrics)
10. [Results in full](#10-results-in-full)
11. [The signal layer](#11-the-signal-layer)
12. [The fusion layer](#12-the-fusion-layer)
13. [The forecast engine](#13-the-forecast-engine)
14. [Reading PCAP files](#14-reading-pcap-files)
15. [The dashboard](#15-the-dashboard)
16. [Cross-dataset validation](#16-cross-dataset-validation)
17. [The horizon experiment](#17-the-horizon-experiment)
18. [Experiments that failed](#18-experiments-that-failed)
19. [Known weaknesses](#19-known-weaknesses)
20. [File reference](#20-file-reference)
21. [Running everything](#21-running-everything)
22. [Glossary](#22-glossary)
23. [Questions judges will ask](#23-questions-judges-will-ask)

---

# 1. What we built and why

## The problem statement

The brief asked for a system that forecasts network attacks from traffic data. The key
word is **forecast** — not detect.

## Why existing tools are not enough

Every intrusion detection system in wide use — Snort, Suricata, and the large
majority of published ML models — treats each network event independently. It
looks at a packet or a flow and asks:

> *Is this bad?*

Three consequences follow, and all three are structural, not fixable by tuning:

**It has no memory.** A slow reconnaissance scan spread over ten minutes looks
completely normal at any single moment. Each individual probe is unremarkable.
Only the *pattern across time* is suspicious, and a memoryless classifier cannot
see patterns across time by construction.

**It alerts after the fact.** By definition it fires once malicious activity is
observable. If it can see the damage, the damage has happened.

**It cannot say what comes next.** It outputs a label — "DoS attack" — with no
notion of where the attacker is in their campaign or where they're heading.

## What we do instead

We ask:

> *Given the last 10 flows, where is this heading over the next 15 minutes?*

The output is a trajectory through the attack lifecycle:

```
Predicted over the next 15 minutes
Lateral movement, then command & control
Confidence 59% · driven by SMB port scan · port 445 probing, 6 SYN/flow, 1 RST
```

Three pieces of information, none of which a conventional IDS produces:

1. **Where it's going** — the next kill-chain stages, not the current label
2. **How sure we are** — a confidence figure
3. **Why** — the specific traffic pattern driving the prediction, named in English

## The pitch line

> Existing tools tell you an attack is happening. We tell you what happens next.

---

# 2. Background: packets, flows, features

## Packets

The lowest level of network traffic. Every piece of data crossing a network is
chopped into packets. A single web page load might be hundreds of them. Each
packet carries a source address, destination address, some control bits, and a
payload.

Working directly with packets is impractical for machine learning — there are far
too many, and individually they carry almost no meaning.

## Flows

A **flow** is one conversation. All packets between `10.0.0.5:51234` and
`10.0.0.9:445` in one session get collapsed into a single row of numbers.

Think of it like a phone bill. You don't get a recording of the call; you get who
called whom, when, how long it lasted, and how much data moved. That summary is
usually enough to tell a normal call from a suspicious one.

**Direction matters.** A flow is bidirectional and we track each way separately:

- **Forward (fwd)** — from whoever initiated the conversation
- **Backward (bwd)** — the replies coming back

This asymmetry is highly informative. A normal web request is small forward, large
backward (you send a short request, receive a big page). Data exfiltration
inverts that ratio, and that inversion is exactly what our "bulk outbound
transfer" rule looks for.

## The 24 features

Every flow becomes these 24 numbers. This is the model's entire view of the world.

### Volume — how much traffic

| # | Feature | Meaning |
|---|---|---|
| 1 | `Flow Duration` | How long the conversation lasted (microseconds) |
| 2 | `Tot Fwd Pkts` | Packet count, initiator → target |
| 3 | `Tot Bwd Pkts` | Packet count, target → initiator |
| 4 | `TotLen Fwd Pkts` | Total bytes sent forward |
| 5 | `TotLen Bwd Pkts` | Total bytes sent backward |

*Why it matters:* a port scan is many tiny short flows. A file transfer is one
long fat flow. Volume alone separates whole categories of behaviour.

### Packet size — the shape of the traffic

| # | Feature | Meaning |
|---|---|---|
| 6 | `Fwd Pkt Len Max` | Biggest forward packet |
| 7 | `Fwd Pkt Len Mean` | Average forward packet size |
| 8 | `Bwd Pkt Len Max` | Biggest backward packet |
| 9 | `Bwd Pkt Len Mean` | Average backward packet size |
| 22 | `Pkt Len Var` | Variance in packet sizes |

*Why it matters:* automated tools produce uniform packet sizes. Humans produce
varied ones. Low variance is a strong hint that software, not a person, is
driving the traffic. C2 beacons in particular are famously uniform.

### Rate — how fast

| # | Feature | Meaning |
|---|---|---|
| 10 | `Flow Byts/s` | Bytes per second |
| 11 | `Flow Pkts/s` | Packets per second |

*Why it matters:* this is the clearest DoS signature. A flood is thousands of
packets per second; normal browsing is tens.

### Timing — the rhythm

| # | Feature | Meaning |
|---|---|---|
| 12 | `Flow IAT Mean` | Average gap between packets ("inter-arrival time") |
| 13 | `Flow IAT Std` | How irregular those gaps are |
| 14 | `Flow IAT Max` | Longest pause |
| 15 | `Fwd IAT Mean` | Average gap, forward direction |
| 16 | `Bwd IAT Mean` | Average gap, backward direction |

*Why it matters:* **this is the most underrated group.** Malware beaconing home
does so on a timer — every 45 seconds, almost exactly. So `IAT Std` divided by
`IAT Mean` (the jitter ratio) is tiny. Humans are erratic; that ratio is large.
Our C2 detection rule is built directly on this.

### TCP flags — control signals

| # | Feature | Meaning |
|---|---|---|
| 17 | `FIN Flag Cnt` | Polite connection closes |
| 18 | `SYN Flag Cnt` | Connection attempts |
| 19 | `RST Flag Cnt` | Abrupt rejections |
| 20 | `PSH Flag Cnt` | "Deliver this immediately" |
| 21 | `ACK Flag Cnt` | Acknowledgements |

*Why it matters:* TCP flags are the handshake grammar of a connection.

- Many **SYN** with few **ACK** = connection attempts that go nowhere = **port scan**
- Many **RST** = connections actively refused = **failed logins / brute force**
- Enormous **SYN** with high rate = **SYN flood**

These are among the most discriminative features the model has.

### Session behaviour

| # | Feature | Meaning |
|---|---|---|
| 23 | `Active Mean` | Average length of a burst of activity |
| 24 | `Idle Mean` | Average length of a quiet gap |

*Why it matters:* separates "steady stream" from "bursts with long silences" —
the low-and-slow pattern stealthy attackers deliberately use to stay under
rate-based thresholds.

## Metadata we keep but don't feed the model

`Dst Port`, `Protocol`, `Timestamp`, `Src IP`, `Dst IP` are carried alongside.
They're used by the **signal rules** (§11) to name what's happening — port 445 is
file sharing, port 22 is remote login — but they are deliberately **not** model
inputs. Section 18 explains that decision, which we tested and reversed.

---

# 3. Background: the kill chain

An attack is never one event. It's a campaign with phases, and MITRE ATT&CK is the
industry-standard naming scheme for them.

## The stages we model

| Stage | ATT&CK ID | What the attacker is doing | Typical traffic |
|---|---|---|---|
| **Benign** | — | Nothing. Normal users. | Ordinary web, DNS, mail |
| **Initial Access** | TA0001 | Getting in — brute force, exploiting a public server | Repeated logins, scanning, many RST |
| **Lateral Movement** | TA0008 | Already inside, moving between machines | SMB/RDP between internal hosts |
| **Command & Control** | TA0011 | Compromised machine takes orders from attacker | Regular small callbacks |
| **Exfiltration** | TA0010 | Stealing the data | Large sustained outbound transfer |
| **DoS / Impact** | TA0040 | Knocking a service offline | Massive packet floods |

## On-chain vs off-chain

Four of these form a sequence:

```
Initial Access  →  Lateral Movement  →  Command & Control  →  Exfiltration
```

**DoS is off-chain.** It's not a step toward stealing data; it's a different
objective entirely (disruption). It's handled separately in the forecast — a DoS
prediction only fires if DoS probability is genuinely rising, never as part of the
progression.

## Why this framing matters

Naming the stage tells a defender **how urgent this is and what to do**:

- Initial Access → block the source IP, you have time
- Lateral Movement → isolate the host, you're already breached
- Exfiltration → cut the link now, data is leaving

"Class 3, confidence 0.87" tells them nothing actionable. "Lateral movement,
heading to exfiltration" tells them exactly what to do.

---

# 4. Background: windows

## The single most important design decision

The model never looks at one flow alone. It looks at a **window** of 10
consecutive flows and predicts the stage of the **next** flow.

From `pipeline.py`:

```python
X.append(vals[i : i + WINDOW])   # flows i, i+1, ... i+9   ← what it SEES
y.append(labels[i + WINDOW])     # label of flow i+10      ← what it PREDICTS
```

That `+ WINDOW` on the label line is the whole thesis of the project.

## Why this is forecasting, not classification

Almost every other CIC-IDS-2018 project writes:

```python
y.append(labels[i + WINDOW - 1])   # the label of the LAST flow in the window
```

That asks *"what was this traffic I just saw?"* — classification. Describing the
past.

We ask *"what comes after this traffic?"* — forecasting. The model must learn how
attacks **progress**, not just what they look like.

It is a one-character difference in code and a completely different problem.

## The sliding window

Windows overlap, stepping forward one flow at a time:

```
window 0:  flows 0–9    → predict flow 10
window 1:  flows 1–10   → predict flow 11
window 2:  flows 2–11   → predict flow 12
```

This gives maximum training data from a fixed dataset — 5.5 million windows from
5.5 million flows. **It also creates a subtle trap that invalidated our first
results entirely.** See §6.

## Window size

We tested 10, 20 and 30. Bigger is better — more history, more context:

| Window | macro-F1 |
|---|---|
| 10 | 0.894 |
| 20 | 0.918 |
| 30 | **0.922** |

Cost: a 30-flow window is 3× the computation and 3× the storage of a 10-flow one.

---

# 5. The dataset

## CIC-IDS-2018 (training)

From the Canadian Institute for Cybersecurity. Seven days of network traffic from
a realistic test network, with real attacks executed on a schedule and every flow
labelled with what it was.

Already converted to flow CSVs with CICFlowMeter, so we start from features rather
than raw packets.

## Class distribution after cleaning

5,549,436 windows:

| Class | Label | Count | Share |
|---|---|---|---|
| 0 | Benign | 4,135,033 | 74.51% |
| 1 | InitialAccess | 381,509 | 6.87% |
| 2 | DoS | 654,300 | 11.79% |
| 3 | Infiltration | 92,403 | **1.67%** |
| 4 | Botnet | 286,191 | 5.16% |

**Imbalance ratio 45:1** between the largest and smallest class. This single fact
drives many later design decisions, and it's why Infiltration is our weakest class
throughout.

## How raw labels map to our 5 classes

```python
LABEL_MAP = {
    'Benign':                   0,
    'FTP-BruteForce':           1,   # password guessing on FTP
    'SSH-Bruteforce':           1,   # password guessing on SSH
    'Brute Force -Web':         1,   # web login guessing
    'Brute Force -XSS':         1,   # cross-site scripting attempts
    'SQL Injection':            1,   # database injection
    'DoS attacks-Hulk':         2,   # HTTP flood
    'DoS attacks-SlowHTTPTest': 2,   # slow-request exhaustion
    'DoS attacks-GoldenEye':    2,   # HTTP flood variant
    'DoS attacks-Slowloris':    2,   # holds connections open
    'Infilteration':            3,   # (sic — the dataset misspells it)
    'Bot':                      4,   # botnet C2 traffic
}
```

Everything that gets an attacker *in* becomes class 1. Everything that takes a
service *down* becomes class 2.

## Why Infiltration is the hard one

Two reasons compound:

1. **It's rare** — 92,403 windows against 4.1 million benign. The model sees 45
   benign examples for every infiltration example.
2. **It's designed to look normal.** That is the entire point of infiltration —
   an attacker already inside, deliberately blending with legitimate traffic.
   Brute force is loud; infiltration is quiet on purpose.

Every other class scores above 0.90. Infiltration sits at 0.751. That gap is
inherent to the problem, not a bug.

## Cleaning

Rows with `inf` or `NaN` (from divide-by-zero in rate calculations on
zero-duration flows) are dropped. Repeated header rows inside the CSVs are
removed. Labels outside the map are excluded.

## DAPT 2020 (testing only, never trained on)

A completely separate dataset used to answer: *does this work on a network it has
never seen?* Full detail in §16.

---

# 6. The bug we found

**This section describes the most important thing in the project.**

## What happened

The first trained model reported strong validation scores. Those scores were
meaningless. Here is why.

## The mechanism

Windows are built with stride 1, so consecutive windows overlap heavily:

```
window 5000:  flows 5000 5001 5002 5003 5004 5005 5006 5007 5008 5009
window 5001:       flows 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010
                         └──────────── 9 of 10 flows identical ────────────┘
```

The original training code then split those windows **randomly**:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Random splitting puts window 5000 in training and window 5001 in validation. They
share 90% of their content. The model is validated on data it memorised.

## We measured it

```
val windows sampled  : 200,000
sharing rows w/ train: 200,000  (100.00%)
```

**Every single validation window contained flows the model trained on.** Not a
partial leak — total contamination.

## Why this matters far beyond our project

The overwhelming majority of published CIC-IDS-2018 work does exactly this, and
reports F1 scores of 0.98–0.99. Those numbers are inflated by this mechanism.

This is not a niche observation. It is a systematic flaw in how a whole
literature evaluates itself.

## Our fix: blocked split with purge gaps

Implemented in `train_v2.py:blocked_split()`:

1. Cut the timeline into **500 contiguous blocks**
2. Assign each **whole block** to train or validation — never split inside a block
3. **Discard 9 windows at every block edge** (9 = WINDOW − 1, the maximum reach of
   an overlap)
4. Stratify block assignment by dominant class, so all 5 classes appear on both
   sides

Step 3 is the critical one. Nine is exactly the number of windows that could
straddle a boundary, so purging them guarantees zero shared rows.

## Verification

```
train 4,421,270  val 1,119,166  purged 9,000
val windows overlapping train: 0 / 300,000  (0.00%)

class            train        val     held out
Benign       3,292,440    835,841       20.2%
InitialAccess  299,215     81,666       21.4%
DoS            532,290    120,940       18.5%
Infiltration    74,270     18,007       19.5%
Botnet         223,055     62,712       21.9%
```

Zero leakage, all classes present on both sides, roughly 20% held out. Cost: 9,000
purged windows out of 5.5 million — 0.16%.

## The consequence

Scores dropped. That is correct and expected. **Every number in this document is
post-fix.** Our 0.922 is a real 0.922. A competitor's 0.99 probably is not.

---

# 7. The model

## What an LSTM is

A **Long Short-Term Memory** network is a neural network built to read sequences.

An ordinary neural network takes a fixed set of inputs and produces an output. It
has no concept of order or history. An LSTM reads items one at a time and keeps a
running internal **memory** that it updates at each step.

The "gates" are learned filters deciding, at each step, what to add to memory,
what to forget, and what to output. In practice this means it can learn things
like *"a SYN spike matters much more if it follows a quiet period than if the line
was already busy."*

That is exactly the reasoning our problem needs, and exactly what a memoryless
classifier cannot do.

## Our architecture

From `model.py`:

```python
class WorldModel(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=2,
                 num_stages=5, dropout=0.3):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head  = nn.Linear(64, num_stages)   # 5-class output
        self.breach_head = nn.Linear(64, 1)            # single probability

    def forward(self, x):
        out, _ = self.lstm(x)          # read all 10 flows
        last = out[:, -1, :]           # keep memory state after the last flow
        shared = self.fc(last)         # shared 64-dim representation
        return self.stage_head(shared), \
               torch.sigmoid(self.breach_head(shared)).squeeze(1)
```

### Layer by layer

**Input** — shape `(batch, 10, 24)`: a batch of windows, each 10 flows, each flow
24 numbers.

**LSTM, 2 layers, 128 hidden units** — reads the 10 flows in order, maintaining a
128-number memory. Two layers means the second reads the first's output, letting
it build higher-level patterns. Dropout 0.3 randomly zeroes 30% of connections
during training, which prevents memorisation.

**Take the last timestep** — `out[:, -1, :]` keeps only the memory state after
reading all 10 flows. That vector is the model's summary of the whole window.

**Shared layer** — compresses 128 → 64 with a ReLU. Both output heads read from
this same representation.

**Two output heads:**

- `stage_head` → 5 numbers, one per class. Highest wins.
- `breach_head` → 1 number through a sigmoid, giving 0–1 probability that
  something malicious is happening.

### Why two heads

The stage head answers *what kind*; the breach head answers *how bad*. Sharing the
body means learning "is this an attack" also improves "which attack" — a standard
multi-task benefit. The breach head also gives a smooth 0–1 score for ranking
alerts, which a hard 5-way choice cannot.

## Size

Roughly 220,000 parameters — small by modern standards. It trains in ~25 minutes
on a laptop GPU and runs inference on CPU with no accelerator. That matters
directly for the "deployable in an air-gapped SOC" claim.

---

# 8. How training works

## The imbalance problem

74.5% of data is benign. A model that always answers "benign" gets 74.5% accuracy
and is completely useless. Training must be forced to care about rare classes.

## Weighted sampling

We use `WeightedRandomSampler` with **square-root inverse frequency**:

```python
w_class = 1.0 / np.sqrt(counts)
w_class /= w_class.sum()
sampler = WeightedRandomSampler(w_class[y[tr_idx]], num_samples=..., replacement=True)
```

**Why square root, not plain inverse?** Plain inverse frequency equalises classes
perfectly, which sounds right but oversamples Infiltration by ~45×. With only
74,270 unique examples, the model sees each one dozens of times per epoch and
memorises them. Square root gives a gentler ~6.7× correction — enough to matter,
not enough to overfit.

## One correction, not two

The original code used a weighted sampler **and** class-weighted loss, both at
inverse frequency — correcting the same imbalance twice, over-shooting toward rare
classes. We now use the sampler alone and leave the loss unweighted.

## Loss function

```python
loss = CrossEntropyLoss()(stage_logits, y) + 0.5 * BCELoss()(breach_prob, y > 0)
```

Two terms, one per head. Cross-entropy for the 5-way choice, binary cross-entropy
for the 0–1 breach score. The `0.5` makes the stage task primary — it's the harder
and more useful one.

## Other training details

| Setting | Value | Why |
|---|---|---|
| Optimiser | Adam, lr 1e-3 | Standard, robust default |
| Batch size | 1024 | Fits 4GB VRAM comfortably |
| Gradient clipping | max norm 5.0 | Stops rare-class batches destabilising training |
| LR schedule | halve after 2 stalled epochs | Fine-tunes once progress slows |
| Early stopping | patience 5, on macro-F1 | Stop when it stops improving |
| Samples/epoch | 1.2M | Enough to learn, fast enough to iterate |

## Model selection on macro-F1, not loss

**This matters.** Validation loss is dominated by the benign class. A model can
lower its loss by getting benign slightly more right while getting Infiltration
much more wrong. Macro-F1 weights all 5 classes equally, so Infiltration counts as
much as Benign. We checkpoint on macro-F1.

## Memory handling

`X.npy` is 5.3GB (16GB at window 30) on a machine with 15GB RAM. The original code
loaded it fully, then `train_test_split` copied it — ~11GB peak, which barely fits
and fails outright at window 30.

`WindowDataset` memory-maps the file and reads only the indices it needs. The
memmap is opened lazily inside each worker process, since a memmap handle cannot
be pickled across processes.

Similarly, `pipeline_v2.py` writes windows **directly into a memmap** rather than
accumulating a Python list. The original approach needs several times the array
size in RAM; the memmap approach has no ceiling. That's why window 30 is feasible
at all.

---

# 9. How to read the metrics

## Precision and recall

Suppose the model flags 100 flows as attacks, 70 are real, and it missed 30 real
attacks elsewhere.

- **Precision** = 70/100 = **0.70** — when it shouts, how often is it right?
  Low precision = false alarms = analysts stop trusting it.
- **Recall** = 70/100 real attacks = **0.70** — of all real attacks, how many did
  it catch? Low recall = misses = attacks get through.

They trade off. Flag everything → perfect recall, terrible precision. Flag only
certainties → perfect precision, terrible recall.

## F1

The harmonic mean:

```
F1 = 2 × (precision × recall) / (precision + recall)
```

Harmonic rather than plain average because it punishes imbalance. Precision 1.0
with recall 0.0 gives F1 = 0, not 0.5. You cannot game it by maximising one side.

## Macro-F1

Compute F1 for each of the 5 classes, then average them **equally**.

This is our headline metric because it refuses to let the model ignore rare
classes. Infiltration is 1.67% of data but 20% of macro-F1. An "always benign"
model scores macro-F1 ≈ 0.17 — correctly terrible.

## Why not accuracy

Accuracy on this dataset is actively misleading:

- Always-benign model: **74.5% accuracy**, completely useless
- Our model: 96% accuracy — sounds barely better, is vastly better

Accuracy hides everything that matters when classes are imbalanced. Ignore it.

## Confusion matrix

A grid of true class (rows) against predicted class (columns). The diagonal is
correct predictions; everything off-diagonal is a specific mistake. It tells you
*which* errors, not just how many — see §10 for how we used it.

---

# 10. Results in full

All numbers on the leak-free blocked split, 1.1M validation windows.

## Model comparison

| Model | Features | Window | macro-F1 | Infiltration | DoS |
|---|---|---|---|---|---|
| Original | 24 | 10 | 0.885 | 0.604 | 0.890 |
| **base_w30** ← best | 24 | 30 | **0.922** | **0.751** | 0.902 |
| ports_w30 | 36 | 30 | 0.925 | 0.755 | 0.907 |

## Per-class, original model (24 features, window 10)

```
                precision  recall     F1    support
Benign            0.985    0.966   0.976   835,841
InitialAccess     0.972    0.965   0.969    81,666
DoS               0.845    0.941   0.890   120,940
Infiltration      0.573    0.638   0.604    18,007
Botnet            0.976    0.999   0.987    62,712
                                   ─────
                        macro-F1   0.885
```

## Per-class, base_w30 (24 features, window 30)

```
                precision  recall     F1    support
Benign            0.987    0.970   0.978   832,795
InitialAccess     0.988    0.994   0.991    81,365
DoS               0.857    0.953   0.902   120,504
Infiltration      0.771    0.732   0.751    17,983
Botnet            0.975    0.999   0.987    62,478
                                   ─────
                        macro-F1   0.922
```

## Where the improvement came from — the confusion matrices

**Original:**

```
                  Benign  InitialAcc     DoS  Infiltration  Botnet
Benign           807,319         701  18,735         7,559   1,527
InitialAccess         29      78,848   2,084           705       0
DoS                5,294       1,564 113,768           314       0
Infiltration       6,499           1      15        11,492       0
Botnet                72           0       0             2  62,638
```

**base_w30:**

```
                  Benign  InitialAcc     DoS  Infiltration  Botnet
Benign           807,780         849  18,766         3,825   1,575
InitialAccess         99      80,908     345            13       0
DoS                5,512         134 114,796            62       0
Infiltration       4,805           5       6        13,167       0
Botnet                75           0       0             1  62,402
```

Two specific cells did the work:

- **Benign wrongly called Infiltration: 7,559 → 3,825.** False alarms halved.
  This is what lifted Infiltration precision 0.573 → 0.771.
- **Infiltration wrongly called Benign: 6,499 → 4,805.** Missed attacks down 26%.
  This lifted recall 0.638 → 0.732.

Both errors were the same underlying confusion — infiltration traffic and normal
traffic look alike — and 20 extra flows of history let the model separate them.

## Is the LSTM worth it? (the baseline)

We trained logistic regression, a simple linear model, on identical data:

| Model | What it sees | macro-F1 | Infiltration |
|---|---|---|---|
| Logistic regression | 1 flow (36 numbers) | 0.607 | 0.077 |
| Logistic regression | 10 flows flattened (360 numbers) | 0.742 | 0.140 |
| **LSTM** | 10 flows as a sequence | **0.889** | **0.627** |
| **LSTM** | 30 flows as a sequence | **0.922** | **0.751** |

This decomposes the value cleanly:

- Giving a linear model **sequence data**: **+0.135**
- Replacing linear with an **LSTM**: **+0.147**

On Infiltration the gap is stark: **0.14 vs 0.75**. The linear model essentially
cannot find infiltration at all.

**This is the direct answer to "why not just use something simple."**

## Stability

`ports_w10` scored 0.8937; a second run with a different random seed scored
0.8949. **Variance ≈ ±0.001**, so differences of 0.03+ between models are real,
not noise.

---

# 11. The signal layer

## The problem it solves

The model outputs `class 3, probability 0.87`. An analyst cannot act on that. They
need to know *what is actually happening*.

`signals.py` contains 14 hand-written rules that inspect the same window and name
the pattern in English.

## How a rule works

```python
if top_port in (445, 139) and dur < 200000 and syn >= 3:
    cands.append(("SMB port scan", 0.88, INITIAL_ACCESS,
                  f"port 445 probing, {syn:.0f} SYN/flow, {rst:.0f} RST"))
```

In words: *if traffic targets file-sharing ports, flows are shorter than 200ms,
and there are 3+ connection attempts per flow — call it an SMB port scan.*

Each rule returns four things:

1. **Name** — `"SMB port scan"`
2. **Confidence** — `0.88`, hand-assigned by how specific the pattern is
3. **Stage hint** — which kill-chain stage it implies
4. **Evidence** — a sentence with the actual numbers filled in

When several rules match, the highest-confidence one wins and the rest are kept as
alternatives.

## All 14 rules

### Reconnaissance / scanning

| Signal | Triggers on | Implies |
|---|---|---|
| Port sweep | 5+ distinct ports, flows < 50ms, 3+ SYN | Initial Access |
| SMB port scan | port 445/139, flows < 200ms, 3+ SYN | Initial Access |

### Brute force

| Signal | Triggers on | Implies |
|---|---|---|
| *(service)* brute force | port 22/21/23/3389/5900, RST ≥ 1, > 50 pkt/s | Initial Access |
| Web credential stuffing | port 80/443, short flows, PSH ≥ 2, > 80 pkt/s | Initial Access |

### Denial of service

| Signal | Triggers on | Implies |
|---|---|---|
| SYN flood | SYN > 15, > 500 pkt/s, flows < 20ms | DoS |
| Volumetric flood | > 400 KB/s, > 40 fwd pkts | DoS |
| Slow HTTP DoS | port 80/443, flows > 0.5s, < 10 pkt/s, high IAT | DoS |

### Command & control

| Signal | Triggers on | Implies |
|---|---|---|
| C2 beaconing | IAT jitter < 0.25, IAT 5–150ms, payload < 2KB, < 40 pkt/s | C2 |
| DNS tunneling | port 53, > 80 KB/s, queries > 2KB | C2 |

### Lateral movement / exfiltration

| Signal | Triggers on | Implies |
|---|---|---|
| *(service)* lateral transfer | port 445/139/3389/5985, > 3KB fwd, > 0.1s | Lateral |
| Bulk outbound transfer | > 0.4s, > 5KB backward, ratio > 2.5:1 | Exfiltration |
| Low-and-slow probing | IAT > 150ms, < 15 pkt/s, > 0.3s, FIN ≥ 1 | Lateral |

### Fallback

| Signal | Triggers on |
|---|---|
| Normal traffic | nothing else matched |

## The C2 beaconing rule, explained

The most interesting one:

```python
periodicity = iat_std / (iat_mean + 1e-6)
if 0 < periodicity < 0.25 and 5000 < iat_mean < 150000 \
   and fwd_len < 2000 and pkts_s < 40:
```

`periodicity` is the **jitter ratio** — how irregular the timing is relative to
its average. Malware phoning home does so on a timer, so its gaps are almost
identical and the ratio approaches zero. Humans are erratic and score high.

A ratio under 0.25 with small consistent payloads is close to a signature for
automated callbacks. Note the extra guards (`pkts_s < 40`, payload cap) — an
earlier looser version fired on ordinary benign DNS traffic, and tightening it was
necessary to stop false alarms.

## These are rules, not AI

**There is no language model here.** Fourteen `if` statements and f-string
templates. Say this plainly if asked — a judge who opens the file will see it
immediately, and pretending otherwise is the only way to lose points for it.

---

# 12. The fusion layer

## Why fuse at all

Two independent sources of evidence about the same window:

- The **LSTM** — learned from 5.5M examples, catches subtle statistical patterns,
  but is a black box and can be wrong on unfamiliar traffic
- The **rules** — transparent and precise, but only cover patterns we thought of

Neither alone is best. `infer.py:_fuse()` combines them:

```python
MODEL_WEIGHT = 0.30
fused = MODEL_WEIGHT * model_probs + (1 - MODEL_WEIGHT) * signal_evidence
```

## Why 0.30 (model) and 0.70 (rules)

Set when the model was mid-training and unreliable on unseen traffic. The rules
were the more trustworthy channel, so they carry the larger share.

**This should be revisited.** With a properly trained model at macro-F1 0.922, a
higher model weight is likely better. It is a tunable constant at `infer.py:36`,
not a fundamental design choice.

## Conceptual note

Combining a learned model with rule-based evidence is essentially what Darktrace's
Bayesian meta-classifier does. Ours is simpler — a fixed weighted average rather
than a learned combiner — but the structure is the same and it is defensible.

---

# 13. The forecast engine

## The core idea

We know the current stage. We want to know where it goes. That requires a model of
**stage transitions** — given that an attacker is in Initial Access, what's next?

We use a **Markov chain**: a table of probabilities of moving from each stage to
each other stage, applied repeatedly to project forward.

## The transition matrix

A 6×6 table. Row = current stage, column = next stage, cell = probability. Each
row sums to 1.

```
              Benign     IA    DoS  Lateral     C2  Exfil
Benign         0.980  0.011  0.005    0.002  0.001  0.001
IA             0.015  0.929  0.006    0.038  0.010  0.001
DoS            0.078  0.011  0.900    0.007  0.003  0.001
Lateral        0.008  0.006  0.003    0.931  0.042  0.010
C2             0.003  0.001  0.001    0.008  0.968  0.019
Exfil          0.012  0.002  0.002    0.008  0.046  0.930
```

## The diagonal is measured, the off-diagonal is doctrine

**Be precise about this if asked.**

- **Diagonal (persistence)** — measured from DAPT 2020's 86,691 labelled flows.
  P(stay in the same stage) is 0.93–0.97.
- **Off-diagonal (where it goes when it moves)** — follows ATT&CK doctrine,
  scaled so each row sums to 1.

**Why the off-diagonal isn't learned:** DAPT contains a single campaign with only
7 stage changes total. Seven observations cannot produce a probability
distribution. You would need many independent campaigns.

## The correction this forced

Our first hand-written matrix guessed persistence at **0.43** — an attacker moves
on about half the time. The measured value is **0.93**.

**The forecast was escalating roughly ten times too fast**, predicting "lateral,
then exfiltration" almost immediately when a real attacker sits in one phase far
longer. Measuring it fixed a genuine error in the system's behaviour.

## Projecting forward

Repeated matrix multiplication:

```python
p = current_stage_distribution
for _ in range(steps):
    p = p @ TRANSITION
```

Each multiplication advances one step, where a step is one window of new traffic
(10 flows ≈ 10 seconds). For a 15-minute horizon that is 90 steps.

## Turning projections into a sentence

`_trajectory()` picks two stages:

1. The **immediate next stage** — the first on-chain stage ahead of the current one
2. The **eventual destination** — the furthest on-chain stage still plausible
   (peak probability ≥ 0.12) within the horizon

Then `_phrase()` glues them:

```python
return f"{_cap(short(seq[0]))}, then {short(seq[1])}"
```

`short(3)` → `"lateral movement"`, `short(4)` → `"command & control"` → **"Lateral
movement, then command & control"**.

Table lookup plus string concatenation. Nothing generative.

## Confidence

```python
confidence = 0.5 * (1 - end[BENIGN]) + 0.5 * concentration
```

Two halves:

- **How sure this isn't benign** — `1 − P(benign)` at the horizon end
- **How concentrated** — is the probability mass actually on the predicted path,
  or spread across many possibilities?

Both high → confident. Either low → hedged.

## Why the horizon is adjustable

DAPT's real campaign advanced roughly **one stage per day**:

```
Reconnaissance      first seen  Jul 16  12:12
Establish Foothold  first seen  Jul 17  14:45   (+26h)
Lateral Movement    first seen  Jul 18  12:28   (+22h)
Data Exfiltration   first seen  Jul 19  16:31   (+28h)
```

At a 60-second horizon with real persistence values, **nothing escalates** — which
is honest but produces a boring demo. At 15 minutes escalation appears with 111
seconds of lead time. At 1 hour+ the chain reaches steady state and saturates.

**The original "next 60 seconds" claim does not survive contact with the measured
data for APT-style intrusions.** 60 seconds is right for volumetric floods, which
genuinely move that fast. The dashboard therefore offers 60s / 15min / 1h / 6h /
24h, defaulting to 15 minutes.

## Lead time

The headline number. Two indices:

- **forecast_idx** — first window where predicted damage risk ≥ 0.45, sustained
  3 windows, while not yet in a damage stage
- **detect_idx** — first window actually in a damage stage, sustained 3 windows
  (what a conventional IDS would need to fire)

`lead_seconds = (detect_idx − forecast_idx) × flow_interval`

On the demo scenario: forecast at T+76s, conventional detection at T+187s →
**111 seconds of advance warning**. The 3-window "sustain" requirement prevents a
single noisy window from creating a fake lead time.

---

# 14. Reading PCAP files

## Why this matters

CSV upload assumes someone already converted traffic to flows. Real network
captures are PCAP files. Accepting PCAP directly is what makes the tool usable
rather than a lab exercise.

## The pipeline

```
capture.pcap
     ↓  tshark -T fields
raw packet rows (time, src, dst, ports, proto, length, TCP flags)
     ↓  group by canonical 5-tuple
bidirectional flows
     ↓  compute statistics
24 features + Dst Port + Protocol + Timestamp
     ↓
same path as CSV input
```

## Grouping into flows

The tricky part: packets A→B and B→A belong to the **same** flow. We build a
canonical key by sorting the two endpoints:

```python
key = (x, y, proto) if x <= y else (y, x, proto)
```

Both directions then produce an identical key. Direction is assigned from the
first packet seen: whoever spoke first is "forward".

## Computing the features

- **IAT** — `np.diff` on packet timestamps, converted to microseconds
- **TCP flags** — `tcp.flags` is a hex bitmask; we test bits
  (FIN 0x01, SYN 0x02, RST 0x04, PSH 0x08, ACK 0x10)
- **Active/Idle** — gaps longer than 1 second end a burst; we average the bursts
  and the gaps separately

## Tested on real captures

`sosem.pcapng` → **642 flows → 632 windows**, and the signal layer flagged *Slow
HTTP DoS*, *Bulk outbound transfer*, and *Low-and-slow probing*.

Small CTF captures (`icmp.pcap`, `exfil_hard.pcap`) produce only 1–2 flows —
single conversations, not enough for a 10-flow window. The app reports this
clearly rather than erroring.

## Honest caveat

**This is our reimplementation of CICFlowMeter, not the real tool.** The model
trained on genuine CICFlowMeter output, so small definitional differences exist —
our active/idle threshold is 1 second, theirs is 5 by default.

That is a **train/serve skew** risk: features at demo time are computed slightly
differently from features at training time. Using the real CICFlowMeter jar would
close the gap, and Java 21 is already installed on this machine.

---

# 15. The dashboard

`streamlit run src/app.py`

## Layout, top to bottom

**Header** — the eye mark (inline SVG, no image file: almond outline, amber iris
doubling as a radar crosshair, rotating once per 9 seconds), the name, and the
live model score.

**Controls** — demo scenario picker, forecast horizon picker, and an upload box
accepting `.pcap`, `.pcapng`, `.cap`, `.csv`.

**Playback slider** — moves through the capture in time. It opens at the forecast
moment, the most interesting point. Below it: `Viewing T+76s of 249s · forecast at
T+76s · signature IDS at T+187s`.

**Stat strip** — escalation risk, current stage, lead time.

**Forecast panel** — the headline prediction, confidence, and driving signal.

**Kill chain** — four boxes. Amber = current, dashed = predicted, dim = not
reached.

**Escalation timeline** — risk over time with two vertical markers: green "we
forecast", red dashed "others alert". **The gap between those two lines is the
product.**

**Detail expander** — everything technical, collapsed by default: feature
attribution, model provenance, horizon curve.

## Design decisions

Earlier versions were rejected for looking generic — glowing cards, tiny uppercase
letterspaced labels, rainbow-coloured boxes, an unlabelled bright green slider
that read as a mystery progress bar. Current version: flat readout strip with
hairline dividers, sentence-case labels, one meaningful accent colour (risk), a
labelled scrubber, everything technical hidden behind one expander.

## Demo determinism

`demo_data.generate()` reseeds its RNG on every call. An earlier version used a
module-level RNG that advanced each call, so the same scenario produced different
traffic on every rerun — fatal for a live demo where you rerun a scenario mid-
sentence.

---

# 16. Cross-dataset validation

## The question

Everything so far is one dataset. Did the model learn **what attacks look like**,
or **what CIC-IDS-2018 looks like**? Only a second dataset can tell you.

## DAPT 2020

A public dataset of a real 4-day APT campaign, free with no registration:
`https://gitlab.com/asu22/dapt2020`

Why it's the right choice:

- **Same feature format** — also CICFlowMeter, so features are compatible after a
  column rename (`Total Fwd Packet` → `Tot Fwd Pkts`, etc.)
- **It has a `Stage` column** — actual kill-chain phase per flow, not just an
  attack name. This is what let us measure real persistence.
- **Different everything else** — different network, year (2019 vs 2018),
  attackers, and traffic mix

## What's in it

86,691 flows, one attack phase per day:

| Day | Phase | Attack flows |
|---|---|---|
| Monday | benign baseline | 0 |
| Tuesday | Reconnaissance | 11,865 |
| Wednesday | Establish Foothold | 8,592 |
| Thursday | Lateral Movement | 2,451 |
| Friday | Data Exfiltration | 15 |

## Results

Trained on CIC-IDS-2018, tested on DAPT, never fine-tuned:

| Model | Attack recall | Precision | Lateral Movement |
|---|---|---|---|
| Original (24, w10) | **0.839** | 0.422 | 93.1% |
| base_w30 (24, w30) | 0.797 | 0.381 | 94.2% |
| ports_w30 (36, w30) | 0.757 | 0.303 | 96.6% |

## Interpretation — the honest version

**The good:** ~80% of attacks caught on a network the model has never seen, with
Lateral Movement — the stage our whole story depends on — at 94%.

**The bad:** precision ~40%. Many benign flows flagged as attacks. In production
that means alert fatigue.

**The specific failure:** the confusion matrix shows nearly everything gets pushed
into the Infiltration class, and `InitialAccess` was predicted **zero times**
across all 86,591 windows. The model generalises on *detection* (is this an
attack) far better than on *classification* (which attack).

**Say this out loud in the pitch.** A team that measured its own generalisation
gap and can explain it is more credible than one that never looked.

## What DAPT could not give us

We hoped to **learn** the transition matrix from it. We could not:

```
cross-stage transitions between consecutive flows: 0
```

Zero — because each attack phase occupies its own day, so consecutive flows never
cross a stage boundary. Bucketing by time recovers the ordering (at 6-hour
granularity it's textbook Recon → Foothold → Lateral → Exfiltration) but yields
only **7 transitions**, nowhere near enough to estimate probabilities.

What it *did* give us: measured persistence (0.93–0.97) and the real escalation
timescale (~1 day per stage). Both changed the system materially.

---

# 17. The horizon experiment

## The question

The system claims to forecast. **How far ahead does that actually work?**

Nobody publishes this. Everyone reports accuracy at detection; nobody reports
accuracy versus lead time.

## Method

Train a separate model for each lead distance *k*, changing only the prediction
target:

```python
y_k[i] = y[i + k]      # predict the stage k flows further ahead
```

No new data needed — the labels already exist, just shifted. `X.npy` untouched.

## A bug we found and fixed mid-experiment

The first sweep produced an impossible result: **k=360 scored 0.885 while k=30
scored 0.873.** Predicting 12× further ahead cannot be easier.

**Cause:** `blocked_split` stratified blocks by the *shifted* labels, so each *k*
got a slightly different validation set. The runs were not comparable.

**Fix:** a `--fixed-split` flag pins the blocks to the unshifted labels so only the
prediction target changes. Rerun, the anomaly vanished, and the curve became
monotonic.

**Use the `fx_k*` numbers. The earlier `ports_w10_k*` numbers are invalid.**

## The corrected curve

| Lead time (flows) | macro-F1 | Infiltration | DoS |
|---|---|---|---|
| 0 | 0.889 | 0.627 | 0.887 |
| 30 | 0.870 | 0.623 | 0.817 |
| 90 | 0.866 | 0.631 | 0.791 |
| 180 | 0.862 | 0.616 | 0.791 |
| 360 | 0.862 | 0.629 | 0.791 |

## What it means

**Predicting 360 flows ahead costs 0.027 macro-F1.** The curve drops slightly then
flattens completely after k=30.

**Lead time is nearly free.** Attack state is highly persistent — corroborated
independently by DAPT's measured 0.93 persistence, from a different dataset by a
different method. Two measurements agreeing is stronger than either alone.

**Per-class asymmetry:**

- **Infiltration is completely flat** (0.627 → 0.629). Intrusions persist, so
  seeing further ahead costs nothing.
- **DoS takes the whole hit** (0.887 → 0.791). Floods are bursty and
  time-sensitive, so forecasting them is genuinely harder.

That asymmetry is good evidence the result is real rather than an artefact — it
matches how these attacks actually behave.

## The caveat to state first

The curve is flat **partly because** attack state persists. The model may be
reading "an attack is ongoing" rather than truly predicting a transition. Say this
before a judge does; it converts a weakness into evidence you understand your own
result.

---

# 18. Experiments that failed

Recording these matters. A judge asking "what didn't work?" gets a real answer.

## Port features

**Idea:** the model never sees which port traffic went to. Port 445 is file
sharing, 22 is remote login — exactly the services an intruder abuses. Adding
this should help Infiltration.

**Implementation:** the raw port number is useless as a scalar (443 and 445 are
numerically adjacent, semantically unrelated), so we expanded it into 12 binary
role features — `svc_web`, `svc_smb`, `svc_rdp`, `svc_remote`, `svc_dns`,
`svc_db`, `svc_mail`, `svc_winrm`, `svc_ephemeral`, `svc_wellknown`, `proto_tcp`,
`proto_udp`. 24 features → 36.

**Sanity check passed:** on the FTP-BruteForce day, `svc_remote` fired on 99.76%
of flows. Real signal.

**Initial result looked excellent:** 0.885 → 0.925, Infiltration 0.604 → 0.755.

**But we changed two things at once** — port features *and* window 10 → 30. So we
ran the control: window 30 with the **original 24 features**.

| Model | Features | Window | macro-F1 | DAPT recall |
|---|---|---|---|---|
| ports_w30 | 36 | 30 | 0.925 | 0.757 |
| base_w30 | 24 | 30 | 0.922 | **0.797** |

**The port features were worth 0.003.** Essentially nothing. All the gain came
from the longer window.

**And they actively hurt generalisation** — cross-dataset recall dropped from
0.797 to 0.757. Ports are environment-specific: the model started memorising which
services *this* network runs instead of learning what attacks look like.

**Decision: dropped.** `base_w30` uses the original 24 features.

**The lesson:** changing two variables at once nearly led us to ship a worse model
while crediting the wrong cause. The control run cost 35 minutes and reversed the
conclusion.

## Learning the transition matrix from DAPT

Covered in §16. Zero cross-stage transitions between consecutive flows, only 7
after time-bucketing. Not enough. We kept doctrine for the ordering and used DAPT
only for persistence.

---

# 19. Known weaknesses

Know these before a judge finds them.

1. **Infiltration is still weakest** — 0.751 against 0.90+ for everything else.
   Rarest class, and designed to look normal. Improved a lot (from 0.604) but
   still the floor.

2. **Cross-dataset precision ~40%** — many false positives on an unfamiliar
   network. Detection generalises; classification doesn't. `InitialAccess` was
   predicted zero times on DAPT.

3. **Port features backfired** — §18. Documented as a finding, not hidden.

4. **Transition ordering is doctrine, not learned** — persistence is measured,
   ordering is ATT&CK convention. One campaign gives 7 stage changes.

5. **"Next 60 seconds" doesn't hold for APT** — real campaigns move ~1 stage per
   day. The horizon is adjustable and defaults to 15 minutes. **The idea document
   still says 60 seconds and should be updated.**

6. **PCAP extraction is our own CICFlowMeter reimplementation** — train/serve skew
   risk. The real jar would close it; Java is installed.

7. **We use gradient × input, not SHAP** — the idea document promises SHAP.
   Gradient × input asks "if I nudge this feature, how much does the output move?"
   — one calculation instead of thousands of feature-subset combinations. Similar
   ranking, far cheaper, **not the same thing.** Either wire up the real `shap`
   library or reword the document to say "gradient-based attribution."

8. **Fusion weight is untuned** — `MODEL_WEIGHT = 0.30` was set when the model was
   mid-training and unreliable. With macro-F1 now at 0.922 it deserves revisiting.

9. **`base_w30` is not deployed** — the dashboard still runs the original
   24-feature window-10 model. Switching requires `infer.py` to build 30-flow
   windows, not just swapping the checkpoint file.

---

# 20. File reference

## Core model and data

| File | What it does |
|---|---|
| `model.py` | The LSTM. 2 layers, 128 hidden, two output heads. |
| `pipeline.py` | Original dataset builder — 24 features, window 10. |
| `pipeline_v2.py` | Parameterised builder. Any window size, optional port features. Writes straight to a memmap so RAM is not a ceiling. |
| `train.py` | Original trainer with the blocked split fix. |
| `train_v2.py` | Parameterised trainer. Any dataset, any horizon, `--fixed-split`. |

## Evaluation

| File | What it does |
|---|---|
| `train_horizon.py` | Drives the horizon sweep across multiple k values. |
| `baseline.py` | Logistic regression comparison, with and without sequence. |
| `eval_dapt.py` | Cross-dataset evaluation on DAPT 2020. |
| `dapt.py` | DAPT loader. Renames CICFlowMeter v4 columns to v3, handles the one file shipped without a header row. |

## Inference and interface

| File | What it does |
|---|---|
| `pcap_ingest.py` | PCAP → flows via tshark. Bidirectional grouping, 24 features. |
| `infer.py` | Runs the model, fuses with signals, calls the forecast, computes lead time. |
| `signals.py` | The 14 rules that name attack patterns. |
| `forecast.py` | Transition matrix, projection, confidence, lead time. |
| `attck_map.py` | Stage definitions, ATT&CK IDs, display names. |
| `app.py` | The Streamlit dashboard. |

## Orchestration

| File | What it does |
|---|---|
| `run_overnight.sh` | Phase 1 — port features at w10/w30, DAPT evals, horizon sweep. |
| `run_phase2.sh` | Phase 2 — logistic baseline, w20, longer horizons, second seed. Resumable. |
| `run_phase3.sh` | Phase 3 — corrected horizon sweep with `--fixed-split`. |

## Outputs

```
models/
  lstm_world_model.pt        currently live (original, 24 feat, window 10)
  base_w30.pt                best model — NOT yet promoted
  ports_w30.pt               port-feature variant (rejected, see §18)
  metrics.json               original model results
  base_w30_metrics.json      best model results
  dapt_*.json                cross-dataset results per model
  horizon_curve_fixed.json   the corrected horizon curve
  baseline_ports_w10.json    logistic regression comparison
  overnight.log              full training log for every run
```

## Data

```
data/
  cic-ids2018/   training data, 7 days
  dapt2020/      cross-dataset test, 10 CSVs, 86,691 flows
  processed/     → symlink to /storage/sih-processed (5.3 GB)
```

The symlink exists because `/home` was at 93% full. `/storage` is a separate
195GB partition. No partitions were resized and the Windows install was not
touched.

---

# 21. Running everything

## The dashboard

```bash
cd ~/netrikan
streamlit run src/app.py
```

Upload a `.pcap`, `.pcapng`, `.cap`, or NetFlow `.csv`, or pick a demo scenario.
Minimum 12 flows to form a window.

## Rebuilding from scratch

```bash
cd ~/netrikan/src

# 1. build the dataset (window 30, original 24 features)
python3 pipeline_v2.py --window 30 --out /storage/sih-base-w30

# 2. train
python3 train_v2.py --data /storage/sih-base-w30 --tag base_w30 --epochs 25

# 3. cross-dataset check
python3 eval_dapt.py --data /storage/sih-base-w30 \
        --model ~/netrikan/models/base_w30.pt --tag dapt_base_w30

# 4. logistic regression comparison
python3 baseline.py --data /storage/sih-base-w30 --tag baseline_w30

# 5. horizon curve
python3 train_v2.py --data /storage/sih-base-w30 --tag fx_k90 \
        --horizon 90 --fixed-split --epochs 10
```

## Verifying the leak fix yourself

```bash
cd ~/netrikan/src && python3 -c "
import numpy as np
from sklearn.model_selection import train_test_split
from train_v2 import blocked_split
y = np.load('/home/pontiff/netrikan/data/processed/y.npy'); n = len(y)

def leak(tr, va, label):
    m = np.zeros(n, np.int8); m[tr] = 1
    s = va[np.random.default_rng(0).choice(len(va), 50000, replace=False)]
    bad = sum(1 for i in s if m[max(0,i-9):min(n,i+10)].any())
    print(f'{label:12} {bad/500:.1f}% of val windows seen in training')

idx = np.arange(n)
leak(*train_test_split(idx, test_size=.2, random_state=42, stratify=y), 'random')
leak(*blocked_split(y, purge=9), 'blocked')
"
```

## Dependencies

`torch`, `streamlit`, `plotly`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
plus `tshark` (from `wireshark-cli`) for PCAP.

---

# 22. Glossary

| Term | Meaning |
|---|---|
| **Flow** | One conversation between two machines, summarised as numbers |
| **Window** | 10 (or 30) consecutive flows the model reads at once |
| **Stride** | How far the window moves each step — ours is 1 flow |
| **IAT** | Inter-arrival time; the gap between packets |
| **Jitter ratio** | IAT std ÷ IAT mean. Low = machine-timed. High = human. |
| **TCP flags** | Control bits: SYN opens, ACK confirms, RST rejects, FIN closes |
| **Kill chain** | The phases of an attack, from getting in to stealing data |
| **ATT&CK** | MITRE's standard catalogue of attacker tactics (TA0001 etc.) |
| **C2** | Command & Control — compromised machine taking orders remotely |
| **Exfiltration** | Stealing data out of the network |
| **Lateral movement** | Moving between machines once already inside |
| **LSTM** | A neural network that reads sequences and keeps a memory |
| **Epoch** | One full pass over the training data |
| **Macro-F1** | F1 averaged equally across all classes; our headline metric |
| **Precision** | Of the things flagged, how many were real |
| **Recall** | Of the real attacks, how many were caught |
| **Data leakage** | Test data contaminating training, inflating scores |
| **Blocked split** | Splitting by contiguous time blocks instead of randomly |
| **Purge gap** | Windows discarded at block edges to guarantee no overlap |
| **Markov chain** | A table of state-transition probabilities, applied repeatedly |
| **Persistence** | P(an attacker stays in the same stage next step) |
| **Horizon** | How far into the future the forecast projects |
| **Lead time** | How much earlier we warn than a conventional IDS |
| **Cross-dataset** | Testing on a completely different dataset |
| **PCAP** | Raw packet capture file |
| **CICFlowMeter** | The tool that converts PCAP to flow features |
| **Train/serve skew** | Features computed differently at training and deployment |

---

# 23. Questions judges will ask

**"How is this different from an IDS?"**
An IDS classifies what already happened. We predict the next kill-chain stage. Our
label is the *next* flow, not the current one — one array index, entirely
different problem.

**"Why not just use logistic regression?"**
We tried. 0.607 single-flow, 0.742 with sequence, versus 0.922 for the LSTM. On
Infiltration it's 0.14 versus 0.75.

**"Your numbers are lower than published papers."**
Because theirs leak. Sliding windows overlap 90%; random splitting puts
near-identical windows on both sides. We measured 100% contamination, fixed it
with a purged blocked split, and our numbers are post-fix.

**"Does it work outside your dataset?"**
Yes, and we measured it. Trained on 2018, tested on an unseen 2019 APT campaign:
80% attack recall, 94% on lateral movement. Precision drops to 40% — that's our
known weakness.

**"How far ahead can it actually predict?"**
Measured, not assumed. 360 flows of lead costs 0.027 macro-F1. Infiltration is
completely flat; DoS degrades most because floods are bursty.

**"Where is the AI?"**
One LSTM does stage classification and breach probability. The forecast is a
Markov chain, the signals are 14 hand-written rules, the text is f-string
templates. No language model anywhere. That's why it runs offline on a laptop and
every alert is traceable.

**"Is the explainability real?"**
Currently gradient × input, not SHAP. Same idea, far cheaper, not identical. The
idea document overstates this and needs correcting.

**"What didn't work?"**
Port features — gained 0.003 in-dataset, cost 0.05 cross-dataset, dropped. And we
couldn't learn the transition matrix from DAPT because one campaign yields only 7
stage changes.

**"What would you do next?"**
Deploy `base_w30`. Get more APT campaigns to learn transitions statistically. Use
the real CICFlowMeter to remove train/serve skew. Fix Infiltration precision.
