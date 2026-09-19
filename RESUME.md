# Overnight run — resume notes

Started 2026-09-08 01:48 IST. Everything below runs detached (`nohup setsid`),
so it survives terminal close, session end, and logout.

## Check state

```bash
tail -40 ~/netrikan/models/overnight.log     # full log
cat ~/netrikan/models/STATUS.txt             # stage-level heartbeat
grep '>>>' ~/netrikan/models/overnight.log   # stage boundaries
cat ~/netrikan/models/BEST.txt               # written at the end
```

## If something died — just re-run

Both scripts skip any stage whose output already exists, so this is safe at
any point and will not redo finished work:

```bash
bash ~/netrikan/run_phase2.sh
```

Phase 1 (`run_overnight.sh`) is not resumable in the same way; if it died
mid-way, the phase-2 script covers the remaining variants.

## What is being measured

One variable changes per run, so each row is attributable.

| run | window | features | tests |
|---|---|---|---|
| `ports_w10` | 10 | 36 | effect of port/protocol features |
| `ports_w20` | 20 | 36 | effect of more temporal context |
| `ports_w30` | 30 | 36 | ditto, further |
| `ports_w10_k*` | 10 | 36 | forecast horizon curve, k = 0…1440 flows |
| `ports_w10_seed2` | 10 | 36 | run-to-run variance |
| `baseline_ports_w10` | 10 | 36 | logistic regression, with and without sequence |

Reference point: the pre-existing 24-feature model scored **macro-F1 0.885**,
Infiltration **0.590**, DAPT cross-dataset recall **83.9%** / precision **42.2%**.

## Known issue found mid-run (fixed in phase 3)

Phase 1's horizon sweep derived the blocked split from the **shifted** labels, so
each k was scored on a slightly different validation set. That is why `k=360`
implausibly beat `k=30` (0.8848 vs 0.8726) — the runs were not comparable.

`train_v2.py --fixed-split` pins the blocks to the unshifted labels so only the
prediction target changes. Phase 3 re-runs k = 0…360 that way and writes
`horizon_curve_fixed.json`. **Use the `fx_k*` numbers for the horizon claim, not
the phase-1 `ports_w10_k*` ones.**

## Cross-dataset warning

Port features improved in-dataset scores but made generalisation worse:

| model | in-dataset macro-F1 | DAPT recall | DAPT precision |
|---|---|---|---|
| 24 feat, w10 | 0.885 | 0.839 | 0.422 |
| 36 feat, w10 | 0.894 | 0.746 | 0.372 |
| 36 feat, w30 | 0.925 | 0.757 | 0.303 |

The likely reason is that port layout is environment-specific: the model learns
CIC-IDS-2018's particular services rather than attack behaviour. Picking the
best model is therefore a real trade-off, not an obvious win — decide which
matters more for the pitch before promoting anything.

## Outputs

- `~/netrikan/models/<tag>.pt` — checkpoints
- `~/netrikan/models/<tag>_metrics.json` — per-run metrics + history
- `~/netrikan/models/dapt_*.json` — cross-dataset results
- `~/netrikan/models/baseline_*.json` — logistic-regression comparison

## Not touched

`lstm_world_model.pt` is the known-good checkpoint the dashboard runs on.
Nothing overnight overwrites it. Promotion to that name is a deliberate
decision to make after reading the results, not automatic.

## Machine settings changed

Auto-suspend on AC was disabled so the run is not killed by sleep. Restore with:

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'suspend'
gsettings set org.gnome.desktop.session idle-delay 300
```
