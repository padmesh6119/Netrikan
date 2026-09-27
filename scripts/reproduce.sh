#!/usr/bin/env bash
#
# Reproduce every committed number in STATUS.md from raw data, in order.
#
# Nothing here is a demo path: each stage writes a JSON under models/ that
# STATUS.md cites, so a reader can regenerate any figure and diff it.
#
#   bash scripts/reproduce.sh            # everything (many hours)
#   STAGES="build train evals" bash scripts/reproduce.sh
#   DRY=1 bash scripts/reproduce.sh      # print the plan, run nothing
#
# STAGES (default: all): build train export evals robustness seeds transfer
#
# `export` runs BEFORE `evals` on purpose: bench/latency.py benchmarks the ONNX
# path and defaults to <model>.onnx, so the export has to exist first.
#
# Wall-clock on the reference machine (Apple Silicon, MPS, 1 CPU thread for the
# latency bench): build ~3 min, each training run ~25-70 min, each DAPT eval
# ~4 min, LODO ~3 h 10 min, the 4-seed transfer study ~2 h.
#
# PREREQUISITES
#   - CIC-IDS-2018 CSVs where configs/train_v2.yaml expects them
#   - DAPT 2020 CSVs under data/dapt2020/ (or data/dapt2020/csv/)
#   - .venv with requirements.txt installed
#   - optional: suricata on PATH for the external-baseline stage
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
DATA="${DATA:-/tmp/netrikan-cic-full-w30}"
STAGES="${STAGES:-build train export evals robustness seeds transfer}"
DRY="${DRY:-0}"

# Training device and dataloader workers. NETRIKAN_WORKERS=0 is not a
# performance choice: at the default 4 this machine drove swap to 5.1 GB and a
# 7-fold sweep died with "Shared memory manager connection has timed out".
# Worker count never changes results -- batch order comes from the sampler.
export NETRIKAN_DEVICE="${NETRIKAN_DEVICE:-mps}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export NETRIKAN_WORKERS="${NETRIKAN_WORKERS:-0}"

have() { [[ " $STAGES " == *" $1 "* ]]; }
run() {
  echo ""
  echo "+ $*"
  [[ "$DRY" == "1" ]] && return 0
  if ! "$@"; then
    echo "!! FAILED: $*" >&2
    echo "!! continuing; later stages may depend on this output" >&2
    return 1
  fi
}

echo "root=$ROOT  data=$DATA  stages=$STAGES  dry=$DRY"
echo "device=$NETRIKAN_DEVICE workers=$NETRIKAN_WORKERS"

# ---------------------------------------------------------------- build
# Writes X.npy (float16, ~8.9 GB), y.npy, onset_k*.npy, scaler.pkl, features.txt.
# load_all() sorts each capture by timestamp -- the CSVs are NOT stored in time
# order, and before that fix windows were not real sequences.
if have build; then
  run "$PY" src/pipeline_v2.py --out "$DATA"
  # file_id.npy tags each window with its capture day; LODO needs it.
  run "$PY" bench/_make_file_id.py --data "$DATA"
fi

# ---------------------------------------------------------------- train
# The deployed recipe: all 7 days, every head, temporal attention, checkpoint
# chosen on `combined` (mean of onset AUC and macro-F1) so one checkpoint is
# good at forecasting AND classification.
if have train; then
  run "$PY" -W ignore src/train_v2.py --data "$DATA" --tag cic_v2_w30 \
      --attention --select-on combined
  run "$PY" src/baseline.py --data "$DATA" --tag baseline_cic_v2_w30
fi

# ---------------------------------------------------------------- export
if have export; then
  run "$PY" src/export_onnx.py --model models/cic_v2_w30.pt \
      --out models/cic_v2_w30.onnx
  if command -v suricata >/dev/null 2>&1; then
    run "$PY" bench/suricata_comparison.py demo_attack_lab/attack_small.pcap
  else
    echo "  (skipping suricata comparison: suricata not on PATH)"
  fi
fi

# ---------------------------------------------------------------- evals
# The numbers STATUS.md section 0 quotes.
if have evals; then
  run "$PY" src/transition_eval.py --data "$DATA" --model models/cic_v2_w30.pt
  run "$PY" src/calibration.py     --data "$DATA" --model models/cic_v2_w30.pt
  run "$PY" src/eval_dapt.py       --data "$DATA" --model models/cic_v2_w30.pt \
      --tag dapt_cic_v2_w30
  run "$PY" bench/rollout_eval.py  --data "$DATA" --model models/cic_v2_w30.pt \
      --steps 10 --out models/rollout_eval_cic_v2_w30.json
  run "$PY" bench/onset_pr.py      --data "$DATA" --model models/cic_v2_w30.pt \
      --out models/onset_pr_cic_v2_w30.json
  run "$PY" bench/lead_time.py     --data "$DATA" --model models/cic_v2_w30.pt \
      --horizons 1 5 15 30 --out models/lead_time_cic_v2_w30.json
  run "$PY" bench/latency.py       --model models/cic_v2_w30.pt \
      --out models/latency.json
  # Cross-dataset operating points. The shipped decision is a hard 5-class
  # argmax with no threshold; this draws the curve it sits on and rolls alerts
  # up to (host, hour) cells, which is the operator-facing rate.
  run "$PY" bench/operating_points.py --data "$DATA" --model models/cic_v2_w30.pt \
      --out models/operating_points_dapt_cic_v2_w30.json
  run "$PY" bench/model_weight_ablation.py --data "$DATA" \
      --out models/model_weight_ablation_cic_v2_w30.json
  # takes no flags; writes models/dapt_persistence.json itself
  run "$PY" bench/dapt_persistence.py
fi

# ------------------------------------------------------------ robustness
# The protocols that make the headline numbers believable -- and the two that
# came back NEGATIVE. Do not drop the negative ones: held-out family scores
# BELOW chance and that bounds the whole generalization claim.
if have robustness; then
  # shuffled-history control: same rows, timestep order destroyed
  run "$PY" -W ignore src/train_v2.py --data "$DATA" --tag shuffled_w30 \
      --attention --select-on combined --shuffle-history
  run "$PY" bench/onset_pr.py --data "$DATA" --model models/shuffled_w30.pt \
      --out models/onset_pr_shuffled_w30.json
  # held-out attack families (3 = Infiltration, 4 = Botnet): both NEGATIVE
  run "$PY" bench/heldout_family.py --data "$DATA" --holdout 3 \
      --out models/heldout_infiltration_w30.json
  run "$PY" bench/heldout_family.py --data "$DATA" --holdout 4 \
      --out models/heldout_botnet_w30.json
  # leave-one-day-out, 7 folds. Quote supported_class_f1, never the 5-class
  # macro-F1: most folds have zero val support for 3-4 classes.
  run "$PY" -W ignore bench/lodo.py --data "$DATA" \
      --out models/lodo_cic_full_w30.json
  run "$PY" -W ignore bench/lodo_ensemble.py --data "$DATA" \
      --out models/lodo_ensemble_dapt.json
fi

# ---------------------------------------------------------------- seeds
# Cross-dataset metrics carry ~8x the seed noise of in-dataset ones, so the
# cross-dataset numbers are meaningless from a single run.
if have seeds; then
  for s in 43 44 45; do
    run "$PY" -W ignore src/train_v2.py --data "$DATA" --tag "xfer_s$s" \
        --attention --select-on combined --epochs 5 --patience 5 \
        --seed "$s" --save-epochs
  done
fi

# -------------------------------------------------------------- transfer
# How cross-dataset transfer moves with training epoch. Reporting the curve is
# legitimate; picking a deployment epoch off its DAPT column is test-set
# selection. Do not promote a checkpoint on this output alone.
if have transfer; then
  run "$PY" -W ignore src/train_v2.py --data "$DATA" --tag xfer_w30 \
      --attention --select-on combined --epochs 15 --patience 15 --save-epochs
  run "$PY" -W ignore bench/transfer_vs_epoch.py --data "$DATA" --tag xfer_w30 \
      --out models/transfer_vs_epoch.json
  for s in 43 44 45; do
    run "$PY" -W ignore bench/transfer_vs_epoch.py --data "$DATA" --tag "xfer_s$s" \
        --out "models/transfer_vs_epoch_s$s.json"
  done
fi

echo ""
echo "done. Outputs are in models/. STATUS.md cites these files by name;"
echo "where a number here disagrees with a doc, STATUS.md is the authority."
