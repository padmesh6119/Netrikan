#!/usr/bin/env bash
# Post-training evaluation for the full CIC-IDS-2018 model.
# Runs every measurement the problem statement and STATUS.md call for, in order,
# against one checkpoint. Safe to re-run; each step writes its own JSON.
#
# Usage: bash bench/run_full_eval.sh <data_dir> <model.pt> [tag]

set -uo pipefail
cd "$(dirname "$0")/.."

DATA="${1:?usage: run_full_eval.sh <data_dir> <model.pt> [tag]}"
MODEL="${2:?need model path}"
TAG="${3:-cic_full_w30}"
PY=.venv/bin/python

echo "=========================================================="
echo " Netrikan full evaluation"
echo " data:  $DATA"
echo " model: $MODEL"
echo "=========================================================="

echo; echo "### 1/6 logistic-regression baseline (mandatory comparison)"
$PY -W ignore src/baseline.py --data "$DATA" --tag "baseline_${TAG}"

echo; echo "### 2/6 transition-only evaluation"
$PY -W ignore src/transition_eval.py --data "$DATA" --model "$MODEL" \
    --out "models/transition_eval_${TAG}.json"

echo; echo "### 3/6 temperature scaling + ECE + reliability diagram"
$PY -W ignore src/calibration.py --data "$DATA" --model "$MODEL" \
    --out models/temperature.json --plot "models/reliability_${TAG}.png"

echo; echo "### 4/6 world-model rollout skill (does state_head learn dynamics?)"
$PY -W ignore bench/rollout_eval.py --data "$DATA" --model "$MODEL" \
    --out "models/rollout_eval_${TAG}.json"

echo; echo "### 5/6 fusion-weight ablation (model vs rules)"
$PY -W ignore bench/model_weight_ablation.py --data "$DATA" --model "$MODEL" \
    --out "models/model_weight_ablation_${TAG}.json"

echo; echo "### 6/6 cross-dataset evaluation on DAPT 2020 (+ onset transfer)"
$PY -W ignore src/eval_dapt.py --data "$DATA" --model "$MODEL" \
    --tag "dapt_${TAG}"

echo; echo "all evaluations complete. JSON in models/"
