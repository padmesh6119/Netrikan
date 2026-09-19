#!/usr/bin/env bash
# Phase 3: corrected horizon sweep. Phase 1 stratified the blocked split on the
# SHIFTED labels, so each k was scored on a slightly different validation set —
# which is why k=360 implausibly beat k=30. --fixed-split pins the blocks to the
# unshifted labels so the only thing changing across runs is the prediction target.
cd "$(dirname "$0")/src"

LOG=~/netrikan/models/overnight.log
STATUS=~/netrikan/models/STATUS.txt
STORE=/storage
MD=~/netrikan/models
PY="python3 -u"
DEADLINE=$(date -d "today 06:45" +%s)

exec >> "$LOG" 2>&1

if [ -n "$1" ]; then
  echo "### phase3 waiting on pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

echo
echo "############ PHASE 3 — corrected horizon sweep  $(date) ############"
echo "$(date +%H:%M:%S) | phase3 started" >> "$STATUS"

for K in 0 30 90 180 360; do
  OUT="$MD/fx_k${K}_metrics.json"
  if [ -e "$OUT" ]; then echo ">>> SKIP fx k=$K"; continue; fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo ">>> SKIP fx k=$K (past deadline)"; continue; fi
  echo
  echo "======================================================================"
  echo ">>> FIXED-SPLIT HORIZON k=$K   $(date +%H:%M:%S)"
  echo "======================================================================"
  t0=$SECONDS
  if $PY train_v2.py --data $STORE/sih-ports-w10 --tag fx_k$K \
       --epochs 10 --patience 3 --samples 600000 --horizon $K --fixed-split; then
    echo "<<< fx k=$K OK ($(( (SECONDS-t0)/60 )) min)"
    echo "$(date +%H:%M:%S) | fx k=$K OK" >> "$STATUS"
  else
    echo "<<< fx k=$K FAILED — continuing"
  fi
done

echo
echo "======================================================================"
echo ">>> CORRECTED HORIZON CURVE   $(date +%H:%M:%S)"
echo "======================================================================"
$PY - <<'PYEOF'
import json, glob, os
md = os.path.expanduser('~/netrikan/models')
rows = []
for p in glob.glob(os.path.join(md, 'fx_k*_metrics.json')):
    try:
        d = json.load(open(p))
        rows.append((d['horizon_k'], d['best_macro_f1'],
                     d['report']['Infiltration']['f1-score'],
                     d['report']['DoS']['f1-score']))
    except Exception:
        pass
rows.sort()
if rows:
    print(f"{'k flows':>8}{'macroF1':>10}{'Infil':>8}{'DoS':>8}   (same split at every k)")
    for k, f1, inf, dos in rows:
        print(f"{k:>8}{f1:>10.4f}{inf:>8.3f}{dos:>8.3f}")
    drop = rows[0][1] - rows[-1][1]
    print(f"\nF1 drop from k={rows[0][0]} to k={rows[-1][0]}: {drop:.4f}")
    json.dump({"corrected_horizon": [
        {"k_flows": k, "macro_f1": f1, "infiltration_f1": inf, "dos_f1": dos}
        for k, f1, inf, dos in rows]},
        open(os.path.join(md, 'horizon_curve_fixed.json'), 'w'), indent=2)
    print("saved horizon_curve_fixed.json")
else:
    print("no fixed-split runs completed")
PYEOF

echo "$(date +%H:%M:%S) | phase3 complete" >> "$STATUS"
echo
echo "############ PHASE 3 DONE  $(date) ############"
