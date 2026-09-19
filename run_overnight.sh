#!/usr/bin/env bash
# Overnight ablation run. Each stage is independent: a failure is
# logged and the run continues, so one bad stage does not waste the night.
cd "$(dirname "$0")/src"

LOG=~/netrikan/models/overnight.log
STORE=/storage
PY="python3 -u"

exec > >(tee -a "$LOG") 2>&1
echo "############ OVERNIGHT RUN  $(date) ############"

stage() {
  local name="$1"; shift
  echo
  echo "======================================================================"
  echo ">>> $name   $(date +%H:%M:%S)"
  echo "======================================================================"
  local t0=$SECONDS
  if "$@"; then
    echo "<<< $name OK  ($(( (SECONDS-t0)/60 )) min)"
  else
    echo "<<< $name FAILED (exit $?) — continuing"
  fi
  df -h /home "$STORE" | grep -E "home|storage"
}

free_gb() { df --output=avail -BG "$STORE" | tail -1 | tr -dc '0-9'; }

echo "free on $STORE: $(free_gb) GB"
[ "$(free_gb)" -lt 40 ] && { echo "ABORT: need >=40GB free on $STORE"; exit 1; }

# ---- Stage 1: dataset with port/protocol features, window 10 -----------------
stage "BUILD ports w10" $PY pipeline_v2.py --window 10 --with-ports \
      --out $STORE/sih-ports-w10

# ---- Stage 2: train it — isolates the effect of the port features -----------
stage "TRAIN ports_w10" $PY train_v2.py --data $STORE/sih-ports-w10 \
      --tag ports_w10 --epochs 25

# ---- Stage 3: cross-dataset check on the new features -----------------------
stage "DAPT ports_w10" $PY eval_dapt.py --data $STORE/sih-ports-w10 \
      --model ~/netrikan/models/ports_w10.pt --tag dapt_ports_w10

# ---- Stage 4: longer window — isolates the effect of temporal context -------
stage "BUILD ports w30" $PY pipeline_v2.py --window 30 --with-ports \
      --out $STORE/sih-ports-w30

stage "TRAIN ports_w30" $PY train_v2.py --data $STORE/sih-ports-w30 \
      --tag ports_w30 --epochs 25

stage "DAPT ports_w30" $PY eval_dapt.py --data $STORE/sih-ports-w30 \
      --model ~/netrikan/models/ports_w30.pt --tag dapt_ports_w30

# ---- Stage 5: horizon sweep on the port features ----------------------------
for K in 0 30 90 180 360; do
  stage "HORIZON k=$K" $PY train_v2.py --data $STORE/sih-ports-w10 \
        --tag ports_w10_k$K --epochs 12 --patience 3 --samples 600000 --horizon $K
done

# ---- Summary ----------------------------------------------------------------
echo
echo "======================================================================"
echo ">>> SUMMARY   $(date +%H:%M:%S)"
echo "======================================================================"
$PY - <<'PYEOF'
import json, glob, os
md = os.path.expanduser('~/netrikan/models')
print(f"{'run':<20}{'win':>5}{'feat':>6}{'k':>6}{'macroF1':>10}{'Infiltration':>14}")
for p in sorted(glob.glob(os.path.join(md, '*_metrics.json'))):
    try:
        d = json.load(open(p))
        inf = d['report']['Infiltration']['f1-score']
        print(f"{d['tag']:<20}{d['window']:>5}{d['features']:>6}"
              f"{d.get('horizon_k',0):>6}{d['best_macro_f1']:>10.4f}{inf:>14.3f}")
    except Exception as e:
        print(f"{os.path.basename(p):<20} unreadable: {e}")
print()
for p in sorted(glob.glob(os.path.join(md, 'dapt_*.json'))):
    try:
        d = json.load(open(p))
        a = d['binary_report']['Attack']
        print(f"{d['tag']:<26} recall {a['recall']:.3f}  precision {a['precision']:.3f}"
              f"  AUC {d['roc_auc']:.3f}")
    except Exception:
        pass
PYEOF

echo
echo "############ DONE  $(date) ############"
