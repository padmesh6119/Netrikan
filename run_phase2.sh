#!/usr/bin/env bash
# Phase 2 of the overnight run. Waits for phase 1 to exit, then continues.
# Every stage is skipped if its output already exists, so this script can be
# re-run at any point and will pick up where it stopped.
cd "$(dirname "$0")/src"

LOG=~/netrikan/models/overnight.log
STATUS=~/netrikan/models/STATUS.txt
STORE=/storage
MD=~/netrikan/models
PY="python3 -u"
DEADLINE=$(date -d "today 06:45" +%s)   # stop starting new work after this

exec >> "$LOG" 2>&1

note() { echo "$(date +%H:%M:%S) | $*" >> "$STATUS"; }

# wait for phase 1 (passed as $1) to finish
if [ -n "$1" ]; then
  echo "### phase2 waiting on pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

echo
echo "############ PHASE 2  $(date) ############"
note "phase2 started"

past_deadline() { [ "$(date +%s)" -gt "$DEADLINE" ]; }

stage() {
  local name="$1" guard="$2"; shift 2
  if [ -e "$guard" ]; then echo ">>> SKIP $name (exists: $guard)"; return; fi
  if past_deadline; then echo ">>> SKIP $name (past deadline)"; return; fi
  echo
  echo "======================================================================"
  echo ">>> $name   $(date +%H:%M:%S)"
  echo "======================================================================"
  local t0=$SECONDS
  if "$@"; then
    echo "<<< $name OK ($(( (SECONDS-t0)/60 )) min)"; note "$name OK"
  else
    echo "<<< $name FAILED — continuing"; note "$name FAILED"
  fi
  df -h "$STORE" | tail -1
}

# --- logistic-regression baseline: the comparison the idea doc promises ------
stage "BASELINE logreg w10" "$MD/baseline_ports_w10.json" \
      $PY baseline.py --data $STORE/sih-ports-w10 --tag baseline_ports_w10

# --- window 20: fills the middle of the context curve (10 / 20 / 30) --------
stage "BUILD ports w20" "$STORE/sih-ports-w20/y.npy" \
      $PY pipeline_v2.py --window 20 --with-ports --out $STORE/sih-ports-w20

stage "TRAIN ports_w20" "$MD/ports_w20_metrics.json" \
      $PY train_v2.py --data $STORE/sih-ports-w20 --tag ports_w20 --epochs 25

stage "DAPT ports_w20" "$MD/dapt_ports_w20.json" \
      $PY eval_dapt.py --data $STORE/sih-ports-w20 --model $MD/ports_w20.pt \
      --tag dapt_ports_w20

# --- longer horizons on the port features -----------------------------------
for K in 720 1440; do
  stage "HORIZON k=$K" "$MD/ports_w10_k${K}_metrics.json" \
        $PY train_v2.py --data $STORE/sih-ports-w10 --tag ports_w10_k$K \
        --epochs 10 --patience 3 --samples 600000 --horizon $K
done

# --- second seed on the base config: variance estimate ----------------------
stage "SEED2 ports_w10" "$MD/ports_w10_seed2_metrics.json" \
      $PY train_v2.py --data $STORE/sih-ports-w10 --tag ports_w10_seed2 --epochs 20

# --- final summary ----------------------------------------------------------
echo
echo "======================================================================"
echo ">>> FINAL SUMMARY   $(date +%H:%M:%S)"
echo "======================================================================"
$PY - <<'PYEOF'
import json, glob, os
md = os.path.expanduser('~/netrikan/models')
rows = []
for p in sorted(glob.glob(os.path.join(md, '*_metrics.json'))):
    try:
        d = json.load(open(p))
        rows.append((d['tag'], d['window'], d['features'], d.get('horizon_k', 0),
                     d['best_macro_f1'], d['report']['Infiltration']['f1-score'],
                     d['report']['DoS']['f1-score']))
    except Exception:
        pass
rows.sort(key=lambda r: -r[4])
print(f"{'run':<22}{'win':>5}{'feat':>6}{'k':>6}{'macroF1':>10}{'Infil':>8}{'DoS':>8}")
for r in rows:
    print(f"{r[0]:<22}{r[1]:>5}{r[2]:>6}{r[3]:>6}{r[4]:>10.4f}{r[5]:>8.3f}{r[6]:>8.3f}")

print()
for p in sorted(glob.glob(os.path.join(md, 'dapt_*.json'))):
    try:
        d = json.load(open(p)); a = d['binary_report']['Attack']
        print(f"{d['tag']:<26} recall {a['recall']:.3f}  precision {a['precision']:.3f}"
              f"  AUC {d['roc_auc']:.3f}")
    except Exception:
        pass

print()
for p in sorted(glob.glob(os.path.join(md, 'baseline_*.json'))):
    try:
        d = json.load(open(p))
        for m, v in d['results'].items():
            print(f"baseline[{m}] {d['tag']:<20} macroF1 {v['macro_f1']:.4f}"
                  f"  ({v['n_features']} features)")
    except Exception:
        pass

if rows:
    best = rows[0]
    print(f"\nBEST: {best[0]}  macroF1={best[4]:.4f}  Infiltration={best[5]:.3f}")
    with open(os.path.join(md, 'BEST.txt'), 'w') as f:
        f.write(f"{best[0]} macroF1={best[4]:.4f} infiltration={best[5]:.3f}\n")
PYEOF

note "phase2 complete"
echo
echo "############ ALL DONE  $(date) ############"
