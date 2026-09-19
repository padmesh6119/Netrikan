#!/usr/bin/env bash
# Launch the Netrikan dashboard. Stops any previous instance first — a stale
# server keeps its old Python modules and old theme in memory, which looks
# exactly like a code bug.
set -u
cd "$(dirname "$0")"
PORT="${1:-8700}"

echo "stopping any previous instance…"
pkill -f "streamlit run src/app.py" 2>/dev/null
sleep 2

echo "checking environment…"
python3 - <<'PY' || { echo; echo "FAILED: python packages are broken."; echo "Fix with: pip install --break-system-packages --upgrade numpy pandas scikit-learn matplotlib bottleneck numexpr pyarrow"; exit 1; }
import sys
mods = ["numpy", "pandas", "sklearn", "torch", "plotly", "streamlit"]
for m in mods:
    __import__(m)
import numpy, pandas
print(f"  numpy {numpy.__version__} · pandas {pandas.__version__} — OK")
PY

echo "starting on http://localhost:$PORT"
exec streamlit run src/app.py --server.port "$PORT"
