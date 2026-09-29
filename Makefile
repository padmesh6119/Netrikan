PY := .venv/bin/python
LGB := .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib

.PHONY: setup fix-lightgbm-macos audit test demo-train z24-train zero-shot demo infer
setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	$(MAKE) fix-lightgbm-macos

# macOS only: LightGBM looks for libomp in Homebrew paths; reuse the copy bundled with torch instead
fix-lightgbm-macos:
	install_name_tool -add_rpath @loader_path/../../torch/lib $(LGB) 2>/dev/null || true
	codesign --force -s - $(LGB)

# Phase 0: regenerate the raw audit output that docs/DATA_AUDIT.md cites
audit:
	$(PY) scripts/audit_data.py | tee results/audit/audit_raw.txt

test:
	.venv/bin/pytest -q tests

# Demo prototype: leave-one-day-out training + evaluation, writes models/demo and results/demo/metrics.json
demo-train:
	$(PY) scripts/train_demo.py

# ZeekData24: technique recognizer + attacker forecaster, leave-one-attack-week-out (~8 min) -> models/z24, results/z24/metrics.json
z24-train:
	$(PY) scripts/train_z24.py

# Leave-one-corpus-out zero-shot transfer across 4 corpora (~2 min) -> models/zero_shot, results/zero_shot/metrics.json
zero-shot:
	$(PY) scripts/zero_shot.py

demo:
	.venv/bin/streamlit run app/demo_app.py

# usage: make infer CSV=path/to/flows.csv
infer:
	$(PY) scripts/infer.py $(CSV)
