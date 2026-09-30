PY := .venv/bin/python
LGB := .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib

.PHONY: setup fix-lightgbm-macos audit test demo-train z24-train zero-shot web-install web-build api web demo infer
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

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

# React console (web/) + FastAPI (api/). Needs Node 20+ once, for the build; runtime is fully offline.
web-install:
	cd web && npm ci

web-build:
	cd web && npm run build

# Development: run `make api` and `make web` in two terminals, open http://localhost:5173
api:
	.venv/bin/uvicorn api.main:app --reload --port 8000

web:
	cd web && npm run dev

# Built console + API on one port: http://localhost:8000
demo: web-build
	.venv/bin/uvicorn api.main:app --port 8000

# usage: make infer CSV=path/to/flows.csv
infer:
	$(PY) scripts/infer.py $(CSV)
