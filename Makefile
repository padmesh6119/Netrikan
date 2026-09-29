PY := .venv/bin/python

.PHONY: setup audit test
setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt

# Phase 0: regenerate the raw audit output that docs/DATA_AUDIT.md cites
audit:
	$(PY) scripts/audit_data.py | tee results/audit/audit_raw.txt

test:
	.venv/bin/pytest -q tests
