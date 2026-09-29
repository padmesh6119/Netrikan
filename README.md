# netrikan-draft

Network-attack forecasting / multi-stage intrusion detection from host-centric traffic telemetry.
Problem statement: `problem-statement.md`. Working rules and gates: see phase status below.

## Status
- [x] Phase 0: data audit (`docs/DATA_AUDIT.md`), awaiting approval
- [ ] Phase 1: eval protocol + per-host bucketed datasets + leakage tests
- [ ] Phase 2: baseline ladder
- [ ] Phase 3: sequence model (only if Phase 2 leaves headroom)
- [ ] Phase 4: demo app

## Setup
```
make setup   # python3.12 venv + pinned requirements
make audit   # regenerates results/audit/audit_raw.txt
make test
```
Raw data lives in `data/` (gitignored, not redistributed here).
