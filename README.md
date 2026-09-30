# netrikan-draft

Network-attack forecasting from host-centric traffic telemetry (SIH problem statement: `problem-statement.md`).

## Status
- [x] Phase 0: data audit (`docs/DATA_AUDIT.md`), awaiting approval
- [x] Demo prototype on DAPT2020 (built ahead of the gates for presentation; see below)
- [ ] Phase 1-3: locked eval protocol, full processed datasets, baseline ladder, sequence-model decision

## Setup
```
make setup        # python3.12 venv, pinned requirements, macOS LightGBM libomp fix
make test         # leakage tests
make demo-train   # DAPT2020: leave-one-day-out training + evaluation (~2 min) -> models/demo, results/demo/metrics.json
make z24-train    # ZeekData24: recognizer + attacker forecaster (~8 min) -> models/z24, results/z24/metrics.json
make web-install  # once: React console dependencies (needs Node 20+)
make demo         # build the React console and serve it with the API on http://localhost:8000 (offline)
make api / make web   # development: FastAPI on :8000 + Vite dev server on :5173 (two terminals)
make infer CSV=flows.csv   # CLI: per-host-minute risk, stage, top drivers
```
Raw data lives in `data/` (gitignored).

## Demo prototype
`src/netrikan/`: `dapt.py` (ingest) → `features.py` (per-host 60 s buckets, signed log1p) → `models.py` (LR, LightGBM, LightGBM + lags, GRU world model with K-step rollout, stage model) → `metrics.py` → `infer.py`.
The console (`web/`, React) talks to a FastAPI layer (`api/`) that runs the same code as training and shows only numbers computed live or read from `results/demo/metrics.json`.

**Task:** for each monitored host that is benign *right now*, P(attack traffic in the next 5 minutes), then the stage of that attack, with TreeSHAP drivers and the simulated-vs-observed future state.
**Evaluation (exploratory, not the locked protocol):** leave-one-day-out over 5 DAPT days is primary; a within-day blocked split is secondary. Every experiment is in `results/registry.csv` with its hypothesis and prediction written before the run.

**What the results say (read `results/demo/metrics.json` or the console's Benchmark & evidence tab):**
- Ranking works: pooled PR-AUC ~0.12 vs 0.045 base rate; within a held-out day ROC-AUC is 0.74-0.95.
- Pooled alarm-budget recall is low; score scales do not transfer between days with different attack types.
- The GRU world model does not beat LightGBM with lags, and **shuffling the history does not hurt either model**: on this data the forecast comes from the current state, not from learned dynamics.
- Stage forecasting is at chance across days (each day has a different stage) and only marginally above majority within a day.
- Data is tiny: 9 monitored hosts, 3 ever attacked, 319 attack minutes. More/richer data (e.g. UWF-ZeekData24) is needed for the forecasting claim.

## ZeekData24 page (attacker campaign)
`src/netrikan/zeek.py` (loader) and `campaign.py` (models), trained by `scripts/train_z24.py`, shown on the console's ZeekData24 page.
- **Task A, recognizer:** which of 5 ATT&CK techniques (T1595 scan, T1190 exploit, T1078 valid accounts, T1110 brute force, T1048 exfil) a host-minute contains, from that minute's behaviour only.
- **Task B, forecaster:** for each attacker, P(technique T fires in the next 5 minutes) from its recent technique history; ladder = renewal hazard, current-behaviour-only, own-technique history, all-technique history, GRU world model; shuffle controls; a clearly-labelled wall-clock ablation.
- **Evaluation:** leave-one-attack-week-out. The 5 attack weeks replay one scripted campaign, so this measures repeatability, not generalisation.
- **Findings** (see `results/z24/metrics.json` or the console): recognition is near-perfect on this stereotyped, pure-bucket data (treat with suspicion); burst timing is forecastable about 3x better than chance and shuffling history order hurts (the model uses temporal order); the GRU world model matches but does not beat hand-built history features; techniques run as roughly-hourly jittered schedules with only weak cross-technique coupling, so there is no kill-chain progression in this corpus.

## Console and response layer
`api/` (FastAPI) wraps `src/netrikan` and the trained models; `web/` (React + TypeScript, Vite, Tailwind, Geist, Phosphor icons, Motion) has two tabs: **Overview** (`/`, a landing page whose chart, numbers and incident preview are live API data) and **Dashboard** (`/dashboard/...`: DAPT2020 forecasting, ZeekData24 campaigns, **Incident response**, CIC-IDS2017, CTU-13, zero-shot evidence). Old URLs such as `/response` redirect. The API warms its caches for the default views at startup (~20 s), so first clicks are fast. API docs at `/api/docs`. UI design follows the taste-skill guidance installed in `.claude/skills/`.

**Incident response** (`api/response.py`, `web/src/pages/Response.tsx`): alert minutes on one host (one host + technique on ZeekData24) are merged into incidents, prioritised, and mapped from the MITRE ATT&CK tactic/technique to MITRE D3FEND countermeasures (Detect, Isolate, Deceive, Evict, Harden). Actions are filled in from the host's real peers and ports around the incident, with example iptables rules as templates for review. Netrikan never applies them. Triage status and checklists are kept in the browser (localStorage).
Honesty rules: on DAPT2020 the stage model is at chance across held-out days, so priority uses risk and duration only, and the stage playbook is shown as a hypothesis behind a stage-agnostic first response. On ZeekData24 the technique comes from the recognizer, which is near-perfect in-corpus (a scripted campaign, so this does not transfer to real attacks).
