# Part F — Demo Attack Replay Button

**Goal**: Add a one-click "Replay real attack" path in `src/app.py` that loads
`demo_attack_lab/attack_flows.csv` (38 real flows, already committed) and
feeds it through `infer.analyze()` without any file upload. Shows the attack
unfolding on the live timeline. No synthetic data. No upload friction.

**Why it matters for the PPT**: Jury members open the dashboard, see a blank
upload box, and give up. One button that immediately shows a real attack
removes that barrier and gives the 2-minute demo a reliable path.

---

## Current state

- `src/app.py:137–138` — `st.selectbox` for `demo_data.SCENARIOS` feeds
  **synthetic** generated data (`demo_data.generate(scenario, 260)` at line 180).
- `demo_attack_lab/attack_flows.csv` — 38 real flows, Sep 15 2026, captured
  against a live Kali box. Columns match `infer.FEATURES` plus Dst Port,
  Protocol, Timestamp, Src IP, Dst IP. NOT currently used anywhere in app.py.
- `up is not None` branch (lines 157–178) handles real uploads.

---

## What to add

### 1. Add "Real attack capture" to the scenario list — `src/app.py`

In the `c3` column (currently only the file uploader), add a second option:

```python
REAL_ATTACK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'demo_attack_lab', 'attack_flows.csv'
)
```

In `c1` — add `"Real attack (Sep 15 2026)"` as the first entry in the
scenario selectbox. When selected, skip `demo_data.generate()` and load
`attack_flows.csv` instead.

### 2. Branch on the new scenario name — replace `df = demo_data.generate(...)` block

```python
REPLAY_LABEL = "Real attack (Sep 15 2026)"

# in the else branch (no upload):
if scenario == REPLAY_LABEL and os.path.exists(REAL_ATTACK_PATH):
    df = pd.read_csv(REAL_ATTACK_PATH)
    source = "demo_attack_lab/attack_flows.csv"
else:
    df = demo_data.generate(scenario, 260)
    source = scenario
```

That's the entire change. `infer.analyze(df, horizon)` already handles a
DataFrame with these columns.

### 3. Optional — frame/step replay (animated)

If the full CSV runs through `analyze()` at once, the dashboard renders
instantly (no drama). For a step-by-step reveal that looks live in the demo:

```python
if scenario == REPLAY_LABEL and st.toggle("Step-by-step replay", value=False):
    placeholder = st.empty()
    for i in range(WINDOW + 1, len(df) + 1):
        result = infer.analyze(df.iloc[:i], horizon)
        with placeholder.container():
            _render_dashboard(result)  # extract current render logic to fn
        time.sleep(0.25)
    st.stop()
```

This is optional — the one-shot path is sufficient for PPT. Only add this
if time allows.

---

## Exact file changes

| File | Change | Lines affected |
|---|---|---|
| `src/app.py` | Add `REAL_ATTACK_PATH` and `REPLAY_LABEL` constants | ~line 20 (after imports) |
| `src/app.py` | Add `REPLAY_LABEL` as first option in scenario selectbox | ~line 138 |
| `src/app.py` | Replace `df = demo_data.generate(...)` with branch | ~line 180 |

No new files. No new dependencies. `demo_data.py` unchanged.

---

## What the demo shows

1. Select "Real attack (Sep 15 2026)" → no upload needed
2. Timeline shows 38 flows: benign → scan → C2 beacon sequence
3. Stage probs shift live — Lateral Movement and C2 lighting up
4. Counterfactual panel shows real risk deltas (note: no SMB in this capture,
   so block-SMB stays flat — use `block_admin_ports` or `isolate_host` for
   the demo)
5. SHA-256 ledger fires when risk > 30%

---

## Known gap this doesn't fix

`attack_flows.csv` has no SMB traffic — block-SMB counterfactual will still
show zero delta. To fix that, re-capture with `attack.sh` generating lateral
movement over port 445. That's a separate task (re-capture demo PCAP in
`MASTER.md` §13 remaining items).

---

## Time estimate

20 minutes. Three lines of real code.
