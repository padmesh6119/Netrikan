# Data audit (Phase 0)

Every number below comes from `make audit` → `results/audit/audit_raw.txt` (script: `scripts/audit_data.py`).
Nothing here is a modelling result. Environment: 11 GB free disk (not 25), Python 3.12 venv in `.venv/`.

## 0. Problem statement, quoted back

Source of truth: `problem-statement.md`.

**The claim (my one-sentence distillation; please confirm or correct):**
> A model that learns network-state transition dynamics `P(S_t+1 | S_t)` from traffic telemetry can forecast, K windows ahead, the probability of infiltration and the MITRE ATT&CK stage of attacker progression *before compromise completes*, and does so measurably better than a logistic-regression baseline on the same features, with per-prediction explanations.

**Success criteria stated in the brief:**
1. Ingest flow (and PCAP-derived) telemetry into a timestamped, normalised feature matrix.
2. A sequence / graph / latent model that learns transition dynamics (not a static classifier); weights and reproducible config included.
3. K-step forecast outputting: infiltration probability, predicted ATT&CK stage (Recon, Initial Access, Lateral Movement, C2, Exfiltration), top driving features.
4. Explainability (SHAP or attention) for each prediction.
5. Offline demo (Streamlit/Flask/CLI) taking CSV/PCAP and showing a probability timeline, flagged flows, stage annotations.
6. Benchmark vs a logistic-regression baseline on F1, precision, recall, FPR, "demonstrating measurable improvement" from temporal dynamics.
7. Generalise to unseen attack patterns, not memorise signatures.

**Ambiguities I need you to resolve (I am not guessing):** see section 6.

## 1. Summary table

| Dataset | Rows (raw → usable) | Host IPs | Real timestamps | Label space | Role you assigned |
|---|---|---|---|---|---|
| DAPT2020 | 86,691 | yes | yes (naive local, 12h clock) | 4 attack stages + benign | dev / LODO |
| ZeekData22 | 18,562,468 → **4,851,829** after exact-dup removal | yes | yes (`ts` epoch; `datetime` unreliable) | 10 ATT&CK tactics + none | training |
| ZeekDataFall22 | 34,621 (CSVs) | yes | yes | 12 tactics + none (+`Duplicate`) | independent test |
| CIC-2018 improved | 5,094,963 (one file only) | **no** | **no** | Benign / Botnet Ares | detection pretrain |

## 2. DAPT2020 (`data/dapt2020/csv/`)

- **Columns:** 85 = Flow ID, Src/Dst IP, Src/Dst Port, Protocol, Timestamp, 76 CICFlowMeter features, `Activity`, `Stage`. No TTL, no IP-fragment or retransmission fields (the brief's "packet-level" list is only partly available: flag counts, init window bytes and IAT exist).
- **Rows:** 86,691 across 10 files (3.4K–29K each). No file is truncated at 1,048,575. No full-row duplicates. `Flow ID` repeats within files (e.g. 14,206 of 29,242 in public-tuesday), which is normal for CICFlowMeter (same 5-tuple, several flows).
- **Data defects:**
  - `enp0s3-pvt-thursday.pcap_Flow.csv` **has no header row** (first data row is line 1). A naive reader silently drops one flow and mis-names columns.
  - Two label vocabularies coexist: `BENIGN/BENIGN` (19,454) and `Benign/Normal` (44,258). `public-tuesday` mixes both.
  - `Timestamp` is a `dd/mm/yyyy hh:mm:ss AM/PM` string with no timezone.
  - No file is time-ordered (`is_monotonic_increasing` is False for all 10). Sort before use.
  - Each file spans only ~7–10 h (approx. 11:30–22:45), not a full day.
- **Date range:** 2019-07-15 (Mon) … 2019-07-19 (Fri).
- **Two capture points** (public vs pvt interface) per day; the same conversation may appear in both files. I have not tested cross-file duplication.
- **Labels (Stage):** Benign 63,712 (both spellings), Reconnaissance 11,909, Establish Foothold 8,604, Lateral Movement 2,451, Data Exfiltration 15. Total attack flows 22,979.
- **Stages are almost one per day:** Tue = Reconnaissance (11,865), Wed = Establish Foothold (8,588), Thu = Lateral Movement (2,451), Fri = Exfiltration (15 flows in total). Mon is entirely benign.
- **Host progression:** 11 malicious source IPs in total. 7 of 11 sources touch >1 stage; 3 of 5 attacked destinations see >1 stage; 7 of 15 (src,dst) pairs see >1 stage. Only one actor, `206.207.50.50` (16,594 flows over 4 days), runs the full chain including exfiltration. Victims are essentially two hosts: `192.168.3.29` (20,633 malicious flows in) and `192.168.3.30` (2,312).
- **Exfiltration has 15 flows.** No model can be trained or evaluated on it.

## 3. ZeekData22 (`data/ZeekData22/`), the training corpus

- **Two formats:** `parquet/` (8 weekly parts, 18,562,468 rows) and `csv/` (5 partial files, 2,044,734 rows). **The CSVs are a strict subset of the parquet** (all 1,100,574 CSV `uid`s are in the parquet) and use different column names (`mitre_attack_tactics`, `src_ip`, `dest_ip` vs `label_tactic`, `src_ip_zeek`, `dest_ip_zeek`). Two CSVs hold ~1,000,000 rows each (a cap, not the Excel 1,048,575 limit). **Use parquet; ignore the CSVs.**
- **Columns (parquet, 23):** `ts`, `datetime`, `uid`, `src_ip_zeek`, `src_port_zeek`, `dest_ip_zeek`, `dest_port_zeek`, `proto`, `service`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `local_orig`, `local_resp`, `missed_bytes`, `history`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes`, `community_id`, `label_tactic`.
- **Massive exact duplication.** 18,562,468 rows contain only 4,851,819 distinct `uid`s and 4,851,829 distinct rows.
  - Benign weeks: every row appears ≈2× (1.9 rows/uid).
  - Attack week (2022-02-06): **9,280,806 rows are 38,331 flows**; each Recon `uid` is repeated exactly **256×** (36,245 uids × 256).
  - The headline "9.28M Reconnaissance flows" is therefore **36,247 real flows**. Any prior result computed on the raw file counted the same 36K flows 256 times, and duplicated rows across train/val would leak.
- **Label counts after dedup:** none 4,813,440; Reconnaissance 36,247; Discovery 2,086; Credential Access 27; Privilege Escalation 13; Exfiltration 7; Resource Development 3; Lateral Movement 3; Persistence 1; Initial Access 1; Defense Evasion 1. No Execution/C2/Collection/Impact.
- **Timestamps:** `ts` (epoch, UTC) covers 2021-12-17 13:00 → 2022-02-19 19:16. `datetime` differs from `ts` by **0 to 6 hours** (naive local, inconsistent). **Use `ts` only.** There is a gap 2022-01-17 → 2022-02-10.
- **Benign and attack traffic are disjoint in time and in hosts.**
  - Benign: 2021-12-17 → 2022-01-17 (49 source hosts). Attack: 2022-02-10 → 2022-02-19 (13 source hosts).
  - Source hosts in both classes: **0**. The attack week has 0 benign rows.
  - Consequence: host identity, calendar date, or any host-specific fingerprint separates classes perfectly. No host in this corpus goes benign → attack, so there is no within-host onset to forecast from Z22 benign context.
- **Attack days:** Recon 02-10…12 (dominated by `143.88.2.10`, `143.88.7.10`, `143.88.5.12`); everything else is on 02-17…19 with 1–27 flows per host.
- **Multi-stage hosts (13 attack hosts):** 5 have >1 tactic, but with tiny volume: `143.88.7.10` (Recon + Discovery, 4,122 flows), `143.88.4.15` (5 tactics, **27 flows**, four labelled at the same instant 12:18:28), `143.88.5.17` (17 flows), `143.88.10.11` (4), `143.88.3.12` (4). Only one host has a non-trivial Recon → Discovery sequence.
- **Order is time-sortable** (`ts`), but file rows were not verified as pre-sorted; sort on load.
- **Benign host population changes:** 17–18 hosts/day until 2022-01-03, then 21–36 hosts/day (new hosts appear from 2022-01-04). Leave-one-day-out over benign will see new hosts in later folds.

## 4. ZeekDataFall22 (`data/ZeekDataFall22/`), the "independent" test corpus

- **`parquet/` contains only one part**, the 2021-12-12 week, 20,745 rows, **all benign**. The attack data is only in `csv_by_tactic/`: 34,621 rows, 13 per-tactic files. Metrics files reference 13 weeks; only this fraction was downloaded.
- **Columns (26):** same as Z22 plus `label_technique` (MITRE technique IDs) and `label_binary` (`True`/`False`/`Duplicate`).
- **Labels (CSV):** none 17,508; Resource Development 13,644; Reconnaissance 2,492; Discovery 861; Privilege Escalation 133; Defense Evasion 133; Execution 30; Initial Access 19; Command and Control 17; Lateral Movement 11; Persistence 10; Credential Access 1; Collection 1. 135 rows have `label_binary=Duplicate`.
- **Attack dates:** 2022-08-31 → 2022-10-23. **Benign dates: 2021-12-17 → 2022-01-08 only.** There is no benign traffic concurrent with the attacks.
- **The benign side is not independent of Z22:** 17,488 of 17,508 benign `uid`s (99.9%) are in ZeekData22, and all 29 benign source hosts are Z22 benign hosts. **Fall22 benign is a subsample of the training corpus.**
- **The attack side is independent by flow:** 0 of 17,113 attack `uid`s appear in Z22. But 3 of the 15 attack source hosts also appear as attack sources in Z22 (0 as Z22 benign).
- **Host progression:** 15 attack hosts; 9 have >1 tactic (e.g. `143.88.10.11`: Lateral Movement, Reconnaissance, Execution, Persistence over 2022-08-31 → 10-23). Row counts per host are small (1–13,652) and the dominant one is a 2-hour Resource Development burst from one host (`143.88.5.11`, 13,644 rows).
- **Consequence for the protocol:** Fall22 can validly test *attack detection/staging on unseen attack episodes*. It **cannot provide independent false-alarm-rate estimates** (its benign is Z22 benign, and it is time-disjoint from the attacks). Decision needed (section 6).

## 5. CIC-2018 improved (`data/cic-2018-improved/`)

- **Only one file is present:** `Botnet-Friday-02-03-2018.parquet` (5,094,963 rows, no truncation, no duplicates). It is not "the V1 CSVs"; the other days are absent.
- **83 columns, no Src/Dst IP, no ports, no timestamp, no Flow ID.** Host-centric bucketing is **impossible** on this dataset. It supports flow-level detection pretraining only, not sequences or forecasting.
- **Labels:** Benign 4,951,834; Botnet Ares 142,921 (2.8%); `Attempted-relabel-as-Benign` 208 (drop these; ambiguous by design).
- A single botnet family/day means pretraining cannot be validated leave-one-day-out.

## 6. Findings that change what is feasible

1. **Forecasting signal is very thin.** Multi-stage attack sequences exist in about 1 DAPT actor with volume (`206.207.50.50`), 1 Z22 host (`143.88.7.10`, two tactics), and a handful of small Fall22 hosts. Any "lead time" or "transition" metric will rest on tens of episodes, not thousands. Confidence intervals must be reported per-episode.
2. **Label/host/time confounding in Z22.** Attack and benign never share hosts or dates. A classifier can score highly by recognising *which host* or *which week*. The feature-hygiene rule (no IPs, no timestamps) is necessary but not sufficient because behavioural fingerprints of the 13 attackers vs 49 benign hosts can still leak.
3. **Duplicated rows (up to 256×) must be removed before any split** or train/test will share identical flows.
4. **Fall22 benign overlaps Z22 training benign** and has no time overlap with attacks; it cannot support an independent FPR.
5. **Stage taxonomies differ:** DAPT (4 stages incl. "Establish Foothold"), Zeek (10–12 ATT&CK tactics, heavily imbalanced), brief (5 phases incl. C2). Cross-dataset stage mapping is a design decision.
6. **The brief's packet-level features (TTL, window size, fragments, retransmissions) are largely absent** in Zeek conn logs; DAPT has flags and init-window sizes only.

## 7. Decisions I need from you before Phase 1

1. **Stage set:** collapse to a coarse common stage map (e.g. Recon / Access+Execution+Foothold / Lateral+Discovery+PrivEsc / C2 / Exfil) or forecast only "attack onset" (binary) and treat stage as secondary?
2. **Fall22 benign:** (a) use only its attack half for lead-time/recall and report no independent FPR; (b) also use Fall22 benign but flag it as a known ≈100% overlap with Z22; (c) allocate a held-out slice of *late-Z22 benign* (Jan 9–17, new hosts) as the benign test side. I recommend (c) with (a), since the alarm budget then rests on benign hosts the model never trained on.
3. **DAPT role:** treat as the only corpus with true benign→recon→foothold→lateral→exfil chains on the same actor (recommended: evaluation of forecasting on `206.207.50.50`, plus LODO), even though it is small.
4. **CIC-2018:** confirm it is excluded from the forecasting task, with only an optional detection-pretraining role.
5. **The brief asks for a logistic-regression baseline with F1/precision/recall/FPR;** your rules ask for lead-time, recall at a fixed alarms/host-hour budget, and calibration. I plan to include LR as a fifth baseline row and report F1/P/R/FPR as secondary metrics, keeping your protocol primary. OK?
