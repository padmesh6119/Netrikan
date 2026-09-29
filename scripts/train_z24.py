"""ZeekData24: technique recognizer + attacker-campaign forecaster, leave-one-attack-week-out.

  Task A  recognizer: technique active in a host-minute (current-minute behaviour only)
  Task B  forecaster: P(technique T fires in the next K=5 min | attacker history), per technique

Folds: each of the 5 attack weeks is held out in turn (the same campaign replayed, so this tests
week-to-week replay, NOT unseen attack patterns). The two benign weeks are held out alternately.
Outputs: results/z24/metrics.json, models/z24/fold_<week>.joblib + all.joblib, rows in results/registry.csv.
Exploratory dev run, not the locked evaluation protocol. Usage: make z24-train
"""
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from netrikan import campaign as C  # noqa: E402
from netrikan import metrics as M  # noqa: E402
from netrikan.features import EPOCH, build_grid  # noqa: E402
from netrikan.zeek import MONITORED, TECH_LABEL, TECH_NAME, TECH_TACTIC, TECHS, load_z24  # noqa: E402

OUT, MODELS = "results/z24", "models/z24"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
BUDGETS = M.BUDGETS
NT, K = C.NT, C.K

# Hypotheses and predictions are fixed here, before any result is computed.
HYPOTHESES = {
    "Z1-recognize": (
        "Technique can be recognised from one host-minute's behaviour in a held-out week of the same campaign.",
        "High for the bursty techniques (Brute Force, Active Scanning: PR-AUC >= 0.9); clearly lower (< 0.7) for at least one of the "
        "low-volume ones (Exploit, Valid Accounts, Exfil)."),
    "Z2-ladder": (
        "Recent technique history predicts the next burst: history models (own/all-technique) and the renewal hazard beat the state-only model.",
        "sched >= 1.5x state (macro PR-AUC); renewal within 10% of sched (the schedule is a renewal process)."),
    "Z3-world": (
        "The GRU world model (dynamics rolled K steps + GBDT head) beats the schedule-feature GBDT.",
        "No win: world <= 1.05x sched. Nothing beyond time-since-last-burst is learnable from these logs."),
    "Z4-shuffle": (
        "Shuffling the order of the 150-minute history hurts the history models.",
        "sched loses >= 15% relative macro PR-AUC (recency is the signal)."),
    "Z5-clock": (
        "ABLATION (breaks the no-timestamp rule, not shipped): adding minute-of-hour shows how much of the future is a fixed schedule.",
        "clock >= 2x sched: bursts fire once per clock hour at a random minute, so the wall clock is far more informative than history."),
    "Z6-progression": (
        "Other techniques' history helps forecast a technique (kill-chain progression exists).",
        "No: own-technique history within 3% of all-technique history (techniques run as independent schedules)."),
}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

# ------------------------------------------------------------------ data
flows = load_z24()
day = (flows["ts"] - EPOCH).dt.days
assert flows.groupby(day)["wk"].nunique().max() == 1, "a calendar day belongs to two capture files"
day2wk = flows.groupby(day)["wk"].first().to_dict()
attack_weeks = sorted(flows.loc[flows["attack"], "wk"].unique())
benign_weeks = sorted(set(flows["wk"]) - set(attack_weeks))
grid = build_grid(flows, 60, MONITORED, tuple(TECH_LABEL))
X = C.log_state(grid)
Y = grid[TECH_LABEL].to_numpy().astype(bool)
traffic = ((grid["in_n_flows"] + grid["out_n_flows"]) > 0).to_numpy()
wk = grid["day"].map(day2wk).to_numpy()
log(f"{len(flows):,} flows, {len(grid):,} host-minutes ({int(traffic.sum()):,} with traffic); attack weeks {attack_weeks}, benign weeks {benign_weeks}")

# ------------------------------------------------------------------ Task A: recognizer, leave-one-attack-week-out
P = np.zeros((len(grid), NT))
recog = {}
benign_fold = {}
for i, aw in enumerate(attack_weeks):
    bw = benign_weeks[i % len(benign_weeks)]
    te = np.isin(wk, [aw, bw])
    models = C.fit_recognizer(X, Y, np.flatnonzero(traffic & ~te))
    recog[aw] = models
    rows = np.flatnonzero(np.isin(wk, [aw]) | ((wk == bw) & (i < len(benign_weeks))))
    P[rows] = C.predict_recognizer(models, X[rows], traffic[rows])
    if i < len(benign_weeks):
        benign_fold[bw] = aw
    log(f"recognizer fold {aw} (benign held out: {bw})")
recog_all = C.fit_recognizer(X, Y, np.flatnonzero(traffic))

is_att = np.isin(wk, attack_weeks)
is_ben = np.isin(wk, benign_weeks)
rec = {"per_technique": {}, "per_week_recall@1/h": {}, "confusion@1/h": {}}
thr1 = {}
for j, t in enumerate(TECHS):
    pos = Y[:, j] & is_att
    ben = is_ben
    oth = is_att & traffic & ~Y[:, j]
    y = np.r_[np.ones(pos.sum()), np.zeros(ben.sum()), np.zeros(oth.sum())]
    s = np.r_[P[pos, j], P[ben, j], P[oth, j]]
    from sklearn.metrics import average_precision_score, roc_auc_score
    r = {"n_pos": int(pos.sum()), "n_benign_minutes": int(ben.sum()), "n_other_attack_minutes": int(oth.sum()),
         "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
         "pr_auc_vs_benign_only": float(average_precision_score(np.r_[np.ones(pos.sum()), np.zeros(ben.sum())], np.r_[P[pos, j], P[ben, j]]))}
    for b in BUDGETS:
        thr = M.threshold_for_budget(P[ben, j], b, 60)
        tp, fpb, fpo = int((P[pos, j] >= thr).sum()), int((P[ben, j] >= thr).sum()), int((P[oth, j] >= thr).sum())
        r[f"thr@{b}/h"] = thr
        r[f"recall@{b}/h"] = tp / max(int(pos.sum()), 1)
        r[f"precision@{b}/h"] = tp / max(tp + fpb + fpo, 1)
        r[f"f1@{b}/h"] = 2 * tp / max(2 * tp + fpb + fpo + (int(pos.sum()) - tp), 1)
        r[f"benign_fpr@{b}/h"] = fpb / int(ben.sum())
        r[f"other_technique_false_fire_rate@{b}/h"] = fpo / max(int(oth.sum()), 1)
    thr1[j] = r["thr@1.0/h"]
    # fixed, untuned operating point p >= 0.5 (what the forecaster and the app use to call a technique)
    tp, fpb, fpo = int((P[pos, j] >= 0.5).sum()), int((P[ben, j] >= 0.5).sum()), int((P[oth, j] >= 0.5).sum())
    r["at_0.5"] = {"recall": tp / max(int(pos.sum()), 1), "precision": tp / max(tp + fpb + fpo, 1),
                   "f1": 2 * tp / max(2 * tp + fpb + fpo + (int(pos.sum()) - tp), 1),
                   "benign_alarms_per_host_hour": fpb / (int(ben.sum()) / 60), "benign_fpr": fpb / int(ben.sum()),
                   "other_technique_false_fire_rate": fpo / max(int(oth.sum()), 1)}
    rec["per_technique"][t] = r
    rec["per_week_recall@1/h"][t] = {w: float((P[pos & (wk == w), j] >= thr1[j]).mean()) if (pos & (wk == w)).any() else None for w in attack_weeks}
excl = Y.sum(1) == 1
for j, t in enumerate(TECHS):
    m = excl & Y[:, j] & is_att
    rec["confusion@1/h"][t] = {TECHS[k]: (float((P[m, k] >= thr1[k]).mean()) if m.any() else None) for k in range(NT)}
rec["confusion@0.5"] = {}
for j, t in enumerate(TECHS):
    m = Y[:, j] & is_att   # minutes containing technique j (a minute can contain several)
    rec["confusion@0.5"][t] = {TECHS[k]: float((P[m, k] >= 0.5).mean()) for k in range(NT)}
anyp = P.max(1)
att_tr = is_att & traffic & Y.any(1)
yy = np.r_[np.ones(att_tr.sum()), np.zeros(is_ben.sum())]
ss = np.r_[anyp[att_tr], anyp[is_ben]]
thr_any = M.threshold_for_budget(anyp[is_ben], 1.0, 60)
rec["any_attack_detector"] = {"pr_auc": float(average_precision_score(yy, ss)), "roc_auc": float(roc_auc_score(yy, ss)),
                              "recall@1.0/h": float((anyp[att_tr] >= thr_any).mean()), "at_0.5_recall": float((anyp[att_tr] >= 0.5).mean()),
                              "at_0.5_benign_alarms_per_host_hour": float((anyp[is_ben] >= 0.5).sum() / (is_ben.sum() / 60))}
rec["alarm_any_technique_on_benign_weeks"] = float((P[is_ben] >= np.array([thr1[j] for j in range(NT)])).any(1).mean())
log("Task A metrics done: " + ", ".join(f"{t} PR-AUC {rec['per_technique'][t]['pr_auc']:.3f} F1@0.5 {rec['per_technique'][t]['at_0.5']['f1']:.2f}" for t in TECHS))

# ------------------------------------------------------------------ Task B: attacker forecasting
Fpred = np.zeros_like(Y)
Fpred[is_att] = P[is_att] >= 0.5
S = C.build_seqs(grid, X, Y, Fpred, day2wk, select=Y)
fut, valid = C.targets(S)
attackers = S.host[np.unique(S.start)]
log(f"attacker sequences: {len(S.seq_bounds())} ({len(set(attackers))} hosts), {S.n:,} minutes; onset rows per technique {[int((fut[valid][:, j] & ~S.F[valid][:, j]).sum()) for j in range(NT)]}")

NAMES = ["renewal", "state", "own", "sched", "sched_shuf", "clock", "world", "world_shuf", "sched_oracle"]
oof = {n: np.zeros((S.n, NT)) for n in NAMES}
bundles = {}
for aw in attack_weeks:
    te_rows = np.flatnonzero(S.wk == aw)
    S_tr, S_te = S.take([w for w in attack_weeks if w != aw]), S.take([aw])
    fut_tr, valid_tr = C.targets(S_tr)
    world = C.World().fit(S_tr)
    models = C.fit_forecasters(S_tr, fut_tr, valid_tr, world=world)
    pr = C.predict_forecasters(models, S_te, S_te.Fp, world=world)
    for n in ("renewal", "state", "own", "sched", "clock", "world"):
        oof[n][te_rows] = pr[n]
    oof["sched_oracle"][te_rows] = C.predict_forecasters({"sched": models["sched"]}, S_te, S_te.F)["sched"]
    world_s = C.World(block_shuffle=True).fit(S_tr)
    models_s = C.fit_forecasters(S_tr, fut_tr, valid_tr, world=world_s, shuffle=True, parts=("sched", "world"))
    ps = C.predict_forecasters(models_s, S_te, S_te.Fp, shuffle=True, world=world_s)
    oof["sched_shuf"][te_rows], oof["world_shuf"][te_rows] = ps["sched"], ps["world"]
    bundles[aw] = {"recog": recog[aw], "models": models, "world": world, "K": K}
    joblib.dump(bundles[aw], f"{MODELS}/fold_{aw}.joblib")
    log(f"forecast fold {aw} done")
world_all = C.World().fit(S)
models_all = C.fit_forecasters(S, fut, valid, world=world_all)
joblib.dump({"recog": recog_all, "models": models_all, "world": world_all, "K": K}, f"{MODELS}/all.joblib")

fc = {n: {} for n in NAMES}
for n in NAMES:
    for j, t in enumerate(TECHS):
        v = C.view(S, j, fut, valid)
        pop = valid & ~S.F[:, j]
        eps = M.onset_episodes(S.F[:, j], S.start, S.end, K)
        fc[n][t] = M.evaluate(oof[n][:, j], v, pop, eps)
boot = {}
sub = (np.arange(S.n) % 3) == 0
for n in ("renewal", "state", "sched", "world", "clock"):
    boot[n] = {}
    for j, t in enumerate(TECHS):
        v = C.view(S, j, fut, valid)
        boot[n][t] = M.block_bootstrap(oof[n][:, j], v, valid & ~S.F[:, j] & sub, reps=60)
macro = {}
for n in NAMES:
    keys = ["pr_auc", "roc_auc", "recall@1.0/h", "precision@1.0/h", "f1@1.0/h", "fpr@1.0/h", "brier", "ece", "onset_detected@1.0/h", "median_lead_min@1.0/h", "prevalence"]
    macro[n] = {k: float(np.nanmean([fc[n][t].get(k, np.nan) for t in TECHS])) for k in keys}
log("Task B metrics: " + ", ".join(f"{n}={macro[n]['pr_auc']:.3f}" for n in NAMES) + f" (base rate {macro['sched']['prevalence']:.3f})")

# ------------------------------------------------------------------ campaign-structure evidence (computed from the flows)
att = flows[flows["attack"]]
minute_flags = {}
for j, t in enumerate(TECHS):
    x = att[att[f"t_{t}"]]
    minute_flags[t] = x.assign(m=((x["ts"] - EPOCH).dt.total_seconds() // 60).astype(int)).groupby(["wk", "Src IP"])["m"].apply(lambda v: np.sort(v.unique())).to_dict()
lags = {}
for a in TECHS[:4]:
    lags[a] = {}
    for b in TECHS[:4]:
        if a == b:
            continue
        ls = []
        for key, ma in minute_flags[a].items():
            mb = minute_flags[b].get(key)
            if mb is None:
                continue
            jdx = np.searchsorted(mb, ma, side="left")
            ls += [int(mb[q] - a_) for a_, q in zip(ma, jdx) if q < len(mb)]
        ls = np.array(ls)
        lags[a][b] = {"n": int(len(ls)), "median_min": float(np.median(ls)), "p_within_5": float((ls <= 5).mean())}
moh, gaps = {}, {}
for t in TECHS[:4]:
    x = att[att[f"t_{t}"]]
    mm = ((x["ts"] - EPOCH).dt.total_seconds() // 60).astype(int)
    first = mm.groupby([x["Src IP"], mm // 60]).min() % 60
    moh[t] = {"mean": float(first.mean()), "std": float(first.std())}
    g = []
    for key, ma in minute_flags[t].items():
        st = ma[np.r_[True, np.diff(ma) > 1]]
        g += list(np.diff(st))
    gaps[t] = {"p5_p25_p50_p75_p95": [float(v) for v in np.percentile(g, [5, 25, 50, 75, 95])]}
structure = {"cross_technique_lags": lags, "minute_of_hour_of_first_flow": moh, "burst_gap_minutes": gaps,
             "uniform_minute_of_hour_std": float(np.sqrt((60 ** 2 - 1) / 12)),
             "attackers": [{"host": h, "week": w, "minutes": int(e - a), **{TECH_NAME[t]: int(S.F[a:e, j].sum()) for j, t in enumerate(TECHS)}}
                           for a, e in S.seq_bounds() for h, w in [(S.host[a], S.wk[a])]]}

# ------------------------------------------------------------------ write
q = lambda n, k="pr_auc": macro[n][k]
dataset = {"source": "UWF-ZeekData24 (Zeek conn logs)", "flows": int(len(flows)), "host_minutes": int(len(grid)), "traffic_minutes": int(traffic.sum()),
           "attack_weeks": attack_weeks, "benign_weeks": benign_weeks, "benign_held_out_by": benign_fold,
           "techniques": {t: {"name": TECH_NAME[t], "tactic": TECH_TACTIC[t], "flows": int(flows[f"t_{t}"].sum())} for t in TECHS},
           "attacker_sequences": len(S.seq_bounds()), "attacker_hosts": int(len(set(attackers))), "sequence_minutes": int(S.n),
           "horizon_K_minutes": K, "history_window_minutes": C.WIN}
result = {"dataset": dataset, "recognizer": rec, "forecast": {"per_technique": fc, "macro": macro, "bootstrap": boot},
          "structure": structure, "note": "Exploratory leave-one-attack-week-out dev run; not the locked evaluation protocol. "
          "The weeks are replays of one campaign, so this measures replay, not generalisation to unseen attacks."}
json.dump(result, open(f"{OUT}/metrics.json", "w"), indent=1)

r = rec["per_technique"]
verdict = {
    "Z1-recognize": (r["T1110"]["pr_auc"] >= 0.9 and r["T1595"]["pr_auc"] >= 0.9 and min(r[t]["pr_auc"] for t in ("T1190", "T1078", "T1048")) < 0.7,
                     "PR-AUC " + ", ".join(f"{TECH_NAME[t]} {r[t]['pr_auc']:.2f}" for t in TECHS)),
    "Z2-ladder": (q("sched") >= 1.5 * q("state") and abs(q("renewal") - q("sched")) / q("sched") <= 0.10,
                  f"macro PR-AUC state {q('state'):.3f}, renewal {q('renewal'):.3f}, own {q('own'):.3f}, sched {q('sched'):.3f} (base rate {q('sched', 'prevalence'):.3f})"),
    "Z3-world": (q("world") <= 1.05 * q("sched"), f"world {q('world'):.3f} vs sched {q('sched'):.3f}"),
    "Z4-shuffle": ((q("sched") - q("sched_shuf")) / q("sched") >= 0.15,
                   f"sched {q('sched'):.3f} -> shuffled {q('sched_shuf'):.3f} ({(q('sched_shuf') - q('sched')) / q('sched'):+.0%}); world {q('world'):.3f} -> shuffled {q('world_shuf'):.3f} ({(q('world_shuf') - q('world')) / q('world'):+.0%})"),
    "Z5-clock": (q("clock") >= 2 * q("sched"), f"clock-aware {q('clock'):.3f} vs sched {q('sched'):.3f} ({q('clock') / q('sched'):.1f}x)"),
    "Z6-progression": (q("own") >= 0.97 * q("sched"), f"own {q('own'):.3f} vs all-technique {q('sched'):.3f} ({q('own') / q('sched'):.2f}x)"),
}
reg = "results/registry.csv"
new = not os.path.exists(reg)
with open(reg, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["run_id", "git_hash", "data_hash", "split", "config", "metrics", "hypothesis", "predicted_outcome", "actual_outcome"])
    gh = sh("git rev-parse --short HEAD") + ("-dirty" if sh("git status --porcelain") else "")
    dh = hashlib.sha256(b"".join(open(f, "rb").read() for f in sorted(os.path.join("data/UWF_Datasets/ZeekData24/parquet", x) for x in os.listdir("data/UWF_Datasets/ZeekData24/parquet") if x.endswith(".parquet")))).hexdigest()[:12]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    for k, (hyp, pred) in HYPOTHESES.items():
        ok, txt = verdict[k]
        w.writerow([f"{stamp}-{k}", gh, dh, "leave-one-attack-week-out (replay), K=5 min", f"WIN={C.WIN} bucket=60s", "results/z24/metrics.json", hyp, pred,
                    f"{'PREDICTION CONFIRMED' if ok else 'PREDICTION REFUTED'}: {txt}"])
for k, (ok, txt) in verdict.items():
    print(f"{k:16s} {'confirmed' if ok else 'REFUTED  '} {txt}")
log("done")
