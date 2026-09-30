"""Train and evaluate the demo forecasting stack on DAPT2020 with leave-one-day-out folds.

Every row is predicted only by models that never saw its day. Outputs:
  results/demo/metrics.json          everything the demo app displays
  results/registry.csv               one line per experiment (hypothesis, prediction, outcome)
  models/demo/fold_<date>.joblib     model trained without that day (used by the app on that day)
  models/demo/all.joblib             model trained on every day (used for uploaded data)

Exploratory dev run, NOT the locked evaluation protocol (docs/EVAL_PROTOCOL.md does not exist yet).
Usage: make demo-train
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

sys.path.insert(0, "src")
from netrikan import metrics as M  # noqa: E402
from netrikan import models  # noqa: E402
from netrikan.dapt import STAGE_NAMES, load_dapt_dir  # noqa: E402
from netrikan.features import build_grid, make_table  # noqa: E402

K, L, BUCKET = 5, 10, 60
OUT, MODELS = "results/demo", "models/demo"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Hypotheses and predictions are fixed here, before any result is computed.
HYPOTHESES = {
    "R1-ladder": (
        "Forecasting skill rises LR < LightGBM(current) < LightGBM(+lags) on onset PR-AUC (pooled LODO).",
        "lag > cur > lr on PR-AUC; modest gains because onsets are few."),
    "R2-world": (
        "A GRU dynamics model rolled K steps and read by a GBDT head beats LightGBM(+lags).",
        "No win: within the bootstrap CI of lag. Data is too small for dynamics learning to add signal."),
    "R3-shuffle": (
        "Shuffling history order hurts models that use dynamics.",
        "lag loses <10% relative PR-AUC (its signal is the current bucket); world loses a similar small amount."),
    "R4-stage": (
        "Stage of the forecast attack is predictable across days.",
        "Fails under leave-one-day-out: each day's stage is unseen in the other days (accuracy <= majority)."),
    "R5-blocked": (
        "SECONDARY split: when attack types are shared between train and test (first 60% of each host-day trains, "
        "last 40% tests, gap >= K+L), forecasting and stage forecasting are much stronger than leave-one-day-out.",
        "lag PR-AUC >= 1.5x its LODO value and stage accuracy above the majority baseline."),
}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def data_hash(path="data/dapt2020"):
    h = hashlib.sha256()
    for f in sorted(os.listdir(path)):
        if f.endswith(".csv"):
            h.update(f.encode())
            h.update(open(os.path.join(path, f), "rb").read())
    return h.hexdigest()[:12]


def day_label(d):
    x = dt.date(1970, 1, 1) + dt.timedelta(days=int(d))
    return x.isoformat()


t0 = time.time()
df, info = load_dapt_dir()
print(f"flows: {info}")
grid = build_grid(df, BUCKET)
tab = make_table(grid, K=K, L=L, bucket_s=BUCKET)
days = sorted(np.unique(tab.day))
print(f"table: {tab.n} host-bucket rows, {len(np.unique(tab.start))} host-days, {len(np.unique(tab.host))} internal hosts, days={[day_label(d) for d in days]}")

pop = tab.valid & ~tab.attack  # onset population: benign right now, full horizon observed
allv = tab.valid
eps_all = M.onset_episodes(tab.attack, tab.start, tab.end, K)
print(f"onset population {int(pop.sum())} rows, {int(tab.fut[pop].sum())} positives, {len(eps_all)} onset episodes")

# ---------------- leave-one-day-out
names = ["lr", "cur", "lag", "world", "lag_shuf", "world_shuf"]
oof = {n: np.zeros(tab.n) for n in names}
oof_stage = np.zeros((tab.n, 5))
for d in days:
    te = np.flatnonzero(tab.day == d)
    tr = np.flatnonzero(tab.day != d)
    b = models.fit(tab, tr)
    pr = models.predict(b, tab, te, which=("lr", "cur", "lag", "world", "stage"))
    for n in ("lr", "cur", "lag", "world"):
        oof[n][te] = pr[n]
    oof_stage[te] = pr["stage"]
    joblib.dump(b, f"{MODELS}/fold_{day_label(d)}.joblib")
    bs = models.fit(tab, tr, parts=("lag", "world"), shuffle=True)
    ps = models.predict(bs, tab, te, which=("lag", "world"))
    oof["lag_shuf"][te], oof["world_shuf"][te] = ps["lag"], ps["world"]
    print(f"fold {day_label(d)} done ({time.time() - t0:.0f}s)")
allb = models.fit(tab, np.arange(tab.n))
joblib.dump(allb, f"{MODELS}/all.joblib")

# ---------------- metrics
pooled, boot, perday, allrows = {}, {}, {}, {}
for n in names:
    pooled[n] = M.evaluate(oof[n], tab, pop, eps_all)
    boot[n] = M.block_bootstrap(oof[n], tab, pop)
    perday[n] = {}
    for d in days:
        m_d = pop & (tab.day == d)
        perday[n][day_label(d)] = M.evaluate(oof[n], tab, m_d, eps_all[tab.day[eps_all] == d])
    allrows[n] = M.evaluate(oof[n], tab, allv, eps_all)
# oracle persistence: predicts "attack soon" iff the true current bucket is an attack
pers = tab.attack.astype(float)
persistence = {"onset_population": M.evaluate(pers, tab, pop, eps_all) if tab.fut[pop].sum() else {},
               "all_rows": M.evaluate(pers, tab, allv, eps_all)}
persistence["onset_recall_by_construction"] = float(pers[pop][tab.fut[pop]].mean())

# stage forecast (LODO), evaluated on true onsets only
on = pop & tab.fut
true_s = tab.fut_stage[on]
pred_s = oof_stage[on].argmax(1)
maj = np.bincount(true_s, minlength=5).argmax()
stage_res = {
    "n": int(on.sum()),
    "accuracy": float((pred_s == true_s).mean()),
    "majority_baseline_accuracy": float((true_s == maj).mean()),
    "true_stage_counts": {STAGE_NAMES[i]: int((true_s == i).sum()) for i in range(1, 5)},
    "per_stage_recall": {STAGE_NAMES[i]: (float((pred_s[true_s == i] == i).mean()) if (true_s == i).any() else None) for i in range(1, 5)},
}
stage_by_day = {}
for d in days:
    m = on & (tab.day == d)
    if m.any():
        stage_by_day[day_label(d)] = {"n": int(m.sum()), "accuracy": float((oof_stage[m].argmax(1) == tab.fut_stage[m]).mean()),
                                      "stages_present": [STAGE_NAMES[i] for i in np.unique(tab.fut_stage[m])]}
stage_res["by_day"] = stage_by_day

# ---------------- secondary: within-day blocked split (shared attack types; never used to choose anything)
i_all = np.arange(tab.n)
bnd = tab.start + np.floor(0.6 * (tab.end - tab.start)).astype(int)
tr_b = np.flatnonzero(i_all < bnd - K)          # horizon of every training row ends before the boundary
te_b = np.flatnonzero(i_all >= bnd + L)         # history of every test row starts after the boundary
bb = models.fit(tab, tr_b)
bs_ = models.fit(tab, tr_b, parts=("lag", "world"), shuffle=True)
pb, pbs = models.predict(bb, tab, te_b, which=("lr", "cur", "lag", "world", "stage")), models.predict(bs_, tab, te_b, which=("lag", "world"))
p_b = {n: np.zeros(tab.n) for n in names}
for n in ("lr", "cur", "lag", "world"):
    p_b[n][te_b] = pb[n]
p_b["lag_shuf"][te_b], p_b["world_shuf"][te_b] = pbs["lag"], pbs["world"]
pop_b = np.zeros(tab.n, dtype=bool)
pop_b[te_b] = True
pop_b &= pop
eps_b = eps_all[(eps_all - K) >= (bnd[eps_all] + L)]
blocked = {n: M.evaluate(p_b[n], tab, pop_b, eps_b) for n in names}
st_b = np.zeros((tab.n, 5))
st_b[te_b] = pb["stage"]
on_b = pop_b & tab.fut
maj_b = np.bincount(tab.fut_stage[on_b], minlength=5).argmax()
blocked_stage = {"n": int(on_b.sum()), "accuracy": float((st_b[on_b].argmax(1) == tab.fut_stage[on_b]).mean()),
                 "majority_baseline_accuracy": float((tab.fut_stage[on_b] == maj_b).mean()),
                 "per_stage_recall": {STAGE_NAMES[s]: (float((st_b[on_b][tab.fut_stage[on_b] == s].argmax(1) == s).mean())
                                                       if (tab.fut_stage[on_b] == s).any() else None) for s in range(1, 5)}}

primary = "lag" if pooled["lag"].get("pr_auc", 0) >= pooled["world"].get("pr_auc", 0) else "world"
dataset = {
    "source": "DAPT2020 (CICFlowMeter flows)", "flows_after_dedup": info["rows_after_dedup"],
    "bucket_seconds": BUCKET, "horizon_K_buckets": K, "history_L_buckets": L,
    "internal_hosts": int(len(np.unique(tab.host))), "host_days": int(len(np.unique(tab.start))),
    "host_bucket_rows": int(tab.n), "attack_buckets": int(tab.attack.sum()),
    "onset_population_rows": int(pop.sum()), "onset_positive_rows": int(tab.fut[pop].sum()),
    "onset_episodes": int(len(eps_all)),
    "onsets_by_day": {day_label(d): int((tab.day[eps_all] == d).sum()) for d in days},
    "stage_buckets": {STAGE_NAMES[i]: int((tab.stage == i).sum()) for i in range(1, 5)},
    "days": [day_label(d) for d in days],
}
result = {"dataset": dataset, "pooled": pooled, "bootstrap": boot, "per_day": perday, "all_rows": allrows,
          "oracle_persistence": persistence, "stage": stage_res, "primary_model": primary,
          "secondary_blocked": {"pooled": blocked, "stage": blocked_stage,
                                "split": "first 60% of each host-day trains, last 40% tests, gap >= K+L buckets"},
          "alert_threshold_1_per_host_hour": pooled[primary].get("thr@1.0/h"),
          "note": "Exploratory leave-one-day-out dev run; not the locked evaluation protocol."}
json.dump(result, open(f"{OUT}/metrics.json", "w"), indent=1)

# ---------------- registry
verdict = {}
p = lambda n: pooled[n].get("pr_auc", float("nan"))
verdict["R1-ladder"] = ("CONFIRMED" if p("lag") > p("cur") > p("lr") else "REFUTED",
                        f"PR-AUC lr={p('lr'):.3f} cur={p('cur'):.3f} lag={p('lag'):.3f} (prevalence {pooled['lr']['prevalence']:.3f})")
lo, hi = boot["lag"]["pr_auc_ci90"]
verdict["R2-world"] = ("CONFIRMED" if p("world") <= hi else "REFUTED",
                       f"world PR-AUC={p('world'):.3f} vs lag={p('lag'):.3f} (lag 90% CI {lo:.3f}-{hi:.3f})")
drop = lambda a, b: (p(a) - p(b)) / p(a) if p(a) else float("nan")
dl, dw = drop("lag", "lag_shuf"), drop("world", "world_shuf")
uses_order = (dl > 0.10) or (dw > 0.10)
verdict["R3-shuffle"] = ("CONFIRMED" if dl < 0.10 else "REFUTED",
                         f"relative PR-AUC change under shuffled history: lag {-dl:+.1%}, world {-dw:+.1%} (negative = worse). "
                         + ("Order matters for at least one model." if uses_order else
                            "Shuffling does NOT hurt: no evidence either model uses temporal order (signal is the current bucket)."))
verdict["R4-stage"] = ("CONFIRMED" if stage_res["accuracy"] <= stage_res["majority_baseline_accuracy"] + 0.05 else "REFUTED",
                       f"stage accuracy {stage_res['accuracy']:.2f} vs majority {stage_res['majority_baseline_accuracy']:.2f} over {stage_res['n']} onset buckets")
bl, ll = blocked["lag"].get("pr_auc", float("nan")), p("lag")
verdict["R5-blocked"] = ("CONFIRMED" if (bl >= 1.5 * ll and blocked_stage["accuracy"] > blocked_stage["majority_baseline_accuracy"]) else "REFUTED",
                         f"blocked lag PR-AUC={bl:.3f} (prevalence {blocked['lag'].get('prevalence', float('nan')):.3f}) vs LODO {ll:.3f}; "
                         f"blocked stage accuracy {blocked_stage['accuracy']:.2f} vs majority {blocked_stage['majority_baseline_accuracy']:.2f}")
reg = "results/registry.csv"
new = not os.path.exists(reg)
with open(reg, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["run_id", "git_hash", "data_hash", "split", "config", "metrics", "hypothesis", "predicted_outcome", "actual_outcome"])
    gh = sh("git rev-parse --short HEAD") + ("-dirty" if sh("git status --porcelain") else "")
    dh = data_hash()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    for k, (hyp, pred) in HYPOTHESES.items():
        w.writerow([f"{stamp}-{k}", gh, dh, "LODO(5 days) onset population", f"K={K} L={L} bucket={BUCKET}s",
                    "results/demo/metrics.json", hyp, pred, f"{verdict[k][0]}: {verdict[k][1]}"])
print(json.dumps({k: v for k, v in verdict.items()}, indent=1))
print(f"primary model: {primary}; total {time.time() - t0:.0f}s")
