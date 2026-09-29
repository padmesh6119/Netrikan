"""Zero-shot transfer across labs: leave-one-corpus-out attack detection on host-minutes.

Corpora (different labs, capture tools, years, attack tooling): DAPT2020, ZeekData24, CIC-IDS2017 slices, CTU-13 s4.
For each held-out corpus the detector is trained on the OTHER THREE only, so every held-out attack family is unseen.
Features: the 22 common host-minute features (flow counts, distinct peers/ports, bytes, duration, port classes).
A learned detector only counts if it beats the flow-volume and fan-out heuristics.

Outputs: results/zero_shot/metrics.json, models/zero_shot/holdout_<corpus>.joblib, rows in results/registry.csv.
Exploratory dev run, not the locked evaluation protocol. Usage: make zero-shot
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
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from netrikan import extra  # noqa: E402
from netrikan.surprise import Surprise  # noqa: E402

OUT, MODELS = "results/zero_shot", "models/zero_shot"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
CF = extra.CF

# Hypotheses and predictions are fixed here, before any result is computed.
HYPOTHESES = {
    "ZS1-volumetric": (
        "A detector trained on three other labs' corpora ranks unseen volumetric families (CIC-2017 DDoS, port scan) above benign host-minutes.",
        "ROC-AUC >= 0.9 for both, but no better than the flow-volume heuristic (gbdt - volume <= 0.05): volume is the whole signal."),
    "ZS2-stealthy": (
        "The same detector transfers to low-volume or stealthy families (Heartbleed, CTU-13 botnet spam/C&C).",
        "No: ROC-AUC < 0.7 on Heartbleed and < 0.75 on the botnet."),
    "ZS3-anomaly": (
        "A benign-only anomaly detector (IsolationForest) transfers better than the supervised detector.",
        "No: its mean ROC-AUC over held-out families is <= the supervised detector's."),
    "ZS5-surprise": (
        "World-model surprise (GRU next-minute prediction error, trained on benign minutes of the other corpora, no attack labels) transfers as well as IsolationForest.",
        "Not better: mean ROC-AUC over held-out families <= IsolationForest + 0.03."),
    "ZS4-reverse": (
        "Detectors trained on the other corpora transfer to DAPT2020's pentest stages.",
        "Mostly no: ROC-AUC <= 0.7 for at least 3 of the 4 DAPT stages."),
}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:4.0f}s] {m}", flush=True)
corp = {}
for n in extra.CORPORA:
    corp[n] = extra.host_minutes(extra.load_corpus(n), n)
    F = corp[n]["fam"]
    log(f"{n}: {len(corp[n]['X']):,} host-minutes with traffic, attack {int(F.any(1).sum()):,}, families {dict(zip(corp[n]['fam_names'], F.sum(0).tolist()))}")

I = {f: CF.index(f) for f in CF}
def volume(X):
    return np.expm1(X[:, I["in_n_flows"]]) + np.expm1(X[:, I["out_n_flows"]])
def fanout(X):
    return sum(np.expm1(X[:, I[f]]) for f in ("in_n_peers", "out_n_peers", "in_n_ports", "out_n_ports"))


def auc_ci(sp, sn, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    sn = sn if len(sn) <= 20000 else rng.choice(sn, 20000, replace=False)
    out = []
    for _ in range(reps):
        p, n = rng.choice(sp, len(sp)), rng.choice(sn, len(sn))
        out.append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


results = {}
for held in extra.CORPORA:
    train = [n for n in extra.CORPORA if n != held]
    Xs, ys, ws = [], [], []
    for n in train:
        X, y = corp[n]["X"], corp[n]["fam"].any(1)
        w = np.where(y, 0.5 / max(y.sum(), 1), 0.5 / max((~y).sum(), 1))  # each corpus and class gets equal total weight
        Xs.append(X); ys.append(y); ws.append(w)
    Xtr, ytr, wtr = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    gbdt = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0).fit(Xtr, ytr, sample_weight=wtr * len(wtr))
    Xb = Xtr[~ytr]
    iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=4).fit(Xb[np.random.default_rng(0).choice(len(Xb), min(len(Xb), 100000), replace=False)])
    seqs = []
    for n in train:
        c = corp[n]
        for g in np.unique(c["gid"]):
            idx = np.flatnonzero(c["gid"] == g)
            seqs.append((c["Xd"][idx], c["Fd"][idx].any(1)))
    sur = Surprise().fit(seqs)
    joblib.dump({"gbdt": gbdt, "iforest": iso, "surprise": sur, "cf": CF, "trained_on": train}, f"{MODELS}/holdout_{held}.joblib")

    H = corp[held]
    scores = {"gbdt": gbdt.predict_proba(H["X"])[:, 1], "iforest": -iso.score_samples(H["X"]), "surprise": sur.score_grid(H["Xd"], H["gid"])[H["keep"]],
              "volume": volume(H["X"]), "fanout": fanout(H["X"])}
    benign = ~H["fam"].any(1)
    res = {"trained_on": train, "n_benign_host_minutes": int(benign.sum()), "families": {}}
    for k, fname in enumerate(H["fam_names"]):
        pos = H["fam"][:, k]
        m = pos | benign
        y = pos[m].astype(int)
        r = {"name": extra.FAMILIES[held][fname], "n_pos": int(pos.sum()), "base_rate": float(y.mean())}
        for s, v in scores.items():
            r[s] = {"roc_auc": float(roc_auc_score(y, v[m])), "pr_auc": float(average_precision_score(y, v[m]))}
        r["gbdt"]["roc_auc_ci90"] = auc_ci(scores["gbdt"][pos], scores["gbdt"][benign])
        r["gbdt"]["recall_at_0.5"] = float((scores["gbdt"][pos] >= 0.5).mean())
        r["gbdt"]["benign_alarm_rate_at_0.5"] = float((scores["gbdt"][benign] >= 0.5).mean())
        res["families"][fname] = r
    results[held] = res
    log(f"held out {held}: " + ", ".join(f"{extra.FAMILIES[held][f]} gbdt {r['gbdt']['roc_auc']:.2f} / iforest {r['iforest']['roc_auc']:.2f} / surprise {r['surprise']['roc_auc']:.2f} / volume {r['volume']['roc_auc']:.2f} (n={r['n_pos']})" for f, r in res["families"].items()))

json.dump({"protocol": "leave-one-corpus-out; features = 22 common host-minute features; positives = host-minutes containing the family, negatives = benign host-minutes of the held-out corpus",
           "corpora": {n: {"label": extra.LABEL[n], "host_minutes": int(len(corp[n]["X"])), "attack_host_minutes": int(corp[n]["fam"].any(1).sum())} for n in extra.CORPORA},
           "features": CF, "results": results,
           "note": "Exploratory dev run. Positive counts are small (tens of host-minutes) and host-minutes within a corpus are not independent, so intervals understate uncertainty."},
          open(f"{OUT}/metrics.json", "w"), indent=1)

g = lambda h, f, s="gbdt": results[h]["families"][f][s]["roc_auc"]
fam_all = [(h, f) for h in results for f in results[h]["families"]]
verdict = {
    "ZS1-volumetric": (all(g("cic17", f) >= 0.9 and g("cic17", f) - g("cic17", f, "volume") <= 0.05 for f in ("fam_ddos", "fam_portscan")),
                       f"DDoS gbdt {g('cic17', 'fam_ddos'):.2f} vs volume {g('cic17', 'fam_ddos', 'volume'):.2f}; port scan gbdt {g('cic17', 'fam_portscan'):.2f} vs volume {g('cic17', 'fam_portscan', 'volume'):.2f}"),
    "ZS2-stealthy": (g("cic17", "fam_heartbleed") < 0.7 and g("ctu13", "fam_botnet") < 0.75,
                     f"Heartbleed {g('cic17', 'fam_heartbleed'):.2f} (n={results['cic17']['families']['fam_heartbleed']['n_pos']}), botnet {g('ctu13', 'fam_botnet'):.2f} (n={results['ctu13']['families']['fam_botnet']['n_pos']})"),
    "ZS3-anomaly": (np.mean([g(h, f, "iforest") for h, f in fam_all]) <= np.mean([g(h, f) for h, f in fam_all]),
                    f"mean ROC-AUC over {len(fam_all)} families: IsolationForest {np.mean([g(h, f, 'iforest') for h, f in fam_all]):.2f}, supervised {np.mean([g(h, f) for h, f in fam_all]):.2f}, volume heuristic {np.mean([g(h, f, 'volume') for h, f in fam_all]):.2f}"),
    "ZS5-surprise": (
        "World-model surprise (GRU next-minute prediction error, trained on benign minutes of the other corpora, no attack labels) transfers as well as IsolationForest.",
        "Not better: mean ROC-AUC over held-out families <= IsolationForest + 0.03."),
    "ZS5-surprise": (np.mean([g(h, f, "surprise") for h, f in fam_all]) <= np.mean([g(h, f, "iforest") for h, f in fam_all]) + 0.03,
                     f"mean ROC-AUC: surprise {np.mean([g(h, f, 'surprise') for h, f in fam_all]):.2f} vs IsolationForest {np.mean([g(h, f, 'iforest') for h, f in fam_all]):.2f}"),
    "ZS4-reverse": (sum(g("dapt", f) <= 0.7 for f in results["dapt"]["families"]) >= 3,
                    "DAPT stages ROC-AUC " + ", ".join(f"{extra.FAMILIES['dapt'][f]} {g('dapt', f):.2f}" for f in results["dapt"]["families"])),
}
reg = "results/registry.csv"
with open(reg, "a", newline="") as fh:
    w = csv.writer(fh)
    gh = sh("git rev-parse --short HEAD") + ("-dirty" if sh("git status --porcelain") else "")
    files = ["data/new-datasets/" + f for f in sorted(os.listdir("data/new-datasets")) if f.endswith(".gz")]
    dh = hashlib.sha256(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:12]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    for k, (hyp, pred) in HYPOTHESES.items():
        ok, txt = verdict[k]
        w.writerow([f"{stamp}-{k}", gh, dh, "leave-one-corpus-out (zero-shot across labs)", "22 common features, 60s buckets", "results/zero_shot/metrics.json", hyp, pred,
                    f"{'PREDICTION CONFIRMED' if ok else 'PREDICTION REFUTED'}: {txt}"])
for k, (ok, txt) in verdict.items():
    print(f"{k:16s} {'confirmed' if ok else 'REFUTED  '} {txt}")
log("done")
