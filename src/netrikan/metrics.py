"""Forecast metrics: ranking, recall at a fixed false-alarm budget, calibration, onset lead time."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BUDGETS = (0.5, 1.0, 2.0)  # false alarms per host-hour


def threshold_for_budget(p_neg: np.ndarray, budget_per_host_hour: float, bucket_s: int) -> float:
    """Lowest threshold whose alarm rate on negative buckets stays within the budget."""
    hours = len(p_neg) * bucket_s / 3600.0
    allowed = int(np.floor(budget_per_host_hour * hours))
    s = np.sort(p_neg)[::-1]
    if allowed >= len(s):
        return 0.0
    return float(np.nextafter(s[allowed], 1.0))  # strictly above the (allowed+1)-th highest


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def onset_episodes(attack, start, end, K):
    """Row index of the first attack bucket of each onset: attack now, K quiet buckets before,
    and the K-bucket history lies inside the sequence."""
    n = len(attack)
    i = np.arange(n)
    quiet = np.ones(n, dtype=bool)
    for k in range(1, K + 1):
        j = np.maximum(i - k, 0)
        quiet &= ~attack[j]
    return np.flatnonzero(attack & quiet & (i - K >= start))


def lead_times(p, eps, K, thr):
    """Minutes-of-lead per onset (0 = missed). An alarm at bucket b in [s-K, s-1] gives s-b buckets."""
    out = []
    for s in eps:
        w = p[s - K:s]
        hit = np.flatnonzero(w >= thr)
        out.append(K - hit[0] if len(hit) else 0)
    return np.array(out)


def evaluate(p, tab, pop, eps=None, budgets=BUDGETS):
    """Onset-forecast metrics on the population `pop` (rows whose current bucket is benign)."""
    y = tab.fut[pop].astype(int)
    pp = p[pop]
    res = {"n_rows": int(pop.sum()), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan")}
    if y.sum() == 0 or y.sum() == len(y):
        return res
    res["pr_auc"] = float(average_precision_score(y, pp))
    res["roc_auc"] = float(roc_auc_score(y, pp))
    res["brier"] = float(np.mean((pp - y) ** 2))
    res["ece"] = ece(y, pp)
    neg = pp[y == 0]
    if eps is None:
        eps = onset_episodes(tab.attack, tab.start, tab.end, tab.K)
    res["n_onsets"] = int(len(eps))
    for b in budgets:
        thr = threshold_for_budget(neg, b, tab.bucket_s)
        tp, fp = int((pp[y == 1] >= thr).sum()), int((neg >= thr).sum())
        res[f"recall@{b}/h"] = tp / int(y.sum())
        res[f"precision@{b}/h"] = tp / (tp + fp) if tp + fp else 0.0
        res[f"f1@{b}/h"] = 2 * tp / (2 * tp + fp + (int(y.sum()) - tp)) if tp else 0.0
        res[f"fpr@{b}/h"] = fp / len(neg)
        res[f"thr@{b}/h"] = thr
        if len(eps):
            lt = lead_times(p, eps, tab.K, thr)
            res[f"onset_detected@{b}/h"] = float((lt > 0).mean())
            res[f"median_lead_min@{b}/h"] = float(np.median(lt[lt > 0]) * tab.bucket_s / 60) if (lt > 0).any() else 0.0
    return res


def block_bootstrap(p, tab, pop, reps=200, seed=0, budget=1.0):
    """90% CI for PR-AUC and recall at the budget, resampling whole (host, day) sequences."""
    rng = np.random.default_rng(seed)
    seq = tab.start
    uniq = np.unique(seq)
    rows_of = {s: np.flatnonzero((seq == s) & pop) for s in uniq}
    ap, rc = [], []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_of[s] for s in pick])
        y = tab.fut[rows].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pp = p[rows]
        ap.append(average_precision_score(y, pp))
        thr = threshold_for_budget(pp[y == 0], budget, tab.bucket_s)
        rc.append((pp[y == 1] >= thr).mean())
    q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 95))] if a else [float("nan")] * 2
    return {"pr_auc_ci90": q(ap), f"recall@{budget}/h_ci90": q(rc)}
