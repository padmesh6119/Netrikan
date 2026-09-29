"""
Lead time in seconds: how much warning the onset head gives before an attack
stage actually begins.

onset AUC answers "can it tell a transition is coming"; it does not answer "how
early". This does, in the unit operators care about, with the false-alarm cost
stated at every operating point.

Definitions (fixed):

  onset event   an index t, inside one capture file, where the stage label goes
                Benign -> any attack class. Non-benign -> non-benign transitions
                (attack escalations) are a SEPARATE event class, reported apart,
                never merged.
  warning       for an event at t, the start of the CONTIGUOUS alarm episode
                active going into t: walk back from t-1 while the onset probability
                stays >= theta within the same file; the run's start is s, and it
                must be at least `min_persist` long to count. lead =
                (t - s) * seconds_per_window. Measuring from any earlier crossing
                since the file start (as a naive "earliest warning" would) reaches
                back across a whole day of benign traffic and yields meaningless
                multi-day leads; the active-episode definition bounds the warning
                to the alarm that actually precedes this event.
  seconds_per_window   the median inter-flow time of the window x WINDOW, taken
                from y.npy's companion timing when present, else a stated default.
                Recorded in the output.
  coverage      fraction of events that got any qualifying warning before t. An
                un-warned event counts against coverage, not into the lead
                distribution. A high median lead over few events is not a result,
                so both are always reported.

FPR is measured on benign windows that precede no event within the horizon.

Baselines in the same table: persistence (0 coverage by construction -- it never
predicts a change) and the flattened-window logistic regression from baseline.py.

Usage:
    python3 bench/lead_time.py --data "$DATA" --model "$CKPT" \
        --horizons 1 5 15 30 --out models/lead_time_cic_full_w30.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import WorldModel, ONSET_HORIZONS   # noqa: E402
from train_v2 import blocked_split             # noqa: E402

# CIC-IDS-2018 flow records carry no reliable per-window timing in the built
# tensor, so we state the assumption explicitly rather than fake precision.
DEFAULT_SECONDS_PER_FLOW = 1.0


def _events(y, file_id):
    """Benign->attack and attack->attack transition indices, within one file."""
    onset_be, onset_esc = [], []
    for i in range(1, len(y)):
        if file_id[i] != file_id[i - 1]:
            continue
        if y[i] != y[i - 1]:
            if y[i - 1] == 0 and y[i] != 0:
                onset_be.append(i)
            elif y[i - 1] != 0 and y[i] != 0:
                onset_esc.append(i)
    return np.array(onset_be, np.int64), np.array(onset_esc, np.int64)


def _run_length(prob, file_id, theta):
    """run_len[i] = number of consecutive windows up to and including i (same file)
    with prob>=theta. So an alarm active at i started run_len[i] windows ago.
    One O(n) pass; every event then reads its active-episode length in O(1)."""
    over = prob >= theta
    n = len(prob)
    run = np.zeros(n, np.int32)
    for i in range(n):
        if over[i]:
            run[i] = run[i - 1] + 1 if (i > 0 and file_id[i] == file_id[i - 1]) else 1
    return run


def _benign_scores(prob, y, file_id, events_all, horizon):
    """Scores of benign windows that precede no event within `horizon` windows.
    Computed once per horizon; thresholds then come from its quantiles in O(1)."""
    is_pre = np.zeros(len(y), bool)
    for t in events_all:
        is_pre[max(0, t - horizon):t] = True
    benign = (y == 0) & ~is_pre
    return prob[benign]


def _dist(leads, spw):
    if not leads:
        return {"median_s": None, "p25_s": None, "p75_s": None, "mean_s": None}
    a = np.array(leads, float) * spw
    return {"median_s": round(float(np.median(a)), 2),
            "p25_s": round(float(np.percentile(a, 25)), 2),
            "p75_s": round(float(np.percentile(a, 75)), 2),
            "mean_s": round(float(a.mean()), 2)}


def _operating_points(prob, y, file_id, ev_be, ev_esc, horizon, spw,
                      target_fprs, min_persist):
    """For each target FPR, set theta = the (1-FPR) quantile of benign scores in
    O(1), then measure coverage + lead distribution for benign->attack onsets."""
    ev_all = np.concatenate([ev_be, ev_esc]) if len(ev_esc) else ev_be
    bscores = _benign_scores(prob, y, file_id, ev_all, horizon)
    n_benign = len(bscores)
    rows = []
    for target in target_fprs:
        if n_benign == 0:
            chosen, achieved = 1.0, None
        else:
            chosen = float(np.quantile(bscores, 1.0 - target))
            achieved = float((bscores >= chosen).mean())
        run = _run_length(prob, file_id, chosen)
        leads, warned = [], 0
        for t in ev_be:
            # alarm episode active at t-1: its length is the lead, if it is at
            # least min_persist long
            active = run[t - 1] if t - 1 >= 0 else 0
            if active >= min_persist:
                warned += 1
                leads.append(int(active))
        rows.append({
            "target_fpr": target, "threshold": round(chosen, 5),
            "achieved_fpr": round(achieved, 5) if achieved is not None else None,
            "coverage": round(warned / max(len(ev_be), 1), 4),
            "events_warned": warned, "events_total": int(len(ev_be)),
            "benign_windows": int(n_benign),
            **_dist(leads, spw)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 15, 30])
    ap.add_argument('--fprs', type=float, nargs='+', default=[0.001, 0.01, 0.05])
    ap.add_argument('--min-persist', type=int, default=2)
    ap.add_argument('--seconds-per-flow', type=float, default=DEFAULT_SECONDS_PER_FLOW)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'lead_time_cic_full_w30.json'))
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))[:n_all]
    file_path = os.path.join(args.data, 'file_id.npy')
    if os.path.exists(file_path):
        file_id = np.load(file_path)[:n_all]
    else:
        file_id = np.zeros(n_all, np.int64)
        print("WARNING: no file_id.npy; treating the corpus as one capture. "
              "Events may span day boundaries.", file=sys.stderr)

    spw = args.seconds_per_flow * W

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    if 'onset_head.weight' not in sd:
        raise SystemExit(f"{args.model} has no trained onset head")
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # onset probs on the val split only, so this is held out
    _, va = blocked_split(y, purge=W - 1)
    va = np.sort(va[va < n_all])
    onset_p = np.zeros((n_all, len(ONSET_HORIZONS)), np.float32)
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            sl = va[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(X[sl], np.float32))
            onset_p[sl] = torch.sigmoid(model(xb)[3]).numpy()

    # restrict everything to the val split: build a val-local view
    vy = y[va]
    vfile = file_id[va]
    ev_be, ev_esc = _events(vy, vfile)
    print(f"val windows {len(va):,} | benign->attack onsets {len(ev_be)} | "
          f"attack->attack escalations {len(ev_esc)} | seconds_per_window {spw:.1f}",
          flush=True)

    out = {"model": args.model, "data": args.data,
           "seconds_per_window": spw, "seconds_per_flow": args.seconds_per_flow,
           "min_persist": args.min_persist,
           "n_val_windows": int(len(va)),
           "n_benign_to_attack_onsets": int(len(ev_be)),
           "n_attack_escalations": int(len(ev_esc)),
           "horizons": {}}

    for ki, k in enumerate(ONSET_HORIZONS):
        if k not in args.horizons:
            continue
        vprob = onset_p[va, ki]
        rows = _operating_points(vprob, vy, vfile, ev_be, ev_esc, k, spw,
                                 args.fprs, args.min_persist)
        out["horizons"][f"k{k}"] = rows
        print(f"\nhorizon k={k}:")
        for r in rows:
            print(f"  FPR~{r['target_fpr']:<5} thr {r['threshold']:.4f} "
                  f"(achieved {r['achieved_fpr']}) coverage {r['coverage']:.2f} "
                  f"({r['events_warned']}/{r['events_total']}) "
                  f"median lead {r['median_s']}s")

    out["baselines"] = {
        "persistence": {"coverage": 0.0,
                        "note": "0 by construction: persistence predicts no change, "
                                "so it never warns before an onset"},
        "logreg_flat": {"note": "flattened-window logreg has no temporal onset "
                                "probability; see models/baseline_*.json for its "
                                "stage macro-F1. It produces no lead-time curve."},
    }
    cov = max((r['coverage'] for rows in out['horizons'].values() for r in rows),
              default=0)
    out["finding"] = (
        f"Benign->attack onsets in val: {len(ev_be)}. Best coverage across "
        f"operating points: {cov:.2f}. " +
        ("Coverage is low; treat lead medians as indicative." if cov < 0.5
         else "Coverage is adequate for a lead-time claim."))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
