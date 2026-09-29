"""CLI: score a flow CSV and write per-host-minute risk.

  python scripts/infer.py flows.csv --out risk.csv [--threshold 0.2]
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from netrikan import infer  # noqa: E402
from netrikan.dapt import STAGE_NAMES, normalise, read_flows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default="results/demo/risk.csv")
ap.add_argument("--model", default="models/demo/all.joblib")
ap.add_argument("--threshold", type=float, default=None)
a = ap.parse_args()

mx = json.load(open("results/demo/metrics.json"))
which = mx["primary_model"]
thr = a.threshold if a.threshold is not None else mx["alert_threshold_1_per_host_hour"]
b = joblib.load(a.model)
flows = normalise(read_flows(a.csv))
tab, pr = infer.analyse(flows, b)
p = pr[which]
out = pd.DataFrame({"host": tab.host, "time": infer.times(tab), "risk": p, "alert": p >= thr,
                    "predicted_stage": [STAGE_NAMES[s] for s in pr["stage"].argmax(1)]})
out.to_csv(a.out, index=False)
print(f"{len(flows)} flows, {out.host.nunique()} hosts, {int(out.alert.sum())} alert minutes (threshold {thr:.3f}, model {which}) -> {a.out}")
top = out.sort_values("risk", ascending=False).head(5)
for _, r in top.iterrows():
    i = int(np.flatnonzero((tab.host == r.host) & (infer.times(tab) == r.time))[0])
    drivers = "; ".join(f"{n} ({v:+.2f})" for n, v in infer.top_drivers(b, pr, i, which, 3))
    print(f"  {r.time:%H:%M} {r.host} risk={r.risk:.2f} stage={r.predicted_stage} | {drivers}")
