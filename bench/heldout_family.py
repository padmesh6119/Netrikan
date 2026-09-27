"""
Held-out attack family: can the model forecast onsets of a class it never trained on?

Trains with --holdout-class (train_v2 removes that class's windows AND any window
whose onset reaches it, so there is no leak), then evaluates the onset head ONLY on
validation transitions INTO the held-out class. If the model warns of an attack
family it never saw, that is genuine generalization, not memorized signatures.

Reports PR-AUC, ROC-AUC and recall-at-FPR with the absolute positive count next to
every number — a held-out class can be rare, and a metric over a handful of
positives is indicative, not precise.

    python3 bench/heldout_family.py --data "$DATA" --holdout 3 \
        --out models/heldout_infiltration_w30.json
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from model import WorldModel, ONSET_HORIZONS   # noqa: E402
from train_v2 import blocked_split, STAGES     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onset_pr import recall_at_fpr             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--holdout', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-train', action='store_true',
                    help='reuse an existing checkpoint instead of retraining')
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    hc = args.holdout
    tag = args.tag or f"heldout_{STAGES[hc].lower()}_w30"
    ckpt = os.path.join(ROOT, 'models', f'{tag}.pt')

    if not args.skip_train:
        cmd = ['.venv/bin/python', '-W', 'ignore', 'src/train_v2.py',
               '--data', args.data, '--tag', tag, '--attention',
               '--select-on', 'combined', '--holdout-class', str(hc),
               '--epochs', str(args.epochs), '--patience', '5']
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1',
                   NETRIKAN_DEVICE=os.environ.get('NETRIKAN_DEVICE', 'mps'))
        print(f"training with class {hc} ({STAGES[hc]}) held out…", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            raise SystemExit(f"training failed (exit {r.returncode})")

    # evaluate: onset of the held-out class on the val split
    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))[:n_all]
    fid_path = os.path.join(args.data, 'file_id.npy')
    file_id = np.load(fid_path)[:n_all] if os.path.exists(fid_path) else np.zeros(n_all, np.int64)

    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    _, va = blocked_split(y, purge=W - 1)
    va = np.sort(va[va < n_all])
    pred = np.zeros((n_all, len(ONSET_HORIZONS)), np.float32)
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            sl = va[i:i + args.batch]
            pred[sl] = torch.sigmoid(model(torch.from_numpy(np.asarray(X[sl], np.float32)))[3]).numpy()

    out = {"holdout_class": hc, "holdout_name": STAGES[hc], "checkpoint": ckpt,
           "data": args.data, "note": "onset positives restricted to transitions "
           "INTO the held-out class; model never trained on it", "horizons": {}}
    print(f"\nheld-out class: {STAGES[hc]}")
    for ki, k in enumerate(ONSET_HORIZONS):
        # positive = window t in val where y[t+k]==hc and y[t]!=hc, same file.
        # negative = benign windows not transitioning into any attack within k.
        t = va[va + k < n_all]
        same = file_id[t] == file_id[t + k]
        into_hc = (y[t + k] == hc) & (y[t] != hc) & same
        benign_stable = (y[t] == 0) & (y[t + k] == 0) & same
        mask = into_hc | benign_stable
        lab = into_hc[mask].astype(int)
        sc = pred[t[mask], ki]
        pos = int(lab.sum())
        if pos == 0 or lab.min() == lab.max():
            out["horizons"][f"k{k}"] = {"positive_count": pos,
                                        "note": "too few positives to score"}
            print(f"  k={k}: {pos} positives — skipped")
            continue
        rec = {t_: recall_at_fpr(lab, sc, t_) for t_ in (0.001, 0.01, 0.05)}
        row = {"positive_count": pos, "n": int(len(lab)),
               "positive_rate": round(float(lab.mean()), 5),
               "roc_auc": round(float(roc_auc_score(lab, sc)), 4),
               "pr_auc": round(float(average_precision_score(lab, sc)), 4),
               "no_skill_pr": round(float(lab.mean()), 5),
               "recall_at_fpr": {t_: {"recall": v[0], "achieved_fpr": v[2]}
                                 for t_, v in rec.items()}}
        out["horizons"][f"k{k}"] = row
        print(f"  k={k}: pos={pos}  PR-AUC {row['pr_auc']} (no-skill {row['no_skill_pr']})  "
              f"ROC-AUC {row['roc_auc']}")

    maxpos = max((h.get('positive_count', 0) for h in out['horizons'].values()), default=0)
    out["finding"] = (f"Held-out {STAGES[hc]} forecast with up to {maxpos} val "
                      f"positives. " + ("Few positives — indicative, not precise."
                                        if maxpos < 200 else
                                        "Positive volume adequate for a real estimate."))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
