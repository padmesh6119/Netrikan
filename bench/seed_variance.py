"""
Five-seed training variance.

All headline numbers are single-seed (seed 42). This trains seeds 42-46 and reports
mean ± std for onset AUC k5, onset PR-AUC k5, stage macro-F1 and transition-window
macro-F1. A wide spread is itself the finding — it means every headline number needs
an error bar.

    python3 bench/seed_variance.py --data "$DATA" --seeds 42 43 44 45 46 \
        --out models/seed_variance_cic_full_w30.json
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def _run(cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'seed_variance_cic_full_w30.json'))
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1',
               NETRIKAN_DEVICE=os.environ.get('NETRIKAN_DEVICE', 'mps'))
    rows = []
    for s in args.seeds:
        tag = f"seed{s}_w30"
        mpath = os.path.join(ROOT, 'models', f'{tag}_metrics.json')
        prpath = os.path.join(ROOT, 'models', f'onset_pr_{tag}.json')
        tepath = os.path.join(ROOT, 'models', f'transition_eval_{tag}.json')
        ck = os.path.join(ROOT, 'models', f'{tag}.pt')
        if not args.skip_train:
            # train_v2 reads seed from config; pass via a temp override env is not
            # supported, so write seed through the CLI-less path: use --tag and the
            # seed arg the trainer already exposes via config. We set it by env-less
            # means: the trainer's blocked_split seed comes from config's `seed`.
            _run(['.venv/bin/python', '-W', 'ignore', 'src/train_v2.py',
                  '--data', args.data, '--tag', tag, '--attention',
                  '--select-on', 'combined', '--epochs', str(args.epochs),
                  '--patience', '4', '--seed', str(s)], env)
            _run(['.venv/bin/python', '-W', 'ignore', 'bench/onset_pr.py',
                  '--data', args.data, '--model', ck, '--out', prpath], env)
            _run(['.venv/bin/python', '-W', 'ignore', 'src/transition_eval.py',
                  '--data', args.data, '--model', ck, '--out', tepath], env)
        m = json.load(open(mpath)); pr = json.load(open(prpath)); te = json.load(open(tepath))
        rows.append({
            "seed": s,
            "onset_auc_k5": m['onset_auc'].get('k5', {}).get('auc'),
            "onset_pr_auc_k5": pr['horizons'].get('k5', {}).get('pr_auc'),
            "stage_macro_f1": round(m['final_macro_f1'], 4),
            "transition_macro_f1": te['subsets']['transition_windows']['model_macro_f1'],
        })
        print(f"  seed {s}: {rows[-1]}", flush=True)

    def ms(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
                "values": v} if v else None

    out = {"data": args.data, "seeds": args.seeds, "epochs": args.epochs,
           "onset_auc_k5": ms('onset_auc_k5'),
           "onset_pr_auc_k5": ms('onset_pr_auc_k5'),
           "stage_macro_f1": ms('stage_macro_f1'),
           "transition_macro_f1": ms('transition_macro_f1'),
           "per_seed": rows}
    spread = max((out[k]['std'] for k in ('onset_auc_k5', 'stage_macro_f1')
                  if out[k]), default=0)
    out["finding"] = ("Spread is small; single-seed numbers are representative."
                      if spread < 0.01 else
                      f"Spread up to ±{spread:.3f}; headline numbers need error bars.")
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
