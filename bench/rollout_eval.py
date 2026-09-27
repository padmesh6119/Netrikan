"""
Does the world model actually learn state-transition dynamics?

The problem statement asks for "a trained world model that demonstrably learns
traffic state transition dynamics -- not a static input-output classifier." The
state_head is what makes that claim, and nothing in the repo tested it. A
state_head can reach a low MSE while learning nothing useful, because the
cheapest way to predict the next flow's feature vector is to copy the current
one: consecutive flows are highly correlated.

So the only measurement that means anything is against that copy. Baselines:

  persistence   next state = the window's last observed row
  window mean   next state = the mean of the window
  model         next state = state_head(window)

If the model does not beat persistence, the state head has learned to echo its
input and the rollout is theatre. Skill is reported as
1 - MSE_model / MSE_persistence, so positive means real dynamics learning.

K-step drift is also reported. Free-running rollout feeds predictions back in, so
errors compound; the curve shows how far ahead the simulation stays usable.

Usage:
    python3 bench/rollout_eval.py --data /tmp/netrikan-ctu13-w30 \\
        --model models/ctu13_w30.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import WorldModel              # noqa: E402
from train_v2 import blocked_split        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--limit', type=int, default=20000,
                    help='cap validation windows evaluated, for speed')
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'rollout_eval.json'))
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))
    n = min(n_all, len(y))

    _, va = blocked_split(y[:n], purge=W - 1)
    va = va[va < n - args.steps - 1]      # need ground truth for every step ahead
    if len(va) > args.limit:
        va = np.sort(np.random.default_rng(0).choice(va, args.limit, replace=False))
    print(f"evaluating {len(va):,} windows, window={W} features={F}", flush=True)

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    if 'state_head.weight' not in sd:
        raise SystemExit(f"{args.model} has no trained state_head -- there is no "
                         f"world model in this checkpoint to evaluate.")
    model = WorldModel(input_size=F, attention='attn.weight' in sd)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ---- one-step ----
    # ground truth for window j's next state is the last row of window j+1
    cur = np.asarray(X[va], dtype=np.float32)                 # (n, W, F)
    truth = np.asarray(X[va + 1], dtype=np.float32)[:, -1, :]  # (n, F)

    with torch.no_grad():
        pred = model(torch.from_numpy(cur))[2].numpy()

    persist = cur[:, -1, :]
    wmean = cur.mean(axis=1)

    def mse(a):
        return float(np.mean((a - truth) ** 2))

    m_model, m_persist, m_mean = mse(pred), mse(persist), mse(wmean)
    skill_p = 1.0 - m_model / m_persist if m_persist > 0 else None
    skill_m = 1.0 - m_model / m_mean if m_mean > 0 else None

    print("\none-step next-state prediction (lower MSE is better)")
    print(f"  model        {m_model:.5f}")
    print(f"  persistence  {m_persist:.5f}   skill vs persistence {skill_p:+.4f}")
    print(f"  window mean  {m_mean:.5f}   skill vs mean        {skill_m:+.4f}")
    # Both baselines have to be cleared. Beating persistence only shows the head
    # is not echoing its input; losing to the window mean shows it has settled on
    # a smoothed central estimate of the window rather than a trajectory, which is
    # not dynamics in any useful sense.
    if not (skill_p and skill_p > 0):
        verdict = ("FAIL: does not beat persistence -- the state head is echoing "
                   "its input and the rollout is not a learned simulation")
    elif not (skill_m and skill_m > 0):
        verdict = ("PARTIAL: beats persistence but LOSES to the window mean. The "
                   "state head predicts a smoothed central estimate of the "
                   "window, not its trajectory. Do not describe this as learned "
                   "dynamics without stating this.")
    else:
        verdict = ("PASS: beats both persistence and the window mean -- the state "
                   "head learned genuine dynamics")
    print(f"  -> {verdict}")

    # ---- K-step free-running drift ----
    print(f"\nK-step free-running drift ({args.steps} steps)")
    with torch.no_grad():
        states, stages = model.rollout(torch.from_numpy(cur), steps=args.steps)
    states = states.numpy()      # (n, steps, F)
    stages = stages.numpy()      # (n, steps, 5)

    drift = []
    for k in range(args.steps):
        t_k = np.asarray(X[va + 1 + k], dtype=np.float32)[:, -1, :]
        e_model = float(np.mean((states[:, k, :] - t_k) ** 2))
        e_persist = float(np.mean((persist - t_k) ** 2))
        sk = 1.0 - e_model / e_persist if e_persist > 0 else None
        drift.append({'step': k + 1, 'model_mse': round(e_model, 5),
                      'persistence_mse': round(e_persist, 5),
                      'skill': round(sk, 4) if sk is not None else None})
        print(f"  step {k+1:>3}  model {e_model:.5f}  persistence {e_persist:.5f}"
              f"  skill {sk:+.4f}")

    # how long the rollout stays better than doing nothing
    usable = 0
    for d in drift:
        if d['skill'] is not None and d['skill'] > 0:
            usable = d['step']
        else:
            break
    print(f"\nrollout beats persistence for {usable} of {args.steps} steps")

    # stage-distribution stability across the rollout
    flips = float(np.mean(stages.argmax(2)[:, 1:] != stages.argmax(2)[:, :-1]))
    print(f"predicted-stage volatility across rollout steps: {flips:.4f}")

    payload = {
        'model': args.model, 'data': args.data,
        'n_windows': int(len(va)), 'window': int(W), 'features': int(F),
        'one_step': {
            'model_mse': round(m_model, 5),
            'persistence_mse': round(m_persist, 5),
            'window_mean_mse': round(m_mean, 5),
            'skill_vs_persistence': round(skill_p, 4) if skill_p else None,
            'skill_vs_window_mean': round(skill_m, 4) if skill_m else None,
            'beats_persistence': bool(skill_p and skill_p > 0),
            'beats_window_mean': bool(skill_m and skill_m > 0),
            'verdict': verdict,
        },
        'k_step_drift': drift,
        'steps_beating_persistence': usable,
        'rollout_stage_volatility': round(flips, 4),
        'note': ('skill = 1 - MSE_model/MSE_persistence. Positive means the state '
                 'head predicts the next flow better than copying the last '
                 'observed flow, which is the only baseline that matters here: '
                 'consecutive flows are highly correlated, so a low raw MSE alone '
                 'does not show that any dynamics were learned.'),
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
