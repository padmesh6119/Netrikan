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

Stage level (TRAINER_BACKLOG R1). State-space MSE says nothing about whether the
rollout forecasts the *stage*, which is what the app shows. At each step k the
argmax of the rolled-out stage distribution is scored against the true stage of
the flow it predicts (y[j+k-1]; step 1 is the ordinary one-step prediction),
against three baselines:

  persistence   the last observed label y[j-1], held
  classifier    the model's own step-1 prediction, held (no rollout at all)
  markov        the step-1 distribution projected k-1 steps through
                forecast.TRANSITION, the app's fallback when there is no state head

Macro-F1 is reported on all windows and on transition windows only
(y[j+k-1] != y[j-1]). Persistence is 0 on transitions by construction. The
comparison that decides the pitch is rollout vs classifier: if holding the
step-1 prediction does as well, the classifier forecasts and the rollout only
illustrates. Transition windows are rare, so a second sample is drawn from
windows whose label changes somewhere in the horizon; it is used only for the
transition rows, where it is an unbiased draw. Windows whose horizon crosses a
capture-file boundary are dropped.

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

from sklearn.metrics import f1_score     # noqa: E402

from model import WorldModel              # noqa: E402
from train_v2 import blocked_split, STAGES  # noqa: E402
from forecast import TRANSITION           # noqa: E402

N_MODEL = len(STAGES)
# forecast.TRANSITION is over the 7 kill-chain stages; the model's 5 classes map
# onto chain ids 0-4 one to one (attck_map.MODEL_TO_CHAIN), so project in chain
# space and read back the first 5.


def markov_project(p5, steps):
    """(n, 5) -> (n, steps, 5): step 1 is p5 itself, step k is k-1 hops of
    forecast.TRANSITION, argmax restricted to the model's 5 classes."""
    n = len(p5)
    p = np.zeros((n, TRANSITION.shape[0]))
    p[:, :N_MODEL] = p5
    out = np.zeros((n, steps, N_MODEL))
    for k in range(steps):
        if k:
            p = p @ TRANSITION
        out[:, k] = p[:, :N_MODEL]
    return out


def macro(t, p):
    return round(float(f1_score(t, p, labels=list(range(N_MODEL)), average='macro',
                                zero_division=0)), 4)


def stage_eval(idx, n_random, stages, y, steps):
    """Stage-level scoring of the rollout at every step, vs persistence, the held
    step-1 classifier prediction, and the Markov projection."""
    roll = stages.argmax(2)                         # (n, steps)
    clf = roll[:, 0]
    mk = markov_project(stages[:, 0, :], steps).argmax(2)
    persist = y[idx - 1]
    rows = []
    print("\nstage-level macro-F1 (all windows | transition windows only)")
    print(f"  {'step':>4}  {'n_tr':>6}  {'rollout':>15}  {'classifier':>15}  "
          f"{'markov':>15}  {'persist':>15}")
    for k in range(steps):
        true = y[idx + k]
        rnd = np.zeros(len(idx), bool)
        rnd[:n_random] = True
        tr = true != persist                        # transitions from both samples
        row = {'step': k + 1, 'n_all': int(rnd.sum()), 'n_transitions': int(tr.sum()),
               'n_transitions_in_random_sample': int((tr & rnd).sum())}
        for name, pred in (('rollout', roll[:, k]), ('classifier_held', clf),
                           ('markov', mk[:, k]), ('persistence', persist)):
            row[f'{name}_all_f1'] = macro(true[rnd], pred[rnd])
            row[f'{name}_transition_f1'] = macro(true[tr], pred[tr]) if tr.any() else None
        rows.append(row)
        print(f"  {k+1:>4}  {tr.sum():>6}  "
              f"{row['rollout_all_f1']:.4f} | {row['rollout_transition_f1']:.4f}  "
              f"{row['classifier_held_all_f1']:.4f} | {row['classifier_held_transition_f1']:.4f}  "
              f"{row['markov_all_f1']:.4f} | {row['markov_transition_f1']:.4f}  "
              f"{row['persistence_all_f1']:.4f} | {row['persistence_transition_f1']:.4f}")

    last = rows[-1]
    beats_clf = [r['step'] for r in rows
                 if r['rollout_transition_f1'] > r['classifier_held_transition_f1']]
    beats_mk = [r['step'] for r in rows
                if r['rollout_transition_f1'] > r['markov_transition_f1']]
    if last['rollout_transition_f1'] > max(last['classifier_held_transition_f1'],
                                           last['markov_transition_f1']):
        verdict = ("world_model_forecasts: at the final step the rolled-out stage "
                   "beats both the held classifier prediction and the Markov "
                   "projection on transition windows")
    else:
        verdict = ("classifier_forecasts_rollout_illustrates: at the final step the "
                   "rollout does not beat the held step-1 prediction and/or the "
                   "Markov projection on transition windows")
    print(f"  -> {verdict}")
    return {
        'target': 'step k predicts y[j+k-1] (step 1 = ordinary one-step prediction)',
        'transition_definition': 'y[j+k-1] != y[j-1] (last observed label)',
        'baselines': {
            'persistence': 'y[j-1] held; 0 on transitions by construction',
            'classifier_held': "model's step-1 argmax held for all k (no rollout)",
            'markov': 'step-1 softmax projected k-1 hops through forecast.TRANSITION',
        },
        'per_step': rows,
        'steps_rollout_beats_classifier_on_transitions': beats_clf,
        'steps_rollout_beats_markov_on_transitions': beats_mk,
        'verdict': verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--limit', type=int, default=20000,
                    help='cap validation windows evaluated, for speed')
    ap.add_argument('--transition-limit', type=int, default=20000,
                    help='extra windows drawn from those whose label changes '
                         'within the horizon, for the stage-level transition rows')
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'rollout_eval.json'))
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    n_all, W, F = X.shape
    y = np.load(os.path.join(args.data, 'y.npy'))
    n = min(n_all, len(y))

    _, va = blocked_split(y[:n], purge=W - 1)
    va = va[(va > 0) & (va < n - args.steps - 1)]   # need y[j-1] and every step ahead
    fid_path = os.path.join(args.data, 'file_id.npy')
    if os.path.exists(fid_path):
        fid = np.load(fid_path)
        va = va[fid[va - 1] == fid[va + args.steps]]   # horizon stays in one capture day
    # does the label change anywhere in the horizon?  y[j-1] vs y[j..j+steps-1]
    changes = np.zeros(len(va), bool)
    for k in range(args.steps):
        changes |= y[va + k] != y[va - 1]
    rng = np.random.default_rng(0)
    pool = va
    if len(va) > args.limit:
        va = np.sort(rng.choice(va, args.limit, replace=False))
    tpool = np.setdiff1d(pool[changes], va)
    if len(tpool) > args.transition_limit:
        tpool = np.sort(rng.choice(tpool, args.transition_limit, replace=False))
    print(f"evaluating {len(va):,} windows (+{len(tpool):,} transition-enriched), "
          f"window={W} features={F}", flush=True)

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
    allidx = np.concatenate([va, tpool])
    states_all = np.zeros((len(allidx), args.steps, F), np.float32)
    stages_all = np.zeros((len(allidx), args.steps, N_MODEL), np.float32)
    with torch.no_grad():
        for i in range(0, len(allidx), args.batch):
            sl = allidx[i:i + args.batch]
            xb = torch.from_numpy(np.asarray(X[sl], dtype=np.float32))
            st, sg = model.rollout(xb, steps=args.steps)
            states_all[i:i + len(sl)] = st.numpy()
            stages_all[i:i + len(sl)] = sg.numpy()
    states = states_all[:len(va)]      # (n, steps, F)
    stages = stages_all[:len(va)]      # (n, steps, 5)

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

    stage_level = stage_eval(allidx, len(va), stages_all, y, args.steps)

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
        'stage_level': stage_level,
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
