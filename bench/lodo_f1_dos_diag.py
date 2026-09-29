"""
LODO fold 1 diagnostic: is the below-chance onset AUC a DoS class-prior effect?

Fold 1 holds out capture day 1 (Benign 447k + DoS 602k). Its onset AUC k1 is
0.425, the only fold below 0.5. The standing hypothesis (STATUS.md, "Task 7
follow-up") is a DoS volume inversion: DoS is 52,498 of 5.1M training windows
(1.0%) but 57% of the held-out day, so the model learns to abandon DoS and its
onset head never learns what a DoS boundary looks like.

Two measurements, both on the identical held-out-day windows:

  1. DoS-only onset curve. Onset events on day 1 are split by type:
       into_dos   label at t is not DoS, label at t+k is DoS
       out_of_dos label at t is DoS,     label at t+k is not DoS
     and each type's AUC is computed against the day's non-onset windows. If the
     below-chance number is a DoS effect, it lives in these rows.
  2. Class-weighted rerun. The same fold retrained with --balance-power 1.0
     (full inverse-frequency sampling: DoS weight 80x Benign instead of 9x) and
     otherwise the sweep's recipe. If the prior gap drives the failure, this
     should lift onset AUC and DoS recall.

Verdict rule, fixed before the numbers were read:
  confirmed  class-weighted onset AUC k1 >= 0.5 AND improves on the sweep by >= 0.05
  killed     class-weighted onset AUC k1 does not improve by >= 0.05
  partial    anything else

    python3 bench/lodo_f1_dos_diag.py --data /tmp/netrikan-cic-full-w30
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import WorldModel, ONSET_HORIZONS   # noqa: E402
from train_v2 import STAGES                     # noqa: E402

DOS = STAGES.index('DoS')
FOLD = 1


def auc_or_none(t, s):
    return round(float(roc_auc_score(t, s)), 4) if 0 < t.sum() < len(t) else None


def score(path, X, idx, batch):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    m = WorldModel(input_size=X.shape[2], attention='attn.weight' in sd)
    m.load_state_dict(sd, strict=False)
    m.eval()
    stage = np.zeros(len(idx), np.int64)
    onset = np.zeros((len(idx), len(ONSET_HORIZONS)), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sl = idx[i:i + batch]
            lg, _, _, o = m(torch.from_numpy(np.asarray(X[sl], dtype=np.float32)))
            stage[i:i + len(sl)] = lg.argmax(1).numpy()
            onset[i:i + len(sl)] = torch.sigmoid(o).numpy()
    return stage, onset


def analyse(stage, onset, y, idx, onset_lab):
    yt = y[idx]
    out = {'stage': {
        'macro_f1_supported': round(float(f1_score(yt, stage, labels=[0, DOS],
                                                   average='macro', zero_division=0)), 4),
        'dos_f1': round(float(f1_score(yt == DOS, stage == DOS, zero_division=0)), 4),
        'dos_recall': round(float(recall_score(yt == DOS, stage == DOS, zero_division=0)), 4),
        'benign_f1': round(float(f1_score(yt == 0, stage == 0, zero_division=0)), 4),
    }, 'onset': {}}
    for ki, k in enumerate(ONSET_HORIZONS):
        lab = onset_lab[ki][idx].astype(bool)
        fut = y[idx + k]
        into = lab & (yt != DOS) & (fut == DOS)
        outof = lab & (yt == DOS) & (fut != DOS)
        other = lab & ~into & ~outof
        neg = ~lab
        s = onset[:, ki]

        def sub(pos):
            m = pos | neg
            return auc_or_none(pos[m].astype(int), s[m])

        in_dos, in_ben = yt == DOS, yt == 0
        out['onset'][f'k{k}'] = {
            'all': auc_or_none(lab.astype(int), s),
            'n_onsets': int(lab.sum()),
            'into_dos': sub(into), 'n_into_dos': int(into.sum()),
            'out_of_dos': sub(outof), 'n_out_of_dos': int(outof.sum()),
            'other': sub(other), 'n_other': int(other.sum()),
            # discrimination within a current stage: does the head rank
            # upcoming changes above steady windows when the stage is held fixed?
            'within_current_dos': auc_or_none(lab[in_dos].astype(int), s[in_dos]),
            'within_current_benign': auc_or_none(lab[in_ben].astype(int), s[in_ben]),
            'mean_score_onset': round(float(s[lab].mean()), 4),
            'mean_score_steady': round(float(s[neg].mean()), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--models', nargs='+', default=[
        'sweep=models/lodo_f1_w30.pt',
        'long_low_lr=models/lodo_f1_diag.pt',
        'class_weighted=models/lodo_f1_cw.pt'])
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'lodo_f1_dos_diag.json'))
    args = ap.parse_args()

    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    y = np.load(os.path.join(args.data, 'y.npy'))
    fid = np.load(os.path.join(args.data, 'file_id.npy'))
    onset_lab = [np.load(os.path.join(args.data, f'onset_k{k}.npy')) for k in ONSET_HORIZONS]
    n = min(len(X), len(y), len(fid))

    # identical to train_v2 --lodo-val-file 1: the day's windows with a valid
    # onset label at every horizon
    idx = np.flatnonzero(fid[:n] == FOLD)
    valid = np.ones(len(idx), bool)
    for o in onset_lab:
        valid &= o[idx] >= 0
    idx = idx[valid]
    train_mask = fid[:n] != FOLD
    prior = {
        'train_dos_windows': int((y[:n][train_mask] == DOS).sum()),
        'train_windows': int(train_mask.sum()),
        'heldout_dos_windows': int((y[idx] == DOS).sum()),
        'heldout_windows': int(len(idx)),
    }
    prior['train_dos_share'] = round(prior['train_dos_windows'] / prior['train_windows'], 4)
    prior['heldout_dos_share'] = round(prior['heldout_dos_windows'] / prior['heldout_windows'], 4)
    print(f"held-out day {FOLD}: {len(idx):,} windows; DoS share train "
          f"{prior['train_dos_share']:.1%} vs held-out {prior['heldout_dos_share']:.1%}")

    runs = {}
    for spec in args.models:
        name, path = spec.split('=', 1)
        path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
        if not os.path.exists(path):
            print(f"  [{name}] {path} missing, skipped")
            continue
        stage, onset = score(path, X, idx, args.batch)
        runs[name] = {'checkpoint': os.path.relpath(path, ROOT),
                      **analyse(stage, onset, y, idx, onset_lab)}
        mj = path.replace('.pt', '_metrics.json')
        if os.path.exists(mj):
            m = json.load(open(mj))
            runs[name]['train_metrics_onset_auc'] = {k: round(v['auc'], 4)
                                                     for k, v in m['onset_auc'].items()}
        o = runs[name]['onset']
        print(f"\n[{name}] DoS F1 {runs[name]['stage']['dos_f1']}  "
              f"DoS recall {runs[name]['stage']['dos_recall']}")
        for k in ONSET_HORIZONS:
            r = o[f'k{k}']
            print(f"  k={k:>2}  all {r['all']}  into_dos {r['into_dos']} (n={r['n_into_dos']})"
                  f"  out_of_dos {r['out_of_dos']} (n={r['n_out_of_dos']})  "
                  f"within DoS {r['within_current_dos']}  within Benign "
                  f"{r['within_current_benign']}")

    verdict = None
    if 'sweep' in runs and 'class_weighted' in runs:
        a = runs['sweep']['onset']['k1']['all']
        b = runs['class_weighted']['onset']['k1']['all']
        if b >= 0.5 and b - a >= 0.05:
            verdict = 'confirmed'
        elif b - a < 0.05:
            verdict = 'killed'
        else:
            verdict = 'partial'
        print(f"\nonset AUC k1: sweep {a} -> class-weighted {b}  => hypothesis {verdict}")

    payload = {
        'fold': FOLD,
        'hypothesis': 'DoS class-prior inversion (1% of training, 57% of held-out '
                      'day) causes the below-chance onset AUC on fold 1',
        'verdict_rule': 'confirmed if class-weighted onset AUC k1 >= 0.5 and '
                        '>= +0.05 over the sweep; killed if < +0.05; else partial',
        'verdict': verdict,
        'class_prior': prior,
        'runs': runs,
        'note': 'into_dos / out_of_dos / other AUCs score that onset type against '
                'the day\'s non-onset windows. within_current_* hold the current '
                'label fixed. class_weighted = train_v2 --balance-power 1.0, '
                'otherwise the LODO sweep recipe (12 ep, patience 4, attention, '
                'select-on combined).',
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
