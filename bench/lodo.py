"""
Leave-one-day-out cross-validation over the 7 CIC-IDS-2018 capture files.

One purged blocked split is correct but gives a single number with no variance and
no domain shift. LODO trains on 6 days and tests on the held-out day, 7 times, so
every metric gets a mean ± std and the worst fold is named. Several CIC days carry
only one attack family, so a fold with zero positives for a class is expected and
is reported as such, never averaged away.

Uses train_v2 --lodo-val-file (val = one file, train = the rest, purged at the
held-out file's edges). Reads each fold's committed metrics JSON — onset AUC per
horizon, stage macro-F1, and the per-class support of the held-out day.

    python3 bench/lodo.py --data "$DATA" --out models/lodo_cic_full_w30.json
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from train_v2 import STAGES   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'lodo_cic_full_w30.json'))
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    fid = np.load(os.path.join(args.data, 'file_id.npy'))
    files = sorted(int(f) for f in np.unique(fid))
    y = np.load(os.path.join(args.data, 'y.npy'))[:len(fid)]

    folds = []
    for f in files:
        tag = f"lodo_f{f}_w30"
        mpath = os.path.join(ROOT, 'models', f'{tag}_metrics.json')
        if not args.skip_train:
            cmd = ['.venv/bin/python', '-W', 'ignore', 'src/train_v2.py',
                   '--data', args.data, '--tag', tag, '--attention',
                   '--select-on', 'combined', '--lodo-val-file', str(f),
                   '--epochs', str(args.epochs), '--patience', '4']
            env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1',
                       NETRIKAN_DEVICE=os.environ.get('NETRIKAN_DEVICE', 'mps'))
            print(f"\n===== fold {f}: hold out file {f} =====", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, env=env)
            if r.returncode != 0:
                raise SystemExit(f"fold {f} training failed")
        m = json.load(open(mpath))
        # class support of the held-out day
        held = y[fid == f]
        support = {STAGES[c]: int((held == c).sum()) for c in range(len(STAGES))}
        rep = m['report']
        # Classes absent from the held-out day score F1 0.0 by absence, not by
        # error, so a 5-class macro mean is not comparable across folds. Average
        # only over classes the held-out day actually contains.
        present = [c for c in STAGES if support[c] > 0]
        sup_f1 = round(float(np.mean([rep[c]['f1-score'] for c in present])), 4)
        # A class confined to this one day is absent from the 6 training days, so
        # that fold is a held-out-FAMILY test wearing a leave-one-day-out costume.
        train_sup = {c: int((y[fid != f] == STAGES.index(c)).sum()) for c in present}
        holdout_families = [c for c in present if train_sup[c] == 0]
        folds.append({
            "fold": f, "val_file": f,
            "onset_auc": {k: round(v['auc'], 4) for k, v in m['onset_auc'].items()},
            "stage_macro_f1": round(m['final_macro_f1'], 4),
            "supported_class_f1": sup_f1,
            "per_class_f1": {c: round(rep[c]['f1-score'], 3) for c in STAGES},
            "held_out_support": support,
            "train_support_of_present_classes": train_sup,
            "family_holdout_classes": holdout_families,
        })
        print(f"  fold {f}: macro-F1 {folds[-1]['stage_macro_f1']}  "
              f"onset {folds[-1]['onset_auc']}  support {support}", flush=True)

    def agg(getter, pool=None):
        vals = [getter(fo) for fo in (folds if pool is None else pool)
                if getter(fo) == getter(fo)]
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)} if vals else None

    onset_k5 = agg(lambda fo: fo['onset_auc'].get('k5', float('nan')))
    macro = agg(lambda fo: fo['stage_macro_f1'])
    sup_macro = agg(lambda fo: fo['supported_class_f1'])
    worst = min(folds, key=lambda fo: fo['supported_class_f1'])

    # Folds with a family_holdout_class test zero-shot family transfer, not day
    # shift (fold 0: Botnet exists on no other day). Report the domain-shift
    # aggregate without them, next to the all-fold one.
    shift = [fo for fo in folds if not fo['family_holdout_classes']]
    excl = {
        "folds": [fo['fold'] for fo in shift],
        "excluded_folds": [fo['fold'] for fo in folds if fo['family_holdout_classes']],
        "supported_class_f1": agg(lambda fo: fo['supported_class_f1'], shift),
        **{f"onset_auc_{k}": agg(lambda fo, k=k: fo['onset_auc'].get(k, float('nan')),
                                 shift)
           for k in ('k1', 'k5', 'k15', 'k30')},
        "note": "family_holdout_classes is a zero-training-support test, so it does "
                "not catch fold 6 (InitialAccess: 566 training windows). That fold "
                "is a near-family-holdout and stays in this aggregate; see "
                "train_support_of_present_classes.",
    }
    out = {
        "data": args.data, "n_folds": len(folds), "epochs_per_fold": args.epochs,
        "aggregate": {"onset_auc_k5": onset_k5,
                      "supported_class_f1": sup_macro,
                      "stage_macro_f1_DO_NOT_REPORT": macro},
        "aggregate_excl_family_holdout": excl,
        "worst_fold": {"fold": worst['fold'], "stage_macro_f1": worst['stage_macro_f1'],
                       "held_out_support": worst['held_out_support']},
        "folds": folds,
        "note": "Days with one attack family produce zero-support classes in some "
                "folds; those are reported per fold, not averaged into a class mean. "
                "stage_macro_f1 averages over all 5 classes including ones with zero "
                "val support, so it is depressed by absence rather than by error and "
                "must NOT be quoted -- use supported_class_f1 and the per-class table. "
                "Folds listing family_holdout_classes are held-out-family tests, not "
                "domain-shift tests: that class occurs on no other capture day.",
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nonset AUC k5 across folds: {onset_k5}")
    print(f"supported-class F1 across folds: {sup_macro}")
    print(f"excluding family-holdout folds {excl['excluded_folds']}: onset k5 "
          f"{excl['onset_auc_k5']}  supported-class F1 {excl['supported_class_f1']}")
    print(f"(5-class macro-F1 {macro} -- depressed by zero-support classes, do not quote)")
    print(f"worst fold: {worst['fold']} (supported-class F1 {worst['supported_class_f1']}, "
          f"family holdouts {worst['family_holdout_classes']})")
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
