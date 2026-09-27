import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             roc_auc_score)

from model import WorldModel, ONSET_HORIZONS

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
STAGES = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']
def _pick_device():
    if os.environ.get('NETRIKAN_DEVICE'):
        return torch.device(os.environ['NETRIKAN_DEVICE'])
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _pick_device()


class WindowDataset(Dataset):
    def __init__(self, path, indices, y, onset=None, shuffle_history=False,
                 seed=42):
        self.path, self.indices, self.y, self.X = path, indices, y, None
        self.onset = onset          # (n_onset, K) int8, or None
        # shuffle_history permutes the timestep axis WITHIN each window, per
        # sample, destroying temporal order while keeping the exact same rows.
        # A model that reads dynamics collapses; one reading a static per-window
        # fingerprint does not. Same rows, same labels, same split.
        self.shuffle_history = shuffle_history
        self._seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(self.path, mmap_mode='r')
        j = self.indices[i]
        win = np.array(self.X[j], dtype=np.float32)
        # X[j+1] is rows j+1..j+W, so its last row IS flow j+W -- the next state.
        # At the tail, fall back to repeating the final observed row.
        nxt = (np.array(self.X[j + 1][-1], dtype=np.float32)
               if j + 1 < len(self.X) else win[-1])
        if self.shuffle_history:
            # deterministic per-sample permutation of the WINDOW axis; the
            # next-state target is left as the true next flow, so the state head
            # cannot trivially recover order from it
            perm = np.random.default_rng(self._seed + int(j)).permutation(win.shape[0])
            win = win[perm]
        # indices are pre-trimmed to the onset array length by the caller, so
        # this never needs to invent a label -- inventing zeros here would add
        # false negatives for the last max(k) windows
        ons = (self.onset[j].astype(np.float32) if self.onset is not None
               else np.zeros(len(ONSET_HORIZONS), dtype=np.float32))
        return (torch.from_numpy(win), int(self.y[j]), torch.from_numpy(nxt),
                torch.from_numpy(ons))


def blocked_split(y, purge, n_blocks=500, val_frac=0.2, seed=42):
    """Whole contiguous blocks to train or val, purged at edges so no val window
    shares a source row with a train window."""
    n = len(y)
    edges = np.linspace(0, n, n_blocks + 1).astype(np.int64)
    dom = np.array([np.bincount(y[edges[i]:edges[i + 1]], minlength=len(STAGES)).argmax()
                    for i in range(n_blocks)])
    rng = np.random.default_rng(seed)
    val_blocks = set()
    for c in np.unique(dom):
        ids = np.where(dom == c)[0]
        rng.shuffle(ids)
        val_blocks.update(ids[:max(1, int(round(len(ids) * val_frac)))].tolist())
    tr, va = [], []
    for b in range(n_blocks):
        s, e = edges[b] + purge, edges[b + 1] - purge
        if e > s:
            (va if b in val_blocks else tr).append(np.arange(s, e))
    return np.concatenate(tr), np.concatenate(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--config', default=None,
                    help='path to configs/train_v2.yaml; CLI flags override it')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=None, help='flows ahead to predict')
    ap.add_argument('--fixed-split', action='store_true',
                    help='derive the blocked split from unshifted labels so every '
                         'horizon is evaluated on the same windows')
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None,
                    help='Adam learning rate (default 1e-3, or `lr` in the config)')
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--state-weight', type=float, default=None,
                    help='weight on the next-state prediction (world-model) loss')
    ap.add_argument('--onset-weight', type=float, default=None,
                    help='weight on the onset/hazard BCE loss (the forecasting target)')
    ap.add_argument('--seed', type=int, default=None,
                    help='overrides the split/sampler seed from the config')
    ap.add_argument('--lodo-val-file', type=int, default=None,
                    help='leave-one-day-out: validate on this capture file id, '
                         'train on all others (needs file_id.npy)')
    ap.add_argument('--holdout-class', type=int, default=None,
                    help='remove this stage id (and its onset signal) from '
                         'training; validation is untouched (held-out-family test)')
    ap.add_argument('--shuffle-history', action='store_true',
                    help='permute the timestep axis within each window (control: '
                         'destroys temporal order, keeps the same rows/labels)')
    ap.add_argument('--attention', action='store_true',
                    help='pool the window with temporal attention instead of the '
                         'last hidden state; yields per-timestep explanation weights')
    ap.add_argument('--select-on', default='auto',
                    choices=['auto', 'macro_f1', 'onset_auc_k5', 'combined'],
                    help='early-stopping metric; auto picks combined when onset '
                         'labels are present, else macro_f1. combined = mean of '
                         'onset AUC (forecasting) and macro-F1 (classification), '
                         'so one checkpoint is good at both.')
    args = ap.parse_args()

    cfg = {}
    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'train_v2.yaml')
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    def _get(name, default):
        cli = getattr(args, name.replace('-', '_'), None)
        return cli if cli is not None else cfg.get(name, default)

    args.epochs       = _get('epochs', 25)
    args.patience     = _get('patience', 5)
    args.horizon      = _get('horizon', 0)
    args.batch        = _get('batch', 1024)
    args.lr           = _get('lr', 1e-3)
    args.samples      = _get('samples', 1_200_000)
    args.state_weight = _get('state_weight', 0.3)
    args.onset_weight = _get('onset_weight', 2.0)
    # split parameters were previously left to blocked_split()'s defaults, so the
    # values in the config file had no effect on the run they claimed to describe
    args.seed         = _get('seed', 42)
    args.n_blocks     = _get('n_blocks', 500)
    args.val_frac     = _get('val_frac', 0.2)
    # seed model init and sampling too, so --seed actually varies the run and
    # multi-seed variance is real, not just a re-split
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    xpath = os.path.join(args.data, 'X.npy')
    Xh = np.load(xpath, mmap_mode='r')
    n_all, W, F = Xh.shape
    del Xh
    y_full = np.load(os.path.join(args.data, 'y.npy'))

    k = args.horizon
    n = len(y_full) - k
    y = y_full[k:n + k]

    print(f"[{args.tag}] device={DEVICE} windows={n:,} window={W} features={F} k={k}",
          flush=True)

    # Stratifying on the shifted labels gives each horizon a slightly different
    # split, which makes k values non-comparable. --fixed-split pins the blocks
    # to the unshifted labels so only the target changes.
    # onset labels, when the dataset was built with them. Stacked in the order
    # of ONSET_HORIZONS so column i corresponds to ONSET_HORIZONS[i].
    onset = None
    onset_paths = [os.path.join(args.data, f'onset_k{kk}.npy') for kk in ONSET_HORIZONS]
    if all(os.path.exists(p) for p in onset_paths):
        onset = np.stack([np.load(p) for p in onset_paths], axis=1)
        print(f"onset labels {onset.shape} positive rate "
              f"{dict(zip(ONSET_HORIZONS, onset.mean(0).round(4).tolist()))}", flush=True)
    else:
        print("no onset_k*.npy in --data; onset head will not be trained. "
              "Rebuild with pipeline_v2.py to enable it.", flush=True)

    if args.lodo_val_file is not None:
        # leave-one-day-out: val = one capture file, train = the rest, with a
        # WINDOW-1 purge around the held-out file's edges so no train window
        # overlaps a val window's source rows
        fid = np.load(os.path.join(args.data, 'file_id.npy'))[:n]
        va_mask = fid == args.lodo_val_file
        if not va_mask.any():
            raise SystemExit(f"no windows for file {args.lodo_val_file}")
        tr_mask = ~va_mask
        span = np.flatnonzero(va_mask)
        lo, hi = span.min(), span.max()
        tr_mask[max(0, lo - (W - 1)):min(n, hi + W)] = False
        tr_idx, va_idx = np.flatnonzero(tr_mask), np.flatnonzero(va_mask)
        print(f"LODO: val=file {args.lodo_val_file} ({va_mask.sum():,} windows), "
              f"train={len(tr_idx):,}", flush=True)
    else:
        split_y = y_full[:n] if args.fixed_split else y
        tr_idx, va_idx = blocked_split(split_y, purge=W - 1, n_blocks=args.n_blocks,
                                       val_frac=args.val_frac, seed=args.seed)
        tr_idx = tr_idx[tr_idx < n]
        va_idx = va_idx[va_idx < n]
    if onset is not None:
        # pipeline_v2 writes onset arrays shorter than y by max(ONSET_HORIZONS);
        # pipeline_identity writes them at full length with -1 at each host's
        # tail. Both mean "no valid label here" — drop those rows rather than
        # padding, so every onset label trained on is a real observation.
        valid = np.flatnonzero((onset >= 0).all(axis=1))
        keep = np.zeros(max(n, len(onset)), dtype=bool)
        keep[valid] = True
        tr_idx = tr_idx[(tr_idx < len(onset)) & keep[tr_idx]]
        va_idx = va_idx[(va_idx < len(onset)) & keep[va_idx]]
    # Held-out attack family: remove from TRAINING every window of the held-out
    # class AND every window whose onset horizon reaches that class, so the model
    # never sees it as a label or as a transition target. Leaving the onset signal
    # in would leak the class the model is supposed to have never seen. Validation
    # is untouched — the point is to test forecasting of an unseen family.
    if args.holdout_class is not None:
        hc = args.holdout_class
        file_id = (np.load(os.path.join(args.data, 'file_id.npy'))[:n]
                   if os.path.exists(os.path.join(args.data, 'file_id.npy'))
                   else np.zeros(n, np.int64))
        reaches_hc = (y == hc)
        maxk = max(ONSET_HORIZONS)
        for kk in ONSET_HORIZONS:
            hi = n - kk
            same = np.zeros(n, bool); same[:hi] = file_id[:hi] == file_id[kk:kk + hi]
            fut = np.zeros(n, bool); fut[:hi] = (y[kk:kk + hi] == hc)
            reaches_hc |= (same & fut)
        before = len(tr_idx)
        tr_idx = tr_idx[~reaches_hc[tr_idx]]
        print(f"HOLDOUT class {hc}: removed {before - len(tr_idx):,} training "
              f"windows (label==class or onset reaches it); val kept whole",
              flush=True)

    print(f"train {len(tr_idx):,}  val {len(va_idx):,}", flush=True)

    counts = np.bincount(y[tr_idx], minlength=len(STAGES))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w /= w.sum()
    sampler = WeightedRandomSampler(torch.as_tensor(w[y[tr_idx]], dtype=torch.double),
                                    num_samples=min(args.samples, len(tr_idx)),
                                    replacement=True)

    pin = DEVICE.type == 'cuda'
    shuf = args.shuffle_history
    if shuf:
        print("SHUFFLE-HISTORY control: timestep order destroyed within every "
              "window (train and val)", flush=True)
    # Worker count only affects how batches are fetched, never what is in them:
    # order comes from `sampler`, so results are unchanged by this value. Override
    # with NETRIKAN_WORKERS=0 on machines where torch's shared-memory manager times
    # out under long multi-run sweeps (seen on Apple Silicon during bench/lodo.py).
    nw = int(os.environ.get('NETRIKAN_WORKERS', '4'))
    persist = nw > 0
    tl = DataLoader(WindowDataset(xpath, tr_idx, y, onset, shuffle_history=shuf),
                    batch_size=args.batch, sampler=sampler, num_workers=nw,
                    pin_memory=pin, persistent_workers=persist)
    vl = DataLoader(WindowDataset(xpath, va_idx, y, onset, shuffle_history=shuf),
                    batch_size=4096, shuffle=False, num_workers=nw,
                    pin_memory=pin, persistent_workers=persist)

    use_attn = args.attention or bool(cfg.get('attention', False))
    model = WorldModel(input_size=F, attention=use_attn).to(DEVICE)
    print(f"pooling: {'temporal attention' if use_attn else 'last hidden state'}",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2, factor=0.5)
    stage_loss, breach_loss = nn.CrossEntropyLoss(), nn.BCELoss()
    state_loss = nn.MSELoss()          # world-model term: predict S_t+1
    onset_loss = nn.BCEWithLogitsLoss()

    # Onset is the forecasting target; stage classification is dominated by
    # persistence. Select on onset AUC when onset labels exist, so early
    # stopping optimises forecasting skill rather than label autocorrelation.
    # A corpus with almost no label transitions produces onset columns that are
    # all zeros, so onset AUC is undefined and selecting on it would never save a
    # checkpoint. Detect that up front rather than discovering it as a silent
    # training failure. CTU-13 is exactly this case: 1 transition in 92k windows.
    MIN_ONSET_POS = 50
    onset_usable = False
    if onset is not None:
        pos = onset[va_idx][:, ONSET_HORIZONS.index(5) if 5 in ONSET_HORIZONS else 0]
        onset_usable = int((pos > 0).sum()) >= MIN_ONSET_POS
        if not onset_usable:
            print(f"onset labels are degenerate: only {int((pos > 0).sum())} "
                  f"positive val windows at k=5 (need {MIN_ONSET_POS}). The onset "
                  f"head will still train but cannot be scored.", flush=True)

    select_on = args.select_on
    if select_on == 'auto':
        select_on = 'combined' if onset_usable else 'macro_f1'
    if select_on == 'combined' and not onset_usable:
        select_on = 'macro_f1'
    if select_on.startswith('onset') and onset is None:
        raise SystemExit("--select-on onset_auc_k5 needs onset_k*.npy in --data")
    if select_on.startswith('onset') and not onset_usable:
        raise SystemExit(
            "--select-on onset_auc_k5 was requested but this corpus has too few "
            "label transitions to score onset. Use --select-on macro_f1, or "
            "build the dataset from a corpus whose attack phases alternate "
            "(CIC-IDS-2018 or DAPT 2020).")
    K_SEL = ONSET_HORIZONS.index(5) if 5 in ONSET_HORIZONS else 0
    print(f"checkpoint selection: {select_on}", flush=True)

    best, bad, hist = -1.0, 0, []
    ckpt = os.path.join(MODEL_DIR, f'{args.tag}.pt')

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nbatch, te = 0.0, 0, time.time()
        for Xb, yb, nb, ob in tl:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            nb = nb.to(DEVICE, non_blocking=True)
            logits, breach, nxt, onset_logits = model(Xb)
            loss = (stage_loss(logits, yb)
                    + 0.5 * breach_loss(breach, (yb > 0).float())
                    + args.state_weight * state_loss(nxt, nb))
            if onset is not None:
                loss = loss + args.onset_weight * onset_loss(
                    onset_logits, ob.to(DEVICE, non_blocking=True))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nbatch += 1

        model.eval()
        P, T, S, OP, OT = [], [], [], [], []
        with torch.no_grad():
            for Xb, yb, nb, ob in vl:
                logits, _, nxt, onset_logits = model(Xb.to(DEVICE, non_blocking=True))
                P.append(logits.argmax(1).cpu().numpy())
                T.append(yb.numpy())
                S.append(float(state_loss(nxt.cpu(), nb)))
                OP.append(torch.sigmoid(onset_logits).cpu().numpy())
                OT.append(ob.numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        f1 = f1_score(T, P, average='macro', zero_division=0)
        per = f1_score(T, P, average=None, zero_division=0, labels=list(range(len(STAGES))))

        onset_aucs = {}
        if onset is not None:
            OP, OT = np.concatenate(OP), np.concatenate(OT)
            for ki, kk in enumerate(ONSET_HORIZONS):
                col = OT[:, ki]
                onset_aucs[kk] = (float(roc_auc_score(col, OP[:, ki]))
                                  if col.min() != col.max() else float('nan'))

        onset_k5 = onset_aucs.get(ONSET_HORIZONS[K_SEL], float('nan'))
        if select_on == 'macro_f1':
            score = f1
        elif select_on == 'combined':
            # reward both forecasting (onset) and classification (macro-F1, which
            # forces attack-class recall and so protects cross-dataset detection).
            # Equal weight; both are in [0,1].
            score = 0.5 * onset_k5 + 0.5 * f1 if onset_k5 == onset_k5 else f1
        else:
            score = onset_k5
        sched.step(score)

        state_mse = float(np.mean(S)) if S else 0.0
        onset_str = ("  " + " ".join(f"onset_k{kk}={v:.3f}"
                                     for kk, v in onset_aucs.items())
                     if onset_aucs else "")
        print(f"  ep{ep:>3} loss={tot/max(nbatch,1):.4f} macroF1={f1:.4f} "
              f"sMSE={state_mse:.4f} ({time.time()-te:.0f}s)  " +
              " ".join(f"{s[:5]}={v:.2f}" for s, v in zip(STAGES, per)) +
              onset_str, flush=True)
        hist.append({"epoch": ep, "macro_f1": float(f1),
                     "onset_auc": {f'k{kk}': v for kk, v in onset_aucs.items()},
                     "per_class": {s: float(v) for s, v in zip(STAGES, per)}})

        if score == score and score > best:   # score == score rejects NaN
            best, bad = score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    P, T, OP, OT = [], [], [], []
    with torch.no_grad():
        for Xb, yb, _n, ob in vl:
            logits, _, _s, onset_logits = model(Xb.to(DEVICE, non_blocking=True))
            P.append(logits.argmax(1).cpu().numpy())
            T.append(yb.numpy())
            OP.append(torch.sigmoid(onset_logits).cpu().numpy())
            OT.append(ob.numpy())
    P, T = np.concatenate(P), np.concatenate(T)

    final_onset = {}
    if onset is not None:
        OP, OT = np.concatenate(OP), np.concatenate(OT)
        print("\nOnset AUC  (persistence = 0.500 by construction: it never "
              "predicts a change)")
        for ki, kk in enumerate(ONSET_HORIZONS):
            col = OT[:, ki]
            if col.min() == col.max():
                continue
            auc = float(roc_auc_score(col, OP[:, ki]))
            final_onset[f'k{kk}'] = {"auc": auc,
                                     "onset_rate": float(col.mean()),
                                     "gap_over_persistence": round(auc - 0.5, 4)}
            print(f"  k={kk:>3}  onset_rate={col.mean():.4f}  AUC={auc:.4f}  "
                  f"gap=+{auc - 0.5:.4f}", flush=True)

    print(f"\n[{args.tag}] FINAL", flush=True)
    # labels= is required: on a corpus that does not contain all 5 classes
    # (CTU-13 is botnet vs benign only) sklearn otherwise raises on the
    # target_names length mismatch
    ALL = list(range(len(STAGES)))
    print(classification_report(T, P, labels=ALL, target_names=STAGES, digits=3,
                                zero_division=0), flush=True)

    out = {
        "tag": args.tag, "data": args.data, "window": int(W), "features": int(F),
        "horizon_k": k,
        "select_on": select_on,
        "attention": bool(use_attn),
        "shuffle_history": bool(shuf),
        "best": float(best),
        "final_macro_f1": float(f1_score(T, P, average='macro', zero_division=0)),
        "onset_auc": final_onset,
        "onset_horizons": list(ONSET_HORIZONS) if onset is not None else None,
        "onset_weight": args.onset_weight if onset is not None else None,
        "persistence_onset_auc": 0.5,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "classes_present": sorted(set(T.tolist()) | set(P.tolist())),
        "report": classification_report(T, P, labels=ALL, target_names=STAGES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(T, P, labels=list(range(len(STAGES)))).tolist(),
        "history": hist, "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(MODEL_DIR, f'{args.tag}_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[{args.tag}] best={best:.4f}  {out['minutes']} min  -> {ckpt}", flush=True)


if __name__ == '__main__':
    main()
