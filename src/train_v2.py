import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from model import WorldModel

MODEL_DIR = os.path.expanduser("~/netrikan/models")
STAGES = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    def __init__(self, path, indices, y):
        self.path, self.indices, self.y, self.X = path, indices, y, None

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
        return torch.from_numpy(win), int(self.y[j]), torch.from_numpy(nxt)


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
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--state-weight', type=float, default=None,
                    help='weight on the next-state prediction (world-model) loss')
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
    args.samples      = _get('samples', 1_200_000)
    args.state_weight = _get('state_weight', 0.3)

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
    split_y = y_full[:n] if args.fixed_split else y
    tr_idx, va_idx = blocked_split(split_y, purge=W - 1)
    tr_idx = tr_idx[tr_idx < n]
    va_idx = va_idx[va_idx < n]
    print(f"train {len(tr_idx):,}  val {len(va_idx):,}", flush=True)

    counts = np.bincount(y[tr_idx], minlength=len(STAGES))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w /= w.sum()
    sampler = WeightedRandomSampler(torch.as_tensor(w[y[tr_idx]], dtype=torch.double),
                                    num_samples=min(args.samples, len(tr_idx)),
                                    replacement=True)

    pin = DEVICE.type == 'cuda'
    tl = DataLoader(WindowDataset(xpath, tr_idx, y), batch_size=args.batch,
                    sampler=sampler, num_workers=4, pin_memory=pin, persistent_workers=True)
    vl = DataLoader(WindowDataset(xpath, va_idx, y), batch_size=4096, shuffle=False,
                    num_workers=4, pin_memory=pin, persistent_workers=True)

    model = WorldModel(input_size=F).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2, factor=0.5)
    stage_loss, breach_loss = nn.CrossEntropyLoss(), nn.BCELoss()
    state_loss = nn.MSELoss()          # world-model term: predict S_t+1

    best, bad, hist = -1.0, 0, []
    ckpt = os.path.join(MODEL_DIR, f'{args.tag}.pt')

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nbatch, te = 0.0, 0, time.time()
        for Xb, yb, nb in tl:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            nb = nb.to(DEVICE, non_blocking=True)
            logits, breach, nxt = model(Xb)
            loss = (stage_loss(logits, yb)
                    + 0.5 * breach_loss(breach, (yb > 0).float())
                    + args.state_weight * state_loss(nxt, nb))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nbatch += 1

        model.eval()
        P, T, S = [], [], []
        with torch.no_grad():
            for Xb, yb, nb in vl:
                logits, _, nxt = model(Xb.to(DEVICE, non_blocking=True))
                P.append(logits.argmax(1).cpu().numpy())
                T.append(yb.numpy())
                S.append(float(state_loss(nxt.cpu(), nb)))
        P, T = np.concatenate(P), np.concatenate(T)
        f1 = f1_score(T, P, average='macro', zero_division=0)
        per = f1_score(T, P, average=None, zero_division=0, labels=list(range(len(STAGES))))
        sched.step(f1)

        state_mse = float(np.mean(S)) if S else 0.0
        print(f"  ep{ep:>3} loss={tot/max(nbatch,1):.4f} macroF1={f1:.4f} sMSE={state_mse:.4f} "
              f"({time.time()-te:.0f}s)  " +
              " ".join(f"{s[:5]}={v:.2f}" for s, v in zip(STAGES, per)), flush=True)
        hist.append({"epoch": ep, "macro_f1": float(f1),
                     "per_class": {s: float(v) for s, v in zip(STAGES, per)}})

        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for Xb, yb, _n in vl:
            logits, _, _s = model(Xb.to(DEVICE, non_blocking=True))
            P.append(logits.argmax(1).cpu().numpy())
            T.append(yb.numpy())
    P, T = np.concatenate(P), np.concatenate(T)

    print(f"\n[{args.tag}] FINAL", flush=True)
    print(classification_report(T, P, target_names=STAGES, digits=3, zero_division=0),
          flush=True)

    out = {
        "tag": args.tag, "data": args.data, "window": int(W), "features": int(F),
        "horizon_k": k, "best_macro_f1": float(best),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "report": classification_report(T, P, target_names=STAGES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(T, P, labels=list(range(len(STAGES)))).tolist(),
        "history": hist, "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(MODEL_DIR, f'{args.tag}_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[{args.tag}] best={best:.4f}  {out['minutes']} min  -> {ckpt}", flush=True)


if __name__ == '__main__':
    main()
