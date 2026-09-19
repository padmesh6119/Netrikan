import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report

from model import WorldModel
from train import WindowDataset, blocked_split, STAGES, DATA_DIR, MODEL_DIR, DEVICE

# how many flows ahead of the window to predict; 0 == current pipeline behaviour
HORIZONS = [0, 30, 90, 180]

BATCH_SIZE = 1024
EVAL_BATCH = 4096
EPOCHS = 12
PATIENCE = 3
LR = 1e-3
SAMPLES_PER_EPOCH = 600_000


def run_one(k, y_full, xpath):
    """Train a model predicting the stage k flows beyond the window."""
    n = len(y_full) - k
    y = y_full[k:n + k]                      # window i -> label of flow i+WINDOW+k
    tr_idx, va_idx = blocked_split(y)

    counts = np.bincount(y[tr_idx], minlength=len(STAGES))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w /= w.sum()
    sampler = WeightedRandomSampler(
        torch.as_tensor(w[y[tr_idx]], dtype=torch.double),
        num_samples=min(SAMPLES_PER_EPOCH, len(tr_idx)), replacement=True)

    tl = DataLoader(WindowDataset(xpath, tr_idx, y), batch_size=BATCH_SIZE,
                    sampler=sampler, num_workers=4,
                    pin_memory=(DEVICE.type == 'cuda'), persistent_workers=True)
    vl = DataLoader(WindowDataset(xpath, va_idx, y), batch_size=EVAL_BATCH,
                    shuffle=False, num_workers=4,
                    pin_memory=(DEVICE.type == 'cuda'), persistent_workers=True)

    model = WorldModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    stage_loss = nn.CrossEntropyLoss()
    breach_loss = nn.BCELoss()

    best, bad, best_per_class, best_preds = -1.0, 0, None, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in tl:
            X_b, y_b = X_b.to(DEVICE, non_blocking=True), y_b.to(DEVICE, non_blocking=True)
            logits, breach = model(X_b)
            loss = stage_loss(logits, y_b) + 0.5 * breach_loss(breach, (y_b > 0).float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        P, T = [], []
        with torch.no_grad():
            for X_b, y_b in vl:
                logits, _ = model(X_b.to(DEVICE, non_blocking=True))
                P.append(logits.argmax(1).cpu().numpy())
                T.append(y_b.numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        f1 = f1_score(T, P, average='macro', zero_division=0)
        per = f1_score(T, P, average=None, zero_division=0, labels=list(range(len(STAGES))))
        print(f"    epoch {epoch:>2}  macroF1={f1:.4f}   " +
              " ".join(f"{s[:5]}={v:.2f}" for s, v in zip(STAGES, per)))

        if f1 > best:
            best, bad, best_per_class, best_preds = f1, 0, per, (T, P)
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f'lstm_horizon_k{k}.pt'))
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    return best, best_per_class, best_preds


def main():
    t0 = time.time()
    y_full = np.load(os.path.join(DATA_DIR, 'y.npy'))
    xpath = os.path.join(DATA_DIR, 'X.npy')

    # seconds of lead time each horizon buys, at the dataset's own flow rate
    results = []
    for k in HORIZONS:
        print(f"\n{'='*66}\nHORIZON k={k} flows ahead\n{'='*66}")
        t = time.time()
        best, per, (T, P) = run_one(k, y_full, xpath)
        mins = (time.time() - t) / 60
        print(f"  best macroF1 = {best:.4f}   ({mins:.1f} min)")
        print(classification_report(T, P, target_names=STAGES, digits=3, zero_division=0))
        results.append({
            "k_flows": k,
            "macro_f1": float(best),
            "per_class_f1": {s: float(v) for s, v in zip(STAGES, per)},
            "minutes": round(mins, 1),
        })
        with open(os.path.join(MODEL_DIR, 'horizon_curve.json'), 'w') as f:
            json.dump({"horizons": results}, f, indent=2)

    print(f"\n{'='*66}\nHORIZON CURVE\n{'='*66}")
    print(f"{'k (flows)':>10}{'macro F1':>12}   per-class")
    for r in results:
        pc = " ".join(f"{s[:5]}={v:.2f}" for s, v in r['per_class_f1'].items())
        print(f"{r['k_flows']:>10}{r['macro_f1']:>12.4f}   {pc}")
    print(f"\nTotal {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
