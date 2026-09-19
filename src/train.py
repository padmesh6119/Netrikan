import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from model import WorldModel

DATA_DIR = os.path.expanduser("~/netrikan/data/processed")
MODEL_DIR = os.path.expanduser("~/netrikan/models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 1024
EVAL_BATCH = 4096
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
WINDOW = 10
N_BLOCKS = 500
VAL_FRAC = 0.2
SAMPLES_PER_EPOCH = 1_200_000
STAGES = ['Benign', 'InitialAccess', 'DoS', 'Infiltration', 'Botnet']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    """Indexes a memmapped array so 5.3 GB never enters RAM."""

    def __init__(self, path, indices, y):
        self.path = path
        self.indices = indices
        self.y = y
        self.X = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.X is None:                       # opened per worker, not pickled
            self.X = np.load(self.path, mmap_mode='r')
        j = self.indices[i]
        x = torch.from_numpy(np.array(self.X[j], dtype=np.float32))
        return x, int(self.y[j])


def blocked_split(y, n_blocks=N_BLOCKS, val_frac=VAL_FRAC, purge=WINDOW - 1, seed=42):
    """
    Contiguous blocks assigned whole to train or val, with `purge` windows dropped
    at every block edge. Guarantees no val window shares a row with a train window.
    Blocks are stratified by dominant class so every class appears on both sides.
    """
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
        if e <= s:
            continue
        (va if b in val_blocks else tr).append(np.arange(s, e))
    return np.concatenate(tr), np.concatenate(va)


def evaluate(model, loader, stage_loss_fn, breach_loss_fn):
    model.eval()
    loss_sum, nb = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE, non_blocking=True), y_b.to(DEVICE, non_blocking=True)
            logits, breach = model(X_b)
            loss = stage_loss_fn(logits, y_b) + 0.5 * breach_loss_fn(breach, (y_b > 0).float())
            loss_sum += loss.item()
            nb += 1
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y_b.cpu().numpy())
    return loss_sum / max(nb, 1), np.concatenate(preds), np.concatenate(trues)


def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")

    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    xpath = os.path.join(DATA_DIR, 'X.npy')
    print(f"Windows: {len(y):,}")

    tr_idx, va_idx = blocked_split(y)
    print(f"Train: {len(tr_idx):,}   Val: {len(va_idx):,}   "
          f"(purged {len(y) - len(tr_idx) - len(va_idx):,} at block edges)")

    tr_counts = np.bincount(y[tr_idx], minlength=len(STAGES))
    va_counts = np.bincount(y[va_idx], minlength=len(STAGES))
    print("\nclass            train        val")
    for i, s in enumerate(STAGES):
        print(f"  {s:<14}{tr_counts[i]:>9,} {va_counts[i]:>10,}")

    # sqrt-inverse frequency: rebalances without the 45x oversampling that
    # pure inverse frequency would apply to Infiltration
    w_class = 1.0 / np.sqrt(np.maximum(tr_counts, 1))
    w_class /= w_class.sum()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(w_class[y[tr_idx]], dtype=torch.double),
        num_samples=min(SAMPLES_PER_EPOCH, len(tr_idx)),
        replacement=True,
    )

    train_loader = DataLoader(
        WindowDataset(xpath, tr_idx, y), batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=True)
    val_loader = DataLoader(
        WindowDataset(xpath, va_idx, y), batch_size=EVAL_BATCH, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=True)

    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           patience=2, factor=0.5)
    # sampler already balances the batches, so the loss stays unweighted
    stage_loss_fn = nn.CrossEntropyLoss()
    breach_loss_fn = nn.BCELoss()

    best_f1, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, nb, t0 = 0.0, 0, time.time()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE, non_blocking=True), y_b.to(DEVICE, non_blocking=True)
            logits, breach = model(X_b)
            loss = stage_loss_fn(logits, y_b) + 0.5 * breach_loss_fn(breach, (y_b > 0).float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            run_loss += loss.item()
            nb += 1

        train_loss = run_loss / max(nb, 1)
        val_loss, preds, trues = evaluate(model, val_loader, stage_loss_fn, breach_loss_fn)
        macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        per_class = f1_score(trues, preds, average=None, zero_division=0,
                             labels=list(range(len(STAGES))))
        scheduler.step(macro_f1)

        print(f"\nEpoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"macroF1={macro_f1:.4f}  ({time.time()-t0:.0f}s)")
        print("  " + "  ".join(f"{s}={f:.3f}" for s, f in zip(STAGES, per_class)))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "macro_f1": float(macro_f1),
                        "per_class_f1": [float(x) for x in per_class]})

        if macro_f1 > best_f1:
            best_f1, bad_epochs = macro_f1, 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_world_model.pt'))
            print(f"  saved (macroF1={macro_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"\nEarly stop: no macro-F1 gain in {PATIENCE} epochs.")
                break

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'lstm_world_model.pt'),
                                     map_location=DEVICE, weights_only=True))
    _, preds, trues = evaluate(model, val_loader, stage_loss_fn, breach_loss_fn)

    print("\n" + "=" * 62)
    print("FINAL — held-out blocks, no window overlap with training")
    print("=" * 62)
    print(classification_report(trues, preds, target_names=STAGES, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted)")
    cm = confusion_matrix(trues, preds, labels=list(range(len(STAGES))))
    print(f"{'':<15}" + "".join(f"{s[:8]:>10}" for s in STAGES))
    for i, s in enumerate(STAGES):
        print(f"{s:<15}" + "".join(f"{v:>10,}" for v in cm[i]))

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            "best_macro_f1": float(best_f1),
            "split": "blocked, purged, stratified by dominant class",
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "history": history,
            "final_report": classification_report(trues, preds, target_names=STAGES,
                                                  output_dict=True, zero_division=0),
            "confusion_matrix": cm.tolist(),
            "minutes": round((time.time() - t_start) / 60, 1),
        }, f, indent=2)

    print(f"\nBest macro-F1: {best_f1:.4f}   |   {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {MODEL_DIR}/lstm_world_model.pt  +  metrics.json")


if __name__ == '__main__':
    main()
