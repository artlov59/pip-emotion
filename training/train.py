"""Train the AudioCNN on MELD audio, locally (Apple-GPU/MPS, CUDA or CPU).

    python -m training.train --run-name wce_sqrt
    python -m training.train --run-name wce_sqrt_mixup --mixup-prob 0.5
    python -m training.train --run-name ce_plain --class-weighting none

The checkpoint with the best *dev* weighted-F1 is saved to checkpoints/<run>/best.pt.
The test split is never touched here (see training/evaluate.py).
"""
import argparse
import json
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from app.config import CHECKPOINT_DIR, EMOTIONS, FEATURE_CONFIG, ROOT
from model import AudioCNN, count_parameters
from training.dataset import MeldMelDataset, collate


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(labels, mode: str) -> torch.Tensor:
    counts = np.maximum(np.bincount(labels, minlength=len(EMOTIONS)), 1).astype(np.float64)
    if mode == "none":
        w = np.ones_like(counts)
    elif mode == "sqrt":       # softened inverse frequency
        w = 1.0 / np.sqrt(counts)
    elif mode == "balanced":   # full inverse frequency
        w = 1.0 / counts
    else:
        raise ValueError(mode)
    w = w / (w * counts).sum() * counts.sum()  # keep the average per-sample weight at 1
    return torch.tensor(w, dtype=torch.float32)


def mixup(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, targets, losses = [], [], []
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float((p == t).mean()),
        "weighted_f1": float(f1_score(t, p, average="weighted")),
        "macro_f1": float(f1_score(t, p, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8, help="early stop after N epochs w/o dev improvement")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--class-weighting", choices=["none", "sqrt", "balanced"], default="sqrt")
    ap.add_argument("--mixup-prob", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=400, help="training crop length (100 frames = 1 s)")
    ap.add_argument("--eval-max-frames", type=int, default=1000)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training clips")
    args = ap.parse_args()

    run = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = CHECKPOINT_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(ROOT / "runs" / run)

    train_ds = MeldMelDataset("train", train=True, max_frames=args.max_frames)
    dev_ds = MeldMelDataset("dev", train=False, max_frames=args.eval_max_frames)
    if args.limit:
        keep = list(range(min(args.limit, len(train_ds))))
        train_ds.ids = [train_ds.ids[i] for i in keep]
        train_ds.labels = [train_ds.labels[i] for i in keep]
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=collate)

    model = AudioCNN(num_classes=len(EMOTIONS), dropout=args.dropout).to(device)
    weights = class_weights(train_ds.labels, args.class_weighting)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    dev_criterion = nn.CrossEntropyLoss()  # unweighted, comparable across runs
    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr / 10, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs,
                           steps_per_epoch=len(train_dl), pct_start=0.1)

    print(f"run={run} device={device} params={count_parameters(model):,}")
    print(f"train={len(train_ds)} dev={len(dev_ds)} class_weights={np.round(weights.numpy(), 3).tolist()}")

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, lengths, y in tqdm(train_dl, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            if args.mixup_prob > 0 and random.random() < args.mixup_prob:
                x, ya, yb, lam = mixup(x, y)
                logits = model(x, lengths)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                loss = criterion(model(x, lengths), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()

        train_loss = total / len(train_dl)
        dev = evaluate(model, dev_dl, device, dev_criterion)
        epoch_s = time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "epoch_s": epoch_s, **{f"dev_{k}": v for k, v in dev.items()}})
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/dev", dev["loss"], epoch)
        writer.add_scalar("dev/weighted_f1", dev["weighted_f1"], epoch)
        writer.add_scalar("dev/macro_f1", dev["macro_f1"], epoch)
        writer.add_scalar("dev/accuracy", dev["accuracy"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        improved = dev["weighted_f1"] > best
        print(f"epoch {epoch + 1:>2} | train {train_loss:.4f} | dev loss {dev['loss']:.4f} "
              f"acc {dev['accuracy']:.3f} wF1 {dev['weighted_f1']:.3f} mF1 {dev['macro_f1']:.3f} "
              f"| {epoch_s:.0f}s{'  *best*' if improved else ''}")
        if improved:
            best, best_epoch = dev["weighted_f1"], epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": EMOTIONS,
                "feature_config": FEATURE_CONFIG,
                "epoch": epoch + 1,
                "dev_metrics": dev,
                "args": vars(args),
                "run": run,
            }, out_dir / "best.pt")
        if epoch + 1 - best_epoch >= args.patience:
            print(f"Early stopping: no dev improvement for {args.patience} epochs")
            break

    writer.close()
    with open(out_dir / "history.json", "w") as f:
        json.dump({"run": run, "args": vars(args), "best_epoch": best_epoch, "best_dev_weighted_f1": best,
                   "device": str(device), "history": history}, f, indent=2)
    print(f"Done. Best dev weighted-F1 {best:.3f} at epoch {best_epoch} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
