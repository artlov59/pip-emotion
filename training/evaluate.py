"""Evaluate one or more checkpoints on a MELD split, against a majority-class baseline.

Pick a model on dev (and copy the winner to checkpoints/emotion_cnn.pt):
    python -m training.evaluate --split dev --checkpoints checkpoints/*/best.pt --promote
Then report it once on test:
    python -m training.evaluate --split test --checkpoints checkpoints/emotion_cnn.pt

Writes to results/: <split>_<name>_metrics.json, _confusion.png, _predictions.csv and
<split>_summary.md (a markdown table that goes straight into the README).
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader  # noqa: E402

from app.config import CHECKPOINT_DIR, EMOTIONS, FEATURE_CONFIG, MELD_DIR, RESULTS_DIR  # noqa: E402
from model import AudioCNN  # noqa: E402
from training.dataset import MeldMelDataset, collate  # noqa: E402
from training.train import pick_device  # noqa: E402


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(EMOTIONS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {e: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, e in enumerate(EMOTIONS)},
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def predict(ckpt_path, ds, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["feature_config"] == FEATURE_CONFIG, "checkpoint was trained with different features"
    model = AudioCNN(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    probs = []
    for x, lengths, _ in DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate):
        probs.append(torch.softmax(model(x.to(device), lengths.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy(), ckpt


def name_for(path: Path) -> str:
    return path.parent.name if path.name == "best.pt" else path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=1000, help="centre-crop clips longer than this (10 s)")
    ap.add_argument("--promote", action="store_true", help="copy the best checkpoint to checkpoints/emotion_cnn.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = MeldMelDataset(args.split, train=False, max_frames=args.max_frames)
    y = np.array(ds.labels)

    tr = pd.read_csv(MELD_DIR / "train.csv")  # majority class is taken from *train*
    majority = int(tr[tr.status == "ok"]["label"].mode()[0])

    rows = []
    base = metrics(y, np.full_like(y, majority))
    rows.append((f"majority class ({EMOTIONS[majority]})", base, None))

    for c in args.checkpoints:
        path = Path(c)
        probs, ckpt = predict(path, ds, device)
        pred = probs.argmax(1)
        m = metrics(y, pred)
        name = name_for(path)
        m.update({"checkpoint": str(path), "run": ckpt.get("run"), "epoch": ckpt.get("epoch"),
                  "train_args": ckpt.get("args"), "split": args.split, "n": int(len(y))})
        stem = f"{args.split}_{name}"
        with open(RESULTS_DIR / f"{stem}_metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        plot_confusion(y, pred, f"{name} on MELD {args.split} (audio only)  wF1={m['weighted_f1']:.3f}",
                       RESULTS_DIR / f"{stem}_confusion.png")
        out = ds.df[["id", "emotion", "utterance"]].copy()
        out["pred"] = [EMOTIONS[i] for i in pred]
        for i, e in enumerate(EMOTIONS):
            out[f"p_{e}"] = probs[:, i].round(4)
        out.to_csv(RESULTS_DIR / f"{stem}_predictions.csv", index=False)
        rows.append((name, m, path))

    lines = [f"MELD **{args.split}** split, audio only, n={len(y)}", "",
             "| model | weighted F1 | macro F1 | accuracy | " + " | ".join(f"F1 {e}" for e in EMOTIONS) + " |",
             "|---|---|---|---|" + "---|" * len(EMOTIONS)]
    for name, m, _ in rows:
        per = " | ".join(f"{m['per_class'][e]['f1']:.2f}" for e in EMOTIONS)
        lines.append(f"| {name} | {m['weighted_f1']:.3f} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} | {per} |")
    table = "\n".join(lines)
    (RESULTS_DIR / f"{args.split}_summary.md").write_text(table + "\n")
    print(table)

    if args.promote:
        name, m, path = max(rows[1:], key=lambda r: r[1]["weighted_f1"])
        dst = CHECKPOINT_DIR / "emotion_cnn.pt"
        shutil.copy(path, dst)
        print(f"\nPromoted {path} (dev weighted F1 {m['weighted_f1']:.3f}) -> {dst}")


if __name__ == "__main__":
    main()
