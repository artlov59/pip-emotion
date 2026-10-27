"""Metrics + confusion-matrix plot shared by all evaluation scripts (no torch dependency)."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)

from app.config import EMOTIONS  # noqa: E402


# Purpose: Calculate overall and per-class emotion classification scores.
# Input: y_true/y_pred: aligned ground-truth and predicted MELD label indices.
# Output: Dict of accuracy, weighted/macro F1, and seven per-class precision/recall/F1/support records.
def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(EMOTIONS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {e: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, e in enumerate(EMOTIONS)},
    }


# Purpose: Save a row-normalized confusion matrix with raw counts.
# Input: y_true/y_pred: label indices; title: plot caption; path: destination image path.
# Output: None; writes the plot and closes its figure.
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


PROBABILITIES = [f"p_{emotion}" for emotion in EMOTIONS]
EPS = 1e-4


# Purpose: Identify an evaluation input by absolute location and checksum.
# Input: path: local source file.
# Output: Dict containing resolved path and SHA-256 digest.
def source_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# Purpose: Validate cached dev probabilities and align them to manifest order.
# Input: manifest: dev ID/emotion frame; path: dev-named CSV; required: reject missing coverage if true.
# Output: Tuple (clipped log-probability array shaped (N, 7), missing-row mask shaped (N,)); invalid data
#          raises errors.
def read_probabilities(manifest, path, required=False):
    path = Path(path)
    if "dev" not in path.stem.lower():
        raise ValueError(f"Development-only input filename must identify dev: {path}")
    frame = pd.read_csv(path)
    if not frame.id.is_unique:
        raise ValueError(f"Duplicate probability IDs: {path}")
    merged = manifest[["id", "emotion"]].merge(frame[["id", "emotion"] + PROBABILITIES],
                                               on="id", how="left", suffixes=("", "_source"),
                                               validate="one_to_one")
    mismatch = merged.emotion_source.notna() & (merged.emotion != merged.emotion_source)
    if mismatch.any():
        raise ValueError(f"Emotion labels disagree with development manifest: {path}")
    values = merged[PROBABILITIES].to_numpy(dtype=float)
    missing = np.isnan(values).all(axis=1)
    present = values[~missing]
    if not np.isfinite(present).all() or ((present < 0) | (present > 1)).any() or (present.sum(axis=1) <= 0).any():
        raise ValueError(f"Invalid/partially missing probability row in {path}")
    if required and missing.any():
        raise ValueError(f"Audio coverage must include every successful development clip: {path}")
    return np.nan_to_num(np.log(np.clip(values, EPS, 1.0)), nan=0.0), missing


# Purpose: Accumulate class counts separately for each dialogue/group.
# Input: truth/predicted: label indices; group_index: per-row group IDs; n_groups: group count.
# Output: Integer confusion tensor shaped (n_groups, 7, 7), indexed by true then predicted label.
def grouped_confusion(truth, predicted, group_index, n_groups):
    result = np.zeros((n_groups, len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)
    np.add.at(result, (group_index, truth, predicted), 1)
    return result


# Purpose: Calculate F1 and accuracy from one or many confusion matrices.
# Input: confusions: array ending in (7, 7), true labels on rows.
# Output: Dict of weighted/macro F1, accuracy, and per-class F1 arrays.
def confusion_scores(confusions):
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1)
    support = confusions.sum(axis=-1)
    predicted = confusions.sum(axis=-2)
    total = support.sum(axis=-1)
    denominator = support + predicted
    per_class = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float),
                          where=denominator > 0)
    return {"weighted_f1": (per_class * support).sum(axis=-1) / total,
            "macro_f1": per_class.mean(axis=-1),
            "accuracy": true_positive.sum(axis=-1) / total, "per_class_f1": per_class}


# Purpose: Estimate paired score-difference intervals by resampling dialogues.
# Input: truth/before/after: aligned labels; group_index: dialogue IDs; bootstrap_counts: resample
#          multiplicities by group.
# Output: Dict of 95% percentile intervals for after-minus-before metrics and class F1.
def bootstrap_comparison(truth, before, after, group_index, bootstrap_counts):
    n_groups = bootstrap_counts.shape[1]
    old = grouped_confusion(truth, before, group_index, n_groups)
    new = grouped_confusion(truth, after, group_index, n_groups)
    old_boot = np.tensordot(bootstrap_counts, old, axes=(1, 0))
    new_boot = np.tensordot(bootstrap_counts, new, axes=(1, 0))
    old_score, new_score = confusion_scores(old_boot), confusion_scores(new_boot)
    intervals = {name: np.quantile(new_score[name] - old_score[name], [0.025, 0.975]).tolist()
                 for name in ("weighted_f1", "macro_f1", "accuracy")}
    per_class = np.quantile(new_score["per_class_f1"] - old_score["per_class_f1"], [0.025, 0.975], axis=0)
    intervals["per_class_f1"] = {emotion: per_class[:, index].tolist() for index, emotion in enumerate(EMOTIONS)}
    return intervals

