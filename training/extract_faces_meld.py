"""Run the face-emotion component over MELD dev/test videos (no training: models are pretrained).

    python -m training.extract_faces_meld            # dev + test
    python -m training.extract_faces_meld --fps 3

For each utterance video: sample ~`fps` frames per second (first 10 s at most), take the largest
face in each frame, classify it, and average over frames with a face. This mirrors the live
system, which receives ~5 webcam frames/s during an utterance.

Writes data/meld/vision_<split>.csv and results/vision_<split>_metrics.json.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.config import DATA_DIR, EMOTIONS, MELD_DIR, RESULTS_DIR
from app.vision import FaceEmotion, mean_probs
from training.metrics_utils import metrics
from training.prepare_meld import find_video_dir

_local = threading.local()


# Purpose: Initialize or reuse one face model per extraction worker thread.
# Input: None explicitly; uses thread-local storage.
# Output: Loaded FaceEmotion instance for the current thread.
def worker_model():
    if not hasattr(_local, "fe"):
        _local.fe = FaceEmotion()
        _local.fe.load()
    return _local.fe


# Purpose: Sample a clip's frames and aggregate facial emotion probabilities.
# Input: video: video Path; fps: frame sampling rate; max_s: maximum sampled duration in seconds.
# Output: Dict containing n_sampled, n_face, and mean probs or None.
def process(video: Path, fps: float, max_s: float):
    import cv2

    if not video.exists():
        return {"n_sampled": 0, "n_face": 0, "probs": None}
    fe = worker_model()
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(src_fps / fps)))
    frames, i = [], 0
    while i < max_s * src_fps:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(fe.analyze(frame))
        i += 1
    cap.release()
    agg = mean_probs(frames)
    return {"n_sampled": len(frames), "n_face": sum(f.get("face", False) for f in frames),
            "probs": agg["probs"] if agg else None}


# Purpose: Build MELD face-probability caches from original videos.
# Input: None explicitly; reads CLI split, sampling, worker, and output options.
# Output: None; saves per-clip face predictions and extraction metadata.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    raw_root = DATA_DIR / "MELD.Raw"
    RESULTS_DIR.mkdir(exist_ok=True)

    for split in args.splits:
        df = pd.read_csv(MELD_DIR / f"{split}.csv")
        df = df[df.status == "ok"].reset_index(drop=True)
        vdir = find_video_dir(raw_root, split)
        videos = [vdir / f"{i}.mp4" for i in df.id]
        with ThreadPoolExecutor(args.workers) as pool:
            res = list(tqdm(pool.map(lambda v: process(v, args.fps, args.max_seconds), videos),
                            total=len(videos), desc=f"faces {split}"))
        out = df[["id", "emotion"]].copy()
        out["n_sampled"] = [r["n_sampled"] for r in res]
        out["n_face"] = [r["n_face"] for r in res]
        for e in EMOTIONS:
            out[f"p_{e}"] = [r["probs"][e] if r["probs"] else np.nan for r in res]
        out.to_csv(MELD_DIR / f"vision_{split}.csv", index=False)

        has = out[f"p_{EMOTIONS[0]}"].notna()
        y = df.label.astype(int).to_numpy()
        probs = out[[f"p_{e}" for e in EMOTIONS]].to_numpy()
        pred = np.where(has, np.nan_to_num(probs).argmax(1), EMOTIONS.index("neutral"))
        m_all = metrics(y, pred)
        m_face = metrics(y[has], pred[has]) if has.any() else None
        summary = {"split": split, "n": int(len(df)), "with_face": int(has.sum()),
                   "face_coverage": round(float(has.mean()), 3),
                   "frames_per_clip_mean": round(float(out.n_sampled.mean()), 2),
                   "all_clips_no_face_as_neutral": m_all, "clips_with_face": m_face}
        (RESULTS_DIR / f"vision_{split}_metrics.json").write_text(json.dumps(summary, indent=2))
        print(f"{split}: face found in {has.sum()}/{len(df)} clips; "
              f"wF1 (face clips) {m_face['weighted_f1']:.3f}; wF1 (all, no-face=neutral) {m_all['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
