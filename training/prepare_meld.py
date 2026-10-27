"""Prepare MELD for audio training.

    python -m training.prepare_meld                       # uses data/MELD.Raw.tar.gz
    python -m training.prepare_meld --raw path/to/MELD.Raw  # already-extracted folder

Steps (each is skipped if its output already exists, so the script can be re-run):
  1. extract MELD.Raw.tar.gz and the inner train/dev/test archives
  2. ffmpeg each utterance .mp4 -> 16 kHz mono 16-bit wav  (data/meld/wav/<split>/)
  3. compute log-mel features for every wav               (data/meld/features/<split>.pkl)
  4. write manifests                                      (data/meld/<split>.csv)
  5. write a prep report with counts and failures         (data/meld/prep_report.json)
"""
import argparse
import json
import os
import pickle
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from app.config import DATA_DIR, EMOTIONS, MELD_DIR, SAMPLE_RATE
from app.features import log_mel

SPLITS = ["train", "dev", "test"]
CSV_NAMES = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
# Folder names inside the official inner archives, with fallbacks.
VIDEO_DIRS = {
    "train": ["train_splits"],
    "dev": ["dev_splits_complete", "dev_splits"],
    "test": ["output_repeated_splits_test", "test_splits"],
}
MOJIBAKE = {"â€™": "'", "â€˜": "'", "â€œ": '"',
            "â€\x9d": '"', "â€¦": "...", "â€”": "-",
            "â€“": "-", "\u0092": "'", "\u0085": "...", "\u0097": "-", "\u0093": '"', "\u0094": '"',
            "\ufffd": "'"}


# Purpose: Repair MELD text encoding artifacts and normalize whitespace.
# Input: s: utterance string.
# Output: Cleaned text with collapsed whitespace.
def clean_text(s: str) -> str:
    s = str(s)
    for bad, good in MOJIBAKE.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


# Purpose: Unpack an existing tar archive into a destination directory.
# Input: archive: tar Path; dest: extraction directory Path.
# Output: None; extracts archive contents.
def extract(archive: Path, dest: Path):
    print(f"Extracting {archive.name} -> {dest} (this can take several minutes)", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(dest)], check=True)


# Purpose: Find MELD.Raw, extracting outer and split archives when necessary.
# Input: raw: directory containing downloaded MELD archives or extracted files.
# Output: MELD.Raw directory Path; may create extracted dataset files.
def find_raw_root(raw: Path) -> Path:
    """Return the MELD.Raw folder, extracting the outer and inner archives as needed.

    Official layout: MELD.Raw.tar.gz -> MELD.Raw/{dev,test}_sent_emo.csv + {train,dev,test}.tar.gz;
    train_sent_emo.csv ships *inside* train.tar.gz next to train_splits/.
    """
    if raw.is_file():
        extracted = DATA_DIR / "MELD.Raw"
        if not (extracted.exists() and (any(extracted.rglob("*_sent_emo.csv")) or any(extracted.glob("*.tar.gz")))):
            extract(raw, DATA_DIR)
        raw = extracted if extracted.exists() else DATA_DIR
    markers = sorted(raw.rglob("*_sent_emo.csv")) + sorted(raw.rglob("train.tar.gz"))
    if not markers:
        raise SystemExit(f"Could not find MELD csv files or inner archives under {raw}")
    root = markers[0].parent
    for split in SPLITS:  # inner archives: train.tar.gz / dev.tar.gz / test.tar.gz
        if find_video_dir(root, split) is None or find_csv(root, split) is None:
            inner = root / f"{split}.tar.gz"
            if inner.exists():
                extract(inner, root)
    return root


# Purpose: Find the source utterance-label CSV for a dataset split.
# Input: root: extracted MELD root; split: train/dev/test.
# Output: Matching CSV Path, or None.
def find_csv(root: Path, split: str):
    hits = sorted(p for p in root.rglob(CSV_NAMES[split]) if not p.name.startswith("._"))
    return hits[0] if hits else None


# Purpose: Find the extracted clip directory for a MELD split.
# Input: root: extracted MELD root; split: train/dev/test.
# Output: Video directory Path, or None.
def find_video_dir(root: Path, split: str):
    for name in VIDEO_DIRS[split]:
        for d in root.rglob(name):
            if d.is_dir() and not d.name.startswith("._"):
                return d
    # fallback: any folder with the split name in it that holds many diaX_uttY.mp4 files
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and split in p.name.lower()):
        if sum(1 for _ in d.glob("dia*_utt*.mp4")) > 100:
            return d
    return None


# Purpose: Convert a source clip into a mono 16-kHz WAV.
# Input: src: input video Path; dst: output WAV Path.
# Output: Status string: ok, missing_video, or ffmpeg_failed; writes WAV when conversion succeeds.
def convert(src: Path, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size > 1000:
        return "ok"
    if not src.exists():
        return "missing_video"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)],
                       capture_output=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 1000:
        tmp.unlink(missing_ok=True)
        return "ffmpeg_failed"
    tmp.rename(dst)
    return "ok"


# Purpose: Prepare MELD audio, manifests, and cached log-mel features.
# Input: None explicitly; reads CLI raw-data, split, and worker options.
# Output: None; writes dataset artifacts and preparation statistics.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DATA_DIR / "MELD.Raw.tar.gz"),
                    help="MELD.Raw.tar.gz or an extracted MELD.Raw folder")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--min-duration", type=float, default=0.2, help="drop clips shorter than this (s)")
    args = ap.parse_args()

    root = find_raw_root(Path(args.raw))
    print(f"MELD root: {root}")
    report = {"splits": {}}

    for split in SPLITS:
        csv_path = find_csv(root, split)
        if csv_path is None:
            raise SystemExit(f"No {CSV_NAMES[split]} found under {root}")
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
        video_dir = find_video_dir(root, split)
        if video_dir is None:
            raise SystemExit(f"No video folder found for split '{split}' under {root}")
        df["id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
        df["emotion"] = df["Emotion"].str.strip().str.lower()
        df["label"] = df["emotion"].map({e: i for i, e in enumerate(EMOTIONS)})
        df["utterance"] = df["Utterance"].map(clean_text)
        wav_dir = MELD_DIR / "wav" / split
        srcs = [video_dir / f"{i}.mp4" for i in df["id"]]
        dsts = [wav_dir / f"{i}.wav" for i in df["id"]]

        with ThreadPoolExecutor(args.workers) as pool:
            status = list(tqdm(pool.map(convert, srcs, dsts), total=len(df), desc=f"{split}: mp4->wav"))
        df["status"] = status
        df["wav_path"] = [str(d.relative_to(MELD_DIR)) for d in dsts]

        feats, durations = {}, []
        for i, row in tqdm(list(df.iterrows()), desc=f"{split}: log-mel"):
            if row["status"] != "ok":
                durations.append(np.nan)
                continue
            try:
                wav, sr = sf.read(str(MELD_DIR / row["wav_path"]), dtype="float32")
                assert sr == SAMPLE_RATE
            except Exception:
                df.at[i, "status"] = "unreadable"
                durations.append(np.nan)
                continue
            dur = len(wav) / SAMPLE_RATE
            durations.append(dur)
            if dur < args.min_duration:
                df.at[i, "status"] = "too_short"
                continue
            feats[row["id"]] = log_mel(wav).astype(np.float16)
        df["duration_s"] = durations

        (MELD_DIR / "features").mkdir(parents=True, exist_ok=True)
        with open(MELD_DIR / "features" / f"{split}.pkl", "wb") as f:
            pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        cols = ["id", "Dialogue_ID", "Utterance_ID", "Speaker", "utterance", "emotion", "label",
                "Sentiment", "duration_s", "status", "wav_path"]
        df[cols].to_csv(MELD_DIR / f"{split}.csv", index=False)

        ok = df[df["status"] == "ok"]
        report["splits"][split] = {
            "rows": int(len(df)),
            "usable": int(len(ok)),
            "status_counts": df["status"].value_counts().to_dict(),
            "failed_ids": df.loc[df["status"] != "ok", "id"].tolist(),
            "emotion_counts": ok["emotion"].value_counts().to_dict(),
            "duration_s": {"mean": round(float(ok["duration_s"].mean()), 2),
                           "median": round(float(ok["duration_s"].median()), 2),
                           "p95": round(float(ok["duration_s"].quantile(0.95)), 2),
                           "max": round(float(ok["duration_s"].max()), 2)},
        }
        print(json.dumps({k: v for k, v in report["splits"][split].items() if k != "failed_ids"}, indent=2))

    with open(MELD_DIR / "prep_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Done. Report: {MELD_DIR / 'prep_report.json'}")


if __name__ == "__main__":
    main()
