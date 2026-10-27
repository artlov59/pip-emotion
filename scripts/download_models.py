"""Fetch the model files that are too large for git, then verify them.

    python -m scripts.download_models

* checkpoints/text_emotion/  - the selected DistilRoBERTa text-emotion model (~330 MB), from this
  repository's GitHub release. Every file is checked against the sha256 recorded in
  results/submission_model_selection.json, so a wrong or partial download fails loudly.
* checkpoints/face_detection_yunet_2023mar.onnx - the YuNet face detector (OpenCV's, ~230 KB).

The audio CNN (checkpoints/emotion_cnn.pt) is committed to git. Whisper and Qwen are cached by
`python -m scripts.cache_models`; HSEmotion downloads its own weights on first load.
Set PIP_MODEL_URL to use a different location for the text-model archive.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from app.config import CHECKPOINT_DIR, RESULTS_DIR
from app.vision import YUNET_PATH, YUNET_URL

GITHUB_REPO = "artlov59/pip-emotion"  # change here if the repository is renamed
RELEASE_TAG = "models-v1"
ARCHIVE = "text_emotion.tar.gz"
ARCHIVE_SHA256 = "1395d52f23017be813708488c0295280ba3097d514966025672178e65b4815dd"
TEXT_URL = os.environ.get("PIP_MODEL_URL",
                          f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{ARCHIVE}")
TEXT_DIR = CHECKPOINT_DIR / "text_emotion"
RECEIPT = RESULTS_DIR / "submission_model_selection.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_text_files() -> dict:
    return json.loads(RECEIPT.read_text())["selected"]["text"]["files"]


def text_model_ok() -> bool:
    return all((TEXT_DIR / name).is_file() and sha256(TEXT_DIR / name) == digest
               for name, digest in expected_text_files().items())


def download(url: str, dest: Path):
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 2**20:6.1f} / {total / 2**20:.1f} MiB", end="", flush=True)
    print()


def fetch_text_model(force: bool):
    if not force and text_model_ok():
        print(f"text model OK: {TEXT_DIR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / ARCHIVE
        download(TEXT_URL, archive)
        if "PIP_MODEL_URL" not in os.environ and sha256(archive) != ARCHIVE_SHA256:
            sys.exit(f"archive checksum mismatch for {TEXT_URL}; not extracting")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        if TEXT_DIR.exists():
            shutil.rmtree(TEXT_DIR)
        shutil.move(str(Path(tmp) / "text_emotion"), TEXT_DIR)
    if not text_model_ok():
        sys.exit("text model files do not match results/submission_model_selection.json")
    print(f"text model OK: {TEXT_DIR}")


def fetch_face_detector(force: bool):
    if YUNET_PATH.exists() and not force:
        print(f"face detector OK: {YUNET_PATH}")
        return
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    download(YUNET_URL, YUNET_PATH)
    print(f"face detector OK: {YUNET_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Download and verify model files that are not stored in git.")
    ap.add_argument("--force", action="store_true", help="download again even if files are present")
    args = ap.parse_args()
    fetch_text_model(args.force)
    fetch_face_detector(args.force)


if __name__ == "__main__":
    main()
