"""Generate a separate, resumable MELD training ASR cache without Metal or downloads.

    python -m training.transcribe_cpu --batch-size 4 --threads 4

This preprocessing backend uses the cached Whisper-small learned weights, with a
different decoder implementation/profile from MLX. Existing MLX caches stay separate.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import MELD_DIR, SAMPLE_RATE
from app.features import load_audio
from app.cpu_asr import BACKEND, CPUWhisperASR
from training.transcribe_meld import edit_distance, words


# Purpose: Hash a local cache or source artifact.
# Input: path: readable file.
# Output: Hexadecimal SHA-256 digest.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Replace a JSON sidecar only after its temporary file is written.
# Input: value: JSON-compatible data; path: destination Path.
# Output: None; atomically replaces the file using strict finite JSON.
def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


# Purpose: Commit transcript rows with an atomic CSV replacement.
# Input: rows: list of transcript records; path: destination Path.
# Output: None; writes a temporary CSV and replaces the destination.
def save_rows(rows, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


# Purpose: Split long utterances, batch-decode pieces, and restore original order.
# Input: asr: loaded transcription backend; clips: (manifest_row, mono_16k_waveform) pairs.
# Output: Transcript/cache records per original clip, with timing, no-speech, chunk, and token-limit
#          metadata.
def transcribe_rows(asr, clips):
    """Batch independent <=30s pieces, then restore each full utterance's order."""
    pieces, owners, durations = [], [], []
    for index, (_, waveform) in enumerate(clips):
        for start in range(0, len(waveform), 30 * SAMPLE_RATE):
            piece = waveform[start:start + 30 * SAMPLE_RATE]
            pieces.append(piece)
            owners.append(index)
            durations.append(len(piece))
    began = time.perf_counter()
    if hasattr(asr, "transcribe_batch"):
        outputs = asr.transcribe_batch(pieces)
    else:
        outputs = [asr.transcribe(piece) for piece in pieces]
    if len(outputs) != len(pieces):
        raise ValueError("ASR returned a different number of pieces")
    wall_ms = (time.perf_counter() - began) * 1000
    rows = []
    for index, (clip, waveform) in enumerate(clips):
        selected = [j for j, owner in enumerate(owners) if owner == index]
        text = " ".join(outputs[j]["text"].strip() for j in selected).strip()
        no_speech = sum(outputs[j]["no_speech_prob"] * durations[j] for j in selected) / len(waveform)
        rows.append({"id": clip.id, "whisper": text, "no_speech_prob": no_speech,
                     "ms": round(wall_ms * len(selected) / len(pieces), 3),
                     "duration_s": len(waveform) / SAMPLE_RATE,
                     "chunks": len(selected),
                     "hit_token_limit": any(outputs[j].get("hit_token_limit", False) for j in selected),
                     "backend": BACKEND})
    return rows


# Purpose: Validate full coverage and commit completed-cache provenance.
# Input: frame: expected manifest rows; rows: cached transcripts; metadata: generation provenance; output:
#          existing cache CSV Path.
# Output: Completed metadata dict; writes its checksum, WER, and quality counts to the sidecar.
def finalize_cache(frame, rows, metadata, output):
    """Commit completion even after interruption between the last CSV and sidecar writes."""
    current_hash = sha256(output)
    if metadata.get("complete") and metadata.get("cache_sha256") != current_hash:
        raise ValueError("Completed cache content changed since its metadata was saved")
    joined = frame[["id", "utterance"]].merge(pd.DataFrame(rows), on="id", validate="one_to_one")
    if len(joined) != len(frame) or len(rows) != len(frame):
        raise ValueError("Cannot mark a partially covered cache complete")
    errors = sum(edit_distance(words(gold), words(asr_text)) for gold, asr_text in zip(joined.utterance, joined.whisper))
    reference_words = sum(len(words(gold)) for gold in joined.utterance)
    result = {**metadata, "complete": True, "n_cached": len(rows), "cache_sha256": current_hash,
              "completed_utc": metadata.get("completed_utc") or datetime.now(timezone.utc).isoformat(),
              "word_error_rate": errors / max(1, reference_words),
              "empty_transcripts": int((joined.whisper.str.strip() == "").sum()),
              "token_limit_hits": int(joined.hit_token_limit.sum())}
    atomic_json(result, Path(str(output) + ".meta.json"))
    return result


# Purpose: Generate or safely resume an offline CPU-ASR cache.
# Input: None explicitly; reads CLI split, shard, batch, thread, and output options.
# Output: None; saves transcripts and provenance, finalizing completion after all expected clips are
#          processed.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error("threads and batch size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard count/index")
    suffix = f".part{args.shard_index}" if args.num_shards > 1 else ""
    output = args.output or MELD_DIR / f"whisper_cpu_{args.split}{suffix}.csv"
    metadata_path = Path(str(output) + ".meta.json")
    manifest_path = MELD_DIR / f"{args.split}.csv"
    frame = pd.read_csv(manifest_path)
    frame = frame[frame.status == "ok"].reset_index(drop=True)
    if frame.empty or not frame.id.is_unique:
        raise ValueError("Manifest must contain unique, usable utterances")
    if output.resolve() in {(MELD_DIR / "whisper_train.csv").resolve(),
                             (MELD_DIR / "whisper_dev.csv").resolve(),
                             (MELD_DIR / "whisper_test.csv").resolve()}:
        raise ValueError("CPU output must not overwrite an MLX cache")
    frame = frame.iloc[args.shard_index::args.num_shards].reset_index(drop=True)
    old_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if output.exists() and not old_metadata:
        raise ValueError("Existing cache has no backend provenance; choose a new output")
    manifest_hash = sha256(manifest_path)
    if old_metadata and (old_metadata.get("backend") != BACKEND or
                         old_metadata.get("split") != args.split or
                         old_metadata.get("manifest_sha256") != manifest_hash or
                         old_metadata.get("num_shards", 1) != args.num_shards or
                         old_metadata.get("shard_index", 0) != args.shard_index):
        raise ValueError("Cannot resume cache from a different backend/split/manifest")
    rows = pd.read_csv(output).fillna({"whisper": ""}).to_dict("records") if output.exists() else []
    done = [row["id"] for row in rows]
    if len(done) != len(set(done)) or set(done) - set(frame.id):
        raise ValueError("Cache has duplicate or unknown utterance IDs")
    remaining = frame[~frame.id.isin(done)]
    if remaining.empty:
        metadata = finalize_cache(frame, rows, old_metadata, output)
        print(json.dumps({"complete": True, "cached": len(rows), "output": str(output),
                          "wer": metadata["word_error_rate"]}, allow_nan=False), flush=True)
        return
    asr = CPUWhisperASR(threads=args.threads)
    asr.load()
    if old_metadata and old_metadata["model"]["source_weights_sha256"] != asr.provenance["source_weights_sha256"]:
        raise ValueError("Whisper source weights changed since cache creation")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"backend": BACKEND, "split": args.split, "manifest_sha256": manifest_hash,
                "script_sha256": sha256(__file__), "model": asr.provenance,
                "n_expected": len(frame), "complete": False,
                "num_shards": args.num_shards, "shard_index": args.shard_index,
                "started_utc": old_metadata.get("started_utc") if old_metadata else datetime.now(timezone.utc).isoformat(),
                "batch_size": args.batch_size,
                "latency_definition": "ms is batch wall time apportioned by number of decoded chunks, not single-clip latency",
                "long_clip_policy": "split >30s utterances into consecutive pieces and concatenate in order",
                "mlx_parity": "not established; CPU cache is separate and MLX dev text remains the selection source"}
    atomic_json(metadata, metadata_path)
    records = list(remaining.itertuples())
    began = time.perf_counter()
    for first in range(0, len(records), args.batch_size):
        clips = [(clip, load_audio(MELD_DIR / clip.wav_path))
                 for clip in records[first:first + args.batch_size]]
        rows.extend(transcribe_rows(asr, clips))
        # Saving every batch makes a long CPU preprocessing run safely resumable.
        save_rows(rows, output)
        if first % (10 * args.batch_size) == 0 or len(rows) == len(frame):
            elapsed = time.perf_counter() - began
            print(json.dumps({"cached": len(rows), "total": len(frame),
                              "new_clips": min(first + len(clips), len(records)),
                              "elapsed_s": round(elapsed, 1),
                              "clips_per_s": round(min(first + len(clips), len(records)) / max(elapsed, 1e-6), 3)}), flush=True)
    metadata = finalize_cache(frame, rows, metadata, output)
    print(json.dumps({"complete": True, "cached": len(rows), "output": str(output),
                      "wer": metadata["word_error_rate"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
