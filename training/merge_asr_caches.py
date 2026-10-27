"""Validate and merge completed CPU ASR shards; never mixes MLX and CPU caches."""
import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import MELD_DIR
from app.cpu_asr import BACKEND
from training.transcribe_cpu import atomic_json, save_rows, sha256


# Purpose: Validate and combine CPU-ASR shards against one MELD split.
# Input: parts: shard CSV paths; output: destination CSV Path; split: target manifest split.
# Output: None; saves the merged transcript cache and finalized provenance sidecar.
def merge(parts, output, split="train"):
    manifest_path = MELD_DIR / f"{split}.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.status == "ok"].reset_index(drop=True)
    records, metadata = [], []
    for part in parts:
        part = Path(part)
        meta = json.loads(Path(str(part) + ".meta.json").read_text())
        if not meta.get("complete") or meta.get("backend") != BACKEND or meta.get("split") != split:
            raise ValueError(f"Incomplete/wrong-backend shard: {part}")
        if meta["manifest_sha256"] != sha256(manifest_path) or meta["cache_sha256"] != sha256(part):
            raise ValueError(f"Shard or manifest hash changed: {part}")
        records.extend(pd.read_csv(part).fillna({"whisper": ""}).to_dict("records"))
        metadata.append({"path": str(part.resolve()), "sha256": sha256(part), "metadata": meta})
    source_hashes = {item["metadata"]["model"]["source_weights_sha256"] for item in metadata}
    if len(source_hashes) != 1:
        raise ValueError("Shards use different source weights")
    frame = pd.DataFrame(records)
    if not frame.id.is_unique or set(frame.id) != set(manifest.id):
        raise ValueError("Merged cache must cover every usable utterance exactly once")
    ordered = manifest[["id"]].merge(frame, on="id", validate="one_to_one")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rows(ordered.to_dict("records"), output)
    combined = {"complete": True, "split": split, "backend": BACKEND,
                "model": metadata[0]["metadata"]["model"],
                "manifest_sha256": sha256(manifest_path), "cache_sha256": sha256(output),
                "n_expected": len(manifest), "n_cached": len(ordered),
                "parts": metadata, "mlx_parity_verified": False}
    atomic_json(combined, output.with_suffix(".meta.json"))
    print(json.dumps({"complete": True, "output": str(output), "n": len(ordered)}), flush=True)


# Purpose: Parse ASR shard paths and merge them.
# Input: None explicitly; reads command-line shard, output, and split options.
# Output: None; writes the validated combined cache.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--output", type=Path, default=MELD_DIR / "whisper_cpu_train.csv")
    args = parser.parse_args()
    merge(args.parts, args.output, args.split)


if __name__ == "__main__":
    main()
