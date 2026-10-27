"""Transcribe MELD dev/test clips with the *same* Whisper model and settings as the live server.

    python -m training.transcribe_meld                # dev + test, resumable
    python -m training.transcribe_meld --splits test

Why: at run time the text-emotion model only ever sees Whisper's transcript, not MELD's gold
text. Fusion weights are therefore tuned (dev) and evaluated (test) on Whisper transcripts,
and the gap to gold text is reported as the cost of ASR errors.

Writes data/meld/whisper_<split>.csv and results/whisper_<split>_wer.json.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from app.asr import WhisperASR
from app.config import MELD_DIR, RESULTS_DIR, Settings


# Purpose: Tokenize reference or recognized text for word-error measurement.
# Input: s: text string.
# Output: Lowercase word-token list, retaining apostrophes.
def words(s: str):
    return re.findall(r"[a-z0-9']+", str(s).lower())


# Purpose: Compute Levenshtein distance between two token sequences.
# Input: a/b: reference and recognized token sequences.
# Output: Integer insertion/deletion/substitution edit count.
def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# Purpose: Transcribe a MELD split with MLX Whisper and measure recognition errors.
# Input: None explicitly; reads the CLI split list and uses Settings plus configured dataset/results paths.
# Output: None; saves the transcript CSV and recognition metrics.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()
    asr = WhisperASR(Settings().asr_repo)
    asr.load()
    RESULTS_DIR.mkdir(exist_ok=True)

    for split in args.splits:
        df = pd.read_csv(MELD_DIR / f"{split}.csv")
        df = df[df.status == "ok"]
        out_path = MELD_DIR / f"whisper_{split}.csv"
        done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["id"])
        todo = df[~df.id.isin(set(done.id))]
        rows = []
        for r in tqdm(list(todo.itertuples()), desc=f"whisper {split}"):
            wav, _ = sf.read(str(MELD_DIR / r.wav_path), dtype="float32")
            t = asr.transcribe(wav)
            rows.append({"id": r.id, "whisper": t["text"], "no_speech_prob": t["no_speech_prob"], "ms": t["ms"]})
            if len(rows) % 200 == 0:  # checkpoint progress so the run can be resumed
                pd.concat([done, pd.DataFrame(rows)]).to_csv(out_path, index=False)
        out = pd.concat([done, pd.DataFrame(rows)]) if rows else done
        out.to_csv(out_path, index=False)

        m = df.merge(out, on="id")
        m["whisper"] = m["whisper"].fillna("")
        errs = sum(edit_distance(words(g), words(h)) for g, h in zip(m.utterance, m.whisper))
        n_ref = sum(len(words(g)) for g in m.utterance)
        summary = {"split": split, "n": int(len(m)), "wer": round(errs / max(n_ref, 1), 4),
                   "empty_transcripts": int((m.whisper.str.strip() == "").sum()),
                   "ms_p50": float(np.percentile(m.ms, 50)), "ms_p95": float(np.percentile(m.ms, 95))}
        (RESULTS_DIR / f"whisper_{split}_wer.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
