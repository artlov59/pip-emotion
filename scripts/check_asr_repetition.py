"""Dev check: should transcripts that fail Whisper's repetition test be dropped from emotion fusion?

    python -m scripts.check_asr_repetition   # -> results/asr_repetition_dev.{md,json}

Whisper flags a decode as failed when its text compresses too well (compression ratio > 2.4), which
catches loops like "No, no, no, ..." or "Sssss...". Pip decodes greedily without Whisper's
temperature fallback, so such loops reach the transcript. This re-scores the deployed recipe on MELD
dev through the runtime fusion code twice: as deployed, and with the words of failing transcripts
treated as missing. The adoption rule is the repo's usual one (dev weighted F1 up, macro F1 not down).
Test transcripts are only counted, after the decision, from the already-reported test predictions.
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app import fusion
from app.asr import REPETITIVE_COMPRESSION_RATIO, compression_ratio
from app.config import EMOTIONS, MELD_DIR, RESULTS_DIR

P = [f"p_{e}" for e in EMOTIONS]
R = RESULTS_DIR / "retraining_20261023"
OUT_MD, OUT_JSON = RESULTS_DIR / "asr_repetition_dev.md", RESULTS_DIR / "asr_repetition_dev.json"


def reading(row):
    return None if row.isna().any() else {"probs": dict(zip(EMOTIONS, row.to_numpy(dtype=float)))}


def scores(gold, pred):
    return {"weighted_f1": round(float(f1_score(gold, pred, average="weighted")), 4),
            "macro_f1": round(float(f1_score(gold, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(gold, pred)), 4)}


def main():
    audio = pd.read_csv(RESULTS_DIR / "runtime_dev" / "dev_runtime_audio_predictions.csv").set_index("id")
    text = pd.read_csv(R / "text_selected" / "text_dev_whisper_predictions.csv").set_index("id")
    vision = pd.read_csv(MELD_DIR / "vision_dev.csv").set_index("id")
    spec = json.loads((RESULTS_DIR / "fusion_weights.json").read_text())
    repetitive = text["text"].fillna("").astype(str).map(compression_ratio) > REPETITIVE_COMPRESSION_RATIO

    def run(drop):
        out = []
        for clip in audio.index:
            r = {"audio": reading(audio.loc[clip, P]), "text": reading(text.loc[clip, P]),
                 "vision": reading(vision.loc[clip, P])}
            if drop and repetitive.loc[clip]:
                r["text"] = None
            out.append(fusion.fuse(r, fusion.runtime_weights({m for m, x in r.items() if x}, spec))["emotion"])
        return pd.Series(out, index=audio.index)

    gold = audio["emotion"]
    deployed, dropped = run(False), run(True)
    before, after = scores(gold, deployed), scores(gold, dropped)
    adopt = after["weighted_f1"] > before["weighted_f1"] and after["macro_f1"] >= before["macro_f1"]
    test_text = pd.read_csv(R / "final_test" / "test_frozen_predictions.csv")["transcript"].fillna("").astype(str)
    report = {"threshold": REPETITIVE_COMPRESSION_RATIO, "dev_clips": int(len(gold)),
              "dev_repetitive": int(repetitive.sum()), "dev_deployed": before, "dev_words_dropped": after,
              "adopt_for_emotion": bool(adopt),
              "test_repetitive_count_descriptive": int((test_text.map(compression_ratio) > REPETITIVE_COMPRESSION_RATIO).sum()),
              "test_clips": int(len(test_text)),
              "decision": "keep words for emotion; mark transcript as garbled in the reply prompt only"}
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Whisper repetition loops (dev check)", "",
             f"{report['dev_repetitive']} of {len(gold)} dev transcripts exceed Whisper's compression-ratio threshold "
             f"({REPETITIVE_COMPRESSION_RATIO}); {report['test_repetitive_count_descriptive']} of {len(test_text)} test "
             "transcripts do (counted after the decision).", "",
             "| dev recipe | weighted F1 | macro F1 | accuracy |", "|---|---:|---:|---:|",
             f"| deployed | {before['weighted_f1']:.4f} | {before['macro_f1']:.4f} | {before['accuracy']:.4f} |",
             f"| words of looping transcripts dropped | {after['weighted_f1']:.4f} | {after['macro_f1']:.4f} | {after['accuracy']:.4f} |",
             "", f"Adopt for emotion: {'yes' if adopt else 'no'}. Decision: {report['decision']}."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
