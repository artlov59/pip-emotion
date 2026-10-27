"""Late fusion + cross-modal agreement, shared by the live server.

Frozen weights were tuned on MELD dev. The runtime rule is:
    score_c = sum over available modalities m of  w_m * log p_m(c)
The fused distribution is softmax(score). A missing modality (no words, no face) is skipped.
"""
import json

import numpy as np

from app.config import EMOTIONS, RESULTS_DIR

EPS = 1e-4
DEFAULT_WEIGHTS = {"audio": 1.0, "text": 1.0, "vision": 0.5}  # used only if fusion was never tuned
VALENCE = {"joy": "positive", "anger": "negative", "disgust": "negative", "fear": "negative",
           "sadness": "negative", "neutral": "neutral", "surprise": "surprise"}


# Purpose: Load saved development fusion weights, or use defaults if absent.
# Input: None explicitly; reads the configured fusion-weight file.
# Output: Tuple (live_weights, all_subset_tables, loaded_from_file).
def load_weights():
    path = RESULTS_DIR / "fusion_weights.json"
    if path.exists():
        d = json.loads(path.read_text())
        return {k: float(v) for k, v in d["live"].items()}, d.get("all", {}), True
    return dict(DEFAULT_WEIGHTS), {}, False


# Purpose: Combine available modality distributions using weighted log probabilities.
# Input: readings: audio/text/vision reading dicts or None; weights: modality coefficients.
# Output: Fused emotion/confidence/probs/used dict, or None when no weighted reading exists.
def fuse(readings: dict, weights: dict):
    """readings: {"audio": {...probs...} | None, "text": ..., "vision": ...} -> fused reading or None."""
    score = np.zeros(len(EMOTIONS))
    used = []
    for m, r in readings.items():
        w = weights.get(m, 0.0)
        if r is None or w == 0:
            continue
        p = np.array([r["probs"][e] for e in EMOTIONS], dtype=np.float64)
        score += w * np.log(np.clip(p, EPS, 1.0))
        used.append(m)
    if not used:
        return None
    p = np.exp(score - score.max())
    p /= p.sum()
    i = int(p.argmax())
    return {"emotion": EMOTIONS[i], "confidence": round(float(p[i]), 4),
            "probs": {e: round(float(v), 4) for e, v in zip(EMOTIONS, p)}, "used": used}


# Purpose: Compare words with voice/face delivery for agreement or conflict.
# Input: words/voice/face: optional emotion readings; min_conf: confidence threshold.
# Output: Dict describing agreement kind, word/nonverbal labels, and a note when known.
def agreement(words: dict, voice: dict, face: dict, min_conf: float = 0.4) -> dict:
    """How the *words* relate to the *non-verbal* signals (voice + face).

    kinds: "agree", "conflict" (opposite valence, e.g. positive words said angrily),
    "tone_only" (neutral words, emotional delivery), "words_only" (emotional words, neutral
    delivery), "unknown" (a side is missing or too unsure).
    """
    nonverbal = fuse({"audio": voice, "vision": face}, {"audio": 1.0, "vision": 1.0})
    if words is None or nonverbal is None:
        return {"kind": "unknown", "words": words and words["emotion"], "nonverbal": nonverbal and nonverbal["emotion"]}
    w, n = words["emotion"], nonverbal["emotion"]
    vw, vn = VALENCE[w], VALENCE[n]
    confident = words["confidence"] >= min_conf and nonverbal["confidence"] >= min_conf
    if w == n:
        kind = "agree"
    elif not confident:
        kind = "unknown"
    elif {vw, vn} == {"positive", "negative"}:
        kind = "conflict"
    elif vw == "neutral" and vn != "neutral":
        kind = "tone_only"
    elif vn == "neutral" and vw != "neutral":
        kind = "words_only"
    else:
        kind = "differ"
    notes = {
        "agree": f"words and delivery agree ({w})",
        "conflict": f"the words sound {w} but voice/face read {n}: possibly sarcasm or masking",
        "tone_only": f"the words are neutral but the delivery reads {n}",
        "words_only": f"the words express {w} but the delivery is flat",
        "differ": f"words suggest {w}, delivery suggests {n}",
        "unknown": "not enough confidence to compare words and delivery",
    }
    return {"kind": kind, "words": w, "nonverbal": n, "note": notes[kind]}


# Purpose: Choose saved weights for the exact set of available modalities.
# Input: available: modality names; specification: live weights and saved subset tables.
# Output: Weight dict with text_whisper normalized to the runtime name text.
def runtime_weights(available, specification):
    # Same subset lookup as Engine.weights_for; text_whisper is the offline key.
    key = "+".join("text_whisper" if mode == "text" else mode
                   for mode in ("audio", "text", "vision") if mode in available)
    chosen = specification.get("all", {}).get(key)
    if chosen:
        return {name.replace("text_whisper", "text"): float(value) for name, value in chosen.items()}
    return {name: float(specification["live"].get(name, 0.0)) for name in available}

