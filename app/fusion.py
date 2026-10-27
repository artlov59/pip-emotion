"""Late fusion + cross-modal agreement, shared by the live server.

Frozen weights were tuned on MELD dev. The runtime rule is:
    score_c = sum over available modalities m of  w_m * log p_m(c)
The fused distribution is softmax(score). A missing modality (no words, no face) is skipped.

The words-vs-delivery check (agreement) is separate and does not change the fused emotion.
It compares temperature-calibrated words and voice+face readings; temperatures and the
decision margin were fitted on MELD dev by scripts/calibrate_agreement.py.
"""
import json

import numpy as np

from app.config import EMOTIONS, RESULTS_DIR

EPS = 1e-4
DEFAULT_WEIGHTS = {"audio": 1.0, "text": 1.0, "vision": 0.5}  # used only if fusion was never tuned
VALENCE = {"joy": "positive", "anger": "negative", "disgust": "negative", "fear": "negative",
           "sadness": "negative", "neutral": "neutral", "surprise": "surprise"}
AGREEMENT_FILE = RESULTS_DIR / "agreement_calibration.json"
# Used only if results/agreement_calibration.json is missing: no calibration, delivery must simply
# be at least as confident as the words.
DEFAULT_AGREEMENT = {"temperatures": {"audio": 1.0, "text": 1.0, "vision": 1.0},
                     "nonverbal_weights": {"audio": 1.0, "vision": 1.0}, "margin": 0.0, "calibrated": False}


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


def load_agreement_calibration(path=AGREEMENT_FILE) -> dict:
    """Dev-fitted temperatures, voice/face weights and margin for agreement(), or uncalibrated defaults."""
    if not path.exists():
        return dict(DEFAULT_AGREEMENT)
    d = json.loads(path.read_text())
    return {"temperatures": {k: float(v) for k, v in d["temperatures"].items()},
            "nonverbal_weights": {k: float(v) for k, v in d["nonverbal_weights"].items()},
            "margin": float(d["margin"]), "calibrated": True}


def temperature_scale(probs, temperature: float):
    """Temperature-scale probabilities, p_T(c) ∝ p(c)^(1/T). Works on (7,) or (n, 7) arrays.

    T > 1 softens an over-confident model, T < 1 sharpens an under-confident one; the
    top class never changes.
    """
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def calibrate(reading, temperature: float):
    """Return a copy of a modality reading with temperature-scaled probabilities, or None."""
    if reading is None:
        return None
    p = temperature_scale([reading["probs"][e] for e in EMOTIONS], temperature)
    i = int(p.argmax())
    return {"emotion": EMOTIONS[i], "confidence": float(p[i]), "probs": dict(zip(EMOTIONS, p.tolist()))}


def agreement(words: dict, voice: dict, face: dict, calibration: dict = None) -> dict:
    """How the *words* relate to the *non-verbal* signals (voice + face).

    Both sides are temperature-calibrated first, so their confidences are comparable. When they
    disagree, the delivery is flagged only if its calibrated confidence is at least the words'
    confidence plus a dev-tuned margin; otherwise the words are treated as the clearer signal.
    On MELD dev/test the uncalibrated rule this replaces flagged ~37% of turns, and in those
    the words were right more often than the delivery (results/agreement_calibration*.md).

    kinds: "agree"; when the delivery leads: "conflict" (opposite valence, e.g. positive words
    said angrily), "tone_only" (neutral words, emotional delivery), "words_only" (emotional words,
    flat delivery), "differ"; "unknown" when a side is missing or the words are the clearer signal.
    lean: None (agree/missing), "words" or "delivery".
    """
    cal = calibration or DEFAULT_AGREEMENT
    temps = cal["temperatures"]
    words_c = calibrate(words, temps.get("text", 1.0))
    nonverbal = fuse({"audio": calibrate(voice, temps.get("audio", 1.0)),
                      "vision": calibrate(face, temps.get("vision", 1.0))}, cal["nonverbal_weights"])
    if words_c is None or nonverbal is None:
        return {"kind": "unknown", "lean": None, "words": words_c and words_c["emotion"],
                "nonverbal": nonverbal and nonverbal["emotion"],
                "note": "words or delivery missing, nothing to compare"}
    w, n = words_c["emotion"], nonverbal["emotion"]
    cw, cn = words_c["confidence"], nonverbal["confidence"]
    vw, vn = VALENCE[w], VALENCE[n]
    lean = None
    if w == n:
        kind = "agree"
    elif cn < cw + cal["margin"]:
        kind, lean = "unknown", "words"
    else:
        lean = "delivery"
        if {vw, vn} == {"positive", "negative"}:
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
        "unknown": f"words suggest {w}, delivery suggests {n}; the words are the clearer signal",
    }
    return {"kind": kind, "lean": lean, "words": w, "nonverbal": n, "words_conf": round(cw, 4),
            "delivery_conf": round(cn, 4), "note": notes[kind]}


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

