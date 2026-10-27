"""Hand-computed prosody (no learned parameters): pitch, loudness, pace.

Used two ways: as structured tags in the state output, and as a plain-language
"delivery" description in the prompt for the reply generator, so the reply can
react to *how* something was said, not only to the emotion label.
"""
import re

import librosa
import numpy as np

SR = 16000
HOP = 160


# Purpose: Measure acoustic delivery without classifying emotion.
# Input: wav: mono 16-kHz NumPy waveform.
# Output: Dict of duration/active speech seconds, loudness dBFS, pitch Hz, and pitch variation semitones;
#          unavailable values are None.
def analyze(wav: np.ndarray) -> dict:
    wav = wav.astype(np.float32)
    if wav.size < 2048:
        wav = np.pad(wav, (0, 2048 - wav.size))
    rms = librosa.feature.rms(y=wav, frame_length=400, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-9)
    active = db > max(db.max() - 30.0, -60.0)          # frames within 30 dB of the peak
    f0 = librosa.yin(wav, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(len(f0), len(active))
    f0, active_f = f0[:n], active[:n]
    voiced = f0[active_f & (f0 > 72) & (f0 < 395)]     # yin saturates at fmin/fmax on unvoiced frames
    pitch = float(np.median(voiced)) if voiced.size >= 10 else None
    pitch_var = float(np.std(12 * np.log2(voiced / pitch))) if pitch else None
    return {
        "duration_s": round(wav.size / SR, 2),
        "active_speech_s": round(float(active.sum()) * HOP / SR, 2),
        "loudness_dbfs": round(float(db[active].mean()), 1) if active.any() else None,
        "pitch_hz": round(pitch, 1) if pitch else None,
        "pitch_variation_st": round(pitch_var, 2) if pitch_var is not None else None,
    }


# Purpose: Add transcript word count and estimated speaking rate to prosody.
# Input: p: prosody dict; transcript: recognized text.
# Output: The same dict, updated with word/rate measurements.
def add_speech_rate(p: dict, transcript: str) -> dict:
    words = len(re.findall(r"[A-Za-z0-9']+", transcript or ""))
    p["words"] = words
    p["rate_wps"] = round(words / max(p["active_speech_s"], 0.3), 2) if words else None
    return p


class ProsodyBaseline:
    """Per-session running reference, so 'loud' / 'high pitch' mean relative to *this* speaker.

    Before two utterances have been heard, only speaker-independent cues are labelled
    (pitch variation, pace, and extreme loudness).
    """

    # Purpose: Create a rolling baseline for speaker-relative delivery tags.
    # Input: keep: maximum number of previous measurements to retain.
    # Output: None; initializes bounded measurement histories.
    def __init__(self, keep: int = 20):
        self.keep = keep
        self.loud, self.pitch = [], []

    # Purpose: Compare current prosody with the rolling speaker baseline.
    # Input: p: current prosody measurement dict.
    # Output: Dict of delivery labels based on loudness, pitch, variation, and speaking rate.
    def describe(self, p: dict) -> dict:
        labels = {"loudness": None, "pitch": None, "pitch_variation": None, "pace": None}
        if p.get("loudness_dbfs") is not None:
            if len(self.loud) >= 2:
                ref = float(np.median(self.loud))
                d = p["loudness_dbfs"] - ref
                labels["loudness"] = "loud" if d > 4 else "quiet" if d < -4 else "normal"
            else:
                v = p["loudness_dbfs"]
                labels["loudness"] = "loud" if v > -12 else "quiet" if v < -42 else "normal"
        if p.get("pitch_hz") and len(self.pitch) >= 2:
            st = 12 * np.log2(p["pitch_hz"] / float(np.median(self.pitch)))
            labels["pitch"] = "high" if st > 2 else "low" if st < -2 else "normal"
        if p.get("pitch_variation_st") is not None:
            v = p["pitch_variation_st"]
            labels["pitch_variation"] = "animated" if v > 4.0 else "flat" if v < 1.5 else "normal"
        if p.get("rate_wps"):
            v = p["rate_wps"]
            labels["pace"] = "fast" if v > 3.6 else "slow" if v < 1.8 else "normal"
        return labels

    # Purpose: Add usable prosody measurements to the rolling baseline.
    # Input: p: current prosody measurement dict.
    # Output: None; appends available measurements to bounded histories.
    def update(self, p: dict):
        if p.get("loudness_dbfs") is not None:
            self.loud = (self.loud + [p["loudness_dbfs"]])[-self.keep:]
        if p.get("pitch_hz"):
            self.pitch = (self.pitch + [p["pitch_hz"]])[-self.keep:]


# Purpose: Turn delivery labels into readable prompt text.
# Input: labels: delivery-tag dict from ProsodyBaseline.describe().
# Output: Comma-separated delivery description, or 'nothing unusual'.
def describe_delivery(labels: dict) -> str:
    phrases = {
        ("loudness", "loud"): "louder than usual", ("loudness", "quiet"): "quieter than usual",
        ("pitch", "high"): "higher-pitched than usual", ("pitch", "low"): "lower-pitched than usual",
        ("pitch_variation", "animated"): "animated, varied intonation", ("pitch_variation", "flat"): "flat, monotone intonation",
        ("pace", "fast"): "speaking fast", ("pace", "slow"): "speaking slowly",
    }
    parts = [phrases[(k, v)] for k, v in labels.items() if (k, v) in phrases]
    return ", ".join(parts) if parts else "nothing unusual"
