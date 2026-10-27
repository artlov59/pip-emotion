"""Turn-taking / event-flow test for app.pipeline with fake models (no GPU, no downloads).

    python -m tests.test_pipeline

Checks: speech start/end detection, short blips discarded, partial updates during speech,
one final_state + streamed reply per utterance, event ordering, turn order, and the
multimodal path (words + face fused, face frames attributed to the right utterance,
words-vs-delivery conflict detected).
"""
import asyncio
import sys
import time
import types

import numpy as np

# Allow running where librosa is not installed: prosody is replaced by a stub below.
if "librosa" not in sys.modules:
    try:
        import librosa  # noqa: F401
    except ImportError:
        sys.modules["librosa"] = types.ModuleType("librosa")

from app import pipeline, prosody  # noqa: E402
from app.config import SAMPLE_RATE, Settings  # noqa: E402
from app.llm import TemplateResponder  # noqa: E402


class EnergyVAD:
    """Dependency-free stand-in used by tests (not for real use)."""
    FRAME = 512

    # Purpose: Configure a deterministic energy detector for pipeline tests.
    # Input: threshold_dbfs: speech energy cutoff.
    # Output: None; stores the cutoff.
    def __init__(self, threshold_dbfs: float = -40.0):
        self.threshold = threshold_dbfs

    # Purpose: Classify a synthetic frame by RMS energy instead of a learned VAD.
    # Input: frame: mono audio array.
    # Output: 1.0 above the cutoff, otherwise 0.0.
    def prob(self, frame: np.ndarray) -> float:
        db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9)
        return 1.0 if db > self.threshold else 0.0

    # Purpose: Provide the VAD reset interface for a stateless fixture.
    # Input: None.
    # Output: None; no state needs resetting.
    def reset(self):
        pass


class FakeEmotion:
    # Purpose: Track audio durations submitted to the fake classifier.
    # Input: None.
    # Output: None; initializes an empty call list.
    def __init__(self):
        self.calls = []

    # Purpose: Provide the audio model loading interface without weights.
    # Input: None.
    # Output: None; no model is loaded.
    def load(self):
        pass

    # Purpose: Simulate voice inference with a short delay and fixed anger probabilities.
    # Input: wav: mono waveform fixture.
    # Output: Emotion reading dict; also records input duration.
    def predict(self, wav):
        self.calls.append(wav.size / SAMPLE_RATE)
        time.sleep(0.01)
        probs = {"anger": 0.6, "disgust": 0.02, "fear": 0.03, "joy": 0.05, "neutral": 0.2, "sadness": 0.05, "surprise": 0.05}
        return {"emotion": "anger", "confidence": 0.6, "probs": probs, "ms": 10.0}


class FakeASR:
    repo = "fake"

    # Purpose: Provide the speech recognizer loading interface without weights.
    # Input: None.
    # Output: None; no model is loaded.
    def load(self):
        pass

    # Purpose: Simulate transcription and processing delay for a test waveform.
    # Input: wav: mono waveform fixture.
    # Output: Transcript/no-speech/timing dict describing the input duration.
    def transcribe(self, wav):
        time.sleep(0.05)
        return {"text": f"utterance of {wav.size / SAMPLE_RATE:.1f} seconds", "no_speech_prob": 0.0, "ms": 50.0}


class FakeText:
    # Purpose: Provide the text classifier loading interface without weights.
    # Input: None.
    # Output: None; no model is loaded.
    def load(self):
        pass

    # Purpose: Simulate words emotion with a fixed joy result.
    # Input: text: transcript fixture.
    # Output: Joy reading dict, or None when the text is empty.
    def predict(self, text):
        if not text:
            return None
        probs = {"anger": 0.05, "disgust": 0.02, "fear": 0.02, "joy": 0.8, "neutral": 0.06, "sadness": 0.02, "surprise": 0.03}
        return {"emotion": "joy", "confidence": 0.8, "probs": probs, "ms": 3.0}


class FakeVision:
    repo = "fake"

    # Purpose: Provide the face classifier loading interface without weights.
    # Input: None.
    # Output: None; no model is loaded.
    def load(self):
        pass

    # Purpose: Simulate a detected face with fixed anger probabilities.
    # Input: img: ignored image fixture.
    # Output: Face/probs/normalized-box/timing dict after a short processing delay.
    def analyze(self, img):
        time.sleep(0.005)
        probs = {"anger": 0.7, "disgust": 0.05, "fear": 0.05, "joy": 0.02, "neutral": 0.1, "sadness": 0.05, "surprise": 0.03}
        return {"face": True, "probs": probs, "box": [0.3, 0.2, 0.3, 0.4], "ms": 5.0}


# Purpose: Supply deterministic acoustic measurements for pipeline tests.
# Input: wav: mono waveform fixture.
# Output: Prosody dict with input duration and fixed loudness/pitch values.
def fake_prosody(wav):
    return {"duration_s": wav.size / SAMPLE_RATE, "active_speech_s": wav.size / SAMPLE_RATE,
            "loudness_dbfs": -20.0, "pitch_hz": 150.0, "pitch_variation_st": 3.0}


# Purpose: Create synthetic speech-like audio for VAD and turn tests.
# Input: seconds: waveform duration; amp: sine amplitude.
# Output: Float32 mono 220-Hz sine wave at SAMPLE_RATE (16 kHz).
def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# Purpose: Create a silent interval for deterministic turn boundaries.
# Input: seconds: waveform duration.
# Output: Float32 zero waveform at SAMPLE_RATE (16 kHz).
def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


# Purpose: Exercise complete turn-taking and multimodal event flow with fake components.
# Input: None.
# Output: Awaited None; assertions verify boundaries, partial/final states, attribution, and reply ordering.
async def main():
    prosody.analyze = fake_prosody
    pipeline.prosody.analyze = fake_prosody
    emotion = FakeEmotion()
    engine = pipeline.Engine(Settings(), emotion=emotion, asr=FakeASR(), responder=TemplateResponder(),
                             vad_factory=EnergyVAD, text=FakeText(), vision=FakeVision())
    session = pipeline.Session(engine)
    stream = np.concatenate([silence(0.5), tone(2.0), silence(1.0),   # utterance 1 (2 s)
                             tone(0.15), silence(1.0),                # blip -> discarded
                             tone(3.2), silence(1.0)])                # utterance 2 (3.2 s)
    chunk = SAMPLE_RATE // 10
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for k, i in enumerate(range(0, stream.size, chunk)):
        session.feed(stream[i:i + chunk])
        if i / SAMPLE_RATE > 4.5 and k % 2 == 0:  # camera only during the 2nd utterance, ~5 fps
            session.feed_frame(frame)
        await asyncio.sleep(0.02)  # let worker results come back, like a real-time stream would
    session.flush()
    await session.drain(10)
    events = []
    while not session.out.empty():
        events.append(session.out.get_nowait())
    engine.close()

    kinds = [e["type"] for e in events]
    vads = [e["state"] for e in events if e["type"] == "vad"]
    finals = [e for e in events if e["type"] == "final_state"]
    partials = [e for e in events if e["type"] == "partial_state"]
    dones = [e for e in events if e["type"] == "response_done"]
    for e in events:
        if e["type"] not in ("response_delta", "face_state"):
            print({k: v for k, v in e.items() if k not in ("probs", "prosody", "modalities")})

    assert vads == ["speech_start", "speech_end", "speech_start", "discarded", "speech_start", "speech_end"], vads
    assert [f["utterance_id"] for f in finals] == [1, 2]
    assert [d["utterance_id"] for d in dones] == [1, 2]
    assert 1.8 <= finals[0]["duration_s"] <= 2.5, finals[0]["duration_s"]
    assert 3.0 <= finals[1]["duration_s"] <= 3.7, finals[1]["duration_s"]
    assert len([p for p in partials if p["utterance_id"] == 1]) >= 2
    assert len([p for p in partials if p["utterance_id"] == 2]) >= 3
    assert all(p["window_s"] <= 3.0 + 1e-6 for p in partials)
    # utterance 1: no camera -> voice (anger) + words (joy) fused; words vs delivery conflict
    assert finals[0]["modalities"]["face"] is None and finals[0]["modalities"]["words"]["emotion"] == "joy"
    assert set(finals[0]["fusion"]["used"]) == {"audio", "text"}
    assert finals[0]["agreement"]["kind"] == "conflict", finals[0]["agreement"]
    assert "agreement:conflict" in finals[0]["tags"] and "voice:anger" in finals[0]["tags"]
    assert finals[0]["asr"]["repetitive"] is False and "asr:repetitive" not in finals[0]["tags"]
    # utterance 2: camera on -> all three modalities used, face frames counted
    assert set(finals[1]["fusion"]["used"]) == {"audio", "text", "vision"}
    assert finals[1]["modalities"]["face"]["frames"] >= 5, finals[1]["modalities"]["face"]
    assert any(e["type"] == "face_state" and e["face"] for e in events)
    assert any(p["modalities"]["face"] for p in partials if p["utterance_id"] == 2)
    for uid in (1, 2):  # per utterance: final_state before its first delta, done last
        idx = [i for i, e in enumerate(events) if e.get("utterance_id") == uid and e["type"] in
               ("final_state", "response_delta", "response_done")]
        types_ = [events[i]["type"] for i in idx]
        assert types_[0] == "final_state" and types_[-1] == "response_done" and "response_delta" in types_, types_
    assert kinds.index("response_done") < max(i for i, e in enumerate(events) if e["type"] == "final_state")
    print(f"\nOK: {len(partials)} partials, {len(finals)} finals, replies: {[d['text'] for d in dones]}")


if __name__ == "__main__":
    asyncio.run(main())
