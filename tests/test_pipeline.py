"""Turn-taking / event-flow test for app.pipeline with fake models (no GPU, no downloads).

    python -m tests.test_pipeline

Checks: speech start/end detection, short blips discarded, partial updates during speech,
one final_state + streamed reply per utterance, event ordering, and turn order.
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
from app.vad import EnergyVAD  # noqa: E402


class FakeEmotion:
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def predict(self, wav):
        self.calls.append(wav.size / SAMPLE_RATE)
        time.sleep(0.01)
        probs = {"anger": 0.6, "disgust": 0.02, "fear": 0.03, "joy": 0.05, "neutral": 0.2, "sadness": 0.05, "surprise": 0.05}
        return {"emotion": "anger", "confidence": 0.6, "probs": probs, "ms": 10.0}


class FakeASR:
    repo = "fake"

    def load(self):
        pass

    def transcribe(self, wav):
        time.sleep(0.05)
        return {"text": f"utterance of {wav.size / SAMPLE_RATE:.1f} seconds", "no_speech_prob": 0.0, "ms": 50.0}


def fake_prosody(wav):
    return {"duration_s": wav.size / SAMPLE_RATE, "active_speech_s": wav.size / SAMPLE_RATE,
            "loudness_dbfs": -20.0, "pitch_hz": 150.0, "pitch_variation_st": 3.0}


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


async def main():
    prosody.analyze = fake_prosody
    pipeline.prosody.analyze = fake_prosody
    emotion = FakeEmotion()
    engine = pipeline.Engine(Settings(), emotion=emotion, asr=FakeASR(), responder=TemplateResponder(),
                             vad_factory=EnergyVAD)
    session = pipeline.Session(engine)
    stream = np.concatenate([silence(0.5), tone(2.0), silence(1.0),   # utterance 1 (2 s)
                             tone(0.15), silence(1.0),                # blip -> discarded
                             tone(3.2), silence(1.0)])                # utterance 2 (3.2 s)
    chunk = SAMPLE_RATE // 10
    for i in range(0, stream.size, chunk):
        session.feed(stream[i:i + chunk])
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
        if e["type"] != "response_delta":
            print({k: v for k, v in e.items() if k not in ("probs", "prosody")})

    assert vads == ["speech_start", "speech_end", "speech_start", "discarded", "speech_start", "speech_end"], vads
    assert [f["utterance_id"] for f in finals] == [1, 2]
    assert [d["utterance_id"] for d in dones] == [1, 2]
    assert 1.8 <= finals[0]["duration_s"] <= 2.5, finals[0]["duration_s"]
    assert 3.0 <= finals[1]["duration_s"] <= 3.7, finals[1]["duration_s"]
    assert len([p for p in partials if p["utterance_id"] == 1]) >= 2
    assert len([p for p in partials if p["utterance_id"] == 2]) >= 3
    assert all(p["window_s"] <= 3.0 + 1e-6 for p in partials)
    assert "emotion:anger" in finals[0]["tags"]
    for uid in (1, 2):  # per utterance: final_state before its first delta, done last
        idx = [i for i, e in enumerate(events) if e.get("utterance_id") == uid and e["type"] in
               ("final_state", "response_delta", "response_done")]
        types_ = [events[i]["type"] for i in idx]
        assert types_[0] == "final_state" and types_[-1] == "response_done" and "response_delta" in types_, types_
    assert kinds.index("response_done") < max(i for i, e in enumerate(events) if e["type"] == "final_state")
    print(f"\nOK: {len(partials)} partials, {len(finals)} finals, replies: {[d['text'] for d in dones]}")


if __name__ == "__main__":
    asyncio.run(main())
