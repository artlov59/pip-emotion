"""Reply prompt framing and Whisper repetition handling (no models needed)."""
import unittest

from app.asr import REPETITIVE_COMPRESSION_RATIO, compression_ratio
from app.config import EMOTIONS
from app.llm import TemplateResponder, build_messages


def state(transcript, repetitive=False, emotion="joy"):
    probs = {e: (0.7 if e == emotion else 0.05) for e in EMOTIONS}
    reading = {"emotion": emotion, "confidence": 0.7, "probs": probs}
    return {"transcript": transcript, "asr": {"repetitive": repetitive}, "emotion": emotion, "probs": probs,
            "modalities": {"voice": reading, "words": reading, "face": None},
            "delivery": "nothing unusual", "agreement": {"kind": "agree"}}


class TranscriptQuality(unittest.TestCase):
    def test_whisper_loops_exceed_threshold(self):
        for loop in ("S" * 300, "No, " * 40, "I'm sorry. " * 12):
            self.assertGreater(compression_ratio(loop), REPETITIVE_COMPRESSION_RATIO, loop[:20])

    def test_normal_speech_does_not(self):
        for line in ("I just got the internship offer, I can't believe it!", "Hey.", "Okay, okay.", "",
                     "Oh my God, he's lost it, he's totally lost it."):
            self.assertLessEqual(compression_ratio(line), REPETITIVE_COMPRESSION_RATIO, line)


class PromptFraming(unittest.TestCase):
    def test_second_person_and_quoted_transcript(self):
        msgs = build_messages(state("I got the job!"), [])
        user = msgs[-1]["content"]
        self.assertIn('You heard: "I got the job!"', user)
        self.assertNotIn("They said", user)
        self.assertIn('as "you"', msgs[0]["content"])

    def test_repetitive_transcript_is_not_quoted(self):
        msgs = build_messages(state("S" * 300, repetitive=True), [])
        self.assertIn("garbled", msgs[-1]["content"])
        self.assertNotIn("SSSS", msgs[-1]["content"])

    def test_history_masks_repetitive_turns(self):
        history = [{"transcript": "No, " * 40, "repetitive": True, "emotion": "anger", "reply": "Sorry?"},
                   {"transcript": "Fine.", "emotion": "neutral", "reply": "Okay."}]
        msgs = build_messages(state("Hello"), history)
        self.assertIn("garbled", msgs[1]["content"])
        self.assertEqual(msgs[3]["content"], '"Fine." (came across as neutral)')

    def test_template_fallback_still_finds_overall_read(self):
        msgs = build_messages(state("I got the job!", emotion="joy"), [])
        reply = "".join(TemplateResponder().stream(msgs))
        self.assertEqual(reply.strip(), "That's lovely to hear! Tell me more.")


if __name__ == "__main__":
    unittest.main()
