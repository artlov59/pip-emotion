"""Words-vs-delivery check: calibration math and the delivery-must-lead rule (no models needed)."""
import unittest

import numpy as np

from app import fusion
from app.config import EMOTIONS


def reading(top, conf):
    rest = (1.0 - conf) / (len(EMOTIONS) - 1)
    probs = {e: (conf if e == top else rest) for e in EMOTIONS}
    return {"emotion": top, "confidence": conf, "probs": probs}


PLAIN = {"temperatures": {"audio": 1.0, "text": 1.0, "vision": 1.0},
         "nonverbal_weights": {"audio": 1.0, "vision": 1.0}, "margin": 0.0}


class TemperatureScaling(unittest.TestCase):
    def test_identity_and_argmax(self):
        p = np.array([[0.1, 0.05, 0.05, 0.5, 0.2, 0.05, 0.05]])
        np.testing.assert_allclose(fusion.temperature_scale(p, 1.0), p, atol=1e-9)
        for t in (0.5, 2.0):
            q = fusion.temperature_scale(p, t)
            self.assertEqual(q.argmax(), p.argmax())
            self.assertAlmostEqual(q.sum(), 1.0)
        self.assertGreater(fusion.temperature_scale(p, 0.5).max(), p.max())  # sharpen
        self.assertLess(fusion.temperature_scale(p, 2.0).max(), p.max())     # soften


class AgreementRule(unittest.TestCase):
    def test_delivery_leads_flags_conflict(self):
        r = fusion.agreement(reading("joy", 0.5), reading("anger", 0.8), None, PLAIN)
        self.assertEqual((r["kind"], r["lean"]), ("conflict", "delivery"))

    def test_words_clearer_is_not_flagged(self):
        r = fusion.agreement(reading("joy", 0.8), reading("anger", 0.5), None, PLAIN)
        self.assertEqual((r["kind"], r["lean"]), ("unknown", "words"))

    def test_margin_raises_the_bar(self):
        cal = dict(PLAIN, margin=0.4)
        r = fusion.agreement(reading("joy", 0.5), reading("anger", 0.8), None, cal)
        self.assertEqual(r["lean"], "words")

    def test_agree_and_missing(self):
        self.assertEqual(fusion.agreement(reading("joy", 0.6), reading("joy", 0.3), None, PLAIN)["kind"], "agree")
        missing = fusion.agreement(None, reading("anger", 0.9), None, PLAIN)
        self.assertEqual((missing["kind"], missing["lean"]), ("unknown", None))

    def test_neutral_words_emotional_delivery(self):
        r = fusion.agreement(reading("neutral", 0.4), reading("sadness", 0.7), None, PLAIN)
        self.assertEqual(r["kind"], "tone_only")

    def test_shipped_calibration_loads(self):
        cal = fusion.load_agreement_calibration()
        self.assertTrue(cal["calibrated"], "results/agreement_calibration.json should be committed")
        self.assertEqual(set(cal["temperatures"]), {"audio", "text", "vision"})
        self.assertTrue(all(t > 0 for t in cal["temperatures"].values()))


if __name__ == "__main__":
    unittest.main()
