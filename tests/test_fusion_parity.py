"""Training/serving parity: the runtime fusion code reproduces the deployed dev predictions.

Rebuilds the fused label for every MELD dev clip from the saved per-input probabilities with
app.fusion.fuse + fusion.runtime_weights (the code the live server uses) and checks it against the
predictions recorded when the recipe was selected. Needs the gitignored MELD face cache
(data/meld/vision_dev.csv), so it skips in a fresh clone.
"""
import json
import unittest

import pandas as pd

from app import fusion
from app.config import EMOTIONS, MELD_DIR, RESULTS_DIR

P = [f"p_{e}" for e in EMOTIONS]
AUDIO = RESULTS_DIR / "runtime_dev" / "dev_runtime_audio_predictions.csv"
TEXT = RESULTS_DIR / "retraining_20261023" / "text_selected" / "text_dev_whisper_predictions.csv"
DEPLOYED = RESULTS_DIR / "retraining_20261023" / "dev_comparison" / "dev_multimodal_predictions.csv"
VISION = MELD_DIR / "vision_dev.csv"


def reading(row):
    return None if row.isna().any() else {"probs": dict(zip(EMOTIONS, row.to_numpy(dtype=float)))}


@unittest.skipUnless(VISION.exists(), "needs data/meld/vision_dev.csv from training.extract_faces_meld")
class FusionParity(unittest.TestCase):
    def test_runtime_fusion_reproduces_deployed_dev_predictions(self):
        audio, text = pd.read_csv(AUDIO).set_index("id"), pd.read_csv(TEXT).set_index("id")
        vision, deployed = pd.read_csv(VISION).set_index("id"), pd.read_csv(DEPLOYED).set_index("id")
        spec = json.loads((RESULTS_DIR / "fusion_weights.json").read_text())
        mismatches = []
        for clip in audio.index:
            readings = {"audio": reading(audio.loc[clip, P]), "text": reading(text.loc[clip, P]),
                        "vision": reading(vision.loc[clip, P])}
            available = {m for m, r in readings.items() if r}
            fused = fusion.fuse(readings, fusion.runtime_weights(available, spec))
            if fused["emotion"] != deployed.loc[clip, "audio0_text1"]:
                mismatches.append(clip)
        self.assertEqual(len(audio), 1092)
        self.assertEqual(mismatches, [], f"{len(mismatches)} dev clips differ from the deployed recipe")


if __name__ == "__main__":
    unittest.main()
