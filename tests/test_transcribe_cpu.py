"""Check long-utterance order and resumable cache writes without ASR model loading."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from training.transcribe_cpu import SAMPLE_RATE, save_rows, transcribe_rows


class FakeASR:
    # Purpose: Label each synthetic chunk so reconstruction order can be tested.
    # Input: pieces: audio chunk list.
    # Output: Ordered transcript/no-speech/token-limit dicts, one per piece.
    def transcribe_batch(self, pieces):
        return [{"text": f"part{index}", "no_speech_prob": index / 10,
                 "hit_token_limit": False} for index, _ in enumerate(pieces)]


class TranscriptionTests(unittest.TestCase):
    # Purpose: Verify long clip pieces keep parent order.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_long_clip_pieces_keep_parent_order(self):
        clips = [(SimpleNamespace(id="long"), np.zeros(41 * SAMPLE_RATE, dtype=np.float32)),
                 (SimpleNamespace(id="short"), np.zeros(SAMPLE_RATE, dtype=np.float32))]
        rows = transcribe_rows(FakeASR(), clips)
        self.assertEqual([row["id"] for row in rows], ["long", "short"])
        self.assertEqual(rows[0]["whisper"], "part0 part1")
        self.assertEqual(rows[1]["whisper"], "part2")
        self.assertEqual(rows[0]["chunks"], 2)
        self.assertAlmostEqual(rows[0]["no_speech_prob"], 0.1 * 11 / 41)
        self.assertEqual(rows[0]["duration_s"], 41)

    # Purpose: Verify cache write is complete and replaces atomically.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_cache_write_is_complete_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.csv"
            save_rows([{"id": "a", "whisper": "hello"}], path)
            save_rows([{"id": "a", "whisper": "hello"}, {"id": "b", "whisper": "world"}], path)
            self.assertEqual(pd.read_csv(path).id.tolist(), ["a", "b"])
            self.assertFalse(path.with_suffix(".csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
