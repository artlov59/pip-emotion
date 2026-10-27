"""Replay must wait for accepted turns even before a final state exists."""
import unittest

from app.replay import ReplayProgress


class ReplayProgressTests(unittest.TestCase):
    # Purpose: Verify slow second turn is not lost.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_slow_second_turn_is_not_lost(self):
        progress = ReplayProgress()
        progress.observe({"type": "vad", "state": "speech_end", "utterance_id": 1})
        progress.observe({"type": "vad", "state": "speech_end", "utterance_id": 2})
        progress.observe({"type": "response_done", "utterance_id": 1})
        self.assertFalse(progress.complete)
        progress.observe({"type": "flushed", "utterance_id": 2})
        self.assertFalse(progress.complete)
        progress.observe({"type": "response_done", "utterance_id": 2})
        self.assertTrue(progress.complete)

    # Purpose: Verify no speech requires server acknowledgement.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_no_speech_requires_server_acknowledgement(self):
        progress = ReplayProgress()
        self.assertFalse(progress.complete)
        progress.observe({"type": "flushed", "utterance_id": 0})
        self.assertTrue(progress.complete)
        self.assertFalse(progress.expected)

    # Purpose: Verify final failure is terminal after flush.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_final_failure_is_terminal_after_flush(self):
        progress = ReplayProgress()
        progress.observe({"type": "flushed", "utterance_id": 1})
        self.assertFalse(progress.complete)
        progress.observe({"type": "error", "where": "final", "utterance_id": 1})
        self.assertTrue(progress.complete)
        self.assertEqual(progress.failed, {1})

    # Purpose: Verify done requires flush to rule out more queued turns.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_done_requires_flush_to_rule_out_more_queued_turns(self):
        progress = ReplayProgress()
        progress.observe({"type": "final_state", "utterance_id": 1})
        progress.observe({"type": "response_done", "utterance_id": 1})
        self.assertFalse(progress.complete)
        progress.observe({"type": "flushed", "utterance_id": 1})
        self.assertTrue(progress.complete)


if __name__ == "__main__":
    unittest.main()
