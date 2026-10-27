"""Benchmark accounting regressions; no model loading, downloads or GPU needed."""
import asyncio
import contextlib
import io
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd

from scripts import benchmark_latency as benchmark


class BenchmarkTests(unittest.TestCase):
    # Purpose: Verify percentiles ignore all nonfinite values.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_percentiles_ignore_all_nonfinite_values(self):
        self.assertEqual(benchmark.pct([None, np.float32('nan'), float('inf'), -float('inf'), 10, 30], 50), 20)
        self.assertIsNone(benchmark.pct([None, float('nan'), float('inf')], 95))

    # Purpose: Verify empty and legacy summaries are strict json.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_empty_and_legacy_summaries_are_strict_json(self):
        empty = benchmark.summarize_latency([])
        self.assertTrue(all(p == {'p50': None, 'p95': None} for p in empty.values()))
        legacy = benchmark.summarize_latency(pd.DataFrame({'asr_ms': [10, float('nan'), 30]}))
        self.assertEqual(legacy['asr_ms']['p50'], 20)
        self.assertIsNone(legacy['face_frame_ms']['p50'])
        json.dumps(legacy, allow_nan=False)
        self.assertIsNone(benchmark.mean_ms([], 'vision'))

    # Purpose: Verify mode and argument guards.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_mode_and_argument_guards(self):
        engine = MagicMock()
        engine.info.return_value = {'multimodal_ready': False}
        with self.assertRaises(RuntimeError):
            benchmark.require_mode(engine, False)
        benchmark.require_mode(engine, True)
        engine.info.return_value = {'multimodal_ready': True}
        benchmark.require_mode(engine, False)
        for argv in [['--n', '0'], ['--n', '-1'], ['--speed', '0'], ['--speed', 'nan'], ['--speed', 'inf']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_args(argv)
        self.assertEqual(benchmark.parse_args(['--n', '1']).n, 1)

    # Purpose: Verify requested count is exact even below seven.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_requested_count_is_exact_even_below_seven(self):
        rows = [{'id': f'{e}{i}', 'status': 'ok', 'duration_s': 2, 'emotion': e}
                for e in benchmark.EMOTIONS for i in range(2)]
        with patch.object(benchmark.pd, 'read_csv', return_value=pd.DataFrame(rows)):
            self.assertEqual(len(benchmark.pick_clips(1, 0)), 1)
            self.assertEqual(len(benchmark.pick_clips(8, 0)), 8)
            self.assertEqual(len(benchmark.pick_clips(35, 0)), 14)

    # Purpose: Verify clip outcomes count drops once and detect partial reply failure.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_clip_outcomes_count_drops_once_and_detect_partial_reply_failure(self):
        stats = {'frames_sent': 10, 'frames_dropped': 3, 'timed_out': False}
        finals = [{'type': 'final_state', 'utterance_id': i, 'fusion': {'used': ['audio', 'text', 'vision']}}
                  for i in [1, 2]]
        dones = [{'type': 'response_done', 'utterance_id': i, 'text': 'hello'} for i in [1, 2]]
        outcome = benchmark.clip_outcome('clip', finals + dones, stats)
        self.assertEqual(outcome['status'], 'complete')
        self.assertEqual(outcome['frames_dropped'], 3)
        self.assertEqual(outcome['n_all_modalities'], 2)
        failed = benchmark.clip_outcome('clip', finals + dones + [{'type': 'error', 'where': 'llm', 'utterance_id': 2}], stats)
        self.assertFalse(failed['reply_complete'])
        self.assertEqual(failed['n_completed_replies'], 1)
        self.assertEqual(benchmark.clip_outcome('silent', [], stats)['status'], 'no_speech')
        pending_failure = benchmark.clip_outcome('clip', finals[:1] + dones[:1] + [
            {'type': 'vad', 'state': 'speech_end', 'utterance_id': 2},
            {'type': 'error', 'where': 'final', 'utterance_id': 2}], stats)
        self.assertFalse(pending_failure['reply_complete'])
        self.assertEqual(pending_failure['n_accepted_turns'], 2)

    # Purpose: Verify load failure cleans up engine and sampler.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_load_failure_cleans_up_engine_and_sampler(self):
        engine, sampler = MagicMock(), MagicMock()
        engine.load.side_effect = RuntimeError('load failed')
        with patch.object(benchmark, 'Engine', return_value=engine), patch.object(benchmark, 'MemorySampler', return_value=sampler), \
                patch.object(benchmark, 'hardware', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                asyncio.run(benchmark.main_async(benchmark.parse_args(['--audio-only'])))
        engine.close.assert_called_once()
        self.assertTrue(sampler.stop_flag)
        sampler.join.assert_called_once()

    # Purpose: Verify run clip handles empty video and closes session.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_run_clip_handles_empty_video_and_closes_session(self):
        session = MagicMock()
        session.out = asyncio.Queue()
        session.tasks = set()
        session.frames_in = session.frames_dropped = 0
        session.drain = AsyncMock()
        with patch.object(benchmark, 'Session', return_value=session), patch.object(benchmark.asyncio, 'sleep', new=AsyncMock()):
            events, stats = asyncio.run(benchmark.run_clip(MagicMock(), np.zeros(0, dtype=np.float32), 1, []))
        self.assertEqual(events, [])
        self.assertEqual(stats, {'frames_sent': 0, 'frames_dropped': 0, 'timed_out': False})
        session.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
