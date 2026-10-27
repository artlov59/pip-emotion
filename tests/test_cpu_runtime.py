"""Offline CPU runtime regressions: synthetic values and mocks; no model loading."""
import builtins
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.config import Settings
from app import backends
from app.cpu_llm import CPUResponder, dequantize_affine_4bit
from app.local_models import PINNED_REVISIONS, cached_model_dir, file_sha256
from scripts import benchmark_latency as benchmark


class CacheTests(unittest.TestCase):
    # Purpose: Verify explicit directory and file hash.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_explicit_directory_and_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(cached_model_dir(root), root.resolve())
            (root / 'a').write_bytes(b'abc')
            self.assertEqual(file_sha256(root / 'a'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    # Purpose: Verify pinned revision ignores different main and missing fails.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_pinned_revision_ignores_different_main_and_missing_fails(self):
        repo = 'mlx-community/Qwen2.5-3B-Instruct-4bit'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'HF_HUB_CACHE': tmp}):
            root = Path(tmp) / 'models--mlx-community--Qwen2.5-3B-Instruct-4bit'
            (root / 'refs').mkdir(parents=True)
            (root / 'refs' / 'main').write_text('other')
            (root / 'snapshots' / 'other').mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                cached_model_dir(repo)
            pinned = root / 'snapshots' / PINNED_REVISIONS[repo]
            pinned.mkdir()
            self.assertEqual(cached_model_dir(repo), pinned.resolve())
            with self.assertRaises(ValueError):
                cached_model_dir(repo, '../bad')


class DequantTests(unittest.TestCase):
    # Purpose: Verify unsigned low nibble first and affine groups.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_unsigned_low_nibble_first_and_affine_groups(self):
        packed = np.tile(np.array([[0x76543210, 0xFEDCBA98]], np.uint32), (1, 8))
        actual = dequantize_affine_4bit(packed, [[2, 0]], [[-3, 11]])
        expected = np.tile(np.arange(16, dtype=np.float32), 8)
        expected[:64] = expected[:64] * 2 - 3
        expected[64:] = 11
        np.testing.assert_array_equal(actual[0], expected)
        self.assertEqual(actual.dtype, np.float32)

    # Purpose: Verify invalid layout is explicit.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_invalid_layout_is_explicit(self):
        for packed, scales, biases in [
            (np.zeros((1, 8), np.int32), [[1]], [[0]]),
            (np.zeros((1, 7), np.uint32), [[1]], [[0]]),
            (np.zeros((1, 8), np.uint32), [[1, 2]], [[0]]),
        ]:
            with self.subTest(shape=packed.shape), self.assertRaises(ValueError):
                dequantize_affine_4bit(packed, scales, biases)


class FakeTokenizer:
    eos_token_id = 7

    # Purpose: Stand in for prompt formatting without a real tokenizer.
    # Input: *_args/**_kwargs: ignored tokenizer-interface arguments.
    # Output: Fixed string 'prompt'.
    def apply_chat_template(self, *_args, **_kwargs):
        return 'prompt'

    # Purpose: Provide deterministic token IDs for generation tests.
    # Input: *_args/**_kwargs: ignored tokenizer-interface arguments.
    # Output: Dict containing a (1, 3) input_ids tensor.
    def __call__(self, *_args, **_kwargs):
        return {'input_ids': torch.tensor([[1, 2, 3]])}

    # Purpose: Simulate incomplete Unicode followed by visible decoded text.
    # Input: tokens: generated token IDs; **_kwargs: ignored decoder options.
    # Output: Replacement character for one token, then accented text with trailing exclamation marks.
    def decode(self, tokens, **_kwargs):
        # First subtoken is incomplete UTF-8 and should never be emitted.
        return '\ufffd' if len(tokens) == 1 else 'é' + ('!' * (len(tokens) - 2))


class FakeModel:
    # Purpose: Prepare a deterministic token sequence and EOS metadata.
    # Input: tokens: sequence of IDs to predict.
    # Output: None; stores token iterator, EOS configurations, and recorded calls.
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.config = SimpleNamespace(eos_token_id=[8])
        self.generation_config = SimpleNamespace(eos_token_id=[9])
        self.calls = []

    # Purpose: Produce synthetic logits and record attention-cache arguments.
    # Input: **kwargs: model forward arguments.
    # Output: Namespace with one-step logits and fake past_key_values.
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.zeros((1, 1, 10))
        logits[0, 0, next(self.tokens)] = 1
        return SimpleNamespace(logits=logits, past_key_values=object())


class CPUGenerationTests(unittest.TestCase):
    # Purpose: Construct a CPU responder with synthetic tokenizer/model components.
    # Input: tokens: predicted token sequence; **kwargs: CPUResponder option overrides.
    # Output: CPUResponder fixture requiring no model files.
    def responder(self, tokens=(4, 5, 9), **kwargs):
        responder = CPUResponder(repo='synthetic/not-loaded', max_tokens=4, **kwargs)
        responder.model = FakeModel(tokens)
        responder.tokenizer = FakeTokenizer()
        return responder

    # Purpose: Verify visible unicode deltas eos union and cache.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_visible_unicode_deltas_eos_union_and_cache(self):
        responder = self.responder()
        with patch('app.cpu_llm.MemoryWatch') as watch:
            watch.return_value.peak = 100
            self.assertEqual(list(responder.stream([])), ['é'])
            watch.return_value.stop.assert_called_once()
        self.assertEqual(responder.last_generation['generated_tokens'], 2)
        self.assertEqual(responder.last_generation['text'], 'é')
        self.assertGreaterEqual(responder.last_generation['first_visible_delta_s'], responder.last_generation['first_token_s'])
        self.assertEqual(responder.model.calls[0]['input_ids'].numel(), 3)
        self.assertEqual(responder.model.calls[1]['input_ids'].numel(), 1)
        self.assertIsNotNone(responder.model.calls[1]['past_key_values'])

    # Purpose: Verify bounds before model and generator close cleanup.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_bounds_before_model_and_generator_close_cleanup(self):
        responder = self.responder(max_context_tokens=6)
        for limit in [0, -1, 4]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                list(responder.stream([], max_tokens=limit))
        self.assertEqual(responder.model.calls, [])
        responder = self.responder(tokens=(4, 5, 6, 6))
        with patch('app.cpu_llm.MemoryWatch') as watch:
            watch.return_value.peak = 100
            stream = responder.stream([])
            self.assertEqual(next(stream), 'é')
            stream.close()
            watch.return_value.stop.assert_called_once()
        self.assertEqual(responder.last_generation['generated_tokens'], 2)
        self.assertEqual(len(responder.model.calls), 2)

    # Purpose: Verify constructor limits and idempotent loaded return.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_constructor_limits_and_idempotent_loaded_return(self):
        for kwargs in [{'threads': 0}, {'threads': 5}, {'max_tokens': 0}, {'max_context_tokens': 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CPUResponder(**kwargs)
        responder = self.responder()
        responder.metadata = {'loaded': True}
        self.assertIs(responder.load(), responder.metadata)


class FactoryAndBenchmarkTests(unittest.TestCase):
    # Purpose: Verify factory preserves default and injects cpu without changing settings.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_factory_preserves_default_and_injects_cpu_without_changing_settings(self):
        settings = Settings()
        original_device = settings.torch_device
        with patch.object(backends, 'Engine') as engine:
            backends.create_engine(settings)
            engine.assert_called_once_with(settings)
        with patch.object(backends, 'Engine') as engine, patch('app.cpu_asr.CPUWhisperASR') as asr, patch('app.cpu_llm.CPUResponder') as llm:
            backends.create_engine(settings, 'cpu', 3)
            args, kwargs = engine.call_args
            self.assertEqual(args[0].torch_device, 'cpu')
            self.assertEqual(settings.torch_device, original_device)
            self.assertEqual(kwargs['backend'], 'cpu')
            self.assertEqual(kwargs['cpu_threads'], 3)
            asr.assert_called_once_with(repo=settings.asr_repo, threads=3)
            llm.assert_called_once_with(settings.llm_repo, settings.max_response_tokens, threads=3)
        for backend, threads in [('unknown', 4), ('cpu', 0), ('cpu', 5)]:
            with self.subTest(backend=backend, threads=threads), self.assertRaises(ValueError):
                backends.create_engine(settings, backend, threads)

    # Purpose: Verify cpu hardware reporting does not import gpu runtime.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_cpu_hardware_reporting_does_not_import_gpu_runtime(self):
        original_import = builtins.__import__
        attempts = []
        # Purpose: Reject MLX imports while allowing other imports during CPU reporting.
        # Input: name: module name; *args/**kwargs: import options; captures original_import and
        #          attempts.
        # Output: Imported module, or AssertionError if CPU reporting touches MLX.
        def guarded_import(name, *args, **kwargs):
            if name == 'mlx' or name.startswith('mlx.'):
                attempts.append(name)
                raise AssertionError('CPU reporting attempted to import MLX')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import):
            self.assertEqual(benchmark.hardware('cpu')['inference_backend'], 'cpu')
            self.assertIsNone(benchmark.mlx_peak_gb('cpu'))
            self.assertIsNone(benchmark.mps_driver_gb('cpu'))
        self.assertEqual(attempts, [])

    # Purpose: Verify cpu flags filenames and default compatibility.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_cpu_flags_filenames_and_default_compatibility(self):
        args = benchmark.parse_args([])
        self.assertEqual((args.backend, args.cpu_threads), ('mlx', 4))
        self.assertEqual(benchmark.output_suffix(False), '_multimodal')
        self.assertEqual(benchmark.output_suffix(True), '_audio_only')
        self.assertEqual(benchmark.output_suffix(False, 'cpu'), '_multimodal_cpu')
        self.assertEqual(benchmark.parse_args(['--backend', 'cpu', '--cpu-threads', '2']).cpu_threads, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(['--cpu-threads', '5'])


if __name__ == '__main__':
    unittest.main()
