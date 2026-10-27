"""Mocked setup acquisition tests: pinned revisions, offline flags and complete shards."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cache_models import acquire_models, MODEL_SPECS, ALLOW_PATTERNS, validate_snapshot


class ModelCacheTests(unittest.TestCase):
    # Purpose: Create temporary pinned Whisper/Qwen cache artifacts for setup tests.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.whisper = self.root / 'whisper'
        self.qwen = self.root / 'qwen'
        self.whisper.mkdir(); self.qwen.mkdir()
        (self.whisper/'config.json').write_text('{"model_type":"whisper"}')
        (self.whisper/'weights.npz').write_bytes(b'whisper weights')
        (self.qwen/'config.json').write_text('{"model_type":"qwen2","quantization":{"bits":4}}')
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
            (self.qwen/name).write_text('{}')
        (self.qwen/'model.safetensors').write_bytes(b'qwen weights')
        (self.qwen/'model.safetensors.index.json').write_text('{"weight_map":{"a":"model.safetensors"}}')

    # Purpose: Verify offline calls are pinned and hashes describe cached files.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_offline_calls_are_pinned_and_hashes_describe_cached_files(self):
        with patch('scripts.cache_models.snapshot_download', side_effect=[str(self.whisper), str(self.qwen)]) as download:
            manifest = acquire_models(offline=True)
        self.assertEqual(download.call_count, 2)
        for call, spec in zip(download.call_args_list, MODEL_SPECS):
            self.assertEqual(call.kwargs['repo_id'], spec['repo_id'])
            self.assertEqual(call.kwargs['revision'], spec['revision'])
            self.assertIs(call.kwargs['local_files_only'], True)
            self.assertEqual(call.kwargs['allow_patterns'], list(ALLOW_PATTERNS))
        self.assertEqual(manifest['models'][0]['files']['weights.npz']['sha256'], hashlib.sha256(b'whisper weights').hexdigest())
        self.assertEqual(manifest['models'][1]['files']['model.safetensors']['bytes'], len(b'qwen weights'))

    # Purpose: Verify online setup is explicitly not local files only.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_online_setup_is_explicitly_not_local_files_only(self):
        with patch('scripts.cache_models.snapshot_download', side_effect=[str(self.whisper), str(self.qwen)]) as download:
            acquire_models(offline=False)
        self.assertTrue(all(call.kwargs['local_files_only'] is False for call in download.call_args_list))

    # Purpose: Verify missing required file and missing mapped shard fail.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_missing_required_file_and_missing_mapped_shard_fail(self):
        (self.whisper/'weights.npz').unlink()
        with self.assertRaises(FileNotFoundError):
            validate_snapshot(self.whisper, MODEL_SPECS[0])
        (self.qwen/'model.safetensors.index.json').write_text('{"weight_map":{"a":"model-missing.safetensors"}}')
        with self.assertRaises(FileNotFoundError):
            validate_snapshot(self.qwen, MODEL_SPECS[1])


if __name__ == '__main__':
    unittest.main()
