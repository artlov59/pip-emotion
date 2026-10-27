"""Text adaptation data/selection regressions. No models or downloads are used."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from app.config import EMOTIONS
from training import train_text


class TextTrainingTests(unittest.TestCase):
    # Purpose: Create a temporary training directory and a labeled three-utterance manifest.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frame = pd.DataFrame({
            'id': ['a', 'b', 'c'], 'status': ['ok', 'ok', 'too_short'],
            'utterance': ['gold A', 'gold B', 'gold C'], 'gold': ['gold A', 'gold B', 'gold C'],
            'label': [0, 3, 4], 'emotion': ['anger', 'joy', 'neutral']})

    # Purpose: Remove the temporary text-training fixture directory.
    # Input: None explicitly; uses the fixture temporary directory.
    # Output: None; cleans up fixture files.
    def tearDown(self):
        self.tmp.cleanup()

    # Purpose: Write synthetic transcript records for cache validation tests.
    # Input: records: row dicts; name: CSV filename under the temporary root.
    # Output: Path to the written CSV.
    def cache(self, records, name='whisper_train.csv'):
        path = self.root / name
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    # Purpose: Verify duplicate unknown and wrong split cache rejected.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_duplicate_unknown_and_wrong_split_cache_rejected(self):
        invalid = [[{'id': 'a', 'whisper': 'one'}, {'id': 'a', 'whisper': 'two'}],
                   [{'id': 'x', 'whisper': 'unknown'}],
                   [{'id': 'a', 'whisper': 'one', 'split': 'dev'}]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                train_text.attach_asr(self.frame, self.cache(records), 'train')

    # Purpose: Verify coverage is strict unless explicit partial threshold.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_coverage_is_strict_unless_explicit_partial_threshold(self):
        path = self.cache([{'id': 'a', 'whisper': 'asr A'}])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            train_text.attach_asr(self.frame, path, 'train')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            train_text.attach_asr(self.frame, path, 'train', allow_partial=True, min_coverage=0.75)
        joined, provenance = train_text.attach_asr(self.frame, path, 'train', True, 0.5)
        self.assertEqual(provenance['covered_audio_usable'], 1)
        self.assertEqual(provenance['expected_audio_usable'], 2)
        self.assertEqual(provenance['coverage'], 0.5)
        self.assertEqual(len(provenance['sha256']), 64)
        self.assertEqual(joined.id.tolist(), self.frame.id.tolist())

    # Purpose: Verify empty asr counts as processed but is not whisper training text.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_empty_asr_counts_as_processed_but_is_not_whisper_training_text(self):
        path = self.cache([{'id': 'a', 'whisper': 'asr A'}, {'id': 'b', 'whisper': ''}])
        joined, provenance = train_text.attach_asr(self.frame, path, 'train')
        self.assertEqual(provenance['coverage'], 1)
        self.assertEqual(provenance['nonempty_audio_usable'], 1)
        self.assertEqual(train_text.prepare_training(joined, 'whisper').id.tolist(), ['a'])
        mixed = train_text.prepare_training(joined, 'mixed')
        self.assertEqual(mixed.id.tolist(), ['a', 'b', 'c'])
        for epoch in range(5):
            texts, labels, counts = train_text.epoch_examples(mixed, 'mixed', 7, epoch)
            self.assertEqual(labels, [0, 3, 4])
            self.assertEqual(len(texts), 3)
            self.assertEqual(texts[1:], ['gold B', 'gold C'])
            self.assertEqual(sum(counts.values()), 3)
            self.assertEqual((texts, labels, counts), train_text.epoch_examples(mixed, 'mixed', 7, epoch))

    # Purpose: Verify generation metadata split is checked.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_generation_metadata_split_is_checked(self):
        path = self.cache([{'id': 'a', 'whisper': 'asr A'}, {'id': 'b', 'whisper': ''}])
        path.with_suffix('.meta.json').write_text(json.dumps({'split': 'test'}))
        with self.assertRaisesRegex(ValueError, 'metadata split'):
            train_text.attach_asr(self.frame, path, 'train')

    # Purpose: Verify capped asr keeps coverage uses gold and preserves raw cache.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_capped_asr_keeps_coverage_uses_gold_and_preserves_raw_cache(self):
        path = self.cache([{'id': 'a', 'whisper': 'truncated repetition', 'hit_token_limit': True},
                           {'id': 'b', 'whisper': 'asr B', 'hit_token_limit': False}])
        original = path.read_bytes()
        joined, provenance = train_text.attach_asr(self.frame, path, 'train')
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(provenance['coverage'], 1)
        self.assertEqual(provenance['excluded_hit_token_limit'], 1)
        self.assertEqual(provenance['excluded_hit_token_limit_audio_usable'], 1)
        self.assertEqual(provenance['nonempty_audio_usable'], 1)
        self.assertEqual(joined.whisper.tolist(), ['', 'asr B', ''])
        self.assertEqual(train_text.prepare_training(joined, 'whisper').id.tolist(), ['b'])
        mixed = train_text.prepare_training(joined, 'mixed')
        for epoch in range(3):
            texts, labels, _ = train_text.epoch_examples(mixed, 'mixed', 0, epoch)
            self.assertEqual(texts[0], 'gold A')
            self.assertEqual(labels, [0, 3, 4])

    # Purpose: Verify csv suffix sidecar fallback is preserved.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_csv_suffix_sidecar_fallback_is_preserved(self):
        path = self.cache([{'id': 'a', 'whisper': 'asr A'}, {'id': 'b', 'whisper': ''}])
        sidecar = Path(str(path) + '.meta.json')
        sidecar.write_text(json.dumps({'split': 'train', 'backend': 'cpu'}))
        _, provenance = train_text.attach_asr(self.frame, path, 'train')
        self.assertEqual(provenance['generation_metadata']['backend'], 'cpu')
        self.assertEqual(provenance['generation_metadata_file']['path'], str(sidecar.resolve()))

    # Purpose: Verify macro guard preserves baseline and breaks no ties.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_macro_guard_preserves_baseline_and_breaks_no_ties(self):
        cases = [(.6, .39, .5, .4, False), (.6, .4, .5, .4, True),
                 (.5, .5, .5, .4, False), (.49, .5, .5, .4, False),
                 (.6, .2, .5, None, True), (float('nan'), .5, .5, .4, False),
                 (.6, float('inf'), .5, .4, False)]
        for weighted, macro, best, floor, expected in cases:
            with self.subTest(weighted=weighted, macro=macro, floor=floor):
                self.assertIs(train_text.should_select_epoch(np.float64(weighted), np.float64(macro), best, floor), expected)

    # Purpose: Verify epoch zero is saved as a usable fallback with input provenance.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_epoch_zero_is_saved_as_a_usable_fallback_with_input_provenance(self):
        output = self.root / 'candidate'
        output.mkdir()
        model, tokenizer = MagicMock(), MagicMock()
        run = {'initialization': 'original_checkpoint', 'selection_source': 'whisper',
               'provenance': {'dev_manifest': {'sha256': 'manifest_hash'},
                              'selection_asr': {'sha256': 'real_dev_asr_hash'}}}
        with patch.object(train_text, 'predict_labels', return_value=np.arange(7)):
            row = train_text.initialize_epoch_zero(model, tokenizer, ['spoken'] * 7, list(range(7)), 'cpu', output, run)
        model.save_pretrained.assert_called_once_with(output)
        tokenizer.save_pretrained.assert_called_once_with(output)
        meta = json.loads((output / 'meld_meta.json').read_text())
        self.assertEqual(meta['epoch'], 0)
        self.assertTrue(meta['evaluation_only'])
        self.assertEqual(meta['selection_macro_floor'], 1)
        self.assertEqual(meta['baseline_selection']['asr_cache']['sha256'], 'real_dev_asr_hash')
        self.assertEqual(json.loads((output / 'history.json').read_text()), [row])

    # Purpose: Verify macro preservation requires explicit training warm start.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_macro_preservation_requires_explicit_training_warm_start(self):
        for argv in [['--preserve-baseline-macro'],
                     ['--preserve-baseline-macro', '--init-model-dir', 'local', '--eval-only']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_text.parse_args(argv)
        args = train_text.parse_args(['--preserve-baseline-macro', '--init-model-dir', 'local',
                                      '--selection-asr-cache', 'real_dev_asr.csv'])
        self.assertTrue(args.preserve_baseline_macro)
        self.assertEqual(args.dev_asr_cache, Path('real_dev_asr.csv'))

    # Purpose: Verify training flow reloads epoch zero when later epoch fails guard.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_training_flow_reloads_epoch_zero_when_later_epoch_fails_guard(self):
        class Encoding(dict):
            # Purpose: Accept a device move while keeping the synthetic encoding unchanged.
            # Input: device: ignored Torch device.
            # Output: The same encoding instance.
            def to(self, device):
                return self

        class Tokenizer:
            # Purpose: Create fixed token IDs for each training example.
            # Input: texts: text sequence; **kwargs: ignored tokenizer options.
            # Output: Encoding containing a zero input_ids tensor shaped (len(texts), 1).
            def __call__(self, texts, **kwargs):
                return Encoding(input_ids=torch.zeros(len(texts), 1, dtype=torch.long))

            # Purpose: Persist a marker for the synthetic saved tokenizer.
            # Input: path: existing model output directory.
            # Output: None; writes fixture_tokenizer.json.
            def save_pretrained(self, path):
                (Path(path) / 'fixture_tokenizer.json').write_text('{}')

        class TinyModel(torch.nn.Module):
            # Purpose: Create a trainable seven-label bias-only classifier.
            # Input: bias: optional seven-element initial bias.
            # Output: None; initializes a learned bias and ordered MELD label config.
            def __init__(self, bias=None):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(7) if bias is None else torch.tensor(bias))
                self.config = SimpleNamespace(num_labels=7, id2label=dict(enumerate(EMOTIONS)))

            # Purpose: Broadcast the fixture's learned bias as classification logits.
            # Input: input_ids: batch token IDs.
            # Output: Namespace containing logits shaped (batch_size, 7).
            def forward(self, input_ids):
                return SimpleNamespace(logits=self.bias.expand(len(input_ids), -1))

            # Purpose: Save the tiny classifier's current bias for reload checks.
            # Input: path: existing checkpoint directory.
            # Output: None; writes fixture_bias.json and a config marker.
            def save_pretrained(self, path):
                (Path(path) / 'fixture_bias.json').write_text(json.dumps(self.bias.detach().tolist()))
                (Path(path) / 'config.json').write_text('{}')

        warm = self.root / 'warm_start'
        warm.mkdir()
        (warm / 'config.json').write_text('{"original":true}')
        self.frame.to_csv(self.root / 'train.csv', index=False)
        self.frame.to_csv(self.root / 'dev.csv', index=False)
        self.cache([{'id': 'a', 'whisper': 'spoken A'}, {'id': 'b', 'whisper': 'spoken B'}], 'whisper_dev.csv')
        output = self.root / 'new_candidate'
        trained = TinyModel()

        # Purpose: Load the warm-start fixture or reconstruct a saved epoch-zero model.
        # Input: path: warm-start/saved model directory; **kwargs: ignored loading options.
        # Output: Existing trained fixture or TinyModel reconstructed from saved bias values.
        def load_model(path, **kwargs):
            return trained if Path(path) == warm else TinyModel(json.loads((Path(path) / 'fixture_bias.json').read_text()))

        old_threads = torch.get_num_threads()
        try:
            with patch.object(train_text, 'MELD_DIR', self.root), \
                    patch('transformers.AutoModelForSequenceClassification.from_pretrained', side_effect=load_model), \
                    patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                    patch.object(train_text, 'predict_labels', side_effect=[np.array([0, 3]), np.array([0, 0])]), \
                    patch.object(train_text, 'evaluate_sources') as evaluate, \
                    patch.object(train_text, 'tqdm', side_effect=lambda it, **kwargs: it), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_text.main(['--init-model-dir', str(warm), '--preserve-baseline-macro',
                                 '--selection-source', 'whisper', '--output-dir', str(output),
                                 '--epochs', '1', '--batch-size', '3', '--threads', '1', '--device', 'cpu', '--lr', '0.01'])
        finally:
            torch.set_num_threads(old_threads)
        self.assertGreater(float(trained.bias.detach().abs().sum()), 0)
        selected = evaluate.call_args.args[0]
        self.assertTrue(torch.equal(selected.bias.detach(), torch.zeros(7)))
        metadata = json.loads((output / 'meld_meta.json').read_text())
        history = json.loads((output / 'history.json').read_text())
        self.assertEqual(metadata['epoch'], 0)
        self.assertEqual(len(history), 2)
        self.assertFalse(history[1]['selected'])
        self.assertFalse(history[1]['passes_macro_guard'])
        self.assertEqual((warm / 'config.json').read_text(), '{"original":true}')

    # Purpose: Verify test evaluation is explicit and eval only.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_test_evaluation_is_explicit_and_eval_only(self):
        self.assertEqual(train_text.parse_args([]).eval_splits, ['dev'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_text.parse_args(['--eval-splits', 'test'])
        self.assertEqual(train_text.parse_args(['--eval-only', '--eval-splits', 'test']).eval_splits, ['test'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_text.parse_args(['--min-asr-coverage', '0.5'])
        args = train_text.parse_args(['--allow-partial-asr', '--min-asr-coverage', '0.5'])
        self.assertEqual(args.min_asr_coverage, 0.5)

    # Purpose: Verify training refuses existing outputs.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_training_refuses_existing_outputs(self):
        existing = self.root / 'baseline'
        existing.mkdir()
        marker = existing / 'weights'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            train_text.validate_training_outputs(existing, self.root / 'new_results')
        self.assertEqual(marker.read_text(), 'keep')

    # Purpose: Verify default evaluation never reads test manifest.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_default_evaluation_never_reads_test_manifest(self):
        self.frame.to_csv(self.root / 'dev.csv', index=False)
        result_dir = self.root / 'results'
        fake_probs = np.array([[0.8, 0.02, 0.02, 0.04, 0.08, 0.02, 0.02],
                               [0.03, 0.02, 0.02, 0.8, 0.08, 0.02, 0.03]], dtype=float)
        with patch.object(train_text, 'predict_probs', return_value=fake_probs), contextlib.redirect_stdout(io.StringIO()):
            train_text.evaluate_sources(object(), object(), 'cpu', result_dir, meld_dir=self.root)
        report = json.loads((result_dir / 'text_dev_gold_metrics.json').read_text())
        self.assertEqual(report['n'], 2)
        self.assertEqual(report['accuracy'], 1)
        self.assertFalse(any(result_dir.glob('*test*')))

    # Purpose: Verify empty selection transcript falls back to neutral.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_empty_selection_transcript_falls_back_to_neutral(self):
        fake_probs = np.zeros((2, len(EMOTIONS)))
        fake_probs[:, 0] = 1
        with patch.object(train_text, 'predict_probs', return_value=fake_probs):
            pred = train_text.predict_labels(object(), object(), ['', 'spoken'], 'cpu')
        self.assertEqual(pred.tolist(), [EMOTIONS.index('neutral'), 0])

    # Purpose: Verify warm start rejects wrong label order.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_warm_start_rejects_wrong_label_order(self):
        config = SimpleNamespace(num_labels=7, id2label=dict(enumerate(EMOTIONS)))
        train_text.validate_warm_start(config)
        config.id2label[0], config.id2label[1] = config.id2label[1], config.id2label[0]
        with self.assertRaisesRegex(ValueError, 'ordered MELD'):
            train_text.validate_warm_start(config)


if __name__ == '__main__':
    unittest.main()
