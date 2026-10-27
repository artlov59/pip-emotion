"""Offline parameter accounting and model label-order guards; no pretrained inference."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


from app import emotion, text_emotion
from app.config import EMOTIONS, FEATURE_CONFIG
from scripts import count_params_offline as count_params


class ModelIntegrityTests(unittest.TestCase):
    # Purpose: Verify fusion counts all stored tables without counting flags or fit metadata.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_fusion_counts_all_stored_tables_without_counting_flags_or_fit_metadata(self):
        specification = {'live': {'audio': 1, 'text': .9, 'vision': .3},
                         'all': {'audio': {'audio': 1}, 'text_whisper': {'text_whisper': 1},
                                 'text_gold': {'text_gold': 1}, 'vision': {'vision': 1},
                                 'audio+text_whisper': {'audio': 1, 'text_whisper': 1},
                                 'audio+vision': {'audio': 1, 'vision': .2},
                                 'audio+text_whisper+vision': {'audio': 1, 'text_whisper': .9, 'vision': .3},
                                 'audio+text_gold+vision': {'audio': 1, 'text_gold': 1.9, 'vision': 1.2}},
                         'fit_n': 1092, 'fitted': True}
        self.assertEqual(count_params.count_scalar_values({"live": specification["live"], "all": specification["all"]}), 17)
        self.assertEqual(count_params.count_scalar_values({'a': [1, .5, True, None, 'rule']}), 2)

    # Purpose: Verify offline count excludes indices but unpacks quantized learned weights.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_offline_count_excludes_indices_but_unpacks_quantized_learned_weights(self):
        tensors = {'encoder.weight': {'shape':[2,3], 'dtype':'F32'},
                   'alignment_heads': {'shape':[10,2], 'dtype':'I64'},
                   'packed.weight': {'shape':[2,4], 'dtype':'U32'},
                   'packed.scales': {'shape':[2,1], 'dtype':'F16'},
                   'packed.biases': {'shape':[2,1], 'dtype':'F16'},
                   'packed.bias': {'shape':[2], 'dtype':'F16'}}
        config={'quantization': {'bits':4,'group_size':32}}
        with self.assertRaisesRegex(ValueError,'explicit classification'):
            count_params.count_tensors(tensors,config)
        result=count_params.count_tensors(tensors,config,excluded=('alignment_heads',))
        self.assertEqual(result['parameters'],72)
        self.assertEqual(result['excluded_quantization_metadata_elements'],4)
        self.assertEqual(result['excluded_nonlearned_tensors'][0]['elements'],20)
        config['quantization']={'bits':8,'group_size':16}
        self.assertEqual(count_params.count_tensors(tensors,config,excluded=('alignment_heads',))['parameters'],40)

    # Purpose: Create a mock classifier with configurable label metadata.
    # Input: labels: optional id-to-label mapping; num_labels: configured class count.
    # Output: MagicMock whose to()/eval() return itself and whose config describes the labels.
    def fake_text_model(self, labels=None, num_labels=7):
        model = MagicMock()
        model.config = SimpleNamespace(num_labels=num_labels, id2label=labels or dict(enumerate(EMOTIONS)))
        model.to.return_value = model
        model.eval.return_value = model
        return model

    # Purpose: Verify text load is local only and checks integer or string label keys.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_text_load_is_local_only_and_checks_integer_or_string_label_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'config.json').write_text('{}')
            for labels in [dict(enumerate(EMOTIONS)), {str(i): e for i, e in enumerate(EMOTIONS)}]:
                with self.subTest(labels=labels):
                    classifier = text_emotion.TextEmotionClassifier(directory, 'cpu')
                    fake = self.fake_text_model(labels)
                    with patch('transformers.AutoTokenizer.from_pretrained') as tokenizer, \
                            patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=fake) as loader, \
                            patch.object(classifier, 'predict') as warmup:
                        classifier.load()
                    tokenizer.assert_called_once_with(directory, local_files_only=True, trust_remote_code=False)
                    loader.assert_called_once_with(directory, local_files_only=True, trust_remote_code=False)
                    warmup.assert_called_once_with('warm up')
                    self.assertEqual(classifier.classes, EMOTIONS)

    # Purpose: Verify wrong text metadata fails before loading weights.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_wrong_text_metadata_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'config.json').write_text('{}')
            (directory / 'meld_meta.json').write_text(json.dumps({'classes': list(reversed(EMOTIONS))}))
            with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'metadata.*label order'):
                    text_emotion.TextEmotionClassifier(directory, 'cpu').load()
            loader.assert_not_called()

    # Purpose: Verify wrong text config fails before device move or inference.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_wrong_text_config_fails_before_device_move_or_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'config.json').write_text('{}')
            labels = dict(enumerate(reversed(EMOTIONS)))
            fake = self.fake_text_model(labels)
            classifier = text_emotion.TextEmotionClassifier(directory, 'cpu')
            with patch('transformers.AutoTokenizer.from_pretrained'), \
                    patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=fake), \
                    patch.object(classifier, 'predict') as warmup:
                with self.assertRaisesRegex(ValueError, 'config.*label order'):
                    classifier.load()
            fake.to.assert_not_called()
            warmup.assert_not_called()

    # Purpose: Verify audio rejects wrong label order before model construction.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_audio_rejects_wrong_label_order_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.pt'
            path.write_bytes(b'fixture')
            checkpoint = {'feature_config': FEATURE_CONFIG, 'classes': list(reversed(EMOTIONS))}
            with patch('torch.load', return_value=checkpoint), patch('model.AudioCNN') as constructor:
                with self.assertRaisesRegex(ValueError, 'classes.*label order'):
                    emotion.EmotionClassifier(path, 'cpu').load()
            constructor.assert_not_called()

    # Purpose: Verify valid audio labels keep legacy pooling and safe weight loading.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_valid_audio_labels_keep_legacy_pooling_and_safe_weight_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.pt'
            path.write_bytes(b'fixture')
            checkpoint = {'feature_config': FEATURE_CONFIG, 'classes': EMOTIONS, 'model_state_dict': {}}
            fake = MagicMock()
            fake.to.return_value = fake
            fake.eval.return_value = fake
            fake.pooling_mode = 'legacy'
            classifier = emotion.EmotionClassifier(path, 'cpu')
            with patch('torch.load', return_value=checkpoint) as loader, \
                    patch('model.AudioCNN', return_value=fake) as constructor, patch.object(classifier, 'predict'):
                classifier.load()
            loader.assert_called_once_with(path, map_location='cpu', weights_only=True)
            constructor.assert_called_once_with(num_classes=7, pooling_mode='legacy')
            self.assertEqual(classifier.classes, EMOTIONS)


if __name__ == '__main__':
    unittest.main()
