"""Fast archive inclusion/hash/reproducibility checks using tiny fake model payloads."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.create_submission_bundle import create_bundle, MANIFEST_PATH, RECEIPT_PATH


class SubmissionBundleTests(unittest.TestCase):
    # Purpose: Create a temporary runnable-repository fixture with selected and excluded artifacts.
    # Input: None.
    # Output: None; initializes isolated fixtures and registers cleanup where needed.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        payload = {
            'model.py': 'model source', 'README.md': 'handoff', '.gitignore': '.venv/',
            'requirements.txt': 'numpy', 'requirements-lock.txt': 'numpy==1.0',
            'app/__init__.py': '', 'app/static/app.js': 'console.log(1)',
            'training/train.py': 'train', 'scripts/helper.py': 'helper',
            'scripts/launch.sh': '#!/bin/sh\nexit 0\n',
            'tests/test_something.py': 'test', 'results/retraining/summary.json': '{"dev":1}',
            'results/dev_predictions.csv': 'id,pred',
            'results/logs/private.log': 'log', 'results/traces/giant.json': '{}',
            'results/retraining/best.pt': 'unselected model',
            'data/meld/raw.wav': 'audio', '.venv/lib/package.py': 'environment',
            'app/__pycache__/cache.pyc': 'cache', 'app/.env': 'TOKEN=secret',
            'scripts/private.pem': 'private key', 'app/secrets.json': '{}',
            'runs/something/events': 'events', '.git/config': 'git',
            'checkpoints/unselected/best.pt': 'other checkpoint',
        }
        for name, text in payload.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (self.repo / 'scripts/launch.sh').chmod(0o755)
        self.audio = self.root / 'chosen_audio.pt'
        self.audio.write_bytes(b'EXACT CHOSEN AUDIO')
        self.text = self.root / 'chosen_text'
        self.text.mkdir()
        for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'meld_meta.json', 'history.json']:
            (self.text / name).write_bytes(f'EXACT CHOSEN TEXT {name}'.encode())
        self.receipt = self.root / 'selection.json'
        self.receipt.write_text('{"selection_split":"dev","test_used_for_selection":false}')

    # Purpose: Package the synthetic repository into a named test archive.
    # Input: name: destination filename under the fixture root.
    # Output: Tuple (archive Path, bundle metadata dict); writes the test archive and sidecars.
    def bundle(self, name):
        output = self.root / name
        result = create_bundle(self.repo, self.audio, self.text, self.receipt, output)
        return output, result

    # Purpose: Verify exclusions selected model paths and every file hash.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_exclusions_selected_model_paths_and_every_file_hash(self):
        output, result = self.bundle('submission.tar.gz')
        with tarfile.open(output, 'r:gz') as archive:
            names = archive.getnames()
            manifest = json.load(archive.extractfile(MANIFEST_PATH))
            self.assertEqual(set(names), set(manifest['files']) | {MANIFEST_PATH})
            self.assertEqual(manifest['model_selection_receipt_path'], RECEIPT_PATH)
            self.assertIn('requirements-lock.txt', names)
            self.assertIn('results/retraining/summary.json', names)
            self.assertEqual(archive.getmember('scripts/launch.sh').mode, 0o755)
            self.assertEqual(archive.extractfile('checkpoints/emotion_cnn.pt').read(), self.audio.read_bytes())
            self.assertEqual(archive.extractfile('checkpoints/text_emotion/model.safetensors').read(), (self.text/'model.safetensors').read_bytes())
            self.assertFalse(any('/logs/' in name or '/traces/' in name or '__pycache__' in name or '.venv' in name or name.startswith('data/') or '/.env' in name or name.endswith('.pem') or name.endswith('secrets.json') or 'unselected' in name for name in names))
            for name, metadata in manifest['files'].items():
                content = archive.extractfile(name).read()
                self.assertEqual(metadata['bytes'], len(content))
                self.assertEqual(metadata['sha256'], hashlib.sha256(content).hexdigest())
        self.assertEqual(result['archive_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())

    # Purpose: Verify identical inputs make identical archives and existing output is preserved.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_identical_inputs_make_identical_archives_and_existing_output_is_preserved(self):
        first, a = self.bundle('first.tar.gz')
        _, b = self.bundle('second.tar.gz')
        self.assertEqual(a['archive_sha256'], b['archive_sha256'])
        with self.assertRaises(FileExistsError):
            create_bundle(self.repo, self.audio, self.text, self.receipt, first)
        self.assertEqual(a['archive_sha256'], hashlib.sha256(first.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
