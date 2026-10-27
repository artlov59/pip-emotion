"""Build a reproducible submission containing source, evidence and selected models.

Example:
  python -m scripts.create_submission_bundle --repo . \
    --audio-checkpoint checkpoints/emotion_cnn.pt \
    --text-model-dir checkpoints/text_emotion \
    --selection-receipt results/submission_model_selection.json \
    --output /path/to/submission.tar.gz

No data/media, environments, Git history, caches, logs, training runs, credentials,
other checkpoints or published upload are included. The selected model files are
installed at the canonical paths expected by app.config.Settings.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT_FILES = ('model.py', 'README.md', '.gitignore')
SOURCE_DIRS = ('app', 'training', 'scripts', 'tests', 'results')
TEXT_REQUIRED = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'meld_meta.json')
TEXT_OPTIONAL = ('history.json', 'special_tokens_map.json', 'added_tokens.json')
RECEIPT_PATH = 'results/submission_model_selection.json'
MANIFEST_PATH = 'BUNDLE_MANIFEST.json'
EXCLUDED_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', '.cache', 'cache', '.pytest_cache',
    'data', 'datasets', 'runs', 'logs', 'traces', 'node_modules', 'checkpoints',
    '.aws', '.ssh', '.gcp', '.azure', 'credentials', 'secrets',
}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.pt', '.pth', '.bin', '.safetensors', '.onnx', '.npz', '.npy',
                     '.wav', '.mp3', '.mp4', '.mov', '.avi', '.pem', '.key', '.p12', '.pfx'}
CREDENTIAL_NAMES = {'credentials', 'credentials.json', 'secrets.json', 'secrets.toml', '.netrc'}
RESULT_SUFFIXES = {'.json', '.md', '.csv', '.png', '.jpg', '.jpeg', '.svg', '.txt'}
MAX_RESULT_BYTES = 25 * 1024 * 1024


# Purpose: Hash one submission payload source in bounded chunks.
# Input: path: readable file Path.
# Output: Hexadecimal SHA-256 digest string.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Identify a payload source without exposing external absolute paths.
# Input: path: source artifact Path; repo: repository root.
# Output: Dict with repository_path, or external_filename for a source outside the repository.
def source_name(path: Path, repo: Path) -> dict:
    path = Path(path).resolve()
    try:
        return {'repository_path': path.relative_to(repo).as_posix()}
    except ValueError:
        return {'external_filename': path.name}


# Purpose: Decide whether a source/evidence file belongs in the submission.
# Input: path: candidate Path; repo: repository root; section: source-directory name.
# Output: Boolean, excluding symlinks, private/generated files, unrelated models, and oversized results.
def is_payload_file(path: Path, repo: Path, section: str) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    relative = path.relative_to(repo)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    name = path.name.lower()
    if name == '.ds_store' or name == '.env' or name.startswith('.env.') or name in CREDENTIAL_NAMES:
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if section == 'results' and (path.suffix.lower() not in RESULT_SUFFIXES or path.stat().st_size > MAX_RESULT_BYTES):
        return False
    return True


# Purpose: Plan selected-model installation and fingerprint every submission file.
# Input: repo: source root; audio_checkpoint/text_model_dir: selected models; selection_receipt: locked
#          recipe; face_detector: optional ONNX Path.
# Output: Tuple (archive_path_to_source_plan, bundle_manifest); missing required artifacts raise errors.
def collect_payload(repo: Path, audio_checkpoint: Path, text_model_dir: Path,
                    selection_receipt: Path, face_detector: Path | None = None) -> tuple[dict, dict]:
    repo = Path(repo).resolve()
    plan = {}
    for name in ROOT_FILES:
        path = repo / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing required repository file: {name}')
        plan[name] = path
    requirements = sorted(repo.glob('requirements*.txt'))
    if not requirements:
        raise FileNotFoundError('No requirements*.txt files in repository root')
    for path in requirements:
        if path.is_file() and not path.is_symlink():
            plan[path.name] = path
    for section in SOURCE_DIRS:
        directory = repo / section
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing repository directory: {section}')
        for path in sorted(directory.rglob('*')):
            if is_payload_file(path, repo, section):
                plan[path.relative_to(repo).as_posix()] = path
    audio_checkpoint = Path(audio_checkpoint).resolve()
    text_model_dir = Path(text_model_dir).resolve()
    selection_receipt = Path(selection_receipt).resolve()
    if not audio_checkpoint.is_file() or not selection_receipt.is_file():
        raise FileNotFoundError('Selected audio checkpoint and model-selection receipt must exist')
    receipt = json.loads(selection_receipt.read_text())
    if not isinstance(receipt, dict):
        raise ValueError('Model-selection receipt must be a JSON object')
    canonical_audio = 'checkpoints/emotion_cnn.pt'
    plan[canonical_audio] = audio_checkpoint
    missing = [name for name in TEXT_REQUIRED if not (text_model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing selected text-model files: {missing}')
    text_names = sorted([*TEXT_REQUIRED, *(name for name in TEXT_OPTIONAL if (text_model_dir / name).is_file())])
    for name in text_names:
        plan[f'checkpoints/text_emotion/{name}'] = text_model_dir / name
    plan[RECEIPT_PATH] = selection_receipt
    models = {
        'audio': {'canonical_path': canonical_audio, 'source': source_name(audio_checkpoint, repo)},
        'text': {'canonical_directory': 'checkpoints/text_emotion', 'source': source_name(text_model_dir, repo),
                 'files': text_names},
    }
    if face_detector is not None:
        face_detector = Path(face_detector).resolve()
        if not face_detector.is_file():
            raise FileNotFoundError(f'Missing selected face detector: {face_detector}')
        canonical_face = 'checkpoints/face_detection_yunet_2023mar.onnx'
        plan[canonical_face] = face_detector
        models['face_detector'] = {'canonical_path': canonical_face, 'source': source_name(face_detector, repo)}
    files = {name: {'bytes': source.stat().st_size, 'sha256': sha256_file(source),
                    'mode': '0755' if source.stat().st_mode & 0o111 else '0644',
                    'source': source_name(source, repo)} for name, source in sorted(plan.items())}
    for model in models.values():
        if 'canonical_path' in model:
            model.update({key: files[model['canonical_path']][key] for key in ('bytes', 'sha256')})
    manifest = {
        'schema': 1, 'artifact': 'Runnable MELD multimodal prototype submission',
        'hash_scope': 'Every payload file; this manifest excludes its own digest.',
        'model_selection_receipt_path': RECEIPT_PATH,
        'model_selection_receipt_sha256': files[RECEIPT_PATH]['sha256'],
        'models': models, 'files': files,
        'excluded': ['MELD data/media', 'environments', 'Git history', 'caches', 'logs/traces',
                     'training runs', 'credentials', 'unselected checkpoints', 'results files larger than 25 MiB'],
        'model_downloads': 'Whisper, Qwen and HSEmotion are obtained separately as described in README; inference remains local.',
    }
    return plan, manifest


class HashingReader:
    # Purpose: Wrap a binary source so packaging can verify bytes as they are copied.
    # Input: stream: readable binary stream.
    # Output: None; stores the stream and initializes digest/byte count.
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    # Purpose: Read payload bytes while updating the running checksum and size.
    # Input: size: maximum bytes to read, or -1 for all remaining bytes.
    # Output: Bytes read from the wrapped stream; updates digest and byte count.
    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block


# Purpose: Create deterministic archive metadata for one payload entry.
# Input: name: archive member path; size: file bytes; mode: permission bits.
# Output: TarInfo with fixed ownership/time metadata and the supplied name/size/mode.
def tar_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mode = mode
    return info


# Purpose: Write a deterministic submission and verify copied payload bytes.
# Input: repo: source root; audio_checkpoint/text_model_dir: selected models; selection_receipt: recipe;
#          output: new archive Path; face_detector: optional ONNX Path.
# Output: Bundle result dict; writes archive/checksum/manifest outputs and refuses an existing archive.
def create_bundle(repo: Path, audio_checkpoint: Path, text_model_dir: Path,
                  selection_receipt: Path, output: Path, face_detector: Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing bundle: {output}')
    plan, manifest = collect_payload(repo, audio_checkpoint, text_model_dir, selection_receipt, face_detector)
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        with temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode='w') as archive:
                    for name, source in sorted(plan.items()):
                        expected = manifest['files'][name]
                        with source.open('rb') as stream:
                            reader = HashingReader(stream)
                            archive.addfile(tar_info(name, expected['bytes'], int(expected['mode'], 8)), reader)
                        if reader.bytes != expected['bytes'] or reader.digest.hexdigest() != expected['sha256'] or source.stat().st_size != expected['bytes']:
                            raise ValueError(f'Payload changed while bundling: {name}; freeze model selection before retrying')
                    archive.addfile(tar_info(MANIFEST_PATH, len(encoded)), io.BytesIO(encoded))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    archive_sha256 = sha256_file(output)
    output.with_name(output.name + '.sha256').write_text(f'{archive_sha256}  {output.name}\n')
    result = {**manifest, 'archive': output.name, 'archive_bytes': output.stat().st_size,
              'archive_sha256': archive_sha256}
    output.with_name(output.name + '.manifest.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


# Purpose: Parse selected artifact paths and build a runnable submission archive.
# Input: None explicitly; reads CLI repository/model/receipt/output options.
# Output: None; writes the bundle and prints its result metadata.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--audio-checkpoint', type=Path, required=True)
    parser.add_argument('--text-model-dir', type=Path, required=True)
    parser.add_argument('--selection-receipt', type=Path, required=True)
    parser.add_argument('--face-detector', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = create_bundle(args.repo, args.audio_checkpoint, args.text_model_dir, args.selection_receipt,
                           args.output, args.face_detector)
    print(json.dumps({key: result[key] for key in ('archive', 'archive_bytes', 'archive_sha256', 'model_selection_receipt_path')}, indent=2))


if __name__ == '__main__':
    main()
