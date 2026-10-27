"""Acquire pinned model snapshots during setup, or verify the existing cache offline.

    python -m scripts.cache_models                  # setup: network/model downloads allowed
    python -m scripts.cache_models --offline        # cache only: no network

This only obtains and hashes local files. It performs no model inference and introduces
no additional models. Use the reported snapshot paths to reproduce these exact weights.
"""
from __future__ import annotations

import argparse
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

MODEL_SPECS = (
    {'name': 'whisper', 'repo_id': 'mlx-community/whisper-small-mlx',
     'revision': '45f3915923c7a79a5a5b5a7d909d39aeb0e5630e',
     'required': ('config.json', 'weights.npz')},
    {'name': 'qwen', 'repo_id': 'mlx-community/Qwen2.5-3B-Instruct-4bit',
     'revision': '4f83f8f146fdf28b512a06562b671d7af4fab457',
     'required': ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')},
)
ALLOW_PATTERNS = (
    'config.json', 'generation_config.json', 'weights.npz', 'weights*.safetensors',
    'model*.safetensors', 'model*.safetensors.index.json', 'tokenizer*.json',
    '*.jinja', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
)


# Purpose: Hash a cached model artifact in bounded-size reads.
# Input: path: readable artifact Path.
# Output: Hexadecimal SHA-256 digest string.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Verify required files and referenced weight shards in a model snapshot.
# Input: snapshot: local snapshot Path; spec: pinned model specification.
# Output: Model provenance dict containing source/revision, file hashes/sizes, and total bytes; raises if
#          incomplete.
def validate_snapshot(snapshot: Path, spec: dict) -> dict:
    snapshot = Path(snapshot)
    missing = [name for name in spec['required'] if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete {spec['name']} snapshot {spec['revision']}: missing/empty {missing}")
    config = json.loads((snapshot / 'config.json').read_text())
    if spec['name'] == 'whisper':
        if config.get('model_type') != 'whisper':
            raise ValueError('Expected a Whisper model config')
    elif spec['name'] == 'qwen':
        if config.get('model_type') != 'qwen2' or config.get('quantization', {}).get('bits') != 4:
            raise ValueError('Expected a Qwen2 4-bit MLX model config')
        index = snapshot / 'model.safetensors.index.json'
        if index.is_file():
            weight_map = json.loads(index.read_text()).get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError('Empty or invalid Qwen weight-map index')
            weights = set(weight_map.values())
            for name in weights:
                if not isinstance(name, str) or PurePosixPath(name).name != name or not fnmatch(name, 'model*.safetensors'):
                    raise ValueError(f'Unexpected weight-map filename: {name!r}')
        else:
            weights = {path.name for path in snapshot.glob('model*.safetensors') if path.is_file()}
        if not weights:
            raise FileNotFoundError('Qwen snapshot contains no model*.safetensors weights')
        missing_weights = [name for name in sorted(weights) if not (snapshot/name).is_file() or (snapshot/name).stat().st_size == 0]
        if missing_weights:
            raise FileNotFoundError(f'Incomplete Qwen weight shards: {missing_weights}')
    files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and any(fnmatch(path.name, pattern) for pattern in ALLOW_PATTERNS):
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    return {
        'name': spec['name'], 'repo_id': spec['repo_id'], 'revision': spec['revision'],
        'snapshot_path': str(snapshot.absolute()), 'files': files,
        'total_bytes': sum(item['bytes'] for item in files.values()),
    }


# Purpose: Acquire or verify every pinned external model for local inference.
# Input: offline: require existing cached files when true; otherwise allow setup downloads.
# Output: Setup manifest dict with verified model artifacts and their provenance.
def acquire_models(offline: bool = False) -> dict:
    models = []
    for spec in MODEL_SPECS:
        snapshot = snapshot_download(
            repo_id=spec['repo_id'], revision=spec['revision'],
            allow_patterns=list(ALLOW_PATTERNS), local_files_only=offline,
        )
        models.append(validate_snapshot(Path(snapshot), spec))
    return {
        'schema': 1, 'purpose': 'Pinned model acquisition/verification during setup; inference is local.',
        'offline': offline, 'models': models,
    }


# Purpose: Run explicit model setup or offline cache verification.
# Input: None explicitly; reads CLI offline/output options.
# Output: None; prints and saves the model acquisition/verification manifest.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Only inspect cached files; never access the network')
    parser.add_argument('--output', type=Path, default=Path('results/model_cache_manifest.json'))
    args = parser.parse_args()
    manifest = acquire_models(offline=args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'offline': args.offline, 'manifest': str(args.output),
                      'models': [{key: model[key] for key in ('name', 'repo_id', 'revision', 'snapshot_path', 'total_bytes')} for model in manifest['models']]}, indent=2))


if __name__ == '__main__':
    main()
