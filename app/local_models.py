"""Read existing local model snapshots, with pinned defaults and no network access."""
import hashlib
import os
from pathlib import Path

PINNED_REVISIONS = {
    "mlx-community/whisper-small-mlx": "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
    "mlx-community/Qwen2.5-3B-Instruct-4bit": "4f83f8f146fdf28b512a06562b671d7af4fab457",
}


# Purpose: Hash a local artifact without reading the entire file into memory.
# Input: path: readable file path.
# Output: Hexadecimal SHA-256 digest string.
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Purpose: Resolve the Hugging Face cache root from environment settings.
# Input: None explicitly; reads HF_HUB_CACHE, HF_HOME, XDG_CACHE_HOME, and home defaults.
# Output: Path to the local hub cache directory.
def model_cache_root():
    direct = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if direct:
        return Path(direct).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser() / "huggingface" / "hub"


# Purpose: Resolve an explicit directory or pinned cached snapshot without downloading.
# Input: source: local directory or model ID; revision: optional snapshot/ref override.
# Output: Resolved snapshot Path; missing or invalid revisions raise errors.
def cached_model_dir(source, revision=None):
    direct = Path(source).expanduser()
    if direct.is_dir():
        return direct.resolve()
    parts = str(source).split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise FileNotFoundError(f"Missing local model directory or invalid cached repository: {source}")
    repository = model_cache_root() / ("models--" + str(source).replace("/", "--"))
    revision = revision or PINNED_REVISIONS.get(str(source), "main")
    if Path(revision).name != revision or revision in {".", ".."}:
        raise ValueError("Model revision must be one commit/ref name")
    snapshot = repository / "snapshots" / revision
    if not snapshot.is_dir():
        reference = repository / "refs" / revision
        if reference.is_file():
            commit = reference.read_text().strip()
            if not commit or Path(commit).name != commit or commit in {".", ".."}:
                raise ValueError(f"Invalid local cache reference: {reference}")
            snapshot = repository / "snapshots" / commit
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Cached model {source}@{revision} is unavailable. CPU mode never downloads; provide its existing local snapshot.")
    return snapshot.resolve()
