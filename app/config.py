"""Shared constants and runtime settings (single source of truth for train + serve)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MELD emotion categories, in label-index order (alphabetical).
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Audio front-end. Training features and live inference MUST use the same values;
# they are stored in every checkpoint and checked at load time.
SAMPLE_RATE = 16_000
N_FFT = 400        # 25 ms window
HOP = 160          # 10 ms hop -> 100 frames / second
N_MELS = 64
FMIN = 50
FMAX = 8_000
FEATURE_CONFIG = dict(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)

DATA_DIR = ROOT / "data"
MELD_DIR = DATA_DIR / "meld"          # prepared manifests, wavs, features
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"


# Purpose: Read an environment setting with a fallback.
# Input: name: environment variable name; default: value used when absent.
# Output: Environment value, or default.
def _env(name, default):
    return os.environ.get(name, default)


@dataclass
class Settings:
    # models
    emotion_checkpoint: Path = field(default_factory=lambda: Path(_env("EMOTION_CKPT", str(CHECKPOINT_DIR / "emotion_cnn.pt"))))
    asr_repo: str = _env("ASR_REPO", "mlx-community/whisper-small-mlx")
    llm_repo: str = _env("LLM_REPO", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    torch_device: str = _env("TORCH_DEVICE", "auto")      # auto | mps | cpu
    use_llm: bool = _env("USE_LLM", "1") == "1"          # 0 -> template replies (debug only)
    # multimodal extension (each degrades gracefully: missing model -> that modality is skipped)
    text_model_dir: Path = field(default_factory=lambda: Path(_env("TEXT_MODEL_DIR", str(CHECKPOINT_DIR / "text_emotion"))))
    use_text: bool = _env("USE_TEXT", "1") == "1"
    use_vision: bool = _env("USE_VISION", "1") == "1"
    face_model: str = _env("FACE_MODEL", "enet_b0_8_best_vgaf")
    face_history_s: float = 60.0        # keep this much face history per session

    # Submission/demo mode: fail startup instead of silently dropping required models.
    require_multimodal: bool = _env("REQUIRE_MULTIMODAL", "0") == "1"

    # streaming / turn-taking (see the README section 'What "real-time" means here')
    chunk_ms: int = 100                 # client sends 100 ms PCM chunks
    vad_threshold: float = 0.5          # speech starts when P(speech) >= this
    vad_neg_threshold: float = 0.35     # silence counted when P(speech) < this
    end_silence_ms: int = 500           # this much silence closes an utterance
    pre_roll_ms: int = 200              # audio kept before speech onset
    min_speech_ms: int = 300            # shorter blips are ignored
    max_utterance_s: float = 15.0       # force-close very long turns
    partial_every_s: float = 0.5        # provisional emotion update period
    partial_min_s: float = 1.0          # need this much speech before first partial
    partial_window_s: float = 3.0       # provisional emotion looks at the last N s
    final_max_s: float = 10.0           # final emotion uses at most the last N s

    # response generation
    max_response_tokens: int = 60
    history_turns: int = 3
