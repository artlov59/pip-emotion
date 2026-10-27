"""Speech-to-text with Whisper (small, MLX port) running on the Apple GPU.

All calls must happen on the single "mlx" worker thread owned by the Engine.
"""
import time

import numpy as np


class WhisperASR:
    # Purpose: Store the Whisper model location; weights are loaded by load().
    # Input: repo — a Hugging Face model repository ID or local model directory.
    # Output: None; initializes self.repo.
    def __init__(self, repo: str = "mlx-community/whisper-small-mlx"):
        self.repo = repo

    # Purpose: Import MLX Whisper, load its weights, and warm up with silence.
    # Input: None; uses self.repo and must run on the Engine's MLX worker thread.
    # Output: None; prepares the model for transcribe(). First use may download weights.
    def load(self):
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self.transcribe(np.zeros(16000, dtype=np.float32))  # downloads (first run) + loads + warms up

    # Purpose: Convert one audio segment into English text using greedy decoding.
    # Input: wav — a one-dimensional, mono NumPy waveform sampled at 16 kHz;
    #        samples should be normalized to [-1, 1]. Call load() first and use
    #        the Engine's MLX worker thread.
    # Output: A dict containing stripped text, mean segment no_speech_prob
    #         (rounded to 3 decimals; 1.0 when no segments exist), and elapsed
    #         processing time in ms (rounded to 1 decimal).
    def transcribe(self, wav: np.ndarray) -> dict:
        t0 = time.perf_counter()
        out = self._mlx_whisper.transcribe(
            wav.astype(np.float32),
            path_or_hf_repo=self.repo,
            language="en",
            temperature=0.0,                  # greedy, no temperature fallback -> predictable latency
            condition_on_previous_text=False,
            verbose=None,
        )
        segs = out.get("segments") or []
        no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segs])) if segs else 1.0
        return {
            "text": (out.get("text") or "").strip(),
            "no_speech_prob": round(no_speech, 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
