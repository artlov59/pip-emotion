"""Speech-to-text with Whisper (small, MLX port) running on the Apple GPU.

All calls must happen on the single "mlx" worker thread owned by the Engine.
"""
import time

import numpy as np


class WhisperASR:
    def __init__(self, repo: str = "mlx-community/whisper-small-mlx"):
        self.repo = repo

    def load(self):
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self.transcribe(np.zeros(16000, dtype=np.float32))  # downloads (first run) + loads + warms up

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

    def num_parameters(self) -> int:
        from mlx_whisper.load_models import load_model

        from app.mlx_params import count_mlx_parameters
        return count_mlx_parameters(load_model(self.repo))
