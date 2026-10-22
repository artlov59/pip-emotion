"""Audio emotion classifier: the trained AudioCNN applied to a waveform."""
import time
from pathlib import Path

import numpy as np

from app.config import EMOTIONS, FEATURE_CONFIG
from app.features import log_mel, normalize


class EmotionClassifier:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.device_name = device
        self.model = None
        self.classes = EMOTIONS
        self.meta = {}

    def load(self):
        import torch

        from model import AudioCNN

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Emotion checkpoint not found: {self.checkpoint}. Train one (README, step 3) "
                f"or set EMOTION_CKPT=/path/to/best.pt")
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("feature_config") != FEATURE_CONFIG:
            raise ValueError(f"Checkpoint features {ckpt.get('feature_config')} != runtime {FEATURE_CONFIG}")
        if self.device_name == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)
        self.classes = ckpt["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.meta = {"run": ckpt.get("run"), "epoch": ckpt.get("epoch"), "dev_metrics": ckpt.get("dev_metrics"),
                     "device": str(self.device)}
        self._torch = torch
        self.predict(np.zeros(16000, dtype=np.float32))  # warm-up (first MPS call is slow)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, wav: np.ndarray) -> dict:
        t0 = time.perf_counter()
        mel = normalize(log_mel(wav))
        x = self._torch.from_numpy(mel)[None, None].to(self.device)
        with self._torch.inference_mode():
            probs = self._torch.softmax(self.model(x), dim=1)[0].float().cpu().numpy()
        i = int(probs.argmax())
        return {
            "emotion": self.classes[i],
            "confidence": round(float(probs[i]), 4),
            "probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
