"""Text emotion classifier: DistilRoBERTa fine-tuned on MELD utterances (training/train_text.py).

Input is the Whisper transcript of an utterance; output has the same shape as the audio
classifier so the two can be fused. Returns None for an empty transcript (modality missing).
"""
import json
import time
from pathlib import Path

import numpy as np

from app.config import EMOTIONS

MAX_LEN = 64


class TextEmotionClassifier:
    # Purpose: Configure the local seven-class text emotion classifier.
    # Input: model_dir: saved model/tokenizer Path; device: auto or explicit Torch device.
    # Output: None; stores model location and device preference.
    def __init__(self, model_dir: Path, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device_name = device
        self.meta = {}

    # Purpose: Validate MELD label order and load local text weights/tokenizer.
    # Input: None explicitly; uses the configured model directory and device.
    # Output: None; initializes the evaluated classifier; invalid metadata or absent local weights raise
    #          errors.
    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"Text model not found in {self.model_dir}. Run: python -m training.train_text")
        meta_path = self.model_dir / "meld_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not isinstance(self.meta, dict) or self.meta.get("classes", EMOTIONS) != EMOTIONS:
            raise ValueError("Text checkpoint metadata must use the shared MELD label order")
        self._torch = torch
        if self.device_name == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)
        self.tok = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        labels = getattr(model.config, "id2label", {})
        ordered = [labels.get(i, labels.get(str(i))) for i in range(len(EMOTIONS))] if isinstance(labels, dict) else []
        if getattr(model.config, "num_labels", None) != len(EMOTIONS) or ordered != EMOTIONS:
            raise ValueError("Text model config must use the shared MELD label order")
        self.model = model.to(self.device).eval()
        self.classes = list(EMOTIONS)
        self.predict("warm up")


    # Purpose: Classify multiple transcripts in a padded batch.
    # Input: texts: sequence of text strings; call load() first.
    # Output: NumPy probabilities shaped (number_of_texts, 7), in MELD label order.
    def predict_batch(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.inference_mode():
            return self._torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()

    # Purpose: Classify one transcript and report its processing time.
    # Input: text: recognized words; call load() first.
    # Output: Emotion/confidence/probs/ms dict, or None for empty/whitespace text.
    def predict(self, text: str):
        if not text or not text.strip():
            return None
        t0 = time.perf_counter()
        probs = self.predict_batch([text])[0]
        i = int(np.argmax(probs))
        return {
            "emotion": self.classes[i],
            "confidence": round(float(probs[i]), 4),
            "probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
