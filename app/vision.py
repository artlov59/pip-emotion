"""Face emotion: OpenCV YuNet face detector + HSEmotion EfficientNet-B0 expression classifier.

Both are pretrained and used as-is (ONNX). HSEmotion predicts 8 AffectNet classes; they are
mapped to the 7 MELD emotions (contempt is merged into disgust). The largest detected face is
used; choosing the *speaking* face in multi-person shots is out of scope (see README).
"""
import time
import urllib.request

import numpy as np

from app.config import CHECKPOINT_DIR, EMOTIONS

YUNET_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
YUNET_PATH = CHECKPOINT_DIR / "face_detection_yunet_2023mar.onnx"
AFFECTNET_TO_MELD = {  # HSEmotion label (lower-case) -> MELD label
    "anger": "anger", "contempt": "disgust", "disgust": "disgust", "fear": "fear",
    "happiness": "joy", "neutral": "neutral", "sadness": "sadness", "surprise": "surprise",
}


class FaceEmotion:
    # Purpose: Configure face detection and facial emotion inference.
    # Input: model_name: HSEmotion model; det_threshold: detector cutoff; max_side: detector resize
    #          limit.
    # Output: None; stores settings without loading models.
    def __init__(self, model_name: str = "enet_b0_8_best_vgaf", det_threshold: float = 0.7, max_side: int = 640):
        self.model_name = model_name
        self.det_threshold = det_threshold
        self.max_side = max_side
        self.repo = f"hsemotion-onnx:{model_name} + YuNet-2023mar"

    # Purpose: Initialize the YuNet detector and HSEmotion classifier.
    # Input: None explicitly; uses the configured models and detector threshold.
    # Output: None; loads detection/classification components and emotion label mapping.
    def load(self):
        import cv2
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self._cv2 = cv2
        if not YUNET_PATH.exists():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"[vision] downloading YuNet face detector -> {YUNET_PATH}", flush=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        self.detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), self.det_threshold, 0.3, 50)
        self.fer = HSEmotionRecognizer(model_name=self.model_name)
        labels = getattr(self.fer, "idx_to_class", None)
        if not labels:
            raise RuntimeError("hsemotion_onnx changed: no idx_to_class mapping")
        self.src_labels = [str(labels[i]).lower() for i in range(len(labels))]
        self.map = np.zeros((len(self.src_labels), len(EMOTIONS)), dtype=np.float32)
        for i, lab in enumerate(self.src_labels):
            self.map[i, EMOTIONS.index(AFFECTNET_TO_MELD[lab])] = 1.0
        self.analyze(np.zeros((240, 320, 3), dtype=np.uint8))  # warm-up

    # Purpose: Find the largest accepted face and map it back to original image pixels.
    # Input: bgr: OpenCV BGR image.
    # Output: Tuple (x, y, width, height, detector_score), or None when no face is found.
    def detect(self, bgr: np.ndarray):
        """Largest face as (x, y, w, h, score) in original pixel coords, or None."""
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = self._cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, fw, fh = (f[:4] / scale).tolist()
        return x, y, fw, fh, float(f[-1])

    # Purpose: Detect and classify the main face in one video frame.
    # Input: bgr: OpenCV BGR image; call load() first.
    # Output: Dict with face flag and ms; detected faces also include MELD probs, normalized box, and
    #          detection score.
    def analyze(self, bgr: np.ndarray) -> dict:
        """One BGR frame -> {"face": bool, "probs": {...}, "box": [x,y,w,h] (0-1), "ms": ...}."""
        t0 = time.perf_counter()
        det = self.detect(bgr)
        if det is None:
            return {"face": False, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        x, y, fw, fh, score = det
        H, W = bgr.shape[:2]
        m = 0.1  # margin around the box, as in HSEmotion's examples
        x0, y0 = int(max(0, x - m * fw)), int(max(0, y - m * fh))
        x1, y1 = int(min(W, x + (1 + m) * fw)), int(min(H, y + (1 + m) * fh))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return {"face": False, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        rgb = self._cv2.cvtColor(bgr[y0:y1, x0:x1], self._cv2.COLOR_BGR2RGB)
        _, scores = self.fer.predict_emotions(rgb, logits=False)
        probs = np.asarray(scores, dtype=np.float32) @ self.map
        probs = probs / max(float(probs.sum()), 1e-6)
        return {
            "face": True,
            "probs": {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)},
            "box": [round(x / W, 3), round(y / H, 3), round(fw / W, 3), round(fh / H, 3)],
            "det_score": round(score, 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }




# Purpose: Average usable face distributions across a captured interval.
# Input: frames: face-analysis dicts; entries without a detected face are ignored.
# Output: Emotion/confidence/probs/frame-count dict, or None when no face probabilities exist.
def mean_probs(frames: list) -> dict:
    """Average per-frame face probabilities (frames without a face are ignored)."""
    ps = [f["probs"] for f in frames if f.get("face")]
    if not ps:
        return None
    arr = np.array([[p[e] for e in EMOTIONS] for p in ps], dtype=np.float32).mean(0)
    i = int(arr.argmax())
    return {"emotion": EMOTIONS[i], "confidence": round(float(arr[i]), 4),
            "probs": {e: round(float(v), 4) for e, v in zip(EMOTIONS, arr)}, "frames": len(ps)}
