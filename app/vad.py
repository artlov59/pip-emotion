"""Voice-activity detection: Silero VAD (MIT licence, ~0.46 M-parameter TorchScript model).

Returns P(speech) for each 32 ms frame (512 samples @ 16 kHz). Turn-taking logic
(hysteresis, end-of-utterance silence) lives in app/pipeline.py so it can be
tested with a fake VAD.
"""
import warnings

import numpy as np


class SileroVAD:
    FRAME = 512  # samples at 16 kHz (the frame size Silero expects at this rate)

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        with warnings.catch_warnings():  # torch>=2.10 warns that torch.jit.load is deprecated
            warnings.simplefilter("ignore", FutureWarning)
            self.model = load_silero_vad()  # TorchScript, CPU; one instance per session (it is stateful)

    def prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame.astype(np.float32)), 16000))

    def reset(self):
        self.model.reset_states()


class EnergyVAD:
    """Dependency-free stand-in used by tests (not for real use)."""
    FRAME = 512

    def __init__(self, threshold_dbfs: float = -40.0):
        self.threshold = threshold_dbfs

    def prob(self, frame: np.ndarray) -> float:
        db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9)
        return 1.0 if db > self.threshold else 0.0

    def reset(self):
        pass
