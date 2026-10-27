"""Voice-activity detection: Silero VAD (MIT licence, ~0.46 M-parameter TorchScript model).

Returns P(speech) for each 32 ms frame (512 samples @ 16 kHz). Turn-taking logic
(hysteresis, end-of-utterance silence) lives in app/pipeline.py so it can be
tested with a fake VAD.
"""
import warnings

import numpy as np


class SileroVAD:
    FRAME = 512  # samples at 16 kHz (the frame size Silero expects at this rate)

    # Purpose: Load the packaged Silero speech detector and initialize its state.
    # Input: None.
    # Output: None; prepares the Torch VAD model used for one conversation.
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        with warnings.catch_warnings():  # torch>=2.10 warns that torch.jit.load is deprecated
            warnings.simplefilter("ignore", FutureWarning)
            self.model = load_silero_vad()  # TorchScript, CPU; one instance per session (it is stateful)

    # Purpose: Estimate speech probability for one Silero frame.
    # Input: frame: 512 mono float samples at 16 kHz.
    # Output: Float speech probability; advances the detector's recurrent state.
    def prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame.astype(np.float32)), 16000))

    # Purpose: Clear the detector's recurrent speech history.
    # Input: None explicitly; uses the instance model.
    # Output: None; resets model state for a new detection sequence.
    def reset(self):
        self.model.reset_states()
