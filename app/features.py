"""Audio loading and log-mel features, shared by data prep, training and live inference."""
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.config import FMAX, FMIN, HOP, N_FFT, N_MELS, SAMPLE_RATE


# Purpose: Decode an audio/video file, downmix to mono, and resample.
# Input: path: input file; sr: target sample rate in Hz, default 16000.
# Output: One-dimensional float32 NumPy waveform; falls back to ffmpeg when needed.
def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio/video file as mono float32 at `sr` Hz."""
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        return wav.astype(np.float32)
    # mp4 / m4a / mp3 / webm ... -> decode with ffmpeg (downmixes 5.1 tracks too)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


# Purpose: Compute the CNN's natural-log mel power features.
# Input: wav: mono waveform at SAMPLE_RATE (16 kHz).
# Output: Float32 array shaped (n_mels, T), at 100 frames per second.
def log_mel(wav: np.ndarray) -> np.ndarray:
    """(n_mels, T) natural-log mel power spectrogram, float32. 100 frames per second."""
    if wav.size < N_FFT:
        wav = np.pad(wav, (0, N_FFT - wav.size))
    mel = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0, center=True)
    return np.log(mel + 1e-6).astype(np.float32)


# Purpose: Remove a clip's overall gain from its log-mel features.
# Input: mel: NumPy log-mel array.
# Output: Float32 features with approximately zero mean and unit variance.
def normalize(mel: np.ndarray) -> np.ndarray:
    """Per-utterance normalisation (zero mean, unit variance over the whole clip).

    Removes absolute gain, which differs a lot between TV audio (MELD) and a live
    laptop microphone. Trade-off: absolute loudness is no longer visible to the CNN;
    loudness is reported separately by app/prosody.py instead.
    """
    return ((mel - mel.mean()) / (mel.std() + 1e-5)).astype(np.float32)
