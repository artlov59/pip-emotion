"""MELD log-mel dataset (features are precomputed by training/prepare_meld.py)."""
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.config import MELD_DIR
from app.features import normalize

MIN_FRAMES = 32  # 0.32 s; shorter clips are zero-padded so the CNN has something to pool


class MeldMelDataset(Dataset):
    def __init__(self, split: str, train: bool, max_frames: int = 400, spec_augment: bool = True):
        df = pd.read_csv(MELD_DIR / f"{split}.csv")
        df = df[df["status"] == "ok"].reset_index(drop=True)
        with open(MELD_DIR / "features" / f"{split}.pkl", "rb") as f:
            self.mels = pickle.load(f)
        df = df[df["id"].isin(self.mels.keys())].reset_index(drop=True)
        self.df = df
        self.ids = df["id"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.train = train
        self.max_frames = max_frames
        self.spec_augment = spec_augment and train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mel = self.mels[self.ids[idx]].astype(np.float32)
        t = mel.shape[1]
        if t > self.max_frames:
            start = random.randint(0, t - self.max_frames) if self.train else (t - self.max_frames) // 2
            mel = mel[:, start:start + self.max_frames]
        mel = normalize(mel)
        if self.spec_augment:
            mel = spec_augment(mel)
        return torch.from_numpy(mel), self.labels[idx]


def spec_augment(mel: np.ndarray, n_freq=2, max_f=8, n_time=2, max_t=40) -> np.ndarray:
    mel = mel.copy()
    n_mels, t = mel.shape
    for _ in range(n_freq):
        w = random.randint(0, max_f)
        f0 = random.randint(0, max(0, n_mels - w))
        mel[f0:f0 + w, :] = 0.0
    max_t = min(max_t, t // 5)
    for _ in range(n_time):
        w = random.randint(0, max_t) if max_t > 0 else 0
        t0 = random.randint(0, max(0, t - w))
        mel[:, t0:t0 + w] = 0.0
    return mel


def collate(batch):
    """Zero-pad to the longest clip in the batch; return (x, lengths, labels)."""
    mels, labels = zip(*batch)
    lengths = torch.tensor([m.shape[1] for m in mels])
    t_max = max(int(lengths.max()), MIN_FRAMES)
    x = torch.zeros(len(mels), 1, mels[0].shape[0], t_max)
    for i, m in enumerate(mels):
        x[i, 0, :, :m.shape[1]] = m
    return x, lengths, torch.tensor(labels)
