"""ResNet-style CNN over log-mel spectrograms.

Originally written for ESC-50 (50 environmental-sound classes); reused here for
the 7 MELD emotion classes. Changes from the ESC-50 version:
  * removed duplicated feature-map/ReLU lines in ResidualBlock.forward
  * removed the feature-map visualisation path (no longer used)
  * added length-aware (masked) temporal pooling so zero-padded frames in a
    batch do not dilute the clip embedding; with lengths=None it is identical
    to global average pooling.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_shortcut = stride != 1 or in_channels != out_channels
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if self.use_shortcut
            else nn.Identity()
        )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class AudioCNN(nn.Module):
    """Input: (B, 1, n_mels, T) normalised log-mel. Output: (B, num_classes) logits."""

    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def embed(self, x, lengths=None):
        t_in = x.shape[-1]
        x = self.conv1(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
        x = x.mean(dim=2)  # pool frequency -> (B, C, T')
        if lengths is None:
            return x.mean(dim=2)
        t_out = x.shape[-1]
        valid = torch.clamp(torch.ceil(lengths.float() * t_out / t_in), min=1, max=t_out)
        mask = (torch.arange(t_out, device=x.device)[None, :] < valid[:, None]).to(x.dtype)
        return (x * mask[:, None, :]).sum(dim=2) / valid[:, None].to(x.dtype)

    def forward(self, x, lengths=None):
        return self.fc(self.dropout(self.embed(x, lengths)))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    m = AudioCNN(num_classes=7)
    print(f"AudioCNN parameters: {count_parameters(m):,}")
    y = m(torch.randn(2, 1, 64, 300), torch.tensor([300, 120]))
    print("output", tuple(y.shape))
