"""Shared trunks for Stage 7.

Two backbones, one per feature family. Heads live in `classifier.py` so a
baseline run is the trunk plus a PD head, with no domain head allocated.

    mlp  40-d handcrafted vectors
    cnn  Mel spectrograms, 1D convolution over time, attention pooling
         (Ibarra et al. 2023's best generaliser)

The CNN takes (batch, n_mels, time). Attention pooling collapses whatever
time axis it is given, so the same weights serve a 26-frame duration window
and a 400-frame utterance chunk. A boolean mask, if given, zeros out padding
before the softmax so chunk padding cannot become a domain cue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpTrunk(nn.Module):
    """Two-layer MLP. `out_dim` is the embedding the heads read."""

    def __init__(self, n_in, width=256, dropout=0.3):
        super().__init__()
        hidden = width // 2
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, x, mask=None):
        return self.net(x)


class CnnTrunk(nn.Module):
    """1D-CNN with convolutional attention pooling."""

    def __init__(self, n_mels=65, width=128, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, width, 5, padding=2), nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv1d(width, width * 2, 3, padding=1),
            nn.BatchNorm1d(width * 2), nn.ReLU(),
        )
        self.attend = nn.Conv1d(width * 2, 1, 1)
        self.out_dim = width * 2

    def forward(self, x, mask=None):
        h = self.conv(x)
        scores = self.attend(h)
        if mask is not None:
            # mask is (batch, time); conv output time can theoretically shrink
            # with unpadded convs, but every layer here is padding-preserving.
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        weights = F.softmax(scores, dim=-1)
        return (h * weights).sum(-1)


def build_trunk(family, n_in=40, n_mels=65, width=None, dropout=None):
    """Construct the trunk for a feature family."""
    if family == "handcrafted":
        return MlpTrunk(n_in, width=width or 256, dropout=0.3 if dropout is None else dropout)
    if family == "mel":
        return CnnTrunk(n_mels, width=width or 128, dropout=0.0 if dropout is None else dropout)
    raise ValueError(f"unknown family {family!r}; expected 'handcrafted' or 'mel'")
