"""Can this GPU carry Stage 7, and where is the real bottleneck?

    python scripts/benchmark_gpu.py

Measures training throughput for the two Stage 7 backbones at the input shapes
Stage 5 actually produced, then projects the run matrix. Also times random reads
from the cached Mel memmap, because for models this small the disk is a more
likely bottleneck than the GPU, and a GPU benchmark alone would hide that.

Re-run this on the KCDH server to get a comparable number.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import store                              # noqa: E402

# Shapes as cached by Stage 5: 65 Mel bands, 10 ms hop.
DURATION_FRAMES = 26        # 250 ms windows
UTTERANCE_FRAMES = 400      # 4 s chunks
N_MELS = 65
N_HANDCRAFTED = 40

DURATION_SEGMENTS = 274_459
UTTERANCE_CHUNKS = 20_397
EPOCHS = 30


class Mlp(nn.Module):
    """Baseline for the handcrafted family: 40 features in, two heads out."""

    def __init__(self, n_in=N_HANDCRAFTED, width=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.ReLU(),
        )
        self.label = nn.Linear(width // 2, 2)
        self.domain = nn.Linear(width // 2, 3)

    def forward(self, x):
        h = self.trunk(x)
        return self.label(h), self.domain(h)


class AttentiveCnn(nn.Module):
    """1D-CNN with attention pooling, the reference paper's best generaliser.

    Mel bands are the channel dimension and convolution runs along time, so the
    same model takes a 26-frame window or a 400-frame chunk without change --
    attention pooling collapses whatever time axis it is given.
    """

    def __init__(self, n_mels=N_MELS, width=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, width, 5, padding=2), nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width * 2, 3, padding=1),
            nn.BatchNorm1d(width * 2), nn.ReLU(),
        )
        self.attend = nn.Conv1d(width * 2, 1, 1)
        self.label = nn.Linear(width * 2, 2)
        self.domain = nn.Linear(width * 2, 3)

    def forward(self, x):
        h = self.conv(x)
        weights = torch.softmax(self.attend(h), dim=-1)
        pooled = (h * weights).sum(-1)
        return self.label(pooled), self.domain(pooled)


def throughput(model, batch, device, steps=40):
    """Samples per second for a full training step, and peak memory."""
    model = model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x = batch.to(device)
    y = torch.randint(0, 2, (len(x),), device=device)
    d = torch.randint(0, 3, (len(x),), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 10):
        if i == 10:                       # discard warm-up, including cuDNN autotune
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        label, domain = model(x)
        loss = loss_fn(label, y) + loss_fn(domain, d)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda" else float("nan"))
    return len(x) * steps / elapsed, peak


def report(name, model_fn, batch, n_samples, device):
    rate, peak = throughput(model_fn(), batch, device)
    epoch = n_samples / rate
    print(f"  {name:34s} {rate:9,.0f} samp/s   epoch {epoch:6.1f}s   "
          f"{EPOCHS}-epoch run {epoch * EPOCHS / 60:5.1f} min   "
          f"peak VRAM {peak:.2f} GB")
    return epoch * EPOCHS / 60


def main():
    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{torch.cuda.get_device_name(0)}, {total:.1f} GB VRAM, "
          f"torch {torch.__version__}\n")

    print("training throughput (batch 256, both heads active):")
    minutes = {}
    minutes["hand_dur"] = report(
        "handcrafted MLP, duration", Mlp,
        torch.randn(256, N_HANDCRAFTED), DURATION_SEGMENTS, device)
    minutes["mel_dur"] = report(
        "Mel CNN, duration (26 frames)", AttentiveCnn,
        torch.randn(256, N_MELS, DURATION_FRAMES), DURATION_SEGMENTS, device)
    minutes["mel_utt"] = report(
        "Mel CNN, utterance (400 frames)", AttentiveCnn,
        torch.randn(256, N_MELS, UTTERANCE_FRAMES), UTTERANCE_CHUNKS, device)

    # For models this small the GPU is rarely the constraint; feeding it is.
    print("\nrandom reads from the cached Mel memmap:")
    for level, n_wanted in (("duration", 20_000), ("utterance", 4_000)):
        data, index = store.load_mel(level)
        rng = np.random.default_rng(0)
        picks = index.iloc[rng.choice(len(index), n_wanted, replace=False)]
        start = time.perf_counter()
        frames = sum(len(store.mel_item(data, item))
                     for item in picks.itertuples())
        elapsed = time.perf_counter() - start
        rate = n_wanted / elapsed
        size = data.shape[0] * data.shape[1] * 4 / 2**30
        print(f"  {level:10s} {rate:9,.0f} items/s  "
              f"({frames / n_wanted:5.1f} frames each, {size:.2f} GB on disk)")
        n_total = DURATION_SEGMENTS if level == "duration" else UTTERANCE_CHUNKS
        print(f"             -> {n_total / rate:6.1f}s to stream one epoch "
              f"cold; RAM-resident after the first pass")

    print("\nrun matrix for the duration level, mixed protocol:")
    per_config = minutes["hand_dur"] + minutes["mel_dur"]
    print(f"  baseline + DANN, both families, 10 configs: "
          f"{2 * per_config * 10 / 60:.1f} GPU-hours")
    print(f"  adding 3 single-corpus baselines per config: "
          f"{5 * per_config * 10 / 60:.1f} GPU-hours")


if __name__ == "__main__":
    main()
