"""Sanity-check a cached Stage 5 feature level against a fresh recomputation.

    python scripts/check_features.py --level word

The extractor computes each file once and slices segments out of it, which is
fast but easy to get subtly wrong: an off-by-one in the frame index would
silently shift every segment. So this recomputes a handful of segments straight
from their audio, in isolation, and compares. It also looks for rows the memmap
never wrote, which is the failure mode that leaves plausible-looking zeros.
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.features.extract import PLOSIVE_N_FFT, chunk_starts  # noqa: E402
FEATURES = REPO_ROOT / "data" / "processed" / "features"
SEGMENTS = REPO_ROOT / "data" / "interim" / "segments.csv"

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 65
FMIN, FMAX = 20.0, 8000.0
CHUNK_FRAMES = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True)
    ap.add_argument("--n", type=int, default=6, help="segments to recompute")
    args = ap.parse_args()

    segments = pd.read_csv(SEGMENTS, low_memory=False)
    mine = segments[segments.level == args.level].set_index("segment_id")

    mel = np.load(FEATURES / f"mel_{args.level}.npy", mmap_mode="r")
    index = pd.read_csv(FEATURES / f"mel_{args.level}_index.csv")
    print(f"mel {mel.shape}, {len(index)} items")

    expected = int(index.n_frames.sum())
    assert expected == mel.shape[0], f"{expected} planned vs {mel.shape[0]}"
    assert index.row_start.is_monotonic_increasing
    print("layout: row_start monotonic, frame count matches the index")

    # An unwritten memmap row is exactly zero across all 65 bands. Real log-mel
    # rows are not: even digital silence lands near the -100 dB floor, and the
    # only legitimate zeros are the padding on short utterance chunks.
    sample = mel[::max(1, mel.shape[0] // 200_000)]
    dead = np.all(sample == 0.0, axis=1).mean()
    print(f"all-zero frames in a {len(sample)}-frame sample: {dead:.2%}")
    if args.level == "utterance":
        padded = 1 - index.n_valid_frames.sum() / index.n_frames.sum()
        print(f"  expected padding share: {padded:.2%}")
    else:
        assert dead < 1e-6, "unwritten rows in the memmap"

    print(f"log-mel range: {sample.min():.1f} .. {sample.max():.1f} dB "
          f"(mean {sample.mean():.1f})")

    rng = np.random.default_rng(0)
    picks = index.iloc[rng.choice(len(index), args.n, replace=False)]
    print(f"\nrecomputing {args.n} items from audio in isolation:")
    worst = 0.0
    for item in picks.itertuples():
        seg = mine.loc[item.segment_id]
        y, _ = sf.read(REPO_ROOT / seg.path, dtype="float64")
        n_fft = PLOSIVE_N_FFT if getattr(seg, "phone_class", "") == "plosive" else N_FFT
        power = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=n_fft, hop_length=HOP, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX)
        full = librosa.power_to_db(power, ref=1.0)

        if args.level == "utterance":
            # The last chunk of an utterance sits at duration - 4 s so the tail
            # is covered, so it is not at k * hop like the others.
            start = chunk_starts(seg.duration)[int(item.id.split("chunk")[1])]
            i0 = int(round(start * SR / HOP))
            want = full[:, i0:i0 + item.n_valid_frames].T
        else:
            i0 = int(round(seg.start * SR / HOP))
            want = full[:, i0:i0 + item.n_frames].T
        got = mel[item.row_start:item.row_start + len(want)]
        diff = np.abs(got - want).max() if len(want) else 0.0
        worst = max(worst, diff)
        print(f"  {item.id[:58]:58s} {len(want):4d} frames  max diff {diff:.2e}")
    assert worst < 1e-3, f"cached mel disagrees with a fresh computation ({worst})"
    print(f"worst disagreement {worst:.2e} dB -- within float32 rounding")

    hand = np.load(FEATURES / f"handcrafted_{args.level}.npy")
    columns = json.loads((FEATURES / "handcrafted_columns.json").read_text())
    hix = pd.read_csv(FEATURES / f"handcrafted_{args.level}_index.csv")
    frame = pd.DataFrame(hand, columns=columns)
    print(f"\nhandcrafted {hand.shape}, "
          f"voice-quality valid {hix.voice_quality_valid.mean():.1%}")
    assert (hix.segment_id.values == mine.index.values).all(), "index misaligned"
    show = ["f0_mean", "jitter_local", "shimmer_local", "f1_mean", "f2_mean",
            "f3_mean", "intensity_mean", "rms_mean", "zcr_mean", "mfcc01_mean"]
    print(frame[show].describe(percentiles=[.05, .5, .95]).round(3).to_string())


if __name__ == "__main__":
    main()
