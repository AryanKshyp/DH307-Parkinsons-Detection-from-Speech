"""Stage 5 feature extraction: Mel spectrograms and handcrafted descriptors.

    python -m src.features.extract
    python -m src.features.extract --levels word --family mel

Writes to data/processed/features/. For each (family, level) pair:

    <family>_<level>.npy         float32 data
    <family>_<level>_index.csv   one row per item, giving its slice of the data

Mel data is time-major, shape (total_frames, 65), so an item is
data[row_start : row_start + n_frames]. Handcrafted data is one fixed-length
row per segment. The index carries `segment_id`, so either family joins back to
segments.csv and from there to the speaker and label.

Features are cached **unnormalised**. Normalisation scope is a modelling choice
-- per-speaker is the current default -- and baking it in here would make
changing it a full re-extraction. Concretely, `power_to_db` uses a fixed
reference rather than each segment's own peak, which would be per-segment
normalisation applied by accident.

Everything is computed once per file and sliced per segment, rather than
recomputed per segment. That is roughly a hundredfold saving on 374k segments,
and it also means neighbouring segments share a framing, so their features are
directly comparable.

Utterance-level spectrograms are stored as fixed 4 s chunks at a 2 s hop, not as
whole utterances. Utterance length is strongly corpus-dependent -- a 3.5 s
NeuroVoz sentence against a 141 s MDVR-KCL phone call -- so a variable-length
tensor would hand the model the domain label through its shape before it saw any
audio. Utterances shorter than one chunk are zero-padded, and the index records
`n_valid_frames` so a mask can be rebuilt.
"""

import argparse
import json
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = REPO_ROOT / "data" / "interim" / "segments.csv"
OUT = REPO_ROOT / "data" / "processed" / "features"

SR = 16000
N_FFT = 400          # 25 ms
HOP = 160            # 10 ms
PLOSIVE_N_FFT = 240  # 15 ms; hop stays 10 ms so the frame grid is shared.
                     # Used only for phoneme-level plosives. Uniform 25 ms is
                     # kept for every other class and level; this is the one
                     # exception, so a plosive DANN is not starved of frames.
N_MELS = 65
N_MFCC = 13
FMIN, FMAX = 20.0, 8000.0

CHUNK_S = 4.0
CHUNK_HOP_S = 2.0
CHUNK_FRAMES = int(round(CHUNK_S * SR / HOP))

# Praat's pitch floor sets the shortest analysable window: it needs three
# periods, so 75 Hz means segments under 40 ms have no F0 at all.
PITCH_FLOOR, PITCH_CEILING = 75.0, 600.0
MAX_FORMANT = 5000.0

HANDCRAFTED_COLUMNS = (
    [f"mfcc{i:02d}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i:02d}_std" for i in range(1, N_MFCC + 1)]
    + ["zcr_mean", "zcr_std", "rms_mean", "rms_std",
       "f0_mean", "f0_std", "voiced_frac",
       "jitter_local", "shimmer_local",
       "f1_mean", "f2_mean", "f3_mean",
       "intensity_mean", "intensity_std"]
)


def frame_at(seconds):
    return int(round(seconds * SR / HOP))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std())


# --------------------------------------------------------------------------
# layout: worked out from segments.csv alone, so the output can be written
# straight into a memmap instead of accumulated in RAM
# --------------------------------------------------------------------------

def chunk_starts(duration):
    """Start times of the fixed chunks covering one utterance."""
    if duration <= CHUNK_S:
        return [0.0]
    starts = list(np.arange(0.0, duration - CHUNK_S + 1e-9, CHUNK_HOP_S))
    tail = duration - CHUNK_S
    if tail - starts[-1] > 1e-9:
        starts.append(tail)
    return starts


def plan_mel(segments, level):
    """One row per item: where its frames go and how many are real."""
    rows, offset = [], 0
    for seg in segments.itertuples():
        if level == "utterance":
            for k, start in enumerate(chunk_starts(seg.duration)):
                valid = min(CHUNK_FRAMES, frame_at(seg.duration) - frame_at(start))
                rows.append({"id": f"{seg.segment_id}__chunk{k:03d}",
                             "segment_id": seg.segment_id, "path": seg.path,
                             "start": start, "row_start": offset,
                             "n_frames": CHUNK_FRAMES,
                             "n_valid_frames": max(0, valid)})
                offset += CHUNK_FRAMES
        else:
            n = frame_at(seg.end) - frame_at(seg.start)
            if n <= 0:
                continue
            rows.append({"id": seg.segment_id, "segment_id": seg.segment_id,
                         "path": seg.path, "start": seg.start,
                         "phone_class": getattr(seg, "phone_class", "") or "",
                         "row_start": offset, "n_frames": n,
                         "n_valid_frames": n})
            offset += n
    return pd.DataFrame(rows), offset


# --------------------------------------------------------------------------
# per-file analysis, computed once and sliced per segment
# --------------------------------------------------------------------------

class FileAnalysis:
    def __init__(self, path, need_handcrafted, need_plosive_mel=False):
        y, sr = sf.read(REPO_ROOT / path, dtype="float64")
        if sr != SR:
            raise ValueError(f"expected {SR} Hz, got {sr}")
        self.y = y

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX)
        # A fixed reference, never ref=np.max: the latter is a per-segment
        # normalisation, which this stage deliberately does not apply.
        self.logmel = librosa.power_to_db(mel, ref=1.0).astype(np.float32)
        self.n_frames = self.logmel.shape[1]
        self.logmel_plosive = None
        if need_plosive_mel:
            pmel = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=PLOSIVE_N_FFT, hop_length=HOP,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
            self.logmel_plosive = librosa.power_to_db(pmel, ref=1.0).astype(
                np.float32)

        if not need_handcrafted:
            return

        self.mfcc = librosa.feature.mfcc(S=self.logmel.astype(np.float64),
                                         n_mfcc=N_MFCC)
        self.zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=N_FFT, hop_length=HOP)[0]
        self.rms = librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP)[0]

        self.sound = parselmouth.Sound(y, sampling_frequency=SR)
        self.f0 = self._pitch_track()
        self.formants = self._formant_tracks()
        self.intensity = self._intensity_track()
        self.point_process = self._point_process()

    # Praat refuses to analyse material shorter than its own window, and the
    # failure is a thrown exception rather than a return value, so every track
    # is optional and its absence becomes NaN downstream.
    def _pitch_track(self):
        try:
            pitch = self.sound.to_pitch(time_step=HOP / SR,
                                        pitch_floor=PITCH_FLOOR,
                                        pitch_ceiling=PITCH_CEILING)
        except Exception:                                   # noqa: BLE001
            return None
        f0 = np.array(pitch.selected_array["frequency"], dtype=np.float64)
        f0[f0 == 0.0] = np.nan
        return self._resample(f0, pitch.xs())

    def _formant_tracks(self):
        try:
            formant = self.sound.to_formant_burg(
                time_step=HOP / SR, maximum_formant=MAX_FORMANT)
        except Exception:                                   # noqa: BLE001
            return None
        times = formant.xs()
        tracks = [np.array([formant.get_value_at_time(n, t) for t in times])
                  for n in (1, 2, 3)]
        return [self._resample(track, times) for track in tracks]

    def _intensity_track(self):
        try:
            intensity = self.sound.to_intensity(minimum_pitch=PITCH_FLOOR,
                                                time_step=HOP / SR)
        except Exception:                                   # noqa: BLE001
            return None
        return self._resample(np.array(intensity.values[0]), intensity.xs())

    def _point_process(self):
        try:
            return call(self.sound, "To PointProcess (periodic, cc)",
                        PITCH_FLOOR, PITCH_CEILING)
        except Exception:                                   # noqa: BLE001
            return None

    def _resample(self, values, times):
        """Put a Praat track on the same 10 ms grid as the spectrogram."""
        grid = np.arange(self.n_frames) * HOP / SR
        return np.interp(grid, times, values, left=np.nan, right=np.nan)

    def handcrafted(self, start, end):
        i0, i1 = frame_at(start), max(frame_at(end), frame_at(start) + 1)
        i1 = min(i1, self.n_frames)
        out = []

        window = self.mfcc[:, i0:i1]
        for row in window:
            out.append(np.nanmean(row) if row.size else np.nan)
        for row in window:
            out.append(np.nanstd(row) if row.size else np.nan)
        out.extend(mean_std(self.zcr[i0:i1]))
        out.extend(mean_std(self.rms[i0:i1]))

        if self.f0 is None:
            out.extend([np.nan, np.nan, np.nan])
            voiced = 0.0
        else:
            f0 = self.f0[i0:i1]
            out.extend(mean_std(f0))
            voiced = float(np.mean(np.isfinite(f0))) if f0.size else 0.0
            out.append(voiced)

        jitter = shimmer = np.nan
        if self.point_process is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    jitter = call(self.point_process, "Get jitter (local)",
                                  start, end, 1e-4, 0.02, 1.3)
                    shimmer = call([self.sound, self.point_process],
                                   "Get shimmer (local)", start, end,
                                   1e-4, 0.02, 1.3, 1.6)
                except Exception:                           # noqa: BLE001
                    pass
        out.extend([jitter, shimmer])

        if self.formants is None:
            out.extend([np.nan, np.nan, np.nan])
        else:
            for track in self.formants:
                segment = track[i0:i1]
                out.append(np.nanmean(segment) if np.any(np.isfinite(segment))
                           else np.nan)
        out.extend(mean_std(self.intensity[i0:i1])
                   if self.intensity is not None else [np.nan, np.nan])

        valid = bool(np.isfinite(jitter) and np.isfinite(shimmer) and voiced > 0)
        return out, valid

    def mel_slice(self, start, n_frames, pad_to=None, plosive=False):
        spec = self.logmel_plosive if (plosive and self.logmel_plosive is not None) \
            else self.logmel
        i0 = frame_at(start)
        block = spec[:, i0:i0 + n_frames].T
        width = pad_to or n_frames
        if block.shape[0] < width:
            block = np.pad(block, ((0, width - block.shape[0]), (0, 0)))
        return block[:width]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+",
                    default=["utterance", "duration", "duration_100",
                             "word", "phoneme"])
    ap.add_argument("--family", default="both",
                    choices=["mel", "handcrafted", "both"])
    args = ap.parse_args()

    want_mel = args.family in ("mel", "both")
    want_hand = args.family in ("handcrafted", "both")
    OUT.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(SEGMENTS, low_memory=False)

    for level in args.levels:
        mine = segments[segments.level == level]
        if mine.empty:
            print(f"{level}: no segments, skipping")
            continue
        print(f"\n=== {level}: {len(mine)} segments, "
              f"{mine.path.nunique()} files ===")

        plan, total_frames = (plan_mel(mine, level) if want_mel
                              else (pd.DataFrame(), 0))
        mel_data = None
        if want_mel:
            gb = total_frames * N_MELS * 4 / 1e9
            print(f"mel: {len(plan)} items, {total_frames} frames, {gb:.2f} GB")
            mel_data = np.lib.format.open_memmap(
                OUT / f"mel_{level}.npy", mode="w+", dtype=np.float32,
                shape=(total_frames, N_MELS))
        by_path = {p: g for p, g in plan.groupby("path")} if want_mel else {}

        hand = np.full((len(mine), len(HANDCRAFTED_COLUMNS)), np.nan,
                       dtype=np.float32)
        valid_flags = np.zeros(len(mine), dtype=bool)
        positions = {p: g for p, g in
                     mine.reset_index(drop=True).groupby("path")}

        for path in tqdm(sorted(mine.path.unique()), unit="file"):
            analysis = FileAnalysis(
                path, want_hand,
                need_plosive_mel=want_mel and level == "phoneme")
            if want_mel:
                for item in by_path[path].itertuples():
                    plosive = (level == "phoneme"
                               and getattr(item, "phone_class", "") == "plosive")
                    block = analysis.mel_slice(
                        item.start, item.n_frames,
                        CHUNK_FRAMES if level == "utterance" else None,
                        plosive=plosive)
                    mel_data[item.row_start:item.row_start + item.n_frames] = block
            if want_hand:
                for row in positions[path].itertuples():
                    values, valid = analysis.handcrafted(row.start, row.end)
                    hand[row.Index] = values
                    valid_flags[row.Index] = valid

        if want_mel:
            mel_data.flush()
            drop = [c for c in ("path", "start", "phone_class") if c in plan.columns]
            plan.drop(columns=drop).to_csv(
                OUT / f"mel_{level}_index.csv", index=False)
            print(f"wrote mel_{level}.npy {mel_data.shape}")

        if want_hand:
            np.save(OUT / f"handcrafted_{level}.npy", hand)
            pd.DataFrame({
                "segment_id": mine.segment_id.values,
                "row": np.arange(len(mine)),
                "voice_quality_valid": valid_flags,
            }).to_csv(OUT / f"handcrafted_{level}_index.csv", index=False)
            frame = pd.DataFrame(hand, columns=HANDCRAFTED_COLUMNS)
            print(f"wrote handcrafted_{level}.npy {hand.shape}")
            print(f"  voice-quality valid: {valid_flags.mean():.1%}")
            missing = frame.isna().mean().sort_values(ascending=False)
            worst = missing[missing > 0]
            if len(worst):
                print("  columns with missing values:")
                print(worst.head(8).mul(100).round(1).to_string())

    (OUT / "handcrafted_columns.json").write_text(
        json.dumps(HANDCRAFTED_COLUMNS, indent=1), encoding="utf-8")
    print(f"\nfeatures in {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
