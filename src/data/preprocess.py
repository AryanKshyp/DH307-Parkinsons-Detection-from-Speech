"""Downmix to mono, resample to 16 kHz, peak-normalise.

    python -m src.data.preprocess                    # all corpora
    python -m src.data.preprocess --dataset NeuroVoz # one corpus

Raw audio is never modified; outputs go to data/processed/<dataset>/ and
data/manifests/manifest_processed.csv.

Peak normalisation only. Loudness normalisation would remove reduced vocal
loudness, which is a PD symptom rather than a recording artefact.

Alongside each output the manifest records three quality measures computed on
the *source* signal, so that the files worth listening to can be picked out
rather than sampled at random: peak level, the fraction of samples at digital
full scale, and the fraction of the file that is silent.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest.csv"
OUT_MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
OUT_AUDIO = REPO_ROOT / "data" / "processed"
BLACKLIST = REPO_ROOT / "configs" / "noise_blacklist.txt"

TARGET_SR = 16000
PEAK_TARGET = 0.99
SILENCE_DB = -50.0        # relative to the file's own peak
FRAME_MS = 20


def load_blacklist():
    if not BLACKLIST.exists():
        return set()
    paths = set()
    for line in BLACKLIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            paths.add(line.replace("\\", "/"))
    return paths


def safe_name(text):
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in text]
    return "".join(keep)


def quality(y, sr):
    """Peak level in dBFS, clipped-sample fraction, silent-frame fraction."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    clipped = float(np.mean(np.abs(y) >= 0.999)) if y.size else 0.0

    frame = max(1, int(sr * FRAME_MS / 1000))
    n = (y.size // frame) * frame
    if n == 0 or peak == 0.0:
        return peak, clipped, 1.0
    frames = y[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    quiet = 20 * np.log10(np.maximum(rms, 1e-12) / peak) < SILENCE_DB
    return peak, clipped, float(np.mean(quiet))


def process(row):
    src = REPO_ROOT / row["path"]
    y, _ = librosa.load(src, sr=TARGET_SR, mono=True)

    peak, clipped, silent = quality(y, TARGET_SR)
    if peak == 0.0:
        return None, "all-zero signal"

    y = y / peak * PEAK_TARGET

    stem = safe_name(f'{row["speaker_id"]}__{row["task"]}__{Path(row["path"]).stem}')
    dst = OUT_AUDIO / row["dataset"] / f"{stem}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, TARGET_SR, subtype="PCM_16")

    out = dict(row)
    out["source_path"] = row["path"]
    out["path"] = dst.relative_to(REPO_ROOT).as_posix()
    out["sample_rate"] = TARGET_SR
    out["n_channels"] = 1
    out["duration"] = round(len(y) / TARGET_SR, 3)
    out["source_sample_rate"] = row["sample_rate"]
    out["peak_dbfs"] = round(float(20 * np.log10(max(peak, 1e-12))), 2)
    out["clipped_frac"] = round(clipped, 6)
    out["silence_frac"] = round(silent, 4)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="process one dataset only")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit("manifest.csv not found; run src.data.build_manifest first")

    manifest = pd.read_csv(MANIFEST)
    if args.dataset:
        manifest = manifest[manifest.dataset == args.dataset]
        if manifest.empty:
            sys.exit(f"no rows for dataset {args.dataset!r}")

    blacklist = load_blacklist()
    unknown = blacklist - set(manifest.path) if not args.dataset else set()
    if unknown:
        print("WARNING: blacklist entries not found in the manifest:")
        for path in sorted(unknown):
            print(f"  {path}")

    kept = manifest[~manifest.path.isin(blacklist)]
    print(f"{len(manifest)} files, {len(manifest) - len(kept)} blacklisted, "
          f"{len(kept)} to process")

    rows, failures = [], []
    for _, row in tqdm(kept.iterrows(), total=len(kept), unit="file"):
        out, error = process(row)
        (failures if error else rows).append((row["path"], error) if error else out)

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for path, error in failures:
            print(f"  {path}: {error}")

    out = pd.DataFrame(rows)
    if args.dataset and OUT_MANIFEST.exists():
        previous = pd.read_csv(OUT_MANIFEST)
        out = pd.concat([previous[previous.dataset != args.dataset], out],
                        ignore_index=True)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MANIFEST, index=False)

    print(f"\nwrote {len(out)} rows to {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    print(out.groupby("dataset").agg(
        files=("path", "size"),
        hours=("duration", lambda s: round(s.sum() / 3600, 2))).to_string())


if __name__ == "__main__":
    main()
