"""Load Stage 5 features and apply a normalisation scope at load time.

Stage 5 caches features unnormalised, so the scope is chosen here rather than
baked into the cache. Scopes:

    none      leave the cached values alone
    global    one mean and std over the whole fitted set
    speaker   one mean and std per speaker
    utterance one mean and std per source file
    corpus    one mean and std per dataset, plus `__all__` from the same
              fitted rows as the fallback when a dataset was not seen
              (strict LODO). This is the optional batch-correction ablation;
              it is not the default. See experiments/corpus_norm_ablation/.

`global` is the default, and must be fitted on the training split only -- hence
`fit_stats` and `apply` being separable rather than one call.

`speaker` and `utterance` are kept for ablation, not for use. They do not leak
across splits, since their statistics come from inside a single speaker or file
and Stage 6 assigns whole speakers to one split. The problem is worse than
leakage: PD/HC is constant within a speaker, so the entire label signal sits in
how one speaker's feature means differ from another's, and subtracting each
speaker's own mean sets that spread to zero. Measured on word-level features,
the between-speaker spread of feature means goes from 13.08 to 1.9e-16 and PD/HC
balanced accuracy falls from 0.679 to 0.510, which is chance. It removes the
recording channel and the target together. `utterance` behaves the same way
because utterances nest inside speakers. See `scripts/probe_normalisation.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES = REPO_ROOT / "data" / "processed" / "features"
SEGMENTS = REPO_ROOT / "data" / "interim" / "segments.csv"

SCOPES = ("none", "global", "speaker", "utterance", "corpus")


def columns():
    return json.loads((FEATURES / "handcrafted_columns.json").read_text(
        encoding="utf-8"))


def load_handcrafted(level, impute=True):
    """Feature matrix and per-row metadata for one segment level."""
    values = np.load(FEATURES / f"handcrafted_{level}.npy").astype(np.float64)
    index = pd.read_csv(FEATURES / f"handcrafted_{level}_index.csv")

    segments = pd.read_csv(SEGMENTS, low_memory=False)
    meta = segments[segments.level == level].reset_index(drop=True)
    if not (meta.segment_id.values == index.segment_id.values).all():
        raise ValueError(f"handcrafted_{level} index does not match segments.csv")
    meta = meta.assign(voice_quality_valid=index.voice_quality_valid.values)

    if impute:
        # Jitter, shimmer and F0 are genuinely undefined on unvoiced material,
        # not merely missing. Filling with the column median keeps the row
        # usable; `voice_quality_valid` says which rows were filled, so a model
        # can exclude them instead if that turns out to matter.
        medians = np.nanmedian(values, axis=0)
        holes = ~np.isfinite(values)
        values[holes] = np.take(medians, np.where(holes)[1])
    return values, meta


def load_mel(level):
    """Memmapped frames plus the index giving each item's slice."""
    data = np.load(FEATURES / f"mel_{level}.npy", mmap_mode="r")
    return data, pd.read_csv(FEATURES / f"mel_{level}_index.csv")


def mel_item(data, item):
    """One item's spectrogram, trimmed to its real frames."""
    return np.asarray(
        data[item.row_start:item.row_start + item.n_valid_frames])


def fit_stats(values, meta, scope):
    """Statistics for a scope, to be fitted on training rows only."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}")
    if scope == "none":
        return None
    if scope == "global":
        return {"__all__": (values.mean(axis=0), values.std(axis=0))}
    if scope == "corpus":
        stats = {"__all__": (values.mean(axis=0), values.std(axis=0))}
        for name, rows in pd.Series(np.arange(len(meta))).groupby(
                meta.dataset.values).apply(list).items():
            stats[name] = (values[rows].mean(axis=0), values[rows].std(axis=0))
        return stats
    key = meta.speaker_id if scope == "speaker" else meta.path
    return {name: (values[rows].mean(axis=0), values[rows].std(axis=0))
            for name, rows in pd.Series(np.arange(len(meta))).groupby(
                key.values).apply(list).items()}


def apply(values, meta, scope, stats):
    """Z-score `values` with previously fitted statistics."""
    if scope == "none" or stats is None:
        return values
    out = np.empty_like(values)
    if scope == "global":
        keys = pd.Series(["__all__"] * len(meta))
        fallback = stats["__all__"]
    elif scope == "corpus":
        keys = meta.dataset
        # Unseen dataset (strict LODO target): pooled source stats, not (0, 1).
        fallback = stats["__all__"]
    elif scope == "speaker":
        keys = meta.speaker_id
        fallback = (np.zeros(values.shape[1]), np.ones(values.shape[1]))
    else:
        keys = meta.path
        fallback = (np.zeros(values.shape[1]), np.ones(values.shape[1]))
    for name, rows in pd.Series(np.arange(len(meta))).groupby(
            np.asarray(keys)).apply(list).items():
        mean, std = stats.get(name, fallback)
        # A speaker or file with one segment has zero variance in every column;
        # dividing by it would produce infinities rather than a useful zero.
        out[rows] = (values[rows] - mean) / np.where(std > 1e-8, std, 1.0)
    return out


def normalise(values, meta, scope):
    """Fit and apply in one step, for use outside a train/test split."""
    return apply(values, meta, scope, fit_stats(values, meta, scope))
