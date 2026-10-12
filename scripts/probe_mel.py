"""Thermometer for the Mel family: corpus vs PD/HC from mean-pooled spectrograms.

    python scripts/probe_mel.py --levels duration utterance

Stage 5 only probed the handcrafted family. The 1D-CNN's domain head starts
from however separable these spectrograms already are; this is that number,
under the same speaker-grouped CV and the same tree classifier, so it can sit
next to the handcrafted table.

Mean-pooling discards the trajectory the CNN will use, so this is a lower
bound on how much domain is sitting in the Mel cache, not an upper bound.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.features import store as fs                            # noqa: E402

SEGMENTS = REPO_ROOT / "data" / "interim" / "segments.csv"
MAX_ROWS = 40_000
SEED = 0


def mean_pool(level, n_max, rng):
    data, index = fs.load_mel(level)
    segments = pd.read_csv(SEGMENTS, low_memory=False)
    segs = segments[segments.level == level][
        ["segment_id", "speaker_id", "dataset", "label"]]
    meta = index.merge(segs, on="segment_id", how="left")
    meta = meta.dropna(subset=["speaker_id"]).reset_index(drop=True)
    if len(meta) > n_max:
        pick = np.sort(rng.choice(len(meta), n_max, replace=False))
        meta = meta.iloc[pick].reset_index(drop=True)
    rows = np.empty((len(meta), data.shape[1]), dtype=np.float64)
    for i, item in enumerate(meta.itertuples()):
        frames = fs.mel_item(data, item)
        rows[i] = frames.mean(axis=0) if len(frames) else 0.0
    return rows, meta


def score(values, meta, target, scope):
    y = meta[target].values
    groups = meta.speaker_id.values
    folds = GroupKFold(n_splits=5)
    truth, predicted = [], []
    for train, test in folds.split(values, y, groups):
        stats = fs.fit_stats(values[train], meta.iloc[train], scope)
        x_train = fs.apply(values[train], meta.iloc[train], scope, stats)
        x_test = fs.apply(values[test], meta.iloc[test], scope, stats)
        model = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
        model.fit(x_train, y[train])
        truth.append(y[test])
        predicted.append(model.predict(x_test))
    return balanced_accuracy_score(np.concatenate(truth),
                                   np.concatenate(predicted))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["duration"])
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    results = []
    for level in args.levels:
        values, meta = mean_pool(level, MAX_ROWS, rng)
        print(f"\n=== {level}: {len(meta)} items, "
              f"{meta.speaker_id.nunique()} speakers, "
              f"{values.shape[1]} mean-pooled bands ===")
        print("corpus mix: " + ", ".join(
            f"{k} {v}" for k, v in meta.dataset.value_counts().items()))
        print(f"chance: corpus {1 / meta.dataset.nunique():.3f}, PD/HC 0.500")
        for scope in ("none", "global"):
            corpus = score(values, meta, "dataset", scope)
            pathology = score(values, meta, "label", scope)
            results.append({"level": level, "scope": scope,
                            "corpus": corpus, "pd_hc": pathology})
            print(f"  {scope:9s} corpus {corpus:.3f}   PD/HC {pathology:.3f}")

    print("\n=== balanced accuracy, mean-pooled Mel ===")
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
