"""Stage 6 splits: speaker-level, stratified, asserted leak-free.

    python -m src.data.splits --protocol mixed --k 10
    python -m src.data.splits --protocol lodo

Two protocols, both assigning whole speakers so no speaker is ever on both
sides of a split:

    mixed  10 configurations. Each stratified fold serves as test once, the next
           fold is validation, the remaining eight are training. Stratified on
           `dataset x label` jointly, so every fold holds a comparable corpus and
           class mix.
    lodo   one configuration per corpus. That corpus is held out entirely as
           test; training and validation come from the others.

Output is one row per (configuration, speaker), carrying the speaker's role.
Speaker-level rather than file-level: leak-freedom is a property of speakers, and
Stage 7 joins on `speaker_id` to reach files, segments and features.

`cohort` is carried through so a later stage can report an elderly-only
robustness number. The ItalianPD young controls average 20.8 years against
patients at 67, with no age overlap at all, so age is an available shortcut for
that corpus. They are kept, per the Stage 6 decision, but the column makes it
possible to quantify their effect without regenerating splits.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
OUT_DIR = REPO_ROOT / "data" / "manifests"

SEED = 0
VAL_FRACTION = 0.15


def speaker_table(manifest):
    """One row per speaker, with the file and duration weight it carries."""
    return (manifest.groupby("speaker_id")
            .agg(dataset=("dataset", "first"), language=("language", "first"),
                 label=("label", "first"), cohort=("cohort", "first"),
                 files=("path", "size"), hours=("duration", lambda s: s.sum() / 3600))
            .reset_index()
            .sort_values("speaker_id")
            .reset_index(drop=True))


def mixed(speakers, k):
    """k configurations: fold i is test, fold i+1 validation, the rest train."""
    stratum = speakers.dataset + "|" + speakers.label
    smallest = stratum.value_counts().min()
    if k > smallest:
        raise ValueError(
            f"k={k} exceeds the smallest dataset x label stratum ({smallest} "
            f"speakers), so some folds would be empty for it")

    folds = np.empty(len(speakers), dtype=int)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for index, (_, held) in enumerate(splitter.split(speakers, stratum)):
        folds[held] = index

    rows = []
    for config in range(k):
        role = np.where(folds == config, "test",
                        np.where(folds == (config + 1) % k, "val", "train"))
        rows.append(speakers.assign(config=config, role=role))
    return pd.concat(rows, ignore_index=True)


def lodo(speakers):
    """One configuration per corpus, that corpus held out entirely as test."""
    rows = []
    for corpus in sorted(speakers.dataset.unique()):
        rest_at = np.where(speakers.dataset != corpus)[0]
        rest = speakers.iloc[rest_at]
        stratum = rest.dataset + "|" + rest.label

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                          random_state=SEED)
        train_within, val_within = next(splitter.split(rest, stratum))

        role = np.full(len(speakers), "test", dtype=object)
        role[rest_at[train_within]] = "train"
        role[rest_at[val_within]] = "val"
        rows.append(speakers.assign(config=corpus, role=role))
    return pd.concat(rows, ignore_index=True)


def assert_leak_free(splits):
    """No speaker in two roles of one configuration, and none unassigned."""
    for config, group in splits.groupby("config"):
        counts = group.speaker_id.value_counts()
        assert counts.max() == 1, (
            f"config {config}: speaker in multiple roles "
            f"({counts[counts > 1].index.tolist()[:3]})")
        by_role = {role: set(g.speaker_id) for role, g in group.groupby("role")}
        for left in by_role:
            for right in by_role:
                if left < right:
                    shared = by_role[left] & by_role[right]
                    assert not shared, (
                        f"config {config}: {len(shared)} speakers shared between "
                        f"{left} and {right}")
        assert set(group.role) <= {"train", "val", "test"}
        assert "test" in by_role and by_role["test"], f"config {config}: no test"

    per_config = splits.groupby("config").speaker_id.nunique()
    assert per_config.nunique() == 1, "configurations cover different speakers"
    print(f"leak-free: {splits.config.nunique()} configurations, "
          f"{per_config.iloc[0]} speakers each, no speaker in two roles")


def report(splits, speakers, protocol):
    print("\nspeakers per role:")
    print(pd.crosstab(splits.config, splits.role).to_string())

    print("\ntest-set composition (dataset x label speakers):")
    test = splits[splits.role == "test"]
    print(pd.crosstab([test.config, test.dataset], test.label).to_string())

    print("\nfiles per role:")
    print(splits.pivot_table(index="config", columns="role", values="files",
                             aggfunc="sum").to_string())

    if protocol == "mixed":
        # A k-fold property worth asserting: every speaker is tested exactly
        # once, which is what makes pooled out-of-fold scoring valid.
        appearances = test.speaker_id.value_counts()
        assert appearances.max() == appearances.min() == 1, \
            "mixed: speakers are not tested exactly once"
        print(f"\nevery one of {len(speakers)} speakers is in the test set of "
              f"exactly one configuration, so pooling out-of-fold predictions "
              f"gives a per-corpus score over all its speakers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["mixed", "lodo"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    speakers = speaker_table(manifest)
    print(f"{len(speakers)} speakers, {len(manifest)} files")
    print(pd.crosstab(speakers.dataset, speakers.label, margins=True).to_string())

    splits = (mixed(speakers, args.k) if args.protocol == "mixed"
              else lodo(speakers))
    assert_leak_free(splits)
    report(splits, speakers, args.protocol)

    out = OUT_DIR / f"splits_{args.protocol}.csv"
    columns = ["config", "role", "speaker_id", "dataset", "language", "label",
               "cohort", "files", "hours"]
    splits[columns].to_csv(out, index=False)
    print(f"\nwrote {len(splits)} rows to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
