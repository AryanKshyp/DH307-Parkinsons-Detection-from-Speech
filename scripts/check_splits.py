"""Verify the Stage 6 split files independently of the code that wrote them.

    python scripts/check_splits.py

`src/data/splits.py` asserts leak-freedom while building, but a split file is
read many times after it is written and a leak in one is silent -- it shows up
only as an implausibly good score much later. So this re-checks the files as
they sit on disk, and does it at **file** level rather than speaker level, which
would catch a speaker whose recordings disagree about which corpus they belong
to.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "data" / "manifests"


def check(protocol, manifest):
    splits = pd.read_csv(MANIFESTS / f"splits_{protocol}.csv")
    print(f"\n=== splits_{protocol}.csv: {len(splits)} rows, "
          f"{splits.config.nunique()} configurations ===")

    # A speaker must mean one corpus and one label everywhere, or the strata and
    # the leakage check are both meaningless.
    per_speaker = manifest.groupby("speaker_id")[["dataset", "label"]].nunique()
    assert (per_speaker == 1).all().all(), "a speaker spans corpora or labels"

    declared = splits.drop_duplicates("speaker_id").set_index("speaker_id")
    actual = manifest.drop_duplicates("speaker_id").set_index("speaker_id")
    common = declared.index.intersection(actual.index)
    assert len(common) == len(actual), "splits do not cover every speaker"
    assert (declared.loc[common, "dataset"] == actual.loc[common, "dataset"]).all()
    assert (declared.loc[common, "label"] == actual.loc[common, "label"]).all()
    print(f"{len(common)} speakers, corpus and label agree with the manifest")

    files = manifest[["path", "speaker_id", "dataset", "label"]]
    for config, group in splits.groupby("config"):
        joined = files.merge(group[["speaker_id", "role"]], on="speaker_id")
        per_file = joined.groupby("path").role.nunique()
        assert per_file.max() == 1, (
            f"config {config}: {(per_file > 1).sum()} files in multiple roles")
        assert len(joined) == len(files), (
            f"config {config}: {len(joined)} file-rows for {len(files)} files")

        roles = {r: set(g.speaker_id) for r, g in group.groupby("role")}
        assert not roles.get("train", set()) & roles.get("test", set())
        assert not roles.get("train", set()) & roles.get("val", set())
        assert not roles.get("val", set()) & roles.get("test", set())
    print(f"no file appears in two roles, in any of "
          f"{splits.config.nunique()} configurations")

    test = splits[splits.role == "test"]
    if protocol == "mixed":
        seen = test.speaker_id.value_counts()
        assert seen.max() == seen.min() == 1
        print("each speaker is tested exactly once across the configurations, "
              "so out-of-fold predictions can be pooled")
        counts = pd.crosstab([test.config, test.dataset], test.label)
        assert (counts > 0).all().all(), "a test fold is missing a corpus/class"
        print("every test fold contains both classes of all three corpora; "
              f"the thinnest cell holds {counts.values.min()} speaker(s)")
    else:
        for config, group in splits.groupby("config"):
            held = group[group.dataset == config]
            assert (held.role == "test").all(), \
                f"{config}: held-out corpus appears in train or val"
            others = group[group.dataset != config]
            assert not (others.role == "test").any(), \
                f"{config}: a non-held-out corpus appears in test"
        print("each configuration tests exactly its held-out corpus, and that "
              "corpus appears nowhere in training or validation")

    print(splits.pivot_table(index="config", columns="role",
                             values="speaker_id", aggfunc="count").to_string())
    return splits


def main():
    manifest = pd.read_csv(MANIFESTS / "manifest_processed.csv")
    print(f"manifest: {len(manifest)} files, "
          f"{manifest.speaker_id.nunique()} speakers")
    for protocol in ("mixed", "lodo"):
        check(protocol, manifest)
    print("\nall split checks passed")


if __name__ == "__main__":
    main()
