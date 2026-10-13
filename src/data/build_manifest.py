"""Build data/manifests/manifest.csv from the dataset registry.

    python -m src.data.build_manifest --dry-run   # inspect matches, write nothing
    python -m src.data.build_manifest             # write the manifest

The dry run reports how many files matched, which files matched no rule, which
task tokens are unmapped, and which speaker ids look suspicious. Run it until
all four are clean before writing.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
OUT = REPO_ROOT / "data" / "manifests" / "manifest.csv"

COLUMNS = ["path", "dataset", "language", "channel", "speaker_id", "label",
           "task", "cohort", "sample_rate", "duration", "n_channels"]


class Unmatched(Exception):
    pass


def match_file(rel_path, rules):
    for rule in rules:
        m = re.match(rule["pattern"], rel_path)
        if m:
            captures = {k: v for k, v in m.groupdict().items() if v is not None}
            fields = {k: v.format(**captures)
                      for k, v in rule.get("fields", {}).items()}
            return captures, fields
    raise Unmatched(rel_path)


def scan(name, spec):
    """Return (rows, unmatched, unmapped_tasks) for one dataset."""
    root = REPO_ROOT / spec["root"]
    if not root.exists():
        sys.exit(f"{name}: root does not exist: {root}")

    task_map = spec.get("task_map", {})
    label_map = spec.get("label_map", {})
    cohort_map = spec.get("cohort_map", {})

    rows, unmatched = [], []
    unmapped = collections.Counter()

    for path in sorted(root.glob(spec["glob"])):
        rel = path.relative_to(root).as_posix()
        try:
            captures, fields = match_file(rel, spec["rules"])
        except Unmatched:
            unmatched.append(rel)
            continue

        token = fields.get("task", captures.get("task"))
        if token not in task_map:
            unmapped[token] += 1
        label = fields["label"]
        cohort = fields["cohort"]

        rows.append({
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "dataset": name,
            "language": spec["language"],
            "channel": spec["channel"],
            "speaker_id": fields["speaker_id"],
            "label": label_map.get(label, label),
            "task": task_map.get(token, "unknown"),
            "cohort": cohort_map.get(cohort, cohort),
        })

    return rows, unmatched, unmapped


def add_audio_info(rows):
    for row in rows:
        info = sf.info(REPO_ROOT / row["path"])
        row["sample_rate"] = info.samplerate
        row["duration"] = round(info.frames / info.samplerate, 3)
        row["n_channels"] = info.channels


def report(name, rows, unmatched, unmapped):
    speakers = {r["speaker_id"] for r in rows}
    labels = collections.Counter(r["label"] for r in rows)
    spk_labels = collections.Counter(
        r["label"] for r in {r["speaker_id"]: r for r in rows}.values())

    print(f"\n{name}")
    print(f"  matched {len(rows)} files, {len(speakers)} speakers "
          f"({spk_labels['HC']} HC / {spk_labels['PD']} PD)")
    print(f"  files by label: {dict(labels)}")
    print("  files by task: " + ", ".join(
        f"{t}={n}" for t, n in sorted(collections.Counter(
            r["task"] for r in rows).items())))

    if unmatched:
        print(f"  UNMATCHED ({len(unmatched)}):")
        for rel in unmatched[:20]:
            print(f"    {rel}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    if unmapped:
        print(f"  UNMAPPED TASK TOKENS: {dict(unmapped)}")

    # A speaker whose files disagree on label would corrupt both the class
    # balance and the speaker-level splits.
    by_speaker = collections.defaultdict(set)
    for r in rows:
        by_speaker[r["speaker_id"]].add(r["label"])
    conflicts = {s: v for s, v in by_speaker.items() if len(v) > 1}
    if conflicts:
        print(f"  SPEAKERS WITH CONFLICTING LABELS: {conflicts}")

    return bool(unmatched or unmapped or conflicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report matches and write nothing")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    all_rows, problems = [], False
    for name, dataset in spec["datasets"].items():
        rows, unmatched, unmapped = scan(name, dataset)
        problems |= report(name, rows, unmatched, unmapped)
        all_rows.extend(rows)

    print(f"\ntotal: {len(all_rows)} files")

    if args.dry_run:
        print("dry run: nothing written" + ("  [PROBLEMS ABOVE]" if problems else ""))
        return

    if problems:
        sys.exit("refusing to write a manifest with unmatched files, "
                 "unmapped tasks, or conflicting labels")

    add_audio_info(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=COLUMNS).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
