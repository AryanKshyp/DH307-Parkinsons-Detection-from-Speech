"""Assemble the transcript for every utterance that needs forced alignment.

    python -m src.data.transcripts

Writes data/interim/transcripts.csv with one row per utterance and a `source`
column recording where the text came from:

    per_file  a verbatim transcript shipped with the corpus (NeuroVoz)
    prompt    the canonical text of a scripted task (configs/prompts.yaml)
    asr       Whisper output cached under data/interim/asr/<dataset>/
    missing   no transcript exists

Sustained vowels and DDK carry no lexical content, so they are excluded here
and segmented by other means.
"""

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
PROMPTS = REPO_ROOT / "configs" / "prompts.yaml"
ASR_DIR = REPO_ROOT / "data" / "interim" / "asr"
OUT = REPO_ROOT / "data" / "interim" / "transcripts.csv"
NEUROVOZ_TRANSCRIPTS = (REPO_ROOT / "data" / "raw" / "NeuroVoz" / "data"
                        / "transcriptions")

LEXICAL_TASKS = {"sentence", "monologue", "read_text", "read_words",
                 "read_phrases", "spontaneous"}


def asr_path(row):
    return ASR_DIR / row.dataset / (Path(row.path).stem + ".txt")


def resolve(row, prompts):
    if row.dataset == "NeuroVoz":
        path = NEUROVOZ_TRANSCRIPTS / (Path(row.source_path).stem + ".txt")
        if path.exists():
            text = " ".join(path.read_text(encoding="utf-8").split())
            if text:
                return "per_file", text

    entry = prompts.get(row.dataset, {}).get(row.task) or {}
    if entry.get("text"):
        return "prompt", " ".join(entry["text"].split())

    cached = asr_path(row)
    if cached.exists():
        text = " ".join(cached.read_text(encoding="utf-8").split())
        if text:
            return "asr", text

    return "missing", ""


def main():
    d = pd.read_csv(MANIFEST)
    lexical = d[d.task.isin(LEXICAL_TASKS)].copy()
    prompts = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["prompts"]

    resolved = [resolve(row, prompts) for row in lexical.itertuples()]
    lexical["source"] = [r[0] for r in resolved]
    lexical["text"] = [r[1] for r in resolved]
    lexical["n_words"] = lexical.text.str.split().str.len()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lexical[["path", "dataset", "speaker_id", "task", "duration",
             "source", "n_words", "text"]].to_csv(OUT, index=False)

    print(f"{len(d)} files total, {len(lexical)} with lexical content\n")
    print(pd.crosstab(
        [lexical.dataset, lexical.task], lexical.source).to_string())

    gap = lexical[lexical.source == "missing"]
    print(f"\nno transcript: {len(gap)} files, "
          f"{gap.duration.sum()/3600:.2f} hours")
    if len(gap):
        print(gap.groupby(["dataset", "task"]).agg(
            files=("path", "size"),
            hours=("duration", lambda s: round(s.sum() / 3600, 2))).to_string())

    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
