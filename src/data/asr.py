"""Whisper ASR for utterances that still have no transcript.

    python -m src.data.asr --dataset MDVR-KCL --task spontaneous

Writes one UTF-8 .txt per file under data/interim/asr/<dataset>/. Re-run
`python -m src.data.transcripts` afterwards so those files become `source=asr`
rows. Local OpenAI Whisper (MIT); no API key.

Only intended for MDVR-KCL spontaneous speech: 36 files, ~1.4 hours. Scripted
tasks use published prompts instead. Interviewer speech is not removed --
previous MDVR-KCL diarization work did not release timestamps, and a gated
diarization model is not cost-efficient here.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
OUT = REPO_ROOT / "data" / "interim" / "asr"

# English-only small: fits a 4 GB card and is enough for a 1.4 h English set.
MODEL = "small.en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MDVR-KCL")
    ap.add_argument("--task", default="spontaneous")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    import whisper

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    d = pd.read_csv(MANIFEST)
    todo = d[(d.dataset == args.dataset) & (d.task == args.task)]
    if todo.empty:
        raise SystemExit(f"no files for {args.dataset} {args.task}")

    dest = OUT / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    remaining = [row for row in todo.itertuples()
                 if not (dest / (Path(row.path).stem + ".txt")).exists()]
    print(f"{len(todo)} {args.dataset} {args.task} files, "
          f"{len(remaining)} to transcribe, device {args.device}, "
          f"model {args.model}")
    if not remaining:
        return

    model = whisper.load_model(args.model, device=args.device)
    for i, row in enumerate(remaining, 1):
        wav = REPO_ROOT / row.path
        result = model.transcribe(str(wav), language="en", fp16=args.device == "cuda")
        text = " ".join(result["text"].split())
        (dest / (Path(row.path).stem + ".txt")).write_text(text, encoding="utf-8")
        print(f"  {i}/{len(remaining)}  {Path(row.path).name}  "
              f"{len(text.split())} words", flush=True)


if __name__ == "__main__":
    main()
