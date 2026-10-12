"""Check forced-alignment quality before trusting it at scale.

    python scripts/inspect_alignments.py                     # all datasets
    python scripts/inspect_alignments.py --dataset NeuroVoz --n 21

Prints quantitative checks over every aligned file and draws waveforms with
word boundaries overlaid for a sample spanning the best, median and worst
alignment scores, so the failure modes get looked at rather than the easy
cases. Figures go to notebooks/figures/alignment_<dataset>/.

Also writes a rejection list to data/interim/alignment_rejects.csv. A file is
rejected on one criterion only: the aligner is not confident about it as a
whole. That catches the failure that matters, which is a transcript describing
speech the audio does not contain -- what happens when a recording was truncated
before the speaker finished the prompt.

Two other criteria were tried and deliberately rejected, both because they flag
properties of the prompt rather than defects in the recording:

  implied speaking rate  31 files exceed 4 words/s, but several score 0.96 and
                         align cleanly, because Spanish sentences of short
                         function words genuinely run that fast.
  share of words pinned  Reads as "words crammed into audio that lacks them",
  to the duration floor  but 9 of 13 files it flagged are the same sentence,
                         whose one-letter words (a, y) are elided in connected
                         speech and correctly given ~20 ms at score 0.99.
                         Rejecting whole files on this would have discarded
                         good word segments, and done so per prompt.

Both are reported as diagnostics. Very short words are a word-level problem and
belong to the minimum-segment-duration decision in Stage 4, not here.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "data" / "interim" / "alignment_index.csv"
REJECTS = REPO_ROOT / "data" / "interim" / "alignment_rejects.csv"
FIGURES = REPO_ROOT / "notebooks" / "figures"

MIN_WORD_S = 0.03      # the shortest span the aligner will emit for a word
MIN_FILE_SCORE = 0.5   # below this the aligner is not confident about the file


def load_words(row):
    data = json.loads((REPO_ROOT / row.alignment).read_text(encoding="utf-8"))
    return data["duration"], data["words"]


def measure(ok):
    """Per-word and per-file quality measures for every successful alignment."""
    words, per_file = [], []
    for row in ok.itertuples():
        duration, ws = load_words(row)
        for w in ws:
            words.append({"path": row.path, "word": w["word"],
                          "score": w["score"], "dur": w["end"] - w["start"]})
        floored = sum(1 for w in ws if w["end"] - w["start"] <= MIN_WORD_S)
        per_file.append({
            "path": row.path,
            "dataset": row.dataset,
            "task": row.task,
            "duration": duration,
            "n_words": len(ws),
            "span": ws[-1]["end"] - ws[0]["start"],
            "lead_silence": ws[0]["start"],
            "tail_silence": duration - ws[-1]["end"],
            "mean_score": row.mean_score,
            "floor_frac": floored / len(ws),
            "words_per_s": len(ws) / duration,
        })
    words = pd.DataFrame(words)
    files = pd.DataFrame(per_file)
    files["coverage"] = files.span / files.duration
    return words, files


def report(words, files):
    print(f"\naligned files {len(files)}, words {len(words)}\n")
    print("--- word confidence ---")
    print(words.score.describe(percentiles=[.05, .25, .5]).round(3).to_string())
    print(f"words scoring below 0.5: {(words.score < 0.5).sum()} "
          f"({100*(words.score < 0.5).mean():.2f}%)")
    print(f"words shorter than {MIN_WORD_S*1000:.0f} ms: "
          f"{(words.dur < MIN_WORD_S).sum()} "
          f"({100*(words.dur < MIN_WORD_S).mean():.2f}%)")

    print("\n--- word duration (s) ---")
    print(words.dur.describe(percentiles=[.05, .5, .95]).round(3).to_string())

    print("\n--- per-file coverage (aligned span / file duration) ---")
    print(files.coverage.describe(percentiles=[.05, .5, .95]).round(3).to_string())
    print(f"files covering under 50% of their audio: "
          f"{(files.coverage < 0.5).sum()}")
    print("\n--- leading / trailing silence (s) ---")
    print(files[["lead_silence", "tail_silence"]].describe(
        percentiles=[.5, .95]).round(3).to_string())
    print("\n--- implied speaking rate (words/s), diagnostic only ---")
    print(files.groupby("task").words_per_s.describe(
        percentiles=[.5, .95]).round(2).to_string())


def draw(rows, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples():
        duration, words = load_words(row)
        y, sr = sf.read(REPO_ROOT / row.path)
        t = np.arange(len(y)) / sr

        fig, ax = plt.subplots(figsize=(min(18, max(8, duration * 1.2)), 3))
        ax.plot(t, y, linewidth=0.4, color="#444444")
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.05, 1.05)
        for i, w in enumerate(words):
            colour = "#c44e52" if w["score"] < 0.5 else "#4c72b0"
            ax.axvspan(w["start"], w["end"], color=colour,
                       alpha=0.14 if i % 2 == 0 else 0.24)
            ax.axvline(w["start"], color=colour, linewidth=0.7)
            ax.text((w["start"] + w["end"]) / 2, 0.88, w["word"],
                    ha="center", va="top", fontsize=7, rotation=90,
                    color=colour)
        ax.set_title(f'{Path(row.path).stem}   mean score {row.mean_score:.3f}',
                     fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / f"{title}_{Path(row.path).stem}.png", dpi=130)
        plt.close(fig)


def write_rejects(files):
    """Flag alignments that should not be segmented, with the reason."""
    rejects = files[files.mean_score < MIN_FILE_SCORE].copy()
    rejects["reason"] = f"mean word score below {MIN_FILE_SCORE}"
    rejects = rejects[["path", "dataset", "task", "duration", "n_words",
                       "mean_score", "floor_frac", "words_per_s", "reason"]]
    REJECTS.parent.mkdir(parents=True, exist_ok=True)
    rejects.to_csv(REJECTS, index=False)

    print(f"\n--- rejected {len(rejects)} of {len(files)} alignments "
          f"({100*len(rejects)/len(files):.1f}%) ---")
    if len(rejects):
        print(rejects.groupby(["dataset", "task"]).size().to_string())
    print(f"list: {REJECTS.relative_to(REPO_ROOT)}")
    return rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="restrict the report and figures; the "
                                      "rejection list always covers everything")
    ap.add_argument("--n", type=int, default=21,
                    help="total files to draw, split across score bands")
    args = ap.parse_args()

    index = pd.read_csv(INDEX)
    ok = index[index.status == "ok"]
    if ok.empty:
        raise SystemExit("no successful alignments in the index")

    words, files = measure(ok)
    # Rejects are recomputed over the whole index every time, so the list can
    # never carry stale rows for files that have since been realigned.
    write_rejects(files)

    scope = [args.dataset] if args.dataset else sorted(ok.dataset.unique())
    for dataset in scope:
        print(f"\n{'=' * 20} {dataset} {'=' * 20}")
        mine = files[files.dataset == dataset]
        report(words[words.path.isin(set(mine.path))], mine)

        out_dir = FIGURES / f"alignment_{dataset}"
        per_band = max(1, args.n // 3)
        ranked = ok[ok.dataset == dataset].sort_values("mean_score")
        bands = {
            "worst": ranked.head(per_band),
            "median": ranked.iloc[len(ranked) // 2 - per_band // 2:][:per_band],
            "best": ranked.tail(per_band),
        }
        for name, rows in bands.items():
            draw(rows, out_dir, name)
        print(f"\ndrew {sum(len(r) for r in bands.values())} alignments to "
              f"{out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
