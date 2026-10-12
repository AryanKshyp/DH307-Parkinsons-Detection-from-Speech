"""Collect the follow-up ablation runs into one table and one figure.

    python scripts/summarise_followup.py

Reads every `probe_mixed_r*.csv`, `probe_lodo_r*.csv` and `moment_ladder_r*.csv`
that the R0-R6 runs wrote into `experiments/corpus_norm_ablation/`, and produces:

    experiments/corpus_norm_ablation/followup_summary.csv
    experiments/corpus_norm_ablation/figures/followup_methods.png
    notebooks/figures/followup_methods.png

The figure is the method-level version of `featimp_scatter.png`: corpus balanced
accuracy on x, PD/HC on y, one point per method, with an arrow from the `global`
reference to each. Up and to the left is the direction a domain-removal method is
supposed to move. Nothing in this ablation moves that way, which is the result.

Only the three-corpus runs go on the figure. The pairwise task-matched probes
have a 0.500 corpus chance rather than 0.333 and would be read wrongly on the
same axes; they stay in the table.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "experiments" / "corpus_norm_ablation"
FIG_A = OUT / "figures"
FIG_B = REPO_ROOT / "notebooks" / "figures"

# Which run produced which row, and how it should be named in the write-up.
NAMES = {
    ("_r0", "global", "none"): "global (reference)",
    ("_r0", "corpus", "none"): "per-corpus z-score",
    ("_r3", "global", "inlp_k1"): "linear projection k=1",
    ("_r3", "global", "inlp_k2"): "linear projection k=2",
    ("_r3", "global", "inlp_k3"): "linear projection k=3",
    ("_r4", "global", "coral"): "CORAL",
}
# A method needs the corpus label at apply time -> it is domain-conditional, so
# its corpus score is not comparable to the reference and it cannot do LODO.
CONDITIONAL = {"per-corpus z-score", "CORAL"}
# Hand offsets: the five points sit within 0.015 BA of each other vertically.
LABEL_OFFSETS = {
    "global (reference)": ((0, 11), "center"),
    "per-corpus z-score": ((-8, 8), "right"),
    "CORAL": ((10, 2), "left"),
    "linear projection k=1": ((8, 5), "left"),
    "linear projection k=2": ((-9, -3), "right"),
    "linear projection k=3": ((-9, -3), "right"),
}


def load_mixed():
    rows = []
    for path in sorted(OUT.glob("probe_mixed_r*.csv")):
        tag = path.stem.replace("probe_mixed", "")
        frame = pd.read_csv(path)
        frame["run"] = tag
        rows.append(frame)
    if not rows:
        raise SystemExit("no probe_mixed_r*.csv found; run the R0-R6 batch first")
    table = pd.concat(rows, ignore_index=True)
    table["method"] = [
        NAMES.get((r.run, r.scope, r.transform),
                  f"{r.scope}+{r.transform}{r.run}")
        for r in table.itertuples()]
    return table


def plot_methods(table):
    three = table[(table.corpus_chance < 0.4)
                  & (table.family == "handcrafted")
                  & (table.level == "duration")].copy()
    three = three[three.method.isin(NAMES.values())]
    if three.empty:
        print("nothing to plot")
        return
    reference = three[three.method == "global (reference)"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axvline(reference.corpus_chance, color="#4c72b0", ls=":", lw=1.0)
    ax.text(reference.corpus_chance + 0.006, three.pd_hc.min() - 0.006,
            "corpus chance 0.333", fontsize=7.5, color="#4c72b0")
    # PD/HC chance (0.500) and the 0.600 floor from dropping the shared 16
    # columns are both far below every point here. Plotting them would squeeze
    # the whole comparison into the top few percent of the axis, so they are
    # stated instead.
    ax.text(0.995, three.pd_hc.min() - 0.006,
            "PD/HC chance 0.500 and the 0.600 shared-column floor are below "
            "this axis", fontsize=7.5, color="#888", ha="right")

    for row in three.itertuples():
        if row.method == "global (reference)":
            continue
        ax.annotate("", xy=(row.corpus, row.pd_hc),
                    xytext=(reference.corpus, reference.pd_hc),
                    arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1.0),
                    zorder=1)
    for row in three.itertuples():
        conditional = row.method in CONDITIONAL
        ax.scatter(row.corpus, row.pd_hc, s=95, zorder=3,
                   marker="s" if conditional else "o",
                   c="#dd8452" if conditional else "#2a9d8f",
                   edgecolors="white", linewidths=0.6)
        offset, align = LABEL_OFFSETS.get(row.method, ((6, 6), "left"))
        ax.annotate(row.method, (row.corpus, row.pd_hc),
                    textcoords="offset points", xytext=offset, ha=align,
                    fontsize=8, color="#333", zorder=4)

    ax.scatter([], [], marker="o", c="#2a9d8f", s=70,
               label="one global map (comparable, LODO-valid)")
    ax.scatter([], [], marker="s", c="#dd8452", s=70,
               label="needs the corpus label at apply time")
    ax.set_xlabel("corpus identity, balanced accuracy  "
                  "(lower is the goal)")
    ax.set_ylabel("PD vs HC, balanced accuracy  (higher is the goal)")
    ax.set_title("No shallow method moves up and to the left\n"
                 "duration + handcrafted, 40k segments, speaker-grouped 5-fold",
                 fontsize=11)
    ax.legend(frameon=False, loc="lower center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.28, 1.03)
    pad = 0.012
    ax.set_ylim(three.pd_hc.min() - pad, three.pd_hc.max() + pad)
    fig.tight_layout()
    FIG_A.mkdir(parents=True, exist_ok=True)
    FIG_B.mkdir(parents=True, exist_ok=True)
    for folder in (FIG_A, FIG_B):
        fig.savefig(folder / "followup_methods.png", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/followup_methods.png")


def main():
    table = load_mixed()
    columns = ["run", "level", "family", "method", "scope", "transform",
               "corpus", "corpus_ci_lo", "corpus_ci_hi",
               "pd_hc", "pd_hc_ci_lo", "pd_hc_ci_hi",
               "corpus_chance", "corpora", "tasks", "dropped_cohorts",
               "n_rows", "n_speakers"]
    columns = [c for c in columns if c in table.columns]
    table[columns].to_csv(OUT / "followup_summary.csv", index=False)
    print(f"wrote {(OUT / 'followup_summary.csv').relative_to(REPO_ROOT)}\n")

    print("=== three-corpus methods (chance: corpus 0.333, PD 0.500) ===")
    three = table[table.corpus_chance < 0.4]
    print(three[["run", "level", "family", "method", "corpus", "pd_hc"]]
          .round(3).to_string(index=False))

    print("\n=== pairwise subsets (chance: corpus 0.500) ===")
    pairwise = table[table.corpus_chance >= 0.4]
    if not pairwise.empty:
        print(pairwise[["run", "method", "corpora", "tasks",
                        "dropped_cohorts", "corpus", "pd_hc"]]
              .round(3).to_string(index=False))

    ladder = sorted(OUT.glob("moment_ladder_r*.csv"))
    if ladder:
        print("\n=== moment ladder ===")
        frame = pd.read_csv(ladder[-1])
        show = [c for c in ("rung", "scope", "real_test", "surrogate_test",
                            "ci_lo", "ci_hi") if c in frame.columns]
        print(frame[show].round(3).to_string(index=False))

    lodo = [pd.read_csv(p).assign(run=p.stem.replace("probe_lodo", ""))
            for p in sorted(OUT.glob("probe_lodo_r*.csv"))]
    if lodo:
        print("\n=== strict LODO PD/HC (reference: 0.591 / 0.599 / 0.521) ===")
        frame = pd.concat(lodo, ignore_index=True)
        wide = frame.pivot_table(index=["run", "transform"],
                                 columns="held_out", values="pd_hc")
        print(wide.round(3).to_string())

    plot_methods(table)


if __name__ == "__main__":
    main()
