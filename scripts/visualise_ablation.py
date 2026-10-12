"""Figures for the shallow-ablation story, including CORAL.

    python scripts/visualise_ablation.py

Does not re-run the GBT probes. It reads the CSVs already on disk and, for
the covariance / t-SNE panels, applies the same transforms the probe used
(global z-score, per-corpus z-score, CORAL, INLP k=3) on one speaker-grouped
fold of the Stage 5 40k sample.

Writes to experiments/corpus_norm_ablation/figures/ and notebooks/figures/:

    ablation_moment_ladder.png   how much of the cue is each moment
    ablation_coral_corr.png      correlation matrices, before vs after CORAL
    ablation_tsne.png            40-d cloud after each transform, by corpus
    ablation_methods.png         four methods on the corpus-vs-PD plane, with CIs
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.features import store as fs                        # noqa: E402
from src.features import coral as coral_mod                 # noqa: E402
from src.features import project as project_mod             # noqa: E402

MAX_ROWS = 40_000
SEED = 0
LEVEL = "duration"
TSNE_PER_CORPUS = 700

OUT = REPO_ROOT / "experiments" / "corpus_norm_ablation"
FIG_A = OUT / "figures"
FIG_B = REPO_ROOT / "notebooks" / "figures"

CORPORA = ["NeuroVoz", "ItalianPD", "MDVR-KCL"]
COLOURS = {"NeuroVoz": "#4c72b0", "ItalianPD": "#dd8452", "MDVR-KCL": "#55a868"}


def savefig(fig, name):
    FIG_A.mkdir(parents=True, exist_ok=True)
    FIG_B.mkdir(parents=True, exist_ok=True)
    for folder in (FIG_A, FIG_B):
        fig.savefig(folder / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


def load_sample():
    rng = np.random.default_rng(SEED)
    values, meta = fs.load_handcrafted(LEVEL)
    if len(meta) > MAX_ROWS:
        pick = np.sort(rng.choice(len(meta), MAX_ROWS, replace=False))
        values, meta = values[pick], meta.iloc[pick].reset_index(drop=True)
    return values, meta


def fold0(values, meta):
    folds = GroupKFold(n_splits=5)
    train, test = next(folds.split(values, meta.dataset, meta.speaker_id))
    return train, test


def apply_global(values, meta, train, test):
    stats = fs.fit_stats(values[train], meta.iloc[train], "global")
    return (fs.apply(values[train], meta.iloc[train], "global", stats),
            fs.apply(values[test], meta.iloc[test], "global", stats))


def apply_corpus(values, meta, train, test):
    stats = fs.fit_stats(values[train], meta.iloc[train], "corpus")
    return (fs.apply(values[train], meta.iloc[train], "corpus", stats),
            fs.apply(values[test], meta.iloc[test], "corpus", stats))


def apply_coral(values, meta, train, test):
    x_train, x_test = apply_global(values, meta, train, test)
    state = coral_mod.fit(x_train, meta.iloc[train])
    worst = coral_mod.verify(x_train, meta.iloc[train], state)
    print(f"  CORAL fitted-set covariance gap: {worst:.4f}")
    return (coral_mod.apply(x_train, meta.iloc[train], state),
            coral_mod.apply(x_test, meta.iloc[test], state))


def apply_inlp(values, meta, train, test):
    x_train, x_test = apply_global(values, meta, train, test)
    state = project_mod.fit(x_train, meta.iloc[train], iters=3, seed=SEED)
    project_mod.verify(state, x_train.shape[1])
    return project_mod.apply(x_train, state), project_mod.apply(x_test, state)


def corr_of(x, meta_slice, corpus):
    rows = np.asarray(meta_slice.dataset.values) == corpus
    return np.corrcoef(x[rows], rowvar=False)


def plot_moment_ladder():
    ladder = pd.read_csv(OUT / "moment_ladder_r1.csv")
    order = ["L0", "L1", "L2", "L3"]
    labels = {
        "L0": "means only",
        "L1": "+ variances",
        "L2": "+ correlations",
        "L3": "real data\n(all orders)",
    }
    rows = [ladder[ladder.rung == r].iloc[0] for r in order]
    x = np.arange(len(order))
    y = [r.real_test for r in rows]
    lo = [r.real_test - r.ci_lo for r in rows]
    hi = [r.ci_hi - r.real_test for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    notes = [
        "+0.001  variances add nothing",
        "+0.098  this is what CORAL removes",
        "+0.088  no covariance match can reach this",
    ]
    colours_note = ["#888", "#6a4c93", "#4c72b0"]
    bars = ax.bar(x, y, color=["#b0b0b0", "#b0b0b0", "#6a4c93", "#4c72b0"],
                  width=0.62, zorder=3)
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#333",
                capsize=3, lw=1.0, zorder=4)
    ax.axhline(1 / 3, color="#4c72b0", ls=":", lw=1.0, zorder=1)
    ax.text(3.42, 1 / 3 + 0.012, "chance 0.333", fontsize=8, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[r] for r in order])
    ax.set_ylabel("corpus identity, balanced accuracy")
    ax.set_ylim(0.28, 0.88)
    ax.set_title("Most of the dataset cue is in the correlations\n"
                 "Gaussian surrogates of each moment, tested on real held-out data")
    for i, (bar, val) in enumerate(zip(bars, y)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018,
                f"{val:.3f}", ha="center", fontsize=8, color="#333")
        if i < 3:
            ax.annotate(notes[i], xy=(i + 0.5, max(y[i], y[i + 1]) + 0.055),
                        ha="center", fontsize=8, color=colours_note[i])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "ablation_moment_ladder.png")


def plot_coral_corr(values, meta, train):
    x_global, _ = apply_global(values, meta, train, train)
    x_coral, _ = apply_coral(values, meta, train, train)
    meta_tr = meta.iloc[train]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), layout="constrained")
    for col, corpus in enumerate(CORPORA):
        before = corr_of(x_global, meta_tr, corpus)
        after = corr_of(x_coral, meta_tr, corpus)
        for row, mat, title in (
                (0, before, f"{corpus}\nbefore CORAL"),
                (1, after, f"{corpus}\nafter CORAL")):
            ax = axes[row, col]
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r",
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
    axes[0, 0].set_ylabel("global z-score\n(correlations still differ)",
                          fontsize=8)
    axes[1, 0].set_ylabel("CORAL\n(correlations matched)", fontsize=8)
    fig.suptitle("CORAL does the covariance job it claims\n"
                 "40 × 40 correlation, training speakers of one fold",
                 fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="Pearson correlation")
    savefig(fig, "ablation_coral_corr.png")


def plot_tsne(values, meta, train, test):
    panels = [
        ("global z-score", apply_global(values, meta, train, test)[1]),
        ("per-corpus z-score", apply_corpus(values, meta, train, test)[1]),
        ("CORAL", apply_coral(values, meta, train, test)[1]),
        ("linear projection  k=3", apply_inlp(values, meta, train, test)[1]),
    ]
    meta_te = meta.iloc[test].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    picks = []
    for corpus in CORPORA:
        at = np.flatnonzero(meta_te.dataset.values == corpus)
        n = min(TSNE_PER_CORPUS, len(at))
        picks.append(rng.choice(at, n, replace=False))
    idx = np.sort(np.concatenate(picks))
    labels = meta_te.dataset.values[idx]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.4))
    for ax, (title, x_test) in zip(axes.ravel(), panels):
        embedded = TSNE(n_components=2, perplexity=30, init="pca",
                        random_state=SEED, learning_rate="auto"
                        ).fit_transform(x_test[idx])
        for corpus in CORPORA:
            at = labels == corpus
            ax.scatter(embedded[at, 0], embedded[at, 1], s=6, alpha=0.55,
                       c=COLOURS[corpus], label=corpus, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8,
                      markerscale=2.2)
    fig.suptitle("The 40-d cloud still clusters by corpus\n"
                 "t-SNE of held-out speakers, same fold the probe uses",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "ablation_tsne.png")


def plot_methods():
    summary = pd.read_csv(OUT / "followup_summary.csv")
    want = {
        "global (reference)": "#2a9d8f",
        "per-corpus z-score": "#dd8452",
        "CORAL": "#dd8452",
        "linear projection k=3": "#2a9d8f",
    }
    markers = {
        "global (reference)": "o",
        "per-corpus z-score": "s",
        "CORAL": "s",
        "linear projection k=3": "o",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.axvline(1 / 3, color="#4c72b0", ls=":", lw=1.0)
    ax.axvspan(0.28, 0.333, color="#4c72b0", alpha=0.04)
    ax.text(0.338, 0.668, "corpus chance  0.333", fontsize=8, color="#4c72b0")
    ax.set_xlim(0.28, 1.08)
    ax.set_ylim(0.705, 0.755)

    ref = summary[summary.method == "global (reference)"].iloc[0]
    for name, colour in want.items():
        row = summary[summary.method == name].iloc[0]
        if name != "global (reference)":
            ax.annotate("", xy=(row.corpus, row.pd_hc),
                        xytext=(ref.corpus, ref.pd_hc),
                        arrowprops=dict(arrowstyle="->", color="#dddddd",
                                        lw=1.1), zorder=1)
        # PD intervals all overlap (~0.03 half-width); corpus CIs do not.
        ax.errorbar(row.corpus, row.pd_hc,
                    xerr=[[row.corpus - row.corpus_ci_lo],
                          [row.corpus_ci_hi - row.corpus]],
                    fmt="none", ecolor="#999", capsize=2.5, lw=0.9, zorder=2)
        ax.scatter(row.corpus, row.pd_hc, s=90, c=colour, zorder=3,
                   marker=markers[name], edgecolors="white", linewidths=0.6)
        labels = {
            "global (reference)": ((0, 11), "center",
                                   "global  (what Stage 7 uses)"),
            "per-corpus z-score": ((8, 8), "left",
                                   "per-corpus z-score  0.991"),
            "CORAL": ((0, -16), "center", "CORAL  0.855"),
            "linear projection k=3": ((-8, -16), "right",
                                      "linear projection  0.701"),
        }
        off, ha, text = labels[name]
        ax.annotate(text, (row.corpus, row.pd_hc),
                    textcoords="offset points", xytext=off, ha=ha,
                    fontsize=8, color="#333", zorder=4)

    ax.scatter([], [], marker="o", c="#2a9d8f", s=70,
               label="one map for every row  (LODO-valid)")
    ax.scatter([], [], marker="s", c="#dd8452", s=70,
               label="needs the corpus label to choose its map")
    ax.set_xlabel("corpus identity, balanced accuracy   (left is the goal)")
    ax.set_ylabel("PD vs HC, balanced accuracy   (up is the goal)")
    ax.set_title("Up and left is the goal. Nothing goes there.")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "ablation_methods.png")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heavy", action="store_true",
                    help="CSV figures only; skip covariance heatmaps and t-SNE")
    args = ap.parse_args()
    plot_moment_ladder()
    plot_methods()
    if args.skip_heavy:
        return
    print("loading duration features for covariance and t-SNE panels...")
    values, meta = load_sample()
    train, test = fold0(values, meta)
    print(f"fold 0: {len(train)} train / {len(test)} test segments, "
          f"{meta.iloc[train].speaker_id.nunique()} / "
          f"{meta.iloc[test].speaker_id.nunique()} speakers")
    plot_coral_corr(values, meta, train)
    plot_tsne(values, meta, train, test)


if __name__ == "__main__":
    main()
