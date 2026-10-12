"""Which of the 40 handcrafted features carry corpus vs Parkinson's.

    python scripts/probe_feature_importance.py
    python scripts/probe_feature_importance.py --level duration

Same 40k-row, seed-0, speaker-grouped 5-fold setup as the Stage 5 probe.
Permutation importance is computed on each held-out fold under global
normalisation fitted on that fold's training rows. Features are binned by
whether they sit above the median importance for corpus, for PD/HC, both,
or neither. The corpus-only bin is then dropped and both targets are
re-scored; dropping the "both" bin is the stress test.

Writes:
    experiments/corpus_norm_ablation/feature_importance.csv
    experiments/corpus_norm_ablation/feature_drop.csv
    experiments/corpus_norm_ablation/figures/featimp_*.png
    notebooks/figures/featimp_*.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.features import store as fs                        # noqa: E402

MAX_ROWS = 40_000
SEED = 0
N_REPEATS = 5
SCOPE = "global"

OUT = REPO_ROOT / "experiments" / "corpus_norm_ablation"
FIG_A = OUT / "figures"
FIG_B = REPO_ROOT / "notebooks" / "figures"

BIN_COLOURS = {
    "corpus-only": "#dd8452",
    "PD-only": "#2a9d8f",
    "both": "#6a4c93",
    "neither": "#b0b0b0",
}


def pretty(name):
    if name.startswith("mfcc"):
        n = int(name[4:6])
        stat = "mean" if name.endswith("mean") else "std"
        return f"MFCC-{n} {stat}"
    return {
        "zcr_mean": "ZCR mean", "zcr_std": "ZCR std",
        "rms_mean": "RMS mean", "rms_std": "RMS std",
        "f0_mean": "F0 mean", "f0_std": "F0 std",
        "voiced_frac": "voiced fraction",
        "jitter_local": "jitter", "shimmer_local": "shimmer",
        "f1_mean": "F1", "f2_mean": "F2", "f3_mean": "F3",
        "intensity_mean": "intensity mean", "intensity_std": "intensity std",
    }.get(name, name)


def load_sample(level):
    rng = np.random.default_rng(SEED)
    values, meta = fs.load_handcrafted(level)
    names = fs.columns()
    if len(meta) > MAX_ROWS:
        pick = np.sort(rng.choice(len(meta), MAX_ROWS, replace=False))
        values, meta = values[pick], meta.iloc[pick].reset_index(drop=True)
    return values, meta, names


def fold_matrices(values, meta, train, test):
    stats = fs.fit_stats(values[train], meta.iloc[train], SCOPE)
    x_train = fs.apply(values[train], meta.iloc[train], SCOPE, stats)
    x_test = fs.apply(values[test], meta.iloc[test], SCOPE, stats)
    return x_train, x_test


def score_cols(values, meta, target, cols):
    y = meta[target].values
    groups = meta.speaker_id.values
    folds = GroupKFold(n_splits=5)
    truth, predicted = [], []
    for train, test in folds.split(values, y, groups):
        x_train, x_test = fold_matrices(values[:, cols], meta, train, test)
        model = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
        model.fit(x_train, y[train])
        truth.append(y[test])
        predicted.append(model.predict(x_test))
    return balanced_accuracy_score(np.concatenate(truth),
                                   np.concatenate(predicted))


def importances(values, meta, target):
    y = meta[target].values
    groups = meta.speaker_id.values
    folds = GroupKFold(n_splits=5)
    n_feat = values.shape[1]
    perm = np.zeros((5, n_feat))
    for i, (train, test) in enumerate(folds.split(values, y, groups)):
        x_train, x_test = fold_matrices(values, meta, train, test)
        model = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
        model.fit(x_train, y[train])
        result = permutation_importance(
            model, x_test, y[test], n_repeats=N_REPEATS,
            random_state=SEED, scoring="balanced_accuracy", n_jobs=-1)
        perm[i] = result.importances_mean
        print(f"  {target:8s} fold {i + 1}/5  "
              f"BA {balanced_accuracy_score(y[test], model.predict(x_test)):.3f}")
    return perm.mean(0), perm.std(0)


def assign_bins(corpus, pd_hc):
    med_c, med_p = np.median(corpus), np.median(pd_hc)
    bins = []
    for c, p in zip(corpus, pd_hc):
        hi_c, hi_p = c > med_c, p > med_p
        if hi_c and hi_p:
            bins.append("both")
        elif hi_c:
            bins.append("corpus-only")
        elif hi_p:
            bins.append("PD-only")
        else:
            bins.append("neither")
    return bins, med_c, med_p


def savefig(fig, name):
    FIG_A.mkdir(parents=True, exist_ok=True)
    FIG_B.mkdir(parents=True, exist_ok=True)
    for folder in (FIG_A, FIG_B):
        fig.savefig(folder / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote experiments/corpus_norm_ablation/figures/{name}")


def _offsets(part):
    """Tiny hand offsets so nearby labels do not sit on the same point."""
    special = {
        "voiced fraction": (4, -9),
        "MFCC-7 mean": (4, 4),
        "F1": (-18, 6),
        "MFCC-9 mean": (4, -8),
        "MFCC-1 std": (4, -8),
        "ZCR std": (4, 4),
        "intensity mean": (6, 10),
        "MFCC-5 mean": (4, -9),
        "MFCC-12 mean": (4, -8),
        "MFCC-11 mean": (4, -8),
        "RMS mean": (4, 4),
        "MFCC-2 std": (4, 3),
        "ZCR mean": (4, 8),
        "shimmer": (4, -10),
    }
    for row in part.itertuples():
        yield row, special.get(row.label, (4, 3))


def plot_scatter(table, med_c, med_p):
    fig, ax = plt.subplots(figsize=(7.6, 7.1))
    xmax = max(table.pd_perm.max() * 1.08, med_p * 1.4)
    ymax = table.corpus_perm.max() * 1.12
    ax.set_xlim(-0.002, xmax)
    ax.set_ylim(-0.004, ymax)
    for bin_name, colour in BIN_COLOURS.items():
        part = table[table.bin == bin_name]
        ax.scatter(part.pd_perm, part.corpus_perm, c=colour, s=48,
                   label=f"{bin_name} (n={len(part)})", zorder=3,
                   edgecolors="white", linewidths=0.4)
        if bin_name == "neither":
            continue
        for row, off in _offsets(part):
            ax.annotate(row.label, (row.pd_perm, row.corpus_perm),
                        textcoords="offset points", xytext=off,
                        fontsize=7, color="#333", zorder=4)
    ax.axvline(med_p, color="#888", ls="--", lw=0.9, zorder=2)
    ax.axhline(med_c, color="#888", ls="--", lw=0.9, zorder=2)
    ax.text(xmax * 0.98, ymax * 0.97, "shared  —  DANN's problem",
            ha="right", va="top", fontsize=8, color="#6a4c93")
    ax.text(0.0005, ymax * 0.97, "corpus shortcut",
            ha="left", va="top", fontsize=8, color="#dd8452")
    ax.text(xmax * 0.98, -0.0025, "PD without domain",
            ha="right", va="bottom", fontsize=8, color="#2a9d8f")
    ax.set_xlabel("importance for PD vs HC  (drop in balanced accuracy)")
    ax.set_ylabel("importance for corpus identity  (drop in balanced accuracy)")
    ax.set_title("Which features carry the dataset, and which carry Parkinson's")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "featimp_scatter.png")


def plot_bars(table):
    ranked = table.sort_values("corpus_perm", ascending=True)
    y = np.arange(len(ranked))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 8.2), sharey=True)
    axes[0].barh(y, ranked.corpus_perm, color=[BIN_COLOURS[b] for b in ranked.bin],
                 height=0.75)
    axes[0].axvline(np.median(table.corpus_perm), color="#888", ls="--", lw=0.9)
    axes[0].set_xlabel("corpus importance")
    axes[0].set_title("Predicting which corpus")
    axes[1].barh(y, ranked.pd_perm, color=[BIN_COLOURS[b] for b in ranked.bin],
                 height=0.75)
    axes[1].axvline(np.median(table.pd_perm), color="#888", ls="--", lw=0.9)
    axes[1].set_xlabel("PD/HC importance")
    axes[1].set_title("Predicting Parkinson's")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ranked.label, fontsize=7.5)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Same features, two jobs  ·  colour is the bin from the scatter",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "featimp_bars.png")


def plot_drop(drop):
    labels = {
        "all 40": "all 40 features",
        "drop corpus-only": "drop corpus-only",
        "drop both + corpus-only": "drop both + corpus-only",
        "PD-only only": "PD-only features only",
    }
    order = [k for k in labels if k in set(drop.subset)]
    x = np.arange(len(order))
    corpus = [drop.loc[drop.subset == k, "corpus"].item() for k in order]
    pd_hc = [drop.loc[drop.subset == k, "pd_hc"].item() for k in order]
    chance_c = drop.corpus_chance.iloc[0]
    chance_p = drop.pd_hc_chance.iloc[0]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    w = 0.36
    ax.bar(x - w / 2, corpus, w, color="#4c72b0", label="corpus identity")
    ax.bar(x + w / 2, pd_hc, w, color="#c44e52", label="PD vs HC")
    ax.axhline(chance_c, color="#4c72b0", ls=":", lw=1.0, label="corpus chance")
    ax.axhline(chance_p, color="#c44e52", ls=":", lw=1.0, label="PD chance")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.25, 1.02)
    ax.set_title("What happens if the corpus-heavy features are removed")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (c, p) in enumerate(zip(corpus, pd_hc)):
        ax.text(i - w / 2, c + 0.015, f"{c:.2f}", ha="center", va="bottom",
                fontsize=8, color="#4c72b0")
        ax.text(i + w / 2, p + 0.015, f"{p:.2f}", ha="center", va="bottom",
                fontsize=8, color="#c44e52")
    fig.tight_layout()
    savefig(fig, "featimp_drop.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="duration")
    ap.add_argument("--from-csv", action="store_true",
                    help="redraw figures from the CSVs already on disk")
    args = ap.parse_args()

    if args.from_csv:
        table = pd.read_csv(OUT / "feature_importance.csv")
        drop = pd.read_csv(OUT / "feature_drop.csv")
        med_c, med_p = np.median(table.corpus_perm), np.median(table.pd_perm)
        plot_scatter(table, med_c, med_p)
        plot_bars(table)
        plot_drop(drop)
        return

    values, meta, names = load_sample(args.level)
    print(f"{args.level}: {len(meta)} segments, "
          f"{meta.speaker_id.nunique()} speakers, {len(names)} features")
    print("permutation importance for corpus identity:")
    c_mean, c_std = importances(values, meta, "dataset")
    print("permutation importance for PD/HC:")
    p_mean, p_std = importances(values, meta, "label")

    bins, med_c, med_p = assign_bins(c_mean, p_mean)
    table = pd.DataFrame({
        "feature": names,
        "label": [pretty(n) for n in names],
        "corpus_perm": c_mean, "corpus_perm_std": c_std,
        "pd_perm": p_mean, "pd_perm_std": p_std,
        "bin": bins,
    }).sort_values("corpus_perm", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "feature_importance.csv", index=False)
    print(f"\nwrote { (OUT / 'feature_importance.csv').relative_to(REPO_ROOT) }")
    print(table.groupby("bin").size().to_string())
    print(f"median corpus importance {med_c:.4f}, PD/HC {med_p:.4f}")

    all_cols = np.arange(len(names))
    corpus_only = np.array([b == "corpus-only" for b in bins])
    both = np.array([b == "both" for b in bins])
    pd_only = np.array([b == "PD-only" for b in bins])
    subsets = [
        ("all 40", all_cols),
        ("drop corpus-only", all_cols[~corpus_only]),
        ("drop both + corpus-only", all_cols[~(corpus_only | both)]),
        ("PD-only only", all_cols[pd_only]),
    ]
    drop_rows = []
    print("\nre-score after dropping bins:")
    for name, cols in subsets:
        if len(cols) == 0:
            print(f"  {name}: no columns, skipped")
            continue
        corpus = score_cols(values, meta, "dataset", cols)
        pathology = score_cols(values, meta, "label", cols)
        drop_rows.append({
            "subset": name, "n_features": int(len(cols)),
            "corpus": corpus, "pd_hc": pathology,
            "corpus_chance": 1 / meta.dataset.nunique(),
            "pd_hc_chance": 0.5,
        })
        print(f"  {name:28s} n={len(cols):2d}  "
              f"corpus {corpus:.3f}   PD/HC {pathology:.3f}")
    drop = pd.DataFrame(drop_rows)
    drop.to_csv(OUT / "feature_drop.csv", index=False)

    plot_scatter(table, med_c, med_p)
    plot_bars(table)
    plot_drop(drop)


if __name__ == "__main__":
    main()
