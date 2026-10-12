"""Publication figures for Stage 8.

    python scripts/plot_stage8.py
    python scripts/plot_stage8.py --family mel --level duration

handcrafted duration keeps the original unsuffixed figure stems; any other
family/level appends "_<family>_<level>" so the two do not overwrite.

Reads the CSVs under reports/ plus speaker-level predictions and DANN
metrics.csv. Writes PNG (220 dpi) and PDF to notebooks/figures/.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import geometry as geo                      # noqa: E402
from src.evaluation import pool                                 # noqa: E402
from src.training.config import TrainConfig, find_runs          # noqa: E402

FIGURES = REPO_ROOT / "notebooks" / "figures"
REPORTS = REPO_ROOT / "reports"
EXPERIMENTS = REPO_ROOT / "experiments"

CORPUS_ORDER = ["ItalianPD", "MDVR-KCL", "NeuroVoz"]

# Set by main(). GBT is handcrafted-only: there is no tree baseline on Mel.
FAMILY = "handcrafted"
LEVEL = "duration"
SUFFIX = ""
REGIME_ORDER = ["gbt", "baseline", "dann"]
REGIME_LABEL = {"gbt": "GBT", "baseline": "baseline", "dann": "DANN"}
REGIME_COLOUR = {"gbt": "#7f7f7f", "baseline": "#4c72b0", "dann": "#c44e52"}
CORPUS_COLOUR = {"NeuroVoz": "#4c72b0", "ItalianPD": "#dd8452", "MDVR-KCL": "#55a868"}
CLASS_COLOUR = {"HC": "#4c72b0", "PD": "#c44e52"}
PAIR_LABEL = {
    "ItalianPD/MDVR-KCL": "IT / MDVR",
    "ItalianPD/NeuroVoz": "IT / NV",
    "MDVR-KCL/NeuroVoz": "MDVR / NV",
}


def _style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem, suffixed=True):
    """`suffixed=False` for stems that already carry the family and level."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    if suffixed:
        stem = f"{stem}{SUFFIX}"
    for ext in ("png", "pdf"):
        dest = FIGURES / f"{stem}.{ext}"
        fig.savefig(dest, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote notebooks/figures/{stem}.png|.pdf")


def spine():
    """Family x level being plotted, for figure titles."""
    return f"{FAMILY} {LEVEL}"


def run(regime, protocol="mixed"):
    """The config of one run family, for the family and level being plotted."""
    return TrainConfig(family=FAMILY, level=LEVEL, protocol=protocol,
                       regime=regime)


def grouped_bars(ax, categories, series, ylabel, ylim=(0, 1.05), chance=None):
    n = len(categories)
    k = len(series)
    width = min(0.8 / k, 0.28)
    x = np.arange(n)
    for i, (name, values, colour) in enumerate(series):
        ax.bar(x + (i - (k - 1) / 2) * width, values, width,
               label=name, color=colour, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if chance is not None:
        ax.axhline(chance, color="#888888", ls="--", lw=1, zorder=0)
        ax.text(n - 0.5, chance + 0.02, f"chance {chance:.2f}",
                ha="right", va="bottom", color="#666666", fontsize=8)
    ax.legend(frameon=False, loc="upper right")


def plot_mixed_ba(clf):
    clf = clf[clf.regime.isin(REGIME_ORDER)]
    cats = ["all"] + CORPUS_ORDER
    series = []
    for regime in REGIME_ORDER:
        vals = []
        for cat in cats:
            row = clf[(clf.regime == regime) & (clf.dataset == cat)]
            vals.append(float(row.balanced_acc.iloc[0]))
        series.append((REGIME_LABEL[regime], vals, REGIME_COLOUR[regime]))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped_bars(ax, ["pooled"] + CORPUS_ORDER, series,
                 "speaker-level balanced accuracy", chance=0.5)
    ax.set_title(f"Mixed protocol · pooled OOF · {spine()}")
    save(fig, "stage8_ba_mixed")


def plot_lodo(lodo):
    lodo = lodo[(lodo.regime.isin(REGIME_ORDER)) & (lodo.dataset == lodo.held_out)]
    series = []
    for regime in REGIME_ORDER:
        vals = [float(lodo[(lodo.regime == regime) & (lodo.held_out == c)].balanced_acc.iloc[0])
                for c in CORPUS_ORDER]
        series.append((REGIME_LABEL[regime], vals, REGIME_COLOUR[regime]))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped_bars(ax, CORPUS_ORDER, series,
                 "speaker-level balanced accuracy", chance=0.5)
    ax.set_title(f"LODO · held-out corpus · {spine()}")
    ax.set_xlabel("held-out corpus")
    save(fig, "stage8_ba_lodo")


def plot_lodo_threshold(lodo):
    """What the fixed 0.5 threshold costs under domain shift."""
    lodo = lodo[(lodo.regime.isin(REGIME_ORDER)) & (lodo.dataset == lodo.held_out)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    series = []
    for name, column, colour in (
            ("BA @ 0.5", "balanced_acc", "#bbbbbb"),
            ("BA @ prior-matched", "balanced_acc_prior", REGIME_COLOUR["baseline"]),
            ("AUC (threshold-free)", "auc", REGIME_COLOUR["dann"])):
        vals = [float(lodo[(lodo.regime == "baseline")
                           & (lodo.held_out == c)][column].iloc[0])
                for c in CORPUS_ORDER]
        series.append((name, vals, colour))
    grouped_bars(ax, CORPUS_ORDER, series, "score", chance=0.5)
    ax.set_xlabel("held-out corpus")
    ax.set_title(f"LODO baseline · {spine()}")

    ax = axes[1]
    x = np.arange(len(CORPUS_ORDER))
    pred = [float(lodo[(lodo.regime == "baseline")
                       & (lodo.held_out == c)].pred_pd_rate.iloc[0])
            for c in CORPUS_ORDER]
    true = [float(lodo[(lodo.regime == "baseline")
                       & (lodo.held_out == c)].true_pd_rate.iloc[0])
            for c in CORPUS_ORDER]
    ax.bar(x - 0.2, pred, 0.4, color="#bbbbbb", label="predicted PD", edgecolor="none")
    ax.bar(x + 0.2, true, 0.4, color=REGIME_COLOUR["baseline"], label="actual PD",
           edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("ItalianPD", "IT").replace("MDVR-KCL", "MDVR")
                        .replace("NeuroVoz", "NV") for c in CORPUS_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of speakers")
    ax.set_title("the actual bug")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("A fixed 0.5 threshold breaks under corpus shift, "
                 "not the representation", y=1.02)
    fig.tight_layout()
    save(fig, "stage8_lodo_threshold")


def plot_folds(folds):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for regime in REGIME_ORDER:
        sub = folds[folds.regime == regime].sort_values("fold")
        ax.plot(sub.fold, sub.balanced_acc, marker="o", ms=5, lw=1.2,
                color=REGIME_COLOUR[regime], label=REGIME_LABEL[regime])
        mean, std = sub.balanced_acc.mean(), sub.balanced_acc.std(ddof=1)
        ax.axhline(mean, color=REGIME_COLOUR[regime], ls=":", lw=1, alpha=0.8)
    ax.axhline(0.5, color="#888888", ls="--", lw=1)
    ax.set_xticks(range(10))
    ax.set_xlabel("mixed configuration (test fold)")
    ax.set_ylabel("test balanced accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Per-fold mixed test BA · 21–22 speakers each")
    ax.legend(frameon=False)
    save(fig, "stage8_ba_folds")


def plot_kl(kl):
    pairs = list(PAIR_LABEL)
    top = 1.15 * float(kl[kl.regime.isin(("baseline", "dann"))].kl.max())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, label in zip(axes, ("HC", "PD")):
        series = []
        for regime in ("baseline", "dann"):
            vals = [float(kl[(kl.regime == regime) & (kl.label == label) & (kl.pair == p)].kl.iloc[0])
                    for p in pairs]
            series.append((REGIME_LABEL[regime], vals, REGIME_COLOUR[regime]))
        grouped_bars(ax, [PAIR_LABEL[p] for p in pairs], series,
                     "symmetric KL", ylim=(0, top))
        ax.set_title(label)
        if label == "PD":
            ax.set_ylabel("")
    fig.suptitle("Intra-class corpus KL · 16-d PCA Gaussians", y=1.02)
    fig.tight_layout()
    save(fig, "stage8_kl")


def plot_tcm(tcm):
    cats = ["all"] + CORPUS_ORDER
    top = 1.15 * float(tcm[tcm.regime.isin(("baseline", "dann"))].tcm.max())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, label in zip(axes, ("HC", "PD")):
        series = []
        for regime in ("baseline", "dann"):
            vals = [float(tcm[(tcm.regime == regime) & (tcm.label == label) & (tcm.dataset == d)].tcm.iloc[0])
                    for d in cats]
            series.append((REGIME_LABEL[regime], vals, REGIME_COLOUR[regime]))
        grouped_bars(ax, ["pooled"] + CORPUS_ORDER, series,
                     "TCM (trace of covariance)", ylim=(0, top))
        ax.set_title(label)
        if label == "PD":
            ax.set_ylabel("")
    fig.suptitle("Intra-class TCM · smaller is tighter", y=1.02)
    fig.tight_layout()
    save(fig, "stage8_tcm")


def plot_roc():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    loaded = {r: pool.mixed_oof(run(r))[0] for r in REGIME_ORDER}
    for ax, corpus in zip(axes, CORPUS_ORDER):
        for regime in REGIME_ORDER:
            pred = loaded[regime]
            sub = pred[pred.dataset == corpus]
            y = (sub.label == "PD").to_numpy()
            fpr, tpr, _ = roc_curve(y, sub.prob_pd)
            ax.plot(fpr, tpr, color=REGIME_COLOUR[regime], lw=1.6,
                    label=f"{REGIME_LABEL[regime]}  {roc_auc_score(y, sub.prob_pd):.2f}")
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1)
        ax.set_title(f"{corpus}  n={int((loaded['baseline'].dataset == corpus).sum())}")
        ax.set_xlabel("false positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("true positive rate")
    fig.suptitle(f"Mixed OOF ROC · speaker-level · {spine()}", y=1.04)
    fig.tight_layout()
    save(fig, "stage8_roc_mixed")


def plot_domain_acc(trace):
    """Domain-head curve per fold, with the checkpoint and last-epoch points.

    Both point sets are already columns of `stage8_*_checkpoints.csv`
    (`pool.selection_trace`), so they are read from `trace` rather than
    re-derived from summary.json here. Only the curves need metrics.csv.
    """
    dann = trace[trace.regime == "dann"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for cfg in find_runs(protocol="mixed", family=FAMILY, level=LEVEL,
                         regime="dann"):
        metrics = EXPERIMENTS / cfg.resolved_run_id() / "metrics.csv"
        if not metrics.exists():
            continue
        m = pd.read_csv(metrics)
        ax.plot(m.epoch, m.train_domain_acc, color=REGIME_COLOUR["dann"],
                alpha=0.25, lw=1)
    ax.scatter(dann.best_epoch, dann.best_train_domain_acc,
               c=REGIME_COLOUR["baseline"], s=36, zorder=3, label="saved checkpoint")
    ax.scatter(dann.last_epoch, dann.last_train_domain_acc,
               c=REGIME_COLOUR["dann"], s=36, marker="D", zorder=3, label="last epoch")
    ax.axhline(1 / 3, color="#888888", ls="--", lw=1)
    ax.text(0.2, 1 / 3 + 0.02, "chance 0.33", color="#666666", fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train domain accuracy")
    lo = min(dann.best_train_domain_acc.min(), dann.last_train_domain_acc.min())
    hi = max(dann.best_train_domain_acc.max(), dann.last_train_domain_acc.max())
    ax.set_ylim(min(0.28, lo - 0.05), max(0.40, hi + 0.05))
    ax.set_title(f"DANN domain head · mixed {spine()}")
    ax.legend(frameon=False)
    save(fig, "stage8_domain_acc")


def plot_checkpoint_lambda(trace):
    dann = trace[trace.regime == "dann"].sort_values("fold")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colours = [REGIME_COLOUR["dann"] if lam > 0.05 else "#bbbbbb"
               for lam in dann.best_lambda]
    ax.bar(dann.fold, dann.best_lambda, color=colours, edgecolor="none")
    ax.set_xticks(range(10))
    ax.set_xlabel("mixed configuration")
    ax.set_ylabel("λ at saved checkpoint")
    zeros = int((dann.best_lambda <= 0.05).sum())
    ax.set_title(f"{zeros} of {len(dann)} DANN folds saved λ = 0")
    handles = [
        Line2D([0], [0], color=REGIME_COLOUR["dann"], lw=8, label="adversary kept"),
        Line2D([0], [0], color="#bbbbbb", lw=8, label="λ = 0 (epoch 0)"),
    ]
    ax.legend(handles=handles, frameon=False)
    save(fig, "stage8_checkpoint_lambda")


def _tsne_panel(ax, frame, colour_by, title):
    cols = geo.embedding_cols(frame)
    x = frame[cols].to_numpy(dtype=np.float64)
    n = len(frame)
    z = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
             init="pca", random_state=0).fit_transform(x)
    palette = CORPUS_COLOUR if colour_by == "dataset" else CLASS_COLOUR
    for name, colour in palette.items():
        at = frame[colour_by].to_numpy() == name
        if not at.any():
            continue
        ax.scatter(z[at, 0], z[at, 1], c=colour, s=22, alpha=0.85,
                   label=name, edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best", fontsize=8)


def plot_tsne():
    _, baseline = pool.mixed_oof(run("baseline"))
    _, dann = pool.mixed_oof(run("dann"))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6))
    _tsne_panel(axes[0, 0], baseline, "dataset", "baseline · corpus")
    _tsne_panel(axes[0, 1], baseline, "label", "baseline · class")
    _tsne_panel(axes[1, 0], dann, "dataset", "DANN · corpus")
    _tsne_panel(axes[1, 1], dann, "label", "DANN · class")
    fig.suptitle(f"Speaker embeddings · mixed OOF · {spine()}", y=0.995)
    fig.tight_layout()
    save(fig, f"tsne_{FAMILY}_{LEVEL}_baseline_dann", suffixed=False)


def plot_geometry_vs_pd(clf, kl, tcm):
    """One-slide summary: PD BA (NeuroVoz) vs mean HC KL vs pooled HC TCM."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))

    nv = clf[(clf.dataset == "NeuroVoz") & (clf.regime.isin(("baseline", "dann")))]
    axes[0].bar([REGIME_LABEL[r] for r in nv.regime], nv.balanced_acc,
                color=[REGIME_COLOUR[r] for r in nv.regime], edgecolor="none")
    axes[0].axhline(0.5, color="#888888", ls="--", lw=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("NeuroVoz mixed OOF")

    hc_kl = kl[kl.label == "HC"]
    pairs = list(PAIR_LABEL)
    x = np.arange(len(pairs))
    w = 0.36
    b = [float(hc_kl[(hc_kl.regime == "baseline") & (hc_kl.pair == p)].kl.iloc[0]) for p in pairs]
    d = [float(hc_kl[(hc_kl.regime == "dann") & (hc_kl.pair == p)].kl.iloc[0]) for p in pairs]
    axes[1].bar(x - w / 2, b, w, color=REGIME_COLOUR["baseline"], label="baseline", edgecolor="none")
    axes[1].bar(x + w / 2, d, w, color=REGIME_COLOUR["dann"], label="DANN", edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PAIR_LABEL[p] for p in pairs], rotation=15)
    axes[1].set_ylabel("symmetric KL")
    axes[1].set_title("HC corpus KL")
    axes[1].legend(frameon=False)

    hc_tcm = tcm[(tcm.label == "HC") & (tcm.dataset == "all")]
    axes[2].bar([REGIME_LABEL[r] for r in hc_tcm.regime], hc_tcm.tcm,
                color=[REGIME_COLOUR[r] for r in hc_tcm.regime], edgecolor="none")
    axes[2].set_ylabel("TCM")
    axes[2].set_title("HC cloud tightness")

    fig.suptitle("DANN mixes corpora (KL) by smearing (TCM), not by helping PD", y=1.03)
    fig.tight_layout()
    save(fig, "stage8_geometry_vs_pd")


def main():
    global FAMILY, LEVEL, SUFFIX, REGIME_ORDER
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="handcrafted",
                    choices=["handcrafted", "mel"])
    ap.add_argument("--level", default="duration")
    args = ap.parse_args()
    FAMILY, LEVEL = args.family, args.level
    if (FAMILY, LEVEL) != ("handcrafted", "duration"):
        SUFFIX = f"_{FAMILY}_{LEVEL}"
    if FAMILY == "mel":
        REGIME_ORDER = ["baseline", "dann"]

    _style()
    stem = REPORTS / f"stage8_{FAMILY}_{LEVEL}"
    clf = pd.read_csv(f"{stem}_classification.csv")
    lodo = pd.read_csv(f"{stem}_lodo.csv")
    folds = pd.read_csv(f"{stem}_folds.csv")
    kl = pd.read_csv(f"{stem}_kl.csv")
    tcm = pd.read_csv(f"{stem}_tcm.csv")
    trace = pd.read_csv(f"{stem}_checkpoints.csv")
    plot_mixed_ba(clf)
    plot_lodo(lodo)
    plot_lodo_threshold(lodo)
    plot_folds(folds)
    plot_kl(kl)
    plot_tcm(tcm)
    plot_roc()
    plot_domain_acc(trace)
    plot_checkpoint_lambda(trace)
    plot_geometry_vs_pd(clf, kl, tcm)
    plot_tsne()


if __name__ == "__main__":
    main()
