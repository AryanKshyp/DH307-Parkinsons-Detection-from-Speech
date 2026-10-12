"""Stage 2 exploratory analysis.

    python scripts/eda.py

Reads manifest_processed.csv, writes figures to notebooks/figures/ and a notes
file to notebooks/eda_notes.md. The headline deliverable is a t-SNE of mean
MFCCs coloured by corpus: if raw acoustic features already separate the
corpora, the corpus-clustering problem is established on our own data before
anything is trained.

Mean MFCCs are cached to data/interim/mean_mfcc.npy; delete it to recompute.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
FIGURES = REPO_ROOT / "notebooks" / "figures"
NOTES = REPO_ROOT / "notebooks" / "eda_notes.md"
CACHE = REPO_ROOT / "data" / "interim" / "mean_mfcc.npy"

N_MFCC = 20
SEED = 0
COLOURS = {"NeuroVoz": "#4c72b0", "ItalianPD": "#dd8452", "MDVR-KCL": "#55a868"}

# The four assigned tasks, which group corpus-specific task codes into the
# families the project is actually asked to compare across corpora. Vowels and
# DDK are in neither family: they are kept for utterance-level work but are not
# part of the assigned comparison.
TASK_FAMILY = {
    "read_text": "reading passage",
    "read_words": "words",
    "read_phrases": "phrases",
    "sentence": "phrases",
    "monologue": "spontaneous",
    "spontaneous": "spontaneous",
}

notes = []


def say(text=""):
    print(text)
    notes.append(text)


def table(df, title):
    say(f"### {title}\n")
    say(df.to_markdown())
    say()


# --------------------------------------------------------------------------
# demographics
# --------------------------------------------------------------------------

def neurovoz_ages():
    root = REPO_ROOT / "data" / "raw" / "NeuroVoz" / "data" / "metadata"
    rows = []
    for group, name in [("HC", "metadata_hc.csv"), ("PD", "metadata_pd.csv")]:
        df = pd.read_csv(root / name, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        df = df.drop_duplicates(subset="ID")
        rows.append(pd.DataFrame({
            "speaker_id": [f"neurovoz_{group}_{int(i):04d}" for i in df.ID],
            "age": pd.to_numeric(df.Age, errors="coerce"),
            "sex": df.Sex.map({0: "F", 1: "M"}),
        }))
    return pd.concat(rows, ignore_index=True)


def _italian_sheet(path):
    """Rows below the header line that holds 'name' and 'surname'."""
    raw = pd.read_excel(path, header=None)
    header = None
    for idx, row in raw.iterrows():
        cells = [str(c).strip().lower() for c in row.tolist()]
        if "name" in cells and "surname" in cells:
            header = idx
            cols = cells
            break
    if header is None:
        return pd.DataFrame(columns=["name", "surname", "sex", "age"])

    body = raw.iloc[header + 1:].copy()
    body.columns = cols
    keep = body.loc[:, ~body.columns.duplicated()]
    return keep[["name", "surname", "sex", "age"]]


def _italian_key(name, surname):
    name = re.sub(r"[*\s]+$", "", str(name)).strip().lower()
    surname = re.sub(r"[*\s]+", "", str(surname)).strip().lower()
    return name, surname[:1]


def italianpd_ages():
    root = REPO_ROOT / "data" / "raw" / "ItalianPD"
    sheets = {
        "italianpd_yhc": root / "15 Young Healthy Control" / "15 YHC.xlsx",
        "italianpd_ehc": root / "22 Elderly Healthy Control" / "Tab 3.xlsx",
        "italianpd_pd": root / "28 People with Parkinson's disease" / "TAB 5.xlsx",
    }
    lookup = {}
    for prefix, path in sheets.items():
        for _, row in _italian_sheet(path).iterrows():
            if pd.isna(row["name"]) or str(row["name"]).strip() in ("", "nan"):
                continue
            age = pd.to_numeric(row["age"], errors="coerce")
            if pd.isna(age):
                continue
            lookup[(prefix, *_italian_key(row["name"], row["surname"]))] = (
                float(age), str(row["sex"]).strip().upper()[:1])
    return lookup


def italianpd_join(speaker_ids):
    """Map italianpd_<cohort>[_<bin>]_<Name S> speaker ids onto the sheets."""
    lookup = italianpd_ages()
    out = []
    for sid in speaker_ids:
        m = re.match(r"^(italianpd_(?:yhc|ehc|pd))(?:_[\d-]+)?_(.+)$", sid)
        if not m:
            continue
        prefix, folder = m.groups()
        parts = folder.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        hit = lookup.get((prefix, *_italian_key(*parts)))
        if hit:
            out.append({"speaker_id": sid, "age": hit[0], "sex": hit[1]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def mean_mfccs(paths):
    if CACHE.exists():
        cached = np.load(CACHE)
        if len(cached) == len(paths):
            print(f"using cached MFCCs from {CACHE.relative_to(REPO_ROOT)}")
            return cached

    out = np.zeros((len(paths), N_MFCC), dtype=np.float32)
    for i, path in enumerate(tqdm(paths, desc="mfcc", unit="file")):
        y, sr = librosa.load(REPO_ROOT / path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=400, hop_length=160)
        out[i] = mfcc.mean(axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, out)
    return out


def separability(X, y, groups, what):
    """How well can `y` be predicted from mean MFCCs, speaker-disjoint?"""
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=SEED))
    scores = cross_val_score(model, X, y, groups=groups,
                             cv=GroupKFold(n_splits=5), scoring="accuracy")
    chance = pd.Series(y).value_counts(normalize=True).max()
    say(f"- {what}: {scores.mean():.3f} +/- {scores.std():.3f} "
        f"(majority class {chance:.3f})")
    return scores.mean(), chance


# --------------------------------------------------------------------------

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(MANIFEST)
    order = ["NeuroVoz", "ItalianPD", "MDVR-KCL"]

    say("# Stage 2 - exploratory analysis\n")
    say(f"Generated from `{MANIFEST.relative_to(REPO_ROOT)}`, "
        f"{len(d)} files / {d.speaker_id.nunique()} speakers / "
        f"{d.duration.sum()/3600:.2f} hours.\n")

    # --- class and domain balance -----------------------------------------
    spk = d.drop_duplicates("speaker_id")
    table(pd.crosstab(spk.dataset, spk.label).loc[order].assign(
        total=lambda t: t.sum(axis=1)), "Speakers per dataset x label")
    table(d.groupby("dataset").agg(
        files=("path", "size"),
        hours=("duration", lambda s: round(s.sum() / 3600, 2)),
        median_s=("duration", lambda s: round(s.median(), 2)),
    ).loc[order], "Files, hours and median utterance length")

    # --- task availability -------------------------------------------------
    avail = pd.crosstab(d.task, d.dataset)[order]
    table(avail, "Task availability (files per dataset)")
    shared = avail[(avail > 0).all(axis=1)]
    say(f"Raw task codes present in all three corpora: "
        f"{list(shared.index) if len(shared) else 'none'}\n")

    # Raw task codes barely overlap, but the four assigned tasks do: grouping
    # the codes into those families is what makes a cross-corpus comparison
    # possible at all.
    fam = d.assign(family=d.task.map(TASK_FAMILY)).dropna(subset=["family"])
    table(pd.crosstab(fam.family, fam.dataset).reindex(columns=order,
                                                       fill_value=0),
          "Assigned task families (files per dataset)")
    say(f"{len(fam)} of {len(d)} files belong to an assigned task family; "
        f"the remaining {len(d) - len(fam)} are vowels and DDK.\n")

    # --- durations ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].boxplot([d[d.dataset == n].duration for n in order],
                    tick_labels=order)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("utterance duration (s, log)")
    axes[0].set_title("Duration by dataset")
    tasks = sorted(d.task.unique())
    axes[1].boxplot([d[d.task == t].duration for t in tasks], tick_labels=tasks)
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set_title("Duration by task")
    fig.tight_layout()
    fig.savefig(FIGURES / "duration.png", dpi=150)
    plt.close(fig)

    # --- age ---------------------------------------------------------------
    demo = pd.concat([neurovoz_ages(),
                      italianpd_join(sorted(d.speaker_id.unique()))],
                     ignore_index=True)
    spk_demo = spk.merge(demo, on="speaker_id", how="left")
    coverage = spk_demo.groupby("dataset").age.agg(
        speakers="size", with_age="count").loc[order]
    table(coverage, "Age metadata coverage (speakers)")

    aged = spk_demo.dropna(subset=["age"])
    table(aged.groupby(["dataset", "label"]).age.agg(
        n="size", mean=lambda s: round(s.mean(), 1),
        min="min", max="max"), "Age by dataset x label")
    table(aged[aged.dataset == "ItalianPD"].groupby(["cohort"]).age.agg(
        n="size", mean=lambda s: round(s.mean(), 1), min="min", max="max"),
        "ItalianPD age by cohort")

    # Controls much younger than the patients let a model separate the classes
    # on age alone. ItalianPD is the documented case; check the others too.
    young = aged[aged.label == "HC"].groupby("dataset").age.agg(
        controls="size", under_50=lambda s: int((s < 50).sum()))
    young["patient_min_age"] = aged[aged.label == "PD"].groupby(
        "dataset").age.min()
    table(young, "Controls younger than the youngest patient")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in order:
        ages = aged[aged.dataset == name].age
        if len(ages):
            ax.hist(ages, bins=np.arange(15, 95, 5), alpha=0.55,
                    label=f"{name} (n={len(ages)})", color=COLOURS[name])
    ax.set_xlabel("age")
    ax.set_ylabel("speakers")
    ax.set_title("Age distribution by corpus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "age.png", dpi=150)
    plt.close(fig)

    # --- mean MFCC t-SNE ---------------------------------------------------
    X = mean_mfccs(list(d.path))
    say("\n## Corpus separability from mean MFCCs\n")
    say("Logistic regression on the 20 mean MFCCs, 5-fold speaker-disjoint:\n")
    separability(X, d.dataset.values, d.speaker_id.values, "predicting corpus")
    separability(X, d.label.values, d.speaker_id.values, "predicting PD/HC")

    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=SEED).fit_transform(StandardScaler().fit_transform(X))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in order:
        m = (d.dataset == name).values
        axes[0].scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                        label=name, color=COLOURS[name])
    axes[0].set_title("mean MFCC t-SNE, coloured by corpus")
    axes[0].legend(markerscale=3)
    for label, colour in [("HC", "#4c72b0"), ("PD", "#c44e52")]:
        m = (d.label == label).values
        axes[1].scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                        label=label, color=colour)
    axes[1].set_title("same embedding, coloured by class")
    axes[1].legend(markerscale=3)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "tsne_mean_mfcc.png", dpi=150)
    plt.close(fig)

    say(f"\nFigures written to `{FIGURES.relative_to(REPO_ROOT)}`.")
    NOTES.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"\nnotes written to {NOTES.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
