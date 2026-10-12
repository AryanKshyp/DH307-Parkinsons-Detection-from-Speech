"""How much does each normalisation scope erase, and of what?

    python scripts/probe_normalisation.py --levels word phoneme

Stage 5 stores features unnormalised and leaves the scope to training time. The
scope is not a neutral choice: per-speaker z-scoring removes whatever is constant
about a speaker's recording chain, which is part of what the domain adversary
exists to remove. This measures that directly, by predicting corpus identity and
PD/HC status from the same features under each scope.

The classifier is a gradient-boosted tree ensemble, chosen because it is
invariant to per-feature scaling. That matters: a linear model would need a
scaler, and the scaler would impose global normalisation on top of whatever scope
is being tested, collapsing the comparison. With trees, `none` and `global` are
expected to score alike -- global z-scoring is an affine map applied to every row
equally, so it preserves information -- while any drop under `speaker` or
`utterance` is real information loss, since those transform each group
differently. `corpus` also applies a different map per dataset, so trees will
see it; that is the optional batch-correction ablation.

Scores are balanced accuracy under speaker-grouped 5-fold CV, so no speaker is
ever in both train and test. Normalisation statistics are fitted on the training
fold only.

    python scripts/probe_normalisation.py --levels duration
    python scripts/probe_normalisation.py --levels duration --lodo-probe

The follow-up ablation (`experiments/corpus_norm_ablation/PLAN_second_order.md`)
adds four things to the same harness, so its numbers land in the same table:

  * per-fold scores and a 95% CI bootstrapped over **speakers**, not segments --
    duration windows are 50%-overlapping and correlated within speaker, so
    resampling segments would understate the interval twice over (R0)

        python scripts/probe_normalisation.py --levels duration \
            --scopes global corpus --per-fold --bootstrap 1000

  * row subsets, so corpus identity can be probed on task-matched and
    cohort-matched material. No task spans all three corpora, so this is
    necessarily pairwise and chance is 0.500, not 0.333 (R2)

        python scripts/probe_normalisation.py --levels duration \
            --corpora ItalianPD NeuroVoz --tasks vowel_a vowel_e vowel_i vowel_o vowel_u

  * `--project inlp`, deleting the linearly corpus-predictive directions. One
    global map, no corpus label at apply time, valid under strict LODO (R3)

        python scripts/probe_normalisation.py --levels duration \
            --project inlp --project-iters 0 1 2 3 --lodo-probe

  * `--align coral`, matching each corpus's full covariance. Domain-conditional,
    so transductive only (R4)

        python scripts/probe_normalisation.py --levels duration --align coral --lodo-probe
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.features import store as fs                        # noqa: E402
from src.features import coral as coral_mod                 # noqa: E402
from src.features import project as project_mod             # noqa: E402

MAX_ROWS = 40_000
SEED = 0


def verify(values, meta):
    """Guard against reading a bug as a finding.

    Both targets dropping to chance is also what a row shuffle inside the
    normaliser would look like, so check the transform before trusting it.
    """
    z = fs.normalise(values, meta, "speaker")
    speakers = meta.speaker_id.values
    first = np.where(speakers == meta.speaker_id.iloc[0])[0]
    order = np.nanmin([np.corrcoef(values[first, j], z[first, j])[0, 1]
                       for j in range(values.shape[1])])
    assert order > 0.999, f"normaliser reordered rows (corr {order})"
    grouped = pd.DataFrame(z).groupby(speakers)
    assert np.abs(grouped.mean().values).max() < 1e-8, "per-speaker mean not 0"

    raw = pd.DataFrame(values).groupby(speakers).mean().std(ddof=0).mean()
    left = pd.DataFrame(z).groupby(speakers).mean().std(ddof=0).mean()
    print(f"normaliser checks out: order preserved, per-speaker mean 0. "
          f"Between-speaker spread of feature means {raw:.2f} -> {left:.1e}, "
          f"which is where a speaker-constant label like PD/HC lives.")
    zc = fs.normalise(values, meta, "corpus")
    by_corpus = pd.DataFrame(zc).groupby(meta.dataset.values)
    assert np.abs(by_corpus.mean().values).max() < 1e-8, "per-corpus mean not 0"
    print("corpus scope checks out: per-dataset mean 0 on the fitted set.")


def subset(values, meta, corpora=None, tasks=None, drop_cohorts=None):
    """Restrict rows before subsampling, so the subset keeps its power."""
    keep = np.ones(len(meta), dtype=bool)
    if corpora:
        keep &= meta.dataset.isin(list(corpora)).to_numpy()
    if tasks:
        keep &= meta.task.isin(list(tasks)).to_numpy()
    if drop_cohorts:
        keep &= ~meta.cohort.isin(list(drop_cohorts)).to_numpy()
    if not keep.any():
        raise ValueError("no rows left after --corpora / --tasks / --drop-cohorts")
    at = np.flatnonzero(keep)
    return values[at], meta.iloc[at].reset_index(drop=True)


def load_level(level, family="handcrafted"):
    """Feature matrix and metadata for one level.

    `mel-meanpool` is the honest shallow analogue of the Mel family: one 65-d
    per-band mean per item, fed to the same tree. It closes the standing Stage 5
    open item on Mel domain signal. It is not a substitute for the CNN, and it
    does not bear on the second-order question.
    """
    if family == "handcrafted":
        values, meta = fs.load_handcrafted(level)
        return values, meta, fs.columns()
    if family != "mel-meanpool":
        raise ValueError(f"unknown family {family!r}")

    data, index = fs.load_mel(level)
    segments = pd.read_csv(fs.SEGMENTS, low_memory=False)
    columns = ["segment_id", "dataset", "language", "channel", "speaker_id",
               "label", "cohort", "task"]
    meta = index.merge(segments[columns], on="segment_id", how="left")
    meta = meta.dropna(subset=["speaker_id"]).reset_index(drop=True)
    # Subsample before touching the memmap: reading 40k slices costs seconds,
    # reading all 274k costs minutes, and the probe only ever uses 40k.
    rng = np.random.default_rng(SEED)
    if len(meta) > MAX_ROWS:
        pick = np.sort(rng.choice(len(meta), MAX_ROWS, replace=False))
        meta = meta.iloc[pick].reset_index(drop=True)
    pooled = np.zeros((len(meta), data.shape[1]))
    for i, item in enumerate(meta.itertuples()):
        frames = fs.mel_item(data, item)
        pooled[i] = frames.mean(axis=0) if len(frames) else 0.0
    names = [f"mel_band_{i:02d}" for i in range(data.shape[1])]
    return pooled, meta, names


def bootstrap_ci(truth, predicted, speakers, n_boot, seed=SEED):
    """95% CI for balanced accuracy, resampling speakers with replacement.

    Speakers, not segments: a speaker contributes thousands of correlated
    windows, and at the `duration` level adjacent windows additionally share
    half their audio. Resampling segments would report an interval several
    times too narrow.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(speakers, return_inverse=True)
    rows_for = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    scores = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([rows_for[i] for i in picked])
        # A resample can lose a class entirely; balanced accuracy over the
        # classes that survive would not be the same quantity, so skip it.
        if len(np.unique(truth[rows])) < len(np.unique(truth)):
            continue
        scores.append(balanced_accuracy_score(truth[rows], predicted[rows]))
    if not scores:
        return float("nan"), float("nan")
    return (float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def _transform_fold(x_train, x_test, meta_train, meta_test, transform):
    """Fit a post-normalisation transform on train rows, apply to both."""
    if transform is None:
        return x_train, x_test
    kind, options = transform
    if kind == "inlp":
        state = project_mod.fit(x_train, meta_train,
                                iters=options["iters"], seed=SEED)
        project_mod.verify(state, x_train.shape[1])
        return (project_mod.apply(x_train, state),
                project_mod.apply(x_test, state))
    if kind == "coral":
        state = coral_mod.fit(x_train, meta_train)
        coral_mod.verify(x_train, meta_train, state)
        return (coral_mod.apply(x_train, meta_train, state),
                coral_mod.apply(x_test, meta_test, state))
    raise ValueError(f"unknown transform {kind!r}")


def score(values, meta, target, scope, transform=None, n_boot=0):
    """Balanced accuracy predicting `target`, with the scope fitted per fold."""
    y = meta[target].values
    groups = meta.speaker_id.values
    folds = GroupKFold(n_splits=5)
    truth, predicted, held_speakers, per_fold = [], [], [], []
    for train, test in folds.split(values, y, groups):
        stats = fs.fit_stats(values[train], meta.iloc[train], scope)
        x_train = fs.apply(values[train], meta.iloc[train], scope, stats)
        x_test = fs.apply(values[test], meta.iloc[test], scope, stats)
        x_train, x_test = _transform_fold(
            x_train, x_test, meta.iloc[train], meta.iloc[test], transform)
        model = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
        model.fit(x_train, y[train])
        prediction = model.predict(x_test)
        per_fold.append(balanced_accuracy_score(y[test], prediction))
        truth.append(y[test])
        predicted.append(prediction)
        held_speakers.append(groups[test])
    truth = np.concatenate(truth)
    predicted = np.concatenate(predicted)
    held_speakers = np.concatenate(held_speakers)
    out = {"ba": balanced_accuracy_score(truth, predicted),
           "per_fold": per_fold}
    if n_boot:
        out["ci_lo"], out["ci_hi"] = bootstrap_ci(
            truth, predicted, held_speakers, n_boot)
    return out


def score_lodo(values, meta, scope, transform=None):
    """PD/HC on each held-out corpus; stats fitted on the other corpora only."""
    rows = []
    for held in sorted(meta.dataset.unique()):
        train = (meta.dataset != held).to_numpy()
        test = ~train
        at_train = np.flatnonzero(train)
        at_test = np.flatnonzero(test)
        stats = fs.fit_stats(values[train], meta.iloc[at_train], scope)
        x_train = fs.apply(values[train], meta.iloc[at_train], scope, stats)
        x_test = fs.apply(values[test], meta.iloc[at_test], scope, stats)
        x_train, x_test = _transform_fold(
            x_train, x_test, meta.iloc[at_train], meta.iloc[at_test], transform)
        model = HistGradientBoostingClassifier(max_iter=120, random_state=SEED)
        model.fit(x_train, meta.iloc[at_train].label.values)
        pred = model.predict(x_test)
        ba = balanced_accuracy_score(meta.iloc[at_test].label.values, pred)
        unseen = held not in stats
        rows.append({"held_out": held, "scope": scope, "pd_hc": ba,
                     "n_test": int(test.sum()),
                     "held_out_in_stats": (not unseen)})
        print(f"  lodo {held:12s} {scope:9s} PD/HC {ba:.3f}"
              f"  target_in_stats={not unseen}")
    return rows


def transforms_from(args):
    """The (name, transform) pairs this invocation should score."""
    if args.project == "inlp":
        return [(f"inlp_k{k}", ("inlp", {"iters": k}))
                for k in args.project_iters]
    if args.align == "coral":
        return [("coral", ("coral", {}))]
    return [("none", None)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=["word"])
    ap.add_argument("--family", default="handcrafted",
                    choices=["handcrafted", "mel-meanpool"])
    ap.add_argument("--scopes", nargs="+", default=list(fs.SCOPES),
                    choices=list(fs.SCOPES),
                    help="normalisation scopes to score (default: all five)")
    ap.add_argument("--corpora", nargs="+",
                    help="restrict to these datasets; chance changes with the "
                         "number kept, so read pairwise scores against 0.500")
    ap.add_argument("--tasks", nargs="+",
                    help="restrict to these tasks, for task-matched probes")
    ap.add_argument("--drop-cohorts", nargs="+",
                    help="drop these cohorts, e.g. young_control")
    ap.add_argument("--project", choices=["inlp"],
                    help="delete the linearly corpus-predictive directions")
    ap.add_argument("--project-iters", nargs="+", type=int, default=[1],
                    help="INLP rounds to score, e.g. 0 1 2 3")
    ap.add_argument("--align", choices=["coral"],
                    help="match each corpus's full covariance (transductive)")
    ap.add_argument("--per-fold", action="store_true",
                    help="print the five per-fold scores as well as the pooled one")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap reps for a 95% CI over speakers")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the normaliser assertions (they need >1 corpus)")
    ap.add_argument("--lodo-probe", action="store_true",
                    help="also score PD/HC holding out each corpus in turn")
    ap.add_argument("--tag", default="",
                    help="suffix for the output CSVs, so runs do not overwrite")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    results = []
    lodo_results = []
    out_dir = REPO_ROOT / "experiments" / "corpus_norm_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = transforms_from(args)

    for level in args.levels:
        values, meta, _ = load_level(level, args.family)
        values, meta = subset(values, meta, args.corpora, args.tasks,
                              args.drop_cohorts)
        if len(meta) > args.max_rows:
            pick = np.sort(rng.choice(len(meta), args.max_rows, replace=False))
            values, meta = values[pick], meta.iloc[pick].reset_index(drop=True)
        n_corpora = meta.dataset.nunique()
        print(f"\n=== {level} [{args.family}]: {len(meta)} segments, "
              f"{meta.speaker_id.nunique()} speakers, "
              f"{values.shape[1]} features ===")
        print("corpus mix: " + ", ".join(
            f"{k} {v}" for k, v in meta.dataset.value_counts().items()))
        if args.tasks:
            print("tasks: " + ", ".join(sorted(meta.task.unique())))
        print("channels: " + ", ".join(sorted(meta.channel.unique()))
              + "   languages: " + ", ".join(sorted(meta.language.unique())))
        print(f"chance: corpus {1 / n_corpora:.3f}, PD/HC 0.500")
        if n_corpora < 2:
            raise ValueError("corpus identity needs at least two datasets")
        if not args.skip_verify and args.family == "handcrafted":
            verify(values, meta)

        for scope in args.scopes:
            for name, transform in pairs:
                corpus = score(values, meta, "dataset", scope, transform,
                               args.bootstrap)
                pathology = score(values, meta, "label", scope, transform,
                                  args.bootstrap)
                row = {"level": level, "family": args.family, "scope": scope,
                       "transform": name,
                       "corpus": corpus["ba"], "pd_hc": pathology["ba"],
                       "corpus_chance": 1 / n_corpora, "pd_hc_chance": 0.5,
                       "n_rows": len(meta),
                       "n_speakers": int(meta.speaker_id.nunique()),
                       "corpora": "+".join(sorted(meta.dataset.unique())),
                       "tasks": ("all" if not args.tasks
                                 else "+".join(sorted(args.tasks))),
                       "dropped_cohorts": ("none" if not args.drop_cohorts
                                           else "+".join(args.drop_cohorts))}
                if args.bootstrap:
                    row.update({"corpus_ci_lo": corpus["ci_lo"],
                                "corpus_ci_hi": corpus["ci_hi"],
                                "pd_hc_ci_lo": pathology["ci_lo"],
                                "pd_hc_ci_hi": pathology["ci_hi"]})
                if args.per_fold:
                    for i, (c, p) in enumerate(zip(corpus["per_fold"],
                                                   pathology["per_fold"])):
                        row[f"corpus_fold{i}"] = c
                        row[f"pd_hc_fold{i}"] = p
                results.append(row)
                label = scope if name == "none" else f"{scope}+{name}"
                line = f"  {label:16s} corpus {corpus['ba']:.3f}"
                if args.bootstrap:
                    line += f" [{corpus['ci_lo']:.3f},{corpus['ci_hi']:.3f}]"
                line += f"   PD/HC {pathology['ba']:.3f}"
                if args.bootstrap:
                    line += f" [{pathology['ci_lo']:.3f},{pathology['ci_hi']:.3f}]"
                print(line)
                if args.per_fold:
                    print("    per-fold corpus " + " ".join(
                        f"{v:.3f}" for v in corpus["per_fold"]))
                    print("    per-fold PD/HC  " + " ".join(
                        f"{v:.3f}" for v in pathology["per_fold"]))

        if args.lodo_probe:
            print("LODO-style PD/HC (fit on the other corpora):")
            for scope in args.scopes:
                for name, transform in pairs:
                    for row in score_lodo(values, meta, scope, transform):
                        lodo_results.append(
                            {"level": level, "transform": name, **row})

    table = pd.DataFrame(results)
    print("\n=== balanced accuracy by normalisation scope ===")
    show = [c for c in ("scope", "transform", "corpus", "pd_hc",
                        "corpus_chance", "corpora", "tasks")
            if c in table.columns]
    print(table[show].round(3).to_string(index=False))
    mixed_path = out_dir / f"probe_mixed{args.tag}.csv"
    table.to_csv(mixed_path, index=False)
    print(f"wrote {mixed_path.relative_to(REPO_ROOT)}")

    if "none" in set(table.scope) and len(pairs) == 1 and pairs[0][0] == "none":
        print("\nchange from leaving the cache alone:")
        for level, group in table.groupby("level"):
            base = group[group.scope == "none"].iloc[0]
            for row in group.itertuples():
                if row.scope == "none":
                    continue
                print(f"  {level:9s} {row.scope:9s} "
                      f"corpus {row.corpus - base.corpus:+.3f}   "
                      f"PD/HC {row.pd_hc - base.pd_hc:+.3f}")

    if lodo_results:
        lodo = pd.DataFrame(lodo_results)
        print("\n=== LODO-style PD/HC ===")
        print(lodo.round(3).to_string(index=False))
        lodo_path = out_dir / f"probe_lodo{args.tag}.csv"
        lodo.to_csv(lodo_path, index=False)
        print(f"wrote {lodo_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
