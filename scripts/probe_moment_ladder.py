"""How much of the corpus cue is reachable from moments alone?

    python scripts/probe_moment_ladder.py --level duration --bootstrap 1000

`reports/ablation_corpus_norm.md` left an open question: per-corpus z-scoring
equalises each corpus's means *and* marginal standard deviations, yet corpus
identity gets *easier* (0.757 -> 0.991). What is left must be higher-order --
but "higher-order" covers two very different things, and only one of them is
removable by a moment-matching method like CORAL:

  * the **correlation** matrices, which survive per-corpus z-scoring exactly
    (correlation is invariant to a per-column positive-scale affine map), and
  * everything above second order -- skew, kurtosis, discrete task clusters,
    non-linear channel structure -- which no covariance matching can touch.

This script apportions them without writing CORAL, by replacing the real data
with a Gaussian surrogate that contains *only* the moments of a chosen rung and
seeing how much of the corpus cue survives:

    L0   per-corpus mean, pooled covariance          first order only
    L1   per-corpus mean, diagonal per-corpus cov    + marginal variance
                                                     (what norm=corpus removes)
    L2   per-corpus mean, full per-corpus cov        + correlations
                                                     (what CORAL removes)
    L2z  zero mean, unit variance, per-corpus corr   correlations alone,
                                                     in the norm=corpus space
    L3   the real data                               all orders (the reference)

Per speaker-grouped fold, the moments are fitted on **training speakers only**,
the surrogate is sampled to match the training per-corpus row counts, the tree
is trained on the surrogate, and it is tested on the **real held-out fold**. Test
sets are therefore identical to `probe_normalisation.py`'s, so every number here
lands in the same table as 0.757 and 0.991.

A surrogate-trained/surrogate-tested score is also reported. It is the cleaner
upper bound on moment *content* -- no Gaussian-vs-real mismatch to pay for -- and
it breaks the tie when the real-test score is low: a low real-test score with a
high surrogate-test score means Gaussian mismatch, not missing moment content.

Reads as (R1 in experiments/corpus_norm_ablation/PLAN_second_order.md):

  L2 - L1 <= 0.03            correlations add nothing beyond what norm=corpus
                             already equalised. CORAL is dead; skip it.
  L2 covers >= 0.70 of the   moments carry most of the cue. CORAL authorised.
    chance-to-L3 span
  L2 low on real, high on    Gaussian mismatch. Inconclusive; test CORAL directly.
    surrogate
  L2 low in both             the cue is higher-order or discrete. Skip CORAL.

Writes `experiments/corpus_norm_ablation/moment_ladder.csv`.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import store as fs                        # noqa: E402
from probe_normalisation import bootstrap_ci                # noqa: E402

MAX_ROWS = 40_000
SEED = 0
RIDGE = 1e-6
RUNGS = ("L0", "L1", "L2", "L2z", "L3")
# L2z lives in the norm=corpus space; every other rung lives in the global one.
SCOPE_FOR = {"L0": "global", "L1": "global", "L2": "global",
             "L2z": "corpus", "L3": "global"}


def moments(values, meta, rung):
    """Per-corpus (mean, covariance) containing only this rung's information."""
    names = np.asarray(meta.dataset.values)
    pooled_cov = np.cov(values, rowvar=False)
    out = {}
    for name in sorted(set(names)):
        rows = np.flatnonzero(names == name)
        mean = values[rows].mean(axis=0)
        cov = np.cov(values[rows], rowvar=False)
        if rung == "L0":
            out[name] = (mean, pooled_cov, len(rows))
        elif rung == "L1":
            out[name] = (mean, np.diag(np.diag(cov)), len(rows))
        elif rung == "L2":
            out[name] = (mean, cov, len(rows))
        elif rung == "L2z":
            # In the norm=corpus space each corpus already has mean 0 and unit
            # marginal variance on the fitted rows, so its covariance *is* its
            # correlation matrix. Forcing the mean to zero makes that explicit:
            # nothing but correlation structure can carry the label.
            scale = np.sqrt(np.maximum(np.diag(cov), RIDGE))
            out[name] = (np.zeros_like(mean),
                         cov / np.outer(scale, scale), len(rows))
        else:
            raise ValueError(f"rung {rung!r} has no surrogate")
    return out


def sample(spec, rng, scale=1.0):
    """Draw a surrogate set from per-corpus moments."""
    parts, labels = [], []
    for name, (mean, cov, count) in spec.items():
        n = max(int(round(count * scale)), len(mean) + 1)
        cov = cov + RIDGE * np.eye(len(mean))
        parts.append(rng.multivariate_normal(mean, cov, size=n,
                                             method="cholesky"))
        labels.append(np.repeat(name, n))
    return np.vstack(parts), np.concatenate(labels)


def check_sampled(spec, drawn, labels, slack=3.0):
    """Assert the surrogate really carries the moments it was given.

    The tolerance is not a magic constant. An empirical covariance from `n`
    draws in `d` dimensions sits about `sqrt(d / n)` away from the truth in
    relative Frobenius norm, so the check is scaled by that and given `slack`
    headroom. A miswired covariance or a mislabelled corpus is an O(1) error and
    still trips it; ordinary sampling noise, which shrinks as the sample grows,
    does not.
    """
    worst_mean = worst_cov = 0.0
    n_dim = drawn.shape[1]
    smallest = min(len(np.flatnonzero(labels == name)) for name in spec)
    tolerance = slack * np.sqrt(n_dim / smallest)
    for name, (mean, cov, _) in spec.items():
        rows = np.flatnonzero(labels == name)
        got_mean = drawn[rows].mean(axis=0)
        got_cov = np.cov(drawn[rows], rowvar=False)
        span = max(np.linalg.norm(mean), 1.0)
        worst_mean = max(worst_mean, np.linalg.norm(got_mean - mean) / span)
        worst_cov = max(worst_cov, np.linalg.norm(got_cov - cov)
                        / max(np.linalg.norm(cov), 1e-12))
    if worst_mean > tolerance or worst_cov > tolerance:
        raise ValueError(
            f"surrogate moments drifted from the fitted ones "
            f"(mean {worst_mean:.3f}, cov {worst_cov:.3f} > {tolerance:.3f} "
            f"for d={n_dim}, smallest corpus n={smallest})")
    return worst_mean, worst_cov


def run_rung(values, meta, rung, bootstrap):
    """Score one rung: train on its surrogate, test on real held-out rows."""
    scope = SCOPE_FOR[rung]
    y = np.asarray(meta.dataset.values)
    groups = np.asarray(meta.speaker_id.values)
    folds = GroupKFold(n_splits=5)
    rng = np.random.default_rng(SEED)

    real_truth, real_pred, real_speakers, real_folds = [], [], [], []
    surrogate_folds, drift = [], []
    for train, test in folds.split(values, y, groups):
        assert not (set(groups[train]) & set(groups[test])), "speaker leak"
        stats = fs.fit_stats(values[train], meta.iloc[train], scope)
        x_train = fs.apply(values[train], meta.iloc[train], scope, stats)
        x_test = fs.apply(values[test], meta.iloc[test], scope, stats)

        held_x = held_y = None
        if rung == "L3":
            fit_x, fit_y = x_train, y[train]
        else:
            spec = moments(x_train, meta.iloc[train], rung)
            fit_x, fit_y = sample(spec, rng)
            drift.append(check_sampled(spec, fit_x, fit_y))
            # Upper bound on moment content: no Gaussian-vs-real mismatch.
            held_x, held_y = sample(spec, rng, scale=len(test) / len(train))

        model = HistGradientBoostingClassifier(
            max_iter=120, random_state=SEED).fit(fit_x, fit_y)
        surrogate_folds.append(
            np.nan if held_x is None
            else balanced_accuracy_score(held_y, model.predict(held_x)))
        prediction = model.predict(x_test)
        real_folds.append(balanced_accuracy_score(y[test], prediction))
        real_truth.append(y[test])
        real_pred.append(prediction)
        real_speakers.append(groups[test])

    real_truth = np.concatenate(real_truth)
    real_pred = np.concatenate(real_pred)
    real_speakers = np.concatenate(real_speakers)
    out = {
        "rung": rung, "scope": scope,
        "real_test": balanced_accuracy_score(real_truth, real_pred),
        "surrogate_test": float(np.nanmean(surrogate_folds)),
        "per_fold": real_folds,
    }
    if drift:
        out["worst_mean_drift"] = max(d[0] for d in drift)
        out["worst_cov_drift"] = max(d[1] for d in drift)
    if bootstrap:
        out["ci_lo"], out["ci_hi"] = bootstrap_ci(
            real_truth, real_pred, real_speakers, bootstrap)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="duration")
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    values, meta = fs.load_handcrafted(args.level)
    rng = np.random.default_rng(SEED)
    if len(meta) > args.max_rows:
        pick = np.sort(rng.choice(len(meta), args.max_rows, replace=False))
        values, meta = values[pick], meta.iloc[pick].reset_index(drop=True)
    chance = 1 / meta.dataset.nunique()
    print(f"{args.level}: {len(meta)} segments, "
          f"{meta.speaker_id.nunique()} speakers, {values.shape[1]} features, "
          f"corpus chance {chance:.3f}")

    rows = []
    for rung in args.rungs:
        print(f"\n--- {rung} ({SCOPE_FOR[rung]} space) ---")
        result = run_rung(values, meta, rung, args.bootstrap)
        line = (f"  corpus BA on real held-out rows {result['real_test']:.3f}")
        if args.bootstrap:
            line += f" [{result['ci_lo']:.3f},{result['ci_hi']:.3f}]"
        print(line)
        if not np.isnan(result["surrogate_test"]):
            print(f"  corpus BA on surrogate held-out rows "
                  f"{result['surrogate_test']:.3f}  (moment-content bound)")
            print(f"  worst moment drift: mean "
                  f"{result['worst_mean_drift']:.3f}, cov "
                  f"{result['worst_cov_drift']:.3f}")
        print("  per-fold " + " ".join(f"{v:.3f}" for v in result["per_fold"]))
        flat = {k: v for k, v in result.items() if k != "per_fold"}
        flat.update({f"fold{i}": v for i, v in enumerate(result["per_fold"])})
        flat["level"] = args.level
        flat["corpus_chance"] = chance
        rows.append(flat)

    table = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "experiments" / "corpus_norm_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"moment_ladder{args.tag}.csv"
    table.to_csv(path, index=False)

    print("\n=== moment ladder ===")
    show = [c for c in ("rung", "scope", "real_test", "surrogate_test",
                        "ci_lo", "ci_hi") if c in table.columns]
    print(table[show].round(3).to_string(index=False))

    by = table.set_index("rung")
    if "L3" in by.index:
        span = by.loc["L3", "real_test"] - chance
        print(f"\nshare of the real corpus cue (L3 = "
              f"{by.loc['L3', 'real_test']:.3f}) reachable from each rung:")
        for rung in ("L0", "L1", "L2"):
            if rung in by.index and span > 0:
                got = (by.loc[rung, "real_test"] - chance) / span
                print(f"  {rung}  {got:+.2f}")
    if "L1" in by.index and "L2" in by.index:
        delta = by.loc["L2", "real_test"] - by.loc["L1", "real_test"]
        print(f"\nL2 - L1 = {delta:+.3f}  "
              f"({'CORAL has no target; skip R4' if delta <= 0.03 else 'correlations carry something; R4 authorised'})")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
