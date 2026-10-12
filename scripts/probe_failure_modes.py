"""Why the LODO numbers are low, measured rather than argued.

    python scripts/probe_failure_modes.py
    python scripts/probe_failure_modes.py --check threshold

Reads finished Stage 7 runs only. Trains nothing, writes no checkpoints.

Three independent failure modes are separated here, because they call for
different fixes and only one of them is about the representation:

  threshold  The decision rule, not the features. Under LODO the model calls
             83-96% of the held-out corpus PD when the true rate is 43-49%.
             Balanced accuracy at a fixed 0.5 therefore reports far less than
             the ranking is worth.

  norm       Gradient reversal has no constraint on the scale of the trunk
             output, and the cheapest way to confuse a domain head is to grow
             it. The measured inflation is 3.8x (MLP) and 14.9x (CNN), which
             is also the order in which DANN damaged the two backbones. It is
             the whole of the intra-class TCM increase reported in Stage 8:
             after L2 normalisation the same quantity *falls* under DANN.

  shortcut   How much of a corpus's PD/HC score is available from recording
             length alone, with no speech content involved.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import geometry as geo                      # noqa: E402

EXPERIMENTS = REPO_ROOT / "experiments"
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
CORPORA = ("ItalianPD", "MDVR-KCL", "NeuroVoz")
FAMILIES = ("handcrafted", "mel")
REGIMES = ("baseline", "dann")


def _best_threshold(y, p):
    grid = np.unique(np.round(p, 4))
    scores = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def check_threshold(level="duration"):
    """Balanced accuracy under four decision rules, three of them label-free."""
    print("=== Decision threshold under LODO ===")
    print("  @0.5    what Stage 8 reports")
    print("  @val    threshold tuned on the source-domain validation speakers")
    print("          (uses no target data at all)")
    print("  @prior  threshold set so the predicted PD rate matches the ~50%")
    print("          design balance (uses unlabelled target -> transductive)")
    print("  @oracle threshold tuned on the target labels: a ceiling, not a method")
    print()
    print(f"{'family':<12}{'held_out':<11}{'regime':<9}"
          f"{'@0.5':>7}{'@val':>7}{'@prior':>8}{'@oracle':>9}{'predPD%':>9}")
    rows = []
    for family in FAMILIES:
        for corpus in CORPORA:
            for regime in REGIMES:
                run = EXPERIMENTS / f"lodo_{corpus}_{family}_{level}_{regime}_s0"
                path = run / "predictions.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                val, test = frame[frame.role == "val"], frame[frame.role == "test"]
                if val.empty or test.empty:
                    continue
                y_val = (val.label == "PD").astype(int).to_numpy()
                p_val = val.prob_pd.to_numpy()
                y, p = (test.label == "PD").astype(int).to_numpy(), test.prob_pd.to_numpy()
                if len(np.unique(y_val)) < 2 or len(np.unique(y)) < 2:
                    continue
                at = {
                    "0.5": balanced_accuracy_score(y, (p >= 0.5).astype(int)),
                    "val": balanced_accuracy_score(
                        y, (p >= _best_threshold(y_val, p_val)).astype(int)),
                    "prior": balanced_accuracy_score(
                        y, (p >= np.median(p)).astype(int)),
                    "oracle": balanced_accuracy_score(
                        y, (p >= _best_threshold(y, p)).astype(int)),
                }
                rows.append(at)
                print(f"{family:<12}{corpus:<11}{regime:<9}"
                      f"{at['0.5']:>7.3f}{at['val']:>7.3f}{at['prior']:>8.3f}"
                      f"{at['oracle']:>9.3f}{100 * (p >= 0.5).mean():>8.0f}%")
    if rows:
        mean = pd.DataFrame(rows).mean()
        print()
        print(f"{'MEAN over ' + str(len(rows)) + ' configs':<32}"
              f"{mean['0.5']:>7.3f}{mean['val']:>7.3f}{mean['prior']:>8.3f}"
              f"{mean['oracle']:>9.3f}")
        print()
        print("  @val below @0.5 is the point: a threshold that is correct on the")
        print("  source corpora is wrong on the target, which is prior shift and")
        print("  not a badly chosen constant.")


def _pooled_test_embeddings(family, level, regime):
    frames = []
    for i in range(10):
        path = EXPERIMENTS / f"mixed_{i}_{family}_{level}_{regime}_s0" / "embeddings.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame[frame.role == "test"])
    return pd.concat(frames, ignore_index=True) if frames else None


def check_norm(level="duration"):
    """Is the DANN TCM increase geometry, or just scale?"""
    print("=== Embedding scale under gradient reversal ===")
    print(f"{'family':<12}{'regime':<10}{'mean|z|':>9}{'vs baseline':>13}")
    tcm = {}
    for family in FAMILIES:
        base = None
        for regime in REGIMES:
            emb = _pooled_test_embeddings(family, level, regime)
            if emb is None:
                continue
            cols = geo.embedding_cols(emb)
            z = emb[cols].to_numpy(dtype=np.float64)
            norms = np.linalg.norm(z, axis=1)
            base = norms.mean() if regime == "baseline" else base
            print(f"{family:<12}{regime:<10}{norms.mean():>9.2f}"
                  f"{norms.mean() / base:>12.1f}x")
            unit = emb.copy()
            unit[cols] = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-9, None)
            raw = geo.tcm(emb)
            normed = geo.tcm(unit)
            tcm[(family, regime)] = (
                raw[raw.dataset == "all"].set_index("label").tcm,
                normed[normed.dataset == "all"].set_index("label").tcm)
    print()
    print("=== Intra-class TCM, before and after removing scale ===")
    print(f"{'family':<12}{'class':<7}{'raw base':>10}{'raw dann':>10}{'raw x':>7}"
          f"{'L2 base':>10}{'L2 dann':>10}{'L2 x':>7}")
    for family in FAMILIES:
        if (family, "baseline") not in tcm or (family, "dann") not in tcm:
            continue
        (rb, nb), (rd, nd) = tcm[(family, "baseline")], tcm[(family, "dann")]
        for label in ("HC", "PD"):
            print(f"{family:<12}{label:<7}{rb[label]:>10.1f}{rd[label]:>10.1f}"
                  f"{rd[label] / rb[label]:>6.0f}x{nb[label]:>10.3f}{nd[label]:>10.3f}"
                  f"{nd[label] / nb[label]:>6.1f}x")
    print()
    print("  Raw TCM rises and unit-norm TCM falls. The Stage 8 'DANN smears the")
    print("  classes' reading is an artefact of unconstrained embedding scale:")
    print("  in direction, the classes do tighten, which is Ibarra's Table 6.")


def check_shortcut():
    """How much PD/HC is available from recording length alone."""
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}")
        return
    frame = pd.read_csv(MANIFEST)
    speaker = frame.groupby(["dataset", "speaker_id", "label"], as_index=False).agg(
        seconds=("duration", "sum"), n_files=("duration", "size"))
    print("=== PD/HC available from recording length, no speech content ===")
    print(f"{'corpus':<12}{'n':>5}{'AUC(seconds)':>14}{'AUC(n_files)':>14}"
          f"{'PD/HC median s':>17}")
    for corpus, group in speaker.groupby("dataset"):
        y = (group.label == "PD").astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc_s = roc_auc_score(y, group.seconds.to_numpy())
        auc_n = roc_auc_score(y, group.n_files.to_numpy())
        med = group.groupby("label").seconds.median()
        ratio = med.get("PD", np.nan) / med.get("HC", np.nan)
        print(f"{corpus:<12}{len(group):>5}{max(auc_s, 1 - auc_s):>14.3f}"
              f"{max(auc_n, 1 - auc_n):>14.3f}{ratio:>17.2f}")
    print()
    print("  A corpus near 0.9 is one where a model needs no speech content to")
    print("  score well, so its per-corpus number is not evidence about the")
    print("  representation. Note the ratio flips sign across corpora.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+",
                    choices=["threshold", "norm", "shortcut"],
                    default=["threshold", "norm", "shortcut"])
    ap.add_argument("--level", default="duration")
    args = ap.parse_args()
    for i, name in enumerate(args.check):
        if i:
            print("\n" + "-" * 78 + "\n")
        if name == "threshold":
            check_threshold(args.level)
        elif name == "norm":
            check_norm(args.level)
        else:
            check_shortcut()


if __name__ == "__main__":
    main()
