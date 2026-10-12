"""Stage 8 scoring of finished Stage 7 runs.

    python scripts/evaluate.py
    python scripts/evaluate.py --family handcrafted --level duration

Reads speaker-level predictions and embeddings already written under
experiments/. Does not load the feature cache or train anything. Mixed metrics
are pooled out-of-fold; LODO is one number per held-out corpus, never averaged.

Writes tables only. The embedding t-SNE is `scripts/plot_stage8.py`'s figure,
which is run straight after this one and drew the same panel at a higher dpi
into the same file.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import geometry as geo                      # noqa: E402
from src.evaluation import pool                                 # noqa: E402
from src.evaluation import tables                               # noqa: E402
from src.training.config import TrainConfig                     # noqa: E402

REPORTS = REPO_ROOT / "reports"


def _print(title, frame):
    print(f"\n=== {title} ===")
    print(frame.round(3).to_string(index=False))
    return frame


def _write(frames, dest):
    pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
    print(f"wrote {dest.relative_to(REPO_ROOT)}")


def score_mixed(stem_name, regimes):
    clf_rows, fold_rows, kl_rows, tcm_rows, trace_rows = [], [], [], [], []
    for regime, cfg in regimes.items():
        pred, emb = pool.mixed_oof(cfg)
        if pred is None:
            print(f"{regime}: no mixed predictions")
            continue
        clf = tables.by_corpus(pred)
        clf.insert(0, "regime", regime)
        clf_rows.append(_print(f"mixed {stem_name} {regime} (pooled OOF)", clf))
        folds = tables.per_fold(pred)
        folds.insert(0, "regime", regime)
        fold_rows.append(folds)
        _print(f"per-fold {regime}", tables.mean_std(folds))
        elderly = tables.elderly_only(pred)
        if elderly is not None:
            extra = tables.by_corpus(elderly)
            extra.insert(0, "regime", regime + "/elderly")
            clf_rows.append(_print(f"elderly-only {regime}", extra))
        if emb is not None:
            kl = geo.pairwise_kl(emb)
            kl.insert(0, "regime", regime)
            kl_rows.append(_print(f"KL {regime}", kl))
            tcm = geo.tcm(emb)
            tcm.insert(0, "regime", regime)
            tcm_rows.append(_print(f"TCM {regime}", tcm))
        trace = pool.selection_trace(cfg)
        if trace is not None and len(trace):
            trace.insert(0, "regime", regime)
            trace_rows.append(_print(f"checkpoint {regime}", trace))
    stem = REPORTS / f"stage8_{stem_name}"
    if clf_rows:
        _write(clf_rows, Path(str(stem) + "_classification.csv"))
    if fold_rows:
        _write(fold_rows, Path(str(stem) + "_folds.csv"))
    if kl_rows:
        _write(kl_rows, Path(str(stem) + "_kl.csv"))
    if tcm_rows:
        _write(tcm_rows, Path(str(stem) + "_tcm.csv"))
    if trace_rows:
        _write(trace_rows, Path(str(stem) + "_checkpoints.csv"))


def score_lodo(stem_name, regimes):
    """LODO scored under both decision rules, with AUC as the headline.

    `balanced_acc` is the inductive number at a fixed 0.5 threshold.
    `balanced_acc_prior` re-thresholds to the corpus design balance and is
    **transductive**. `auc` is threshold-free and so is unaffected by the
    prior shift that breaks the fixed rule -- it is the fair headline for a
    cross-corpus claim.
    """
    rows = []
    for regime, cfg in regimes.items():
        for corpus in ("ItalianPD", "MDVR-KCL", "NeuroVoz"):
            pred = pool.lodo_test(cfg, corpus)
            if pred is None:
                print(f"{regime} LODO {corpus}: missing")
                continue
            clf = tables.by_corpus(pred)
            adj = tables.by_corpus(pred, rule="prior").set_index("dataset")
            for column in ("accuracy", "balanced_acc", "precision", "recall"):
                clf[f"{column}_prior"] = clf.dataset.map(adj[column])
            clf["pred_pd_rate"] = clf.dataset.map(
                lambda d: float(((pred if d == "all" else pred[pred.dataset == d])
                                 .pred == "PD").mean()))
            clf["true_pd_rate"] = clf.dataset.map(
                lambda d: float(((pred if d == "all" else pred[pred.dataset == d])
                                 .label == "PD").mean()))
            hold = clf[clf.dataset.isin(["all", corpus])].copy()
            hold.insert(0, "held_out", corpus)
            hold.insert(0, "regime", regime)
            rows.append(hold)
    if rows:
        frame = pd.concat(rows, ignore_index=True)
        head = frame[frame.dataset == frame.held_out][
            ["regime", "held_out", "n", "auc", "balanced_acc",
             "balanced_acc_prior", "pred_pd_rate", "true_pd_rate"]]
        _print(f"LODO {stem_name} -- AUC is the headline; "
               f"_prior is transductive", head)
        dest = REPORTS / f"stage8_{stem_name}_lodo.csv"
        frame.to_csv(dest, index=False)
        print(f"wrote {dest.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="handcrafted")
    ap.add_argument("--level", default="duration")
    ap.add_argument("--phone-class",
                    help="phoneme runs are trained within one class and "
                         "carry it in the run id")
    args = ap.parse_args()
    pc = f"_{args.phone_class}" if args.phone_class else ""
    stem = f"{args.family}_{args.level}{pc}"
    order = ["gbt", "baseline", "dann"]
    if args.family == "mel":
        order.remove("gbt")   # GBT is handcrafted-only: no tree baseline on Mel
    base = TrainConfig(family=args.family, level=args.level,
                       phone_class=args.phone_class)
    mixed = {r: replace(base, protocol="mixed", regime=r) for r in order}
    lodo = {r: replace(base, protocol="lodo", regime=r) for r in order}
    score_mixed(stem, mixed)
    if args.phone_class:
        # MDVR-KCL has no phoneme segments, so holding it out is a no-op and
        # holding out NeuroVoz leaves one training domain. See PIPELINE Stage 6.
        print("phoneme is mixed-protocol only; skipping LODO")
    else:
        score_lodo(stem, lodo)


if __name__ == "__main__":
    main()
