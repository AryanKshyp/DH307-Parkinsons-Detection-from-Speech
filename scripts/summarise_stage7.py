"""Summarise Stage 7 experiment directories that already have predictions."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "experiments"


def pool_mixed(pattern, title):
    rows = []
    print(f"=== {title} ===")
    for i in range(10):
        folder = next(ROOT.glob(pattern.format(i=i)), None)
        if folder is None or not (folder / "predictions.csv").exists():
            print(f"  fold {i}: missing")
            continue
        pred = pd.read_csv(folder / "predictions.csv")
        test = pred[pred.role == "test"]
        y = (test.label == "PD").astype(int)
        pp = (test.pred == "PD").astype(int)
        ba = balanced_accuracy_score(y, pp)
        by = {}
        for dataset, group in test.groupby("dataset"):
            yy = (group.label == "PD").astype(int)
            ppp = (group.pred == "PD").astype(int)
            if len(group) >= 2 and yy.nunique() == 2:
                by[dataset] = round(float(balanced_accuracy_score(yy, ppp)), 3)
        print(f"  fold {i}: test BA {ba:.3f}  n={len(test)}  {by}")
        rows.append(test)
    if not rows:
        return
    all_ = pd.concat(rows, ignore_index=True)
    y = (all_.label == "PD").astype(int)
    pred = (all_.pred == "PD").astype(int)
    print(f"pooled OOF: {len(all_)} speakers  "
          f"BA {balanced_accuracy_score(y, pred):.3f}  "
          f"AUC {roc_auc_score(y, all_.prob_pd):.3f}")
    for dataset, group in all_.groupby("dataset"):
        yy = (group.label == "PD").astype(int)
        pp = (group.pred == "PD").astype(int)
        auc = roc_auc_score(yy, group.prob_pd) if yy.nunique() == 2 else float("nan")
        print(f"  {dataset:12s} n={len(group):3d}  "
              f"BA {balanced_accuracy_score(yy, pp):.3f}  "
              f"AUC {auc:.3f}")
    print()


def lodo(pattern, title):
    print(f"=== {title} ===")
    for corpus in ("ItalianPD", "MDVR-KCL", "NeuroVoz"):
        folder = next(ROOT.glob(pattern.format(c=corpus)), None)
        if folder is None or not (folder / "predictions.csv").exists():
            print(f"  hold {corpus}: missing")
            continue
        pred = pd.read_csv(folder / "predictions.csv")
        test = pred[pred.role == "test"]
        y = (test.label == "PD").astype(int)
        pp = (test.pred == "PD").astype(int)
        ba = balanced_accuracy_score(y, pp)
        auc = roc_auc_score(y, test.prob_pd) if y.nunique() == 2 else float("nan")
        print(f"  hold {corpus:12s}  n={len(test):3d}  BA {ba:.3f}  AUC {auc:.3f}")
    print()


if __name__ == "__main__":
    pool_mixed("mixed_{i}_handcrafted_duration_gbt_norm-global_s0",
               "GBT mixed duration")
    lodo("lodo_{c}_handcrafted_duration_gbt_norm-global_s0",
         "GBT LODO duration")
    pool_mixed("mixed_{i}_handcrafted_duration_baseline_s0",
               "MLP baseline mixed duration")
    pool_mixed("mixed_{i}_handcrafted_duration_dann_s0",
               "MLP DANN mixed duration")
    lodo("lodo_{c}_handcrafted_duration_baseline_s0",
         "MLP baseline LODO duration")
    lodo("lodo_{c}_handcrafted_duration_dann_s0",
         "MLP DANN LODO duration")
    pool_mixed("mixed_{i}_mel_duration_baseline_s0",
               "CNN baseline mixed duration")
    pool_mixed("mixed_{i}_mel_duration_dann_s0",
               "CNN DANN mixed duration")
    lodo("lodo_{c}_mel_duration_baseline_s0",
         "CNN baseline LODO duration")
    lodo("lodo_{c}_mel_duration_dann_s0",
         "CNN DANN LODO duration")
