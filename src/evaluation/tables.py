"""Classification tables from speaker-level predictions.

Two decision rules are available. `fixed` thresholds P(PD) at 0.5 and is
the inductive default. `prior` sets the threshold so the predicted PD rate
matches the corpus design balance; it reads the held-out corpus's
predictions, so any score derived from it is **transductive** -- target
features are seen, target labels never are. It exists because under LODO a
0.5 threshold calls 83-96% of the held-out corpus PD when the true rate is
43-49% (`scripts/probe_failure_modes.py`).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score)


def _scores(y, pred, prob):
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    else:
        out["auc"] = float("nan")
    return out


def prior_matched(prob, prior=0.5):
    """Predictions whose positive rate matches `prior`.

    Transductive: the threshold is a quantile of the target corpus's own
    predictions. No target label is read. Ties can leave the realised rate a
    little off `prior`, which is harmless for a speaker-level table.
    """
    prob = np.asarray(prob, dtype=float)
    if prob.size == 0:
        return np.zeros(0, dtype=bool)
    return prob >= float(np.quantile(prob, 1.0 - prior))


def _decide(group, rule, prior):
    if rule == "fixed":
        return (group.pred == "PD").to_numpy()
    if rule == "prior":
        return prior_matched(group.prob_pd.to_numpy(), prior)
    raise ValueError(f"unknown rule {rule!r}")


def by_corpus(pred, rule="fixed", prior=0.5):
    """One row per corpus plus `all`. `rule` selects the decision threshold.

    The threshold is fitted per corpus, which is what makes it meaningful
    under LODO -- there `all` and the held-out corpus are the same rows.
    """
    rows = [{"dataset": "all", **_scores(
        (pred.label == "PD").to_numpy(),
        _decide(pred, rule, prior),
        pred.prob_pd.to_numpy())}]
    for dataset, group in pred.groupby("dataset"):
        rows.append({"dataset": dataset, **_scores(
            (group.label == "PD").to_numpy(),
            _decide(group, rule, prior),
            group.prob_pd.to_numpy())})
    return pd.DataFrame(rows)


def elderly_only(pred):
    """Drop ItalianPD young_control if that cohort is present."""
    if "cohort" not in pred.columns:
        return None
    if not (pred.cohort == "young_control").any():
        return None
    return pred[pred.cohort != "young_control"]


def per_fold(pred):
    """One row per mixed configuration. `fold` must already be on the frame."""
    rows = []
    for fold, group in pred.groupby("fold"):
        y = (group.label == "PD").to_numpy()
        hat = (group.pred == "PD").to_numpy()
        rows.append({"fold": int(fold), **_scores(y, hat, group.prob_pd.to_numpy())})
    return pd.DataFrame(rows)


def mean_std(folds, columns=("balanced_acc", "auc")):
    rows = []
    for col in columns:
        rows.append({
            "metric": col,
            "mean": float(folds[col].mean()),
            "std": float(folds[col].std(ddof=1)) if len(folds) > 1 else float("nan"),
            "n_folds": int(len(folds)),
        })
    return pd.DataFrame(rows)
