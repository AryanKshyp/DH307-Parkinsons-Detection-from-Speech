"""Speaker-level metrics.

The label is a speaker label. Scoring at segment level inflates every number
because a speaker contributes hundreds of highly correlated windows. Pooling
is mean-of-probabilities, then argmax; that is Ibarra's joint-probability
aggregation in linear space, and is more stable than majority vote.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


LABEL_TO_ID = {"HC": 0, "PD": 1}
ID_TO_LABEL = {0: "HC", 1: "PD"}


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pool_speakers(logits, y, speakers, datasets=None, cohorts=None,
                  domain_logits=None, domain_y=None):
    """Collapse item-level outputs to one row per speaker.

    `logits` is (N, 2). Speakers are an array of ids, not necessarily sorted.
    Returns a dict of arrays aligned on the unique speaker order of first
    appearance, plus scalar metrics.
    """
    speakers = np.asarray(speakers)
    y = np.asarray(y)
    probs = softmax_np(np.asarray(logits))
    order, first = np.unique(speakers, return_index=True)
    # unique sorts; restore first-seen order so a later join on input order
    # is not required. Sort by first index.
    ranked = first.argsort()
    names = speakers[first[ranked]]

    rows = []
    for name in names:
        at = speakers == name
        p = probs[at].mean(axis=0)
        rows.append({
            "speaker_id": name,
            "y": int(y[at][0]),
            "prob_pd": float(p[1]),
            "pred": int(p.argmax()),
            "n_items": int(at.sum()),
            "dataset": (None if datasets is None
                        else np.asarray(datasets)[at][0]),
            "cohort": (None if cohorts is None
                       else np.asarray(cohorts)[at][0]),
        })

    truth = np.array([r["y"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    scores = np.array([r["prob_pd"] for r in rows])
    out = {
        "rows": rows,
        "balanced_acc": float(balanced_accuracy_score(truth, pred)),
        "n_speakers": len(rows),
        "by_dataset": {},
    }
    if len(np.unique(truth)) == 2:
        out["auc"] = float(roc_auc_score(truth, scores))
    else:
        out["auc"] = float("nan")

    if datasets is not None:
        names = np.array([r["dataset"] for r in rows])
        for name in sorted(set(names)):
            at = names == name
            if at.sum() < 2 or len(np.unique(truth[at])) < 2:
                continue
            out["by_dataset"][str(name)] = float(
                balanced_accuracy_score(truth[at], pred[at]))

    if domain_logits is not None and domain_y is not None:
        d_prob = softmax_np(np.asarray(domain_logits))
        d_y = np.asarray(domain_y)
        known = d_y >= 0
        if known.any():
            out["domain_acc"] = float(
                (d_prob[known].argmax(axis=1) == d_y[known]).mean())
        else:
            out["domain_acc"] = float("nan")
    return out


def inverse_weights(y, n_classes, ignore_index=-1):
    """Class weights that sum to n_classes, ignoring `ignore_index`."""
    y = np.asarray(y)
    y = y[y != ignore_index]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return counts.sum() / (n_classes * counts)
