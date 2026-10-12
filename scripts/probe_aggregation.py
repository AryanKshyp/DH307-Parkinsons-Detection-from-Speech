"""Compare speaker-aggregation rules on finished Stage 7 runs.

    python scripts/probe_aggregation.py --family handcrafted --level duration
    python scripts/probe_aggregation.py --family mel --level duration --regime dann

The assigned pipeline asks for aggregation to be treated as a design axis, not
a default: "Mean pooling, attention pooling, or majority vote." The spine uses
mean-of-probabilities then argmax (`src/training/metrics.py`), which is
Ibarra's joint-probability aggregation in linear space. This script measures
the alternatives instead of asserting that choice.

It is inference-only. Segment-level probabilities are not saved by Stage 7 --
`predictions.csv` is already pooled -- so the probe reloads `best.pt` and
re-runs the test split of every mixed configuration. No model is retrained, and
no checkpoint is rewritten. Runs on CPU by default so it does not contend with
a training job for the GPU.

Speakers are pooled out-of-fold across all ten mixed configurations, so each
speaker is scored exactly once, by the model that did not train on them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbones import build_trunk                    # noqa: E402
from src.models.classifier import Classifier                    # noqa: E402
from src.training.data import Bundle                            # noqa: E402
from src.training.loop import _run_loader                       # noqa: E402
from src.training.metrics import softmax_np                     # noqa: E402

EXPERIMENTS = REPO_ROOT / "experiments"
REPORTS = REPO_ROOT / "reports"


# --- aggregation rules -------------------------------------------------
# Each takes the (n_items, 2) softmax probabilities of one speaker's segments
# and returns that speaker's P(PD). The decision is score >= 0.5 throughout,
# so the rules differ only in how segments are combined, never in the
# threshold applied afterwards.

def agg_mean_prob(probs):
    """Current spine: arithmetic mean of probabilities."""
    return float(probs[:, 1].mean())


def agg_mean_logit(probs):
    """Geometric mean -- joint probability in log space, renormalised."""
    logs = np.log(np.clip(probs, 1e-12, 1.0)).mean(axis=0)
    e = np.exp(logs - logs.max())
    return float((e / e.sum())[1])


def agg_median_prob(probs):
    """Median: one badly-scored segment cannot drag the speaker."""
    return float(np.median(probs[:, 1]))


def agg_trimmed_mean(probs, trim=0.1):
    """Mean of the middle 80% of segment probabilities."""
    p = np.sort(probs[:, 1])
    k = int(len(p) * trim)
    core = p[k:len(p) - k] if len(p) - 2 * k > 0 else p
    return float(core.mean())


def agg_majority_vote(probs):
    """Fraction of segments that individually call PD."""
    return float((probs[:, 1] >= 0.5).mean())


def agg_confident_mean(probs, keep=0.25):
    """Mean over the segments furthest from the decision boundary.

    Most windows of connected speech carry no pathology signal at all; this
    asks whether restricting to the confident ones helps or just amplifies
    whatever the model is most wrong about.
    """
    conf = np.abs(probs[:, 1] - 0.5)
    k = max(1, int(len(probs) * keep))
    at = np.argsort(-conf)[:k]
    return float(probs[at, 1].mean())


RULES = {
    "mean_prob": agg_mean_prob,
    "mean_logit": agg_mean_logit,
    "median_prob": agg_median_prob,
    "trimmed_mean": agg_trimmed_mean,
    "majority_vote": agg_majority_vote,
    "confident_mean": agg_confident_mean,
}


def infer_fold(run_dir, device, batch_size=256):
    """Segment-level probabilities on one run's test split."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    bundle = Bundle(
        family=cfg["family"],
        level=cfg["level"],
        protocol=cfg["protocol"],
        config=cfg["split"],
        scope=cfg.get("norm", "global"),
        phone_class=cfg.get("phone_class"),
        dataset=cfg.get("dataset"),
        tasks=cfg.get("tasks"),
        seed=int(cfg.get("seed", 0)),
    )
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device,
                      weights_only=False)
    trunk = build_trunk(bundle.family, n_in=bundle.n_in or 40,
                        n_mels=bundle.n_mels or 65,
                        width=cfg.get("width"), dropout=cfg.get("dropout"))
    model = Classifier(trunk, n_domain=ckpt.get("n_domain")).to(device)
    model.load_state_dict(ckpt["model"])

    loader = bundle.loader("test", int(cfg.get("batch_size", batch_size)),
                           num_workers=0)
    out = _run_loader(model, loader, device, train=False, optimiser=None,
                      label_loss=torch.nn.CrossEntropyLoss(),
                      domain_loss=None, lam=0.0)
    return pd.DataFrame({
        "speaker_id": out["speakers_raw"],
        "dataset": out["item_dataset"],
        "y": out["item_y"],
        "prob_pd": softmax_np(out["item_logits"])[:, 1],
    })


def score(frame, rule):
    """Speaker-level balanced accuracy and AUC under one aggregation rule."""
    rows = []
    for speaker, group in frame.groupby("speaker_id", sort=False):
        probs = np.stack([1.0 - group.prob_pd.to_numpy(),
                          group.prob_pd.to_numpy()], axis=1)
        rows.append({
            "speaker_id": speaker,
            "dataset": group.dataset.iloc[0],
            "y": int(group.y.iloc[0]),
            "score": rule(probs),
        })
    return pd.DataFrame(rows)


def summarise(pred, regime, rule_name):
    rows = []
    for name, group in [("all", pred)] + list(pred.groupby("dataset")):
        y = group.y.to_numpy()
        s = group.score.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({
            "regime": regime,
            "rule": rule_name,
            "dataset": name,
            "n": len(group),
            "balanced_acc": balanced_accuracy_score(y, (s >= 0.5).astype(int)),
            "auc": roc_auc_score(y, s),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="handcrafted")
    ap.add_argument("--level", default="duration")
    ap.add_argument("--phone-class")
    ap.add_argument("--regimes", nargs="+", default=["baseline", "dann"])
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so a training job keeps the GPU")
    args = ap.parse_args()

    device = torch.device(args.device)
    suffix = f"_{args.phone_class}" if args.phone_class else ""
    rows = []
    for regime in args.regimes:
        frames = []
        for i in range(10):
            name = (f"mixed_{i}_{args.family}_{args.level}_{regime}"
                    f"{suffix}_s0")
            run_dir = EXPERIMENTS / name
            if not (run_dir / "checkpoints" / "best.pt").exists():
                print(f"skip {name}: no checkpoint")
                continue
            print(f"inferring {name}")
            frames.append(infer_fold(run_dir, device).assign(fold=i))
        if not frames:
            print(f"{regime}: nothing to score")
            continue
        # Out-of-fold: every speaker appears in exactly one test split.
        items = pd.concat(frames, ignore_index=True)
        seen = items.groupby("speaker_id").fold.nunique()
        assert (seen == 1).all(), (
            f"{int((seen > 1).sum())} speakers appear in more than one test fold")
        dup = items.groupby("speaker_id").dataset.nunique()
        assert (dup == 1).all(), "a speaker appeared under two corpora"
        for rule_name, rule in RULES.items():
            pred = score(items, rule)
            rows.extend(summarise(pred, regime, rule_name))

    if not rows:
        raise SystemExit("no runs scored")
    frame = pd.DataFrame(rows)
    print()
    print(frame.round(3).to_string(index=False))
    dest = REPORTS / f"stage7_aggregation_{args.family}_{args.level}{suffix}.csv"
    frame.to_csv(dest, index=False)
    print(f"\nwrote {dest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
