"""Load Stage 7 experiment outputs for Stage 8 scoring.

Mixed protocol: concatenate the test role of every configuration. Each speaker
appears once, so per-corpus numbers are over the whole speaker population
rather than over a two-patient fold.

Each loader takes the `TrainConfig` of the run family it is pooling and varies
only `split`, so the run-id grammar lives in `resolved_run_id()` alone. A
format string with a `{i}` or `{c}` hole used to stand in for it, which meant
one parameter name carried two incompatible schemas.
"""

from dataclasses import replace
from pathlib import Path
import json

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"


def mixed_oof(cfg):
    """Test-role predictions and embeddings across 10 mixed configurations."""
    preds, embs = [], []
    missing = []
    for i in range(10):
        folder = EXPERIMENTS / replace(cfg, split=i).resolved_run_id()
        pred_path = folder / "predictions.csv"
        if not pred_path.exists():
            missing.append(str(folder.name))
            continue
        pred = pd.read_csv(pred_path)
        test = pred[pred.role == "test"].copy()
        test["fold"] = i
        preds.append(test)
        emb_path = folder / "embeddings.csv"
        if emb_path.exists():
            emb = pd.read_csv(emb_path)
            test_emb = emb[emb.role == "test"].copy()
            test_emb["fold"] = i
            embs.append(test_emb)
    if missing:
        print(f"missing {len(missing)} mixed configs: {missing}")
    if not preds:
        return None, None
    pred = pd.concat(preds, ignore_index=True)
    dup = pred.speaker_id.duplicated().sum()
    if dup:
        raise ValueError(f"{dup} speakers appear in more than one test fold")
    emb = pd.concat(embs, ignore_index=True) if embs else None
    return pred, emb


def lodo_test(cfg, corpus):
    folder = EXPERIMENTS / replace(cfg, split=corpus).resolved_run_id()
    path = folder / "predictions.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    return pred[pred.role == "test"].copy()


def selection_trace(cfg, n=10):
    """Best-checkpoint vs last-epoch, to see whether early-stop kept the adversary."""
    rows = []
    for i in range(n):
        folder = EXPERIMENTS / replace(cfg, split=i).resolved_run_id()
        summary = folder / "summary.json"
        metrics = folder / "metrics.csv"
        if not summary.exists() or not metrics.exists():
            continue
        meta = json.loads(summary.read_text())
        m = pd.read_csv(metrics)
        if "train_domain_acc" not in m.columns:
            continue
        best = int(meta["best_epoch"])
        last = int(m.epoch.max())
        brow = m.loc[m.epoch == best].iloc[0]
        lrow = m.loc[m.epoch == last].iloc[0]
        rows.append({
            "fold": i,
            "best_epoch": best,
            "last_epoch": last,
            "best_val_ba": float(brow.val_ba),
            "best_lambda": float(brow["lambda"]),
            "best_train_domain_acc": float(brow.train_domain_acc),
            "last_train_domain_acc": float(lrow.train_domain_acc),
            "last_lambda": float(lrow["lambda"]),
        })
    return pd.DataFrame(rows) if rows else None
