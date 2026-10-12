"""GBT baseline on the same splits the neural runs use.

    python -m src.training.classical --level duration --split 0
    python -m src.training.classical --protocol lodo --split MDVR-KCL

This is the Stage 5 thermometer scored the Stage 7 way: speaker-level, on a
Stage 6 configuration, with train-only imputation and global stats. It is not
a replacement for the MLP; it is the number the MLP should land near.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.training.config import TrainConfig
from src.training.data import Bundle
from src.training.metrics import pool_speakers

REPO_ROOT = Path(__file__).resolve().parents[2]


def _eval(model, ds, role):
    X = ds.X.numpy()
    proba = model.predict_proba(X)
    # pool_speakers softmaxes logits; log-proba is enough
    logits = np.log(np.clip(proba, 1e-7, 1.0))
    pooled = pool_speakers(
        logits, ds.y, ds.speakers, datasets=ds.datasets, cohorts=ds.cohorts)
    print(f"{role:5s} speaker BA {pooled['balanced_acc']:.3f}  "
          f"AUC {pooled['auc']:.3f}  n={pooled['n_speakers']}"
          + ("" if not pooled["by_dataset"] else "  " + "  ".join(
              f"{k} {v:.3f}" for k, v in pooled["by_dataset"].items())))
    rows = []
    for r in pooled["rows"]:
        rows.append({
            "role": role,
            "speaker_id": r["speaker_id"],
            "dataset": r["dataset"],
            "cohort": r["cohort"],
            "label": "PD" if r["y"] == 1 else "HC",
            "prob_pd": r["prob_pd"],
            "pred": "PD" if r["pred"] == 1 else "HC",
            "n_items": r["n_items"],
        })
    return pd.DataFrame(rows), pooled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are not repeated here: they come from TrainConfig, so this
    # script and the neural runs cannot drift apart on the values they share.
    for flag in ("--level", "--protocol", "--split", "--dataset",
                 "--phone-class", "--norm"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(TrainConfig.from_yaml(), family="handcrafted", **{
        name: value for name, value in vars(args).items()
        if name != "dry_run" and value is not None})

    bundle = Bundle(
        family=cfg.family, level=cfg.level, protocol=cfg.protocol,
        config=cfg.split, scope=cfg.norm, phone_class=cfg.phone_class,
        dataset=cfg.dataset, seed=cfg.seed)
    print(bundle.summary().to_string(index=False))
    if args.dry_run:
        return

    train = bundle.datasets["train"]
    model = HistGradientBoostingClassifier(
        max_iter=120, random_state=cfg.seed)
    model.fit(train.X.numpy(), train.y)

    # Kept as it is: renaming these directories would orphan the finished
    # `*_gbt_norm-*` runs under experiments/.
    name = (f"{cfg.protocol}_{cfg.split}_handcrafted_{cfg.level}_gbt"
            f"_norm-{cfg.norm}_s{cfg.seed}")
    out = REPO_ROOT / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    parts, summary = [], {
        "model": "HistGradientBoostingClassifier",
        "norm": cfg.norm,
        "protocol": cfg.protocol,
        "split": cfg.split,
        "target_stats": cfg.target_stats,
    }
    for role in ("val", "test"):
        ds = bundle.datasets[role]
        if ds is None:
            continue
        frame, pooled = _eval(model, ds, role)
        parts.append(frame)
        summary[f"{role}_ba"] = pooled["balanced_acc"]
        summary[f"{role}_auc"] = pooled["auc"]
        summary[f"{role}_by_dataset"] = pooled["by_dataset"]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            out / "predictions.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
