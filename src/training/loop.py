"""Train one run: baseline or DANN, write an experiment directory.

Early-stops on speaker-level PD balanced accuracy on the validation speakers.
Item-level accuracy is logged but never used for selection.

    python -m src.training.run --dry-run
    python -m src.training.run --family handcrafted --level duration --split 0
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.backbones import build_trunk
from src.models.classifier import Classifier
from src.models.grl import lambda_at
from src.training.metrics import inverse_weights, pool_speakers

REPO_ROOT = Path(__file__).resolve().parents[2]


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _cat(chunks):
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _run_loader(model, loader, device, *, train, optimiser, label_loss,
                domain_loss, lam, max_steps=None):
    model.train(train)
    totals = {"label": 0.0, "domain": 0.0, "n": 0}
    logits, ys, speakers, datasets, cohorts = [], [], [], [], []
    domain_logits, domain_ys, features = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            x = batch["x"].to(device)
            mask = None if batch["mask"] is None else batch["mask"].to(device)
            y = batch["y"].to(device)
            d = batch["d"].to(device)
            out = model(x, mask=mask, lam=lam if train else 0.0)
            loss_y = label_loss(out["label"], y)
            loss = loss_y
            loss_d = None
            if domain_loss is not None and "domain" in out:
                loss_d = domain_loss(out["domain"], d)
                loss = loss + loss_d
            if train:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            n = len(y)
            totals["label"] += float(loss_y.detach()) * n
            totals["n"] += n
            if loss_d is not None:
                totals["domain"] += float(loss_d.detach()) * n
            logits.append(out["label"].detach().cpu().numpy())
            features.append(out["features"].detach().cpu().numpy())
            ys.append(batch["y"].numpy())
            speakers.extend(batch["speaker"])
            datasets.extend(batch["dataset"])
            cohorts.extend(batch["cohort"])
            if "domain" in out:
                domain_logits.append(out["domain"].detach().cpu().numpy())
                domain_ys.append(batch["d"].numpy())

    pooled = pool_speakers(
        _cat(logits), _cat(ys), np.array(speakers),
        datasets=np.array(datasets), cohorts=np.array(cohorts),
        domain_logits=_cat(domain_logits), domain_y=_cat(domain_ys))
    pooled["label_loss"] = totals["label"] / max(totals["n"], 1)
    pooled["domain_loss"] = (totals["domain"] / max(totals["n"], 1)
                             if domain_loss is not None else float("nan"))
    pooled["features"] = _cat(features)
    pooled["speakers_raw"] = np.array(speakers)
    pooled["item_y"] = _cat(ys)
    pooled["item_logits"] = _cat(logits)
    pooled["item_dataset"] = np.array(datasets)
    pooled["item_cohort"] = np.array(cohorts)
    return pooled


def _mean_embeddings(features, speakers):
    speakers = np.asarray(speakers)
    features = np.asarray(features)
    _, first = np.unique(speakers, return_index=True)
    names = speakers[first[np.argsort(first)]]
    rows = []
    for name in names:
        rows.append(features[speakers == name].mean(axis=0))
    return names, np.stack(rows, axis=0)


def _prediction_frame(pooled, role):
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
    return pd.DataFrame(rows)


def _embedding_frame(pooled, role):
    names, vecs = _mean_embeddings(pooled["features"], pooled["speakers_raw"])
    lookup = {r["speaker_id"]: r for r in pooled["rows"]}
    frame = pd.DataFrame(vecs, columns=[f"e{i}" for i in range(vecs.shape[1])])
    frame.insert(0, "speaker_id", names)
    frame.insert(1, "role", role)
    frame.insert(2, "dataset", [lookup[n]["dataset"] for n in names])
    frame.insert(3, "label", ["PD" if lookup[n]["y"] == 1 else "HC" for n in names])
    return frame


def train(bundle, cfg, out_dir):
    """Run baseline or DANN. `cfg` is a `TrainConfig`; `out_dir` is created."""
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Init and dropout read this. The sampler keeps its own generator, seeded
    # from the same value. Runs written before this call existed did not use
    # it, so re-running their directory names would not reproduce them.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _device(cfg.device)
    regime = cfg.regime_for_model
    n_domain = bundle.n_domain if regime == "dann" else None
    if regime == "dann" and bundle.n_domain < 2:
        raise ValueError(
            "DANN needs at least two training domains; this split has "
            f"{bundle.domain_names}. Use --regime baseline, or do not "
            "filter to a single corpus.")

    trunk = build_trunk(
        bundle.family, n_in=bundle.n_in or 40, n_mels=bundle.n_mels or 65,
        width=cfg.width, dropout=cfg.dropout)
    model = Classifier(trunk, n_domain=n_domain).to(device)

    train_y = bundle.y[bundle.meta.role.to_numpy() == "train"]
    train_d = bundle.d[bundle.meta.role.to_numpy() == "train"]
    label_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(inverse_weights(train_y, 2),
                            device=device, dtype=torch.float32))
    domain_loss = None
    if n_domain:
        domain_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(inverse_weights(train_d, n_domain),
                                device=device, dtype=torch.float32),
            ignore_index=-1)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    batch_size = cfg.batch_size
    workers = cfg.num_workers
    train_loader = bundle.loader(
        "train", batch_size, seed=cfg.seed,
        samples_per_epoch=cfg.samples_per_epoch, num_workers=workers)
    val_loader = bundle.loader("val", batch_size, num_workers=workers)
    test_loader = bundle.loader("test", batch_size, num_workers=workers)

    epochs = cfg.epochs
    patience = cfg.patience
    # Checkpoint selection does not open until this epoch. Without it the run
    # can save epoch 0 -- where lambda is still 0 -- and call the result DANN;
    # five of ten mixed folds did exactly that. Applied to baseline too, so the
    # two regimes are selected under one rule and stay comparable. The default
    # lives on TrainConfig, so a caller that skips the yaml still gets it.
    # A floor at or past the last epoch would leave best.pt unwritten.
    min_select_epoch = min(cfg.min_select_epoch, epochs - 1)
    max_steps = cfg.max_steps
    n_steps = max(1, len(train_loader)) * epochs
    gamma = cfg.lambda_gamma
    lam_max = cfg.lambda_max

    history = []
    best_ba = -1.0
    best_epoch = -1
    stale = 0
    global_step = 0

    print(f"device {device}, {sum(p.numel() for p in model.parameters())} "
          f"parameters, {bundle.n_domain} training domains "
          f"{bundle.domain_names}, regime {regime}")

    for epoch in range(epochs):
        lam = lambda_at(global_step, n_steps, gamma=gamma, lam_max=lam_max) \
            if regime == "dann" else 0.0
        train_out = _run_loader(
            model, train_loader, device, train=True, optimiser=optimiser,
            label_loss=label_loss, domain_loss=domain_loss, lam=lam,
            max_steps=max_steps)
        global_step += len(train_loader) if max_steps is None else min(
            len(train_loader), max_steps)

        row = {
            "epoch": epoch,
            "lambda": lam,
            "train_ba": train_out["balanced_acc"],
            "train_auc": train_out["auc"],
            "train_loss": train_out["label_loss"],
            "train_domain_acc": train_out.get("domain_acc", float("nan")),
            "train_domain_loss": train_out["domain_loss"],
        }
        if val_loader is not None:
            val_out = _run_loader(
                model, val_loader, device, train=False, optimiser=None,
                label_loss=label_loss, domain_loss=domain_loss, lam=0.0)
            row["val_ba"] = val_out["balanced_acc"]
            row["val_auc"] = val_out["auc"]
            row["val_loss"] = val_out["label_loss"]
            row["val_domain_acc"] = val_out.get("domain_acc", float("nan"))
            for corpus, score in val_out["by_dataset"].items():
                row[f"val_ba_{corpus}"] = score
            metric = val_out["balanced_acc"]
        else:
            val_out = None
            metric = train_out["balanced_acc"]
            row["val_ba"] = float("nan")

        history.append(row)
        print(f"epoch {epoch:02d}  lam {lam:.3f}  "
              f"train BA {row['train_ba']:.3f}  val BA {row['val_ba']:.3f}  "
              f"domain {row['train_domain_acc']}")

        if epoch < min_select_epoch:
            continue
        if metric > best_ba + 1e-4:
            best_ba = metric
            best_epoch = epoch
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_ba": best_ba,
                "domain_names": bundle.domain_names,
                "n_in": bundle.n_in,
                "n_mels": bundle.n_mels,
                "n_domain": n_domain,
                "family": bundle.family,
            }, out_dir / "checkpoints" / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}, best was {best_epoch} "
                      f"(val BA {best_ba:.3f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)

    ckpt = torch.load(out_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    pred_parts, emb_parts = [], []
    for role, loader in (("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        pooled = _run_loader(
            model, loader, device, train=False, optimiser=None,
            label_loss=label_loss, domain_loss=domain_loss, lam=0.0)
        pred_parts.append(_prediction_frame(pooled, role))
        emb_parts.append(_embedding_frame(pooled, role))
        print(f"{role:5s} speaker BA {pooled['balanced_acc']:.3f}  "
              f"AUC {pooled['auc']:.3f}  n={pooled['n_speakers']}"
              + ("" if not pooled["by_dataset"] else "  " + "  ".join(
                  f"{k} {v:.3f}" for k, v in pooled["by_dataset"].items())))

    if pred_parts:
        pd.concat(pred_parts, ignore_index=True).to_csv(
            out_dir / "predictions.csv", index=False)
        pd.concat(emb_parts, ignore_index=True).to_csv(
            out_dir / "embeddings.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_ba": best_ba,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "domain_names": bundle.domain_names,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
