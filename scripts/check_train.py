"""Sanity-check Stage 7 pieces that do not need the feature cache.

    python scripts/check_train.py
    python scripts/check_train.py --bundle    # needs features + splits on disk

The GRL test is the one that matters: a reversal layer that does not reverse
is a silent failure, and it would only show up as a DANN that never moves.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbones import build_trunk                    # noqa: E402
from src.models.classifier import Classifier                    # noqa: E402
from src.models.grl import grl, lambda_at                       # noqa: E402
from src.training.metrics import pool_speakers                  # noqa: E402


def check_grl():
    x = torch.ones(4, 3, requires_grad=True)
    y = grl(x, 0.5)
    assert torch.equal(y, x), "GRL changed the forward pass"
    y.sum().backward()
    want = torch.full_like(x, -0.5)
    assert torch.allclose(x.grad, want), f"GRL grad was {x.grad}"
    print("GRL: forward is identity, backward is -lambda")


def check_lambda():
    start = lambda_at(0, 100)
    end = lambda_at(99, 100)
    assert start < 0.05, f"lambda starts at {start}, should be near 0"
    assert end > 0.95, f"lambda ends at {end}, should be near 1"
    print(f"lambda schedule: {start:.4f} -> {end:.4f}")


def check_pool():
    logits = np.array([[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    speakers = np.array(["a", "a", "b", "b"])
    out = pool_speakers(logits, y, speakers)
    assert out["n_speakers"] == 2
    assert out["balanced_acc"] == 1.0
    print("speaker pooling: two speakers, perfect BA")


def check_step():
    trunk = build_trunk("handcrafted", n_in=8, width=16, dropout=0.0)
    model = Classifier(trunk, n_domain=3)
    x = torch.randn(12, 8)
    y = torch.randint(0, 2, (12,))
    d = torch.randint(0, 3, (12,))
    out = model(x, lam=0.5)
    loss = torch.nn.functional.cross_entropy(out["label"], y)
    loss = loss + torch.nn.functional.cross_entropy(out["domain"], d)
    loss.backward()
    assert out["features"].shape == (12, 8)
    grad = next(model.trunk.parameters()).grad
    assert grad is not None and torch.isfinite(grad).all()
    print(f"one DANN step: {sum(p.numel() for p in model.parameters())} params")

    cnn = build_trunk("mel", n_mels=5, width=8, dropout=0.0)
    spec = torch.randn(4, 5, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, 7:] = False
    pooled = cnn(spec, mask=mask)
    assert pooled.shape == (4, 16)
    print("CNN attention pooling respects the padding mask")


def check_bundle():
    from src.training.data import Bundle
    bundle = Bundle("handcrafted", "duration", "mixed", 0)
    print(bundle.summary().to_string(index=False))
    loader = bundle.loader("train", batch_size=32, samples_per_epoch=64)
    batch = next(iter(loader))
    assert batch["x"].shape[0] == 32
    assert batch["x"].shape[1] == bundle.n_in
    print(f"bundle: batch {tuple(batch['x'].shape)}, "
          f"{bundle.n_domain} domains {bundle.domain_names}")


def check_corpus_scope():
    """Unseen dataset must use pooled source stats, not identity (0, 1)."""
    import pandas as pd
    from src.features import store as fs

    rng = np.random.default_rng(0)
    n = 60
    meta = pd.DataFrame({
        "dataset": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        "speaker_id": [f"s{i}" for i in range(n)],
        "path": [f"p{i}" for i in range(n)],
    })
    values = rng.normal(size=(n, 4))
    values[:20] += 10.0
    values[20:40] -= 5.0
    train = meta.dataset != "C"
    stats = fs.fit_stats(values[train.to_numpy()], meta.loc[train], "corpus")
    assert "A" in stats and "B" in stats and "__all__" in stats
    assert "C" not in stats, "held-out corpus leaked into corpus stats"
    z = fs.apply(values, meta, "corpus", stats)
    assert np.abs(z[:20].mean()) < 1e-6
    assert np.abs(z[20:40].mean()) < 1e-6
    assert not np.allclose(z[40:], values[40:]), (
        "unseen corpus was left unscaled; fallback should be __all__")
    print("corpus scope: unseen dataset uses __all__, not identity")


def _toy_frames(seed=0):
    """Synthetic (meta, X) for Bundle.from_frames: no features, no splits.

    Eight speakers, four items each. A and B are seen in training, C is a
    test-only corpus (the LODO shape). Test rows sit far from the training
    rows so a statistic fitted on the wrong rows is visible.
    """
    import pandas as pd

    spec = [("a1", "A", "HC", "train"), ("a2", "A", "PD", "train"),
            ("b1", "B", "HC", "train"), ("b2", "B", "PD", "train"),
            ("a3", "A", "PD", "val"), ("b3", "B", "HC", "val"),
            ("c1", "C", "PD", "test"), ("c2", "C", "HC", "test")]
    rows = [{"speaker_id": speaker, "dataset": dataset, "label": label,
             "role": role, "item_id": f"{speaker}_{k}", "path": f"{speaker}.wav",
             "cohort": "elderly", "task": "read",
             "phone_class": "vowel" if k % 2 == 0 else "plosive"}
            for speaker, dataset, label, role in spec for k in range(4)]
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(meta), 3))
    X[(meta.role == "test").to_numpy()] += 50.0
    return meta, X


def _rejects(build):
    """The ValueError text from `build`, or None if it was accepted."""
    try:
        build()
    except ValueError as exc:
        return str(exc)
    return None


def check_bundle_leak():
    """Invariant 1: a speaker in two roles must not build."""
    from src.training.data import Bundle

    meta, X = _toy_frames()
    leaky = meta.copy()
    # c1 is a test speaker; give it a training speaker's id.
    leaky.loc[leaky.speaker_id == "c1", "speaker_id"] = "a1"
    said = _rejects(lambda: Bundle.from_frames(leaky, X=X))
    assert said and "leak" in said, (
        f"a speaker in both train and test was accepted ({said!r})")
    val_leak = meta.copy()
    val_leak.loc[val_leak.speaker_id == "a3", "speaker_id"] = "a2"
    said_val = _rejects(lambda: Bundle.from_frames(val_leak, X=X))
    assert said_val and "leak" in said_val, (
        f"a speaker in both train and val was accepted ({said_val!r})")
    assert Bundle.from_frames(meta, X=X).n_domain == 2, "clean split rejected"
    print("bundle leak check: train/val and train/test speaker overlap rejected")


def check_bundle_impute():
    """Imputation medians come from training rows only."""
    from src.training.data import Bundle

    meta, X = _toy_frames()
    train = (meta.role == "train").to_numpy()
    test = (meta.role == "test").to_numpy()
    X = X.copy()
    X[:, 0] = np.where(train, 1.0, 100.0)
    X[np.flatnonzero(train)[0], 0] = np.nan
    X[np.flatnonzero(test)[0], 0] = np.nan

    bundle = Bundle.from_frames(meta, X=X, scope="none")
    assert bundle.medians[0] == 1.0, (
        f"median is {bundle.medians[0]}, so non-train rows were used "
        "(all rows would give 50.5)")
    filled = bundle.datasets["test"].X.numpy()[0, 0]
    assert filled == 1.0, f"held-out NaN was filled with {filled}, not 1.0"
    print("bundle impute: NaNs filled with the train-only median")


def check_bundle_norm_train_only():
    """Invariant 2: global stats are fitted on training rows only."""
    from src.training.data import Bundle

    meta, X = _toy_frames()
    train = (meta.role == "train").to_numpy()
    bundle = Bundle.from_frames(meta, X=X)
    mean, std = bundle.stats["__all__"]
    assert np.allclose(mean, X[train].mean(axis=0)), (
        "global mean is not the training-row mean")
    assert np.allclose(std, X[train].std(axis=0)), (
        "global std is not the training-row std")
    assert not np.allclose(mean, X.mean(axis=0)), (
        "training mean equals the all-row mean; test rows leaked into the fit")
    z_train = bundle.datasets["train"].X.numpy()
    assert np.abs(z_train.mean()) < 1e-5, "train rows are not centred"
    z_test = bundle.datasets["test"].X.numpy()
    assert z_test.mean() > 5.0, (
        "test rows were centred too, so their own statistics were used")
    print("bundle norm: mean/std fitted on train rows, applied to val and test")


def check_bundle_labels_domains():
    """PD/HC mapping, and an unseen corpus encoding to -1 for ignore_index."""
    from src.training.data import Bundle

    meta, X = _toy_frames()
    bundle = Bundle.from_frames(meta, X=X)
    want = (meta.label == "PD").to_numpy().astype(np.int64)
    assert (bundle.y == want).all(), "PD/HC did not map to 1/0"
    assert bundle.domain_names == ["A", "B"], (
        f"domains are {bundle.domain_names}; C is test-only and must not be one")
    for name, ids in (("A", {0}), ("B", {1}), ("C", {-1})):
        got = set(bundle.d[(meta.dataset == name).to_numpy()].tolist())
        assert got == ids, f"corpus {name} encoded to {got}, expected {ids}"
    said = _rejects(lambda: Bundle.from_frames(
        meta.assign(label=meta.label.replace({"HC": "CONTROL"})), X=X))
    assert said and "unrecognised" in said, f"unknown label accepted ({said!r})"
    print("bundle labels: HC/PD -> 0/1, train corpora -> 0..D-1, unseen -> -1")


def check_bundle_shuffle():
    """Shuffled domain labels keep the training multiset and spare val/test.

    Agreement has to land near chance. An identity permutation would leave the
    control identical to real DANN, and shuffling val/test would change the
    logged domain accuracy without changing the adversary.
    """
    from src.training.data import Bundle

    meta, X = _toy_frames()
    plain = Bundle.from_frames(meta, X=X)
    shuffled = Bundle.from_frames(meta, X=X, shuffle_domains=True, seed=0)
    again = Bundle.from_frames(meta, X=X, shuffle_domains=True, seed=0)
    train = (shuffled.meta.role == "train").to_numpy()
    assert sorted(shuffled.d[train].tolist()) == sorted(plain.d[train].tolist()), (
        "shuffle changed the training label counts, so class weights would move")
    assert not np.array_equal(shuffled.d[train], plain.d[train]), (
        "seed 0 left the training labels identical to the corpus")
    assert np.array_equal(shuffled.d[~train], plain.d[~train]), (
        "shuffle touched val or test labels")
    assert np.array_equal(again.d, shuffled.d), "the same seed gave a new permutation"
    other = Bundle.from_frames(meta, X=X, shuffle_domains=True, seed=1)
    assert not np.array_equal(other.d[train], shuffled.d[train]), (
        "seed 1 reproduced seed 0")
    # Toy train is 8 items of A and 8 of B, so chance is 0.5.
    assert abs(shuffled.domain_chance - 0.5) < 1e-9, shuffled.domain_chance
    assert shuffled.domain_agreement < 0.85, (
        f"agreement {shuffled.domain_agreement:.3f} is far above chance 0.5")
    print(f"bundle shuffle: train labels permuted "
          f"(agree {shuffled.domain_agreement:.3f}, chance 0.500), "
          "val/test unchanged")


def check_shuffle_run_id():
    """The token appears only on the control, so published run dirs stay put."""
    from src.training.config import TrainConfig

    control = TrainConfig(regime="dann", shuffle_domains=True)
    plain = TrainConfig(regime="dann")
    assert control.resolved_run_id() == (
        "mixed_0_handcrafted_duration_dann_shuffle-domains_s0"), (
        control.resolved_run_id())
    assert plain.resolved_run_id() == "mixed_0_handcrafted_duration_dann_s0"
    print("run id: shuffle-domains is a suffix, plain DANN is unchanged")


def check_bundle_filter():
    """Filters cut rows before any statistic is fitted."""
    from src.training.data import Bundle

    meta, X = _toy_frames()
    vowels = Bundle.from_frames(meta, X=X, phone_class="vowel")
    assert set(vowels.meta.phone_class) == {"vowel"}, (
        f"phone_class filter left {set(vowels.meta.phone_class)}")
    assert len(vowels.meta) == len(meta) // 2, (
        f"phone_class filter kept {len(vowels.meta)} of {len(meta)} rows")
    train = ((meta.role == "train") & (meta.phone_class == "vowel")).to_numpy()
    assert np.allclose(vowels.stats["__all__"][0], X[train].mean(axis=0)), (
        "stats were fitted before the filter")
    only_a = Bundle.from_frames(meta, X=X, dataset="A")
    assert set(only_a.meta.dataset) == {"A"} and only_a.n_domain == 1
    said = _rejects(lambda: Bundle.from_frames(
        meta.drop(columns=["phone_class"]), X=X, phone_class="vowel"))
    assert said and "phone_class" in said, (
        f"phone_class filter on a table without the column was accepted ({said!r})")
    print("bundle filter: phone_class/dataset applied before stats are fitted")


def main():
    check_grl()
    check_lambda()
    check_pool()
    check_step()
    check_corpus_scope()
    check_bundle_leak()
    check_bundle_impute()
    check_bundle_norm_train_only()
    check_bundle_labels_domains()
    check_bundle_shuffle()
    check_shuffle_run_id()
    check_bundle_filter()
    if "--bundle" in sys.argv:
        check_bundle()
    else:
        print("skipping --bundle (needs features and splits on disk)")
    print("all training checks passed")


if __name__ == "__main__":
    main()
