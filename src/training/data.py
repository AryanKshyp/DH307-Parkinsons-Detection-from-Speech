"""Join Stage 6 splits to Stage 5 features, without leaking test statistics.

The cache is unnormalised. Global mean and std are fitted on training rows of
the requested configuration only, then applied to val and test. Imputation of
undefined voice-quality columns uses the training median for the same reason.

Sampling is speaker-balanced and stratified on `dataset x label`. A uniform
draw over segments would let ItalianPD (most duration windows) or MDVR-KCL
(most utterance chunks) dominate the gradient, which is the opposite of what
the domain head is for. WeightedRandomSampler gives each of the six cells
equal mass, and each speaker inside a cell equal mass.

    python -m src.training.run --dry-run
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.features import store as fs
from src.training.metrics import LABEL_TO_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = REPO_ROOT / "data" / "manifests"
SEGMENTS = REPO_ROOT / "data" / "interim" / "segments.csv"

HC, PD = "HC", "PD"


def _require(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} is missing. {hint}")
    return path


def _read_splits(protocol):
    path = _require(
        MANIFESTS / f"splits_{protocol}.csv",
        "Run Stage 6: python -m src.data.splits --protocol "
        f"{protocol}" + (" --k 10" if protocol == "mixed" else ""))
    return pd.read_csv(path)


def configs_for(protocol):
    """Split configuration ids as they appear in the split file."""
    return list(_read_splits(protocol).config.unique())


def _config_equals(series, value):
    """mixed configs are ints, lodo configs are corpus names."""
    if pd.api.types.is_numeric_dtype(series):
        return series == int(value)
    return series.astype(str) == str(value)


def load_items(family, level):
    """One row per training item, plus the array or memmap it indexes.

    Handcrafted: one vector per segment.
    Mel: one spectrogram per index row (a chunk, at utterance level).
    """
    _require(SEGMENTS, "Run Stage 4: python -m src.data.segments")
    if family == "handcrafted":
        npy = fs.FEATURES / f"handcrafted_{level}.npy"
        _require(npy, f"Run Stage 5: python -m src.features.extract --levels {level} --family handcrafted")
        values, meta = fs.load_handcrafted(level, impute=False)
        meta = meta.copy()
        meta["item_id"] = meta["segment_id"].astype(str)
        return {"kind": "vector", "X": values, "meta": meta.reset_index(drop=True)}

    if family == "mel":
        npy = fs.FEATURES / f"mel_{level}.npy"
        _require(npy, f"Run Stage 5: python -m src.features.extract --levels {level} --family mel")
        data, index = fs.load_mel(level)
        segments = pd.read_csv(SEGMENTS, low_memory=False)
        segs = segments[segments.level == level]
        keep = ["segment_id", "path", "dataset", "language", "channel",
                "speaker_id", "label", "cohort", "task", "phone_class"]
        keep = [c for c in keep if c in segs.columns]
        meta = index.merge(segs[keep], on="segment_id", how="left")
        missing = meta.speaker_id.isna().sum()
        if missing:
            meta = meta.dropna(subset=["speaker_id"]).reset_index(drop=True)
            print(f"dropped {missing} mel items whose segment_id is not in "
                  f"segments.csv (features and segments are out of date)")
        meta = meta.copy()
        meta["item_id"] = meta["id"].astype(str) if "id" in meta.columns \
            else meta["segment_id"].astype(str)
        return {"kind": "mel", "data": data, "meta": meta.reset_index(drop=True)}

    raise ValueError(f"unknown family {family!r}")


def attach_split(meta, protocol, config):
    """Add a `role` column from the requested split configuration."""
    splits = _read_splits(protocol)
    slice_ = splits[_config_equals(splits.config, config)]
    if slice_.empty:
        known = splits.config.unique().tolist()
        raise ValueError(f"config {config!r} not in splits_{protocol}.csv "
                         f"(have {known})")
    roles = slice_[["speaker_id", "role"]]
    out = meta.merge(roles, on="speaker_id", how="left")
    unmatched = out.role.isna().sum()
    if unmatched:
        extra = out.loc[out.role.isna(), "speaker_id"].nunique()
        raise ValueError(
            f"{extra} speakers in the features are not in splits_{protocol}.csv; "
            "regenerate splits against the current manifest")
    # A speaker in two roles of one configuration is a leak. The Stage 6
    # checker asserts this on the split file; re-assert after the join in
    # case a speaker_id collision was introduced in segments.csv.
    clash = (out.groupby("speaker_id").role.nunique() > 1).sum()
    if clash:
        raise ValueError(f"{clash} speakers have two roles after the join")
    return out


def _impute_from_train(X, train_mask):
    """Fill NaNs with the training-set column median."""
    X = np.array(X, dtype=np.float64, copy=True)
    medians = np.nanmedian(X[train_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    holes = ~np.isfinite(X)
    if holes.any():
        X[holes] = np.take(medians, np.where(holes)[1])
    return X, medians


def _mel_stats(data, meta, train_mask, max_items=30_000, seed=0):
    """Per-band mean and std of training frames only."""
    train = meta.loc[train_mask]
    if len(train) > max_items:
        rng = np.random.default_rng(seed)
        train = train.iloc[rng.choice(len(train), max_items, replace=False)]
    total = np.zeros(data.shape[1], dtype=np.float64)
    total_sq = np.zeros(data.shape[1], dtype=np.float64)
    n = 0
    for item in train.itertuples():
        frames = fs.mel_item(data, item)
        if len(frames) == 0:
            continue
        total += frames.sum(axis=0)
        total_sq += np.square(frames, dtype=np.float64).sum(axis=0)
        n += len(frames)
    if n == 0:
        raise ValueError("no training frames to fit Mel statistics on")
    mean = total / n
    var = total_sq / n - np.square(mean)
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std > 1e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def encode_domains(meta, train_mask):
    """Map training corpora to 0..D-1. Unseen (LODO test) corpora get -1."""
    names = sorted(meta.loc[train_mask, "dataset"].unique())
    to_id = {name: i for i, name in enumerate(names)}
    encoded = meta["dataset"].map(to_id).fillna(-1).astype(np.int64).values
    return names, encoded


def shuffle_domain_labels(encoded, train_mask, seed):
    """Permute domain ids on training rows only.

    Item-level, so the label multiset (and the domain-loss class weights) is
    unchanged and the adversary receives no corpus information. A speaker-level
    permutation would leave every segment of a speaker sharing one random
    label, which lets the head learn speaker identity instead. Val and test
    rows keep the true corpus id.

    Returns the new ids, the fraction of training rows whose label still
    matches the corpus, and that fraction's chance level (sum of squared
    label frequencies).
    """
    out = np.array(encoded, copy=True)
    train_idx = np.flatnonzero(train_mask)
    true = out[train_idx].copy()
    out[train_idx] = np.random.default_rng(seed).permutation(true)
    _, counts = np.unique(true, return_counts=True)
    freq = counts / counts.sum()
    chance = float(np.square(freq).sum())
    agree = float((out[train_idx] == true).mean())
    return out, agree, chance


def speaker_weights(meta):
    """Per-row weights: equal mass per dataset x label cell, then per speaker."""
    counts = meta.groupby("speaker_id").size()
    speakers = meta.drop_duplicates("speaker_id")
    stratum = speakers.dataset.astype(str) + "|" + speakers.label.astype(str)
    n_in_cell = stratum.groupby(stratum).transform("size")
    cell_size = speakers.assign(_n=n_in_cell.values).set_index("speaker_id")["_n"]
    n_seg = meta.speaker_id.map(counts).to_numpy(dtype=np.float64)
    n_spk = meta.speaker_id.map(cell_size).to_numpy(dtype=np.float64)
    w = 1.0 / np.maximum(n_seg * n_spk, 1.0)
    return w.astype(np.float64)


class VectorDataset(Dataset):
    def __init__(self, X, y, d, speakers, segment_ids, cohorts, datasets):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = np.asarray(y, dtype=np.int64)
        self.d = np.asarray(d, dtype=np.int64)
        self.speakers = np.asarray(speakers)
        self.segment_ids = np.asarray(segment_ids)
        self.cohorts = np.asarray(cohorts)
        self.datasets = np.asarray(datasets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], int(self.y[i]), int(self.d[i]),
                self.speakers[i], self.segment_ids[i],
                self.cohorts[i], self.datasets[i])


class MelDataset(Dataset):
    def __init__(self, data, row_start, n_valid, mean, std,
                 y, d, speakers, segment_ids, cohorts, datasets):
        self.data = data
        self.row_start = np.asarray(row_start, dtype=np.int64)
        self.n_valid = np.asarray(n_valid, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.d = np.asarray(d, dtype=np.int64)
        self.speakers = np.asarray(speakers)
        self.segment_ids = np.asarray(segment_ids)
        self.cohorts = np.asarray(cohorts)
        self.datasets = np.asarray(datasets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        sl = slice(self.row_start[i], self.row_start[i] + self.n_valid[i])
        frames = np.asarray(self.data[sl], dtype=np.float32)
        frames = (frames - self.mean) / self.std
        # (time, bands) -> (bands, time) for Conv1d
        x = torch.from_numpy(np.ascontiguousarray(frames.T))
        return (x, int(self.y[i]), int(self.d[i]),
                self.speakers[i], self.segment_ids[i],
                self.cohorts[i], self.datasets[i])


def collate_vector(batch):
    xs, ys, ds, speakers, sids, cohorts, datasets = zip(*batch)
    return {
        "x": torch.stack(xs),
        "mask": None,
        "y": torch.tensor(ys, dtype=torch.long),
        "d": torch.tensor(ds, dtype=torch.long),
        "speaker": speakers,
        "segment_id": sids,
        "cohort": cohorts,
        "dataset": datasets,
    }


def collate_mel(batch):
    xs, ys, ds, speakers, sids, cohorts, datasets = zip(*batch)
    n_mels = xs[0].shape[0]
    lengths = [x.shape[1] for x in xs]
    T = max(lengths)
    out = torch.zeros(len(xs), n_mels, T)
    mask = torch.zeros(len(xs), T, dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[1]
        out[i, :, :t] = x
        mask[i, :t] = True
    return {
        "x": out,
        "mask": mask,
        "y": torch.tensor(ys, dtype=torch.long),
        "d": torch.tensor(ds, dtype=torch.long),
        "speaker": speakers,
        "segment_id": sids,
        "cohort": cohorts,
        "dataset": datasets,
    }


def _filter(meta, phone_class, dataset, tasks):
    keep = np.ones(len(meta), dtype=bool)
    if phone_class:
        if "phone_class" not in meta.columns:
            raise ValueError("phone_class filter requested but the table has "
                             "no phone_class column")
        keep &= (meta.phone_class.astype(str) == str(phone_class)).to_numpy()
    if dataset:
        keep &= (meta.dataset.astype(str) == str(dataset)).to_numpy()
    if tasks:
        keep &= meta.task.isin(list(tasks)).to_numpy()
    return keep


class Bundle:
    """Train / val / test datasets for one (family, level, protocol, config).

    Two ways in, one body. `from_disk` (equivalently `Bundle(...)`) reads the
    Stage 5 cache and the Stage 6 split file; `from_frames` takes an array and
    a `meta` that already carries `role`. Everything after the load -- filter,
    label map, domain encode, leak check, train-only impute, fit/apply stats,
    Datasets -- is the same code either way, so it can be checked without the
    corpus on disk (`scripts/check_train.py`).
    """

    def __init__(self, family, level, protocol, config, scope="global",
                 phone_class=None, dataset=None, tasks=None, seed=0,
                 items=None, shuffle_domains=False):
        # `items` is how `from_frames` skips the disk load; leave it None and
        # this is the original path, unchanged.
        if items is None:
            items = load_items(family, level)
            meta = attach_split(items["meta"], protocol, config)
        else:
            meta = items["meta"]
        keep = _filter(meta, phone_class, dataset, tasks)
        at = np.flatnonzero(keep)
        meta = meta.iloc[at].reset_index(drop=True)
        X = items["X"][at] if items["kind"] == "vector" else None
        data = items["data"] if items["kind"] == "mel" else None

        if meta.empty:
            raise ValueError("no items left after filters "
                             f"(level={level}, phone_class={phone_class}, "
                             f"dataset={dataset})")

        train_mask = (meta.role == "train").to_numpy()
        if not train_mask.any():
            raise ValueError("training split is empty after filters")

        mapped = meta.label.map(LABEL_TO_ID)
        if mapped.isna().any():
            unknown = meta.loc[mapped.isna(), "label"].unique()
            raise ValueError(f"unrecognised labels: {unknown}")
        y = mapped.to_numpy(dtype=np.int64)

        domain_names, d = encode_domains(meta, train_mask)
        self.domain_agreement = 1.0
        self.domain_chance = 1.0
        if shuffle_domains:
            d, self.domain_agreement, self.domain_chance = shuffle_domain_labels(
                d, train_mask, seed)
        train_speakers = set(meta.loc[train_mask, "speaker_id"])
        for role, mask in (("val", meta.role == "val"),
                           ("test", meta.role == "test")):
            leaked = set(meta.loc[mask, "speaker_id"]) & train_speakers
            if leaked:
                raise ValueError(f"{len(leaked)} speakers leak into {role}")

        kind = items["kind"]
        if kind == "mel" and scope == "corpus":
            raise ValueError(
                "norm=corpus is defined for handcrafted vectors only; "
                "Mel still uses a single per-band mean/std. See "
                "experiments/corpus_norm_ablation/README.md")
        if kind == "vector":
            X, medians = _impute_from_train(X, train_mask)
            if scope == "none":
                stats = None
            else:
                stats = fs.fit_stats(X[train_mask], meta.loc[train_mask], scope)
                X = fs.apply(X, meta, scope, stats)
            self.medians = medians
            self.mel_mean = self.mel_std = None
        else:
            if scope == "none":
                mean = np.zeros(data.shape[1], dtype=np.float32)
                std = np.ones(data.shape[1], dtype=np.float32)
            else:
                mean, std = _mel_stats(data, meta, train_mask, seed=seed)
            self.mel_mean, self.mel_std = mean, std
            self.medians = None
            stats = {"mean": mean, "std": std}

        self.family = family
        self.level = level
        self.protocol = protocol
        self.config = config
        self.scope = scope
        self.kind = kind
        self.meta = meta
        self.y = y
        self.d = d
        self.domain_names = domain_names
        self.n_domain = len(domain_names)
        self.stats = stats
        self.n_in = None if kind != "vector" else X.shape[1]
        self.n_mels = None if kind != "mel" else int(data.shape[1])

        self.datasets = {}
        for role in ("train", "val", "test"):
            at = (meta.role == role).to_numpy()
            if not at.any():
                self.datasets[role] = None
                continue
            speakers = meta.speaker_id.to_numpy()[at]
            sids = meta.item_id.to_numpy()[at]
            cohorts = (meta.cohort.to_numpy()[at] if "cohort" in meta.columns
                       else np.array(["unknown"] * at.sum()))
            datasets = meta.dataset.to_numpy()[at]
            if kind == "vector":
                self.datasets[role] = VectorDataset(
                    X[at], y[at], d[at], speakers, sids, cohorts, datasets)
            else:
                self.datasets[role] = MelDataset(
                    data, meta.row_start.to_numpy()[at],
                    meta.n_valid_frames.to_numpy()[at],
                    mean, std, y[at], d[at], speakers, sids, cohorts, datasets)

    @classmethod
    def from_disk(cls, family, level, protocol, config, **kwargs):
        """The Stage 5 cache joined to a Stage 6 split file."""
        return cls(family, level, protocol, config, **kwargs)

    @classmethod
    def from_frames(cls, meta, X=None, data=None, level="frames",
                    scope="global", phone_class=None, dataset=None, tasks=None,
                    seed=0, shuffle_domains=False):
        """The same bundle from arrays already in memory.

        `meta` must carry `role`, `speaker_id`, `dataset`, `label`, `item_id`
        (and `row_start` / `n_valid_frames` for Mel), one row per item.
        """
        if (X is None) == (data is None):
            raise ValueError("pass exactly one of X (handcrafted) or data (mel)")
        kind = "vector" if X is not None else "mel"
        family = "handcrafted" if kind == "vector" else "mel"
        items = {"kind": kind, "meta": meta.reset_index(drop=True),
                 "X": X, "data": data}
        return cls(family, level, "frames", 0, scope=scope,
                   phone_class=phone_class, dataset=dataset, tasks=tasks,
                   seed=seed, items=items, shuffle_domains=shuffle_domains)

    def loader(self, role, batch_size, seed=0, samples_per_epoch=None,
               num_workers=0):
        ds = self.datasets[role]
        if ds is None:
            return None
        collate = collate_vector if self.kind == "vector" else collate_mel
        if role != "train":
            return DataLoader(ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=num_workers)
        meta_train = self.meta[self.meta.role == "train"]
        weights = speaker_weights(meta_train)
        n = samples_per_epoch or len(ds)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(n), replacement=True, generator=generator)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          collate_fn=collate, num_workers=num_workers,
                          drop_last=n >= 2 * batch_size)

    def summary(self):
        rows = []
        for role in ("train", "val", "test"):
            at = self.meta.role == role
            sub = self.meta[at]
            if sub.empty:
                continue
            rows.append({
                "role": role,
                "items": int(at.sum()),
                "speakers": sub.speaker_id.nunique(),
                "corpora": ",".join(sorted(sub.dataset.unique())),
                "pd": int((self.y[at.to_numpy()] == 1).sum()),
                "hc": int((self.y[at.to_numpy()] == 0).sum()),
            })
        return pd.DataFrame(rows)
