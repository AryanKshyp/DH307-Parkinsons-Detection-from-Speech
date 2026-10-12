"""Resolved configuration for one Stage 7 run.

One field, one default, one place. `configs/train.yaml` is a profile layered
over the dataclass defaults, and flags are layered over the profile. Nothing
downstream defaults a value again: `loop.py` and `classical.py` read
attributes off a resolved `TrainConfig` rather than calling `cfg.get(key,
fallback)` with a fallback of their own.

That rule is not stylistic. `min_select_epoch` was 5 in the yaml and 0 in
`loop.py`, so any entry point that did not come through `run.py` plus the yaml
silently got the value that saved five of ten mixed folds at lambda = 0 and
called the result DANN (see `reports/stage7_training.md`).

    cfg = TrainConfig.from_args(parse_args())
    cfg.run_id()            # experiments/<this>/
    cfg.epochs              # never cfg.get("epochs", 30)
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
import json

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = REPO_ROOT / "configs" / "train.yaml"
EXPERIMENTS = REPO_ROOT / "experiments"

# Fields the CLI exposes, and how argparse should read each one. A field with
# no entry here is settable from the yaml only. Adding a knob means adding a
# field plus one line in this table -- not four edits across three files.
_FLAGS = {
    "family": {"choices": ["handcrafted", "mel"]},
    "level": {},
    "protocol": {"choices": ["mixed", "lodo"]},
    "split": {"help": "mixed fold index, or held-out corpus for lodo"},
    "regime": {"choices": ["baseline", "dann", "single"]},
    "dataset": {"help": "restrict to one corpus (regime=single)"},
    "phone_class": {},
    "tasks": {"nargs": "+"},
    "norm": {"choices": ["none", "global", "speaker", "utterance", "corpus"]},
    "epochs": {"type": int},
    "batch_size": {"type": int},
    "lr": {"type": float},
    "weight_decay": {"type": float},
    "patience": {"type": int},
    "min_select_epoch": {"type": int,
                         "help": "no checkpoint is saved before this epoch"},
    "seed": {"type": int},
    "samples_per_epoch": {"type": int},
    "num_workers": {"type": int},
    "device": {"choices": ["auto", "cpu", "cuda"]},
    "width": {"type": int},
    "dropout": {"type": float},
    "lambda_max": {"type": float},
    "lambda_gamma": {"type": float},
    "max_steps": {"type": int, "help": "stop each epoch after this many steps"},
    "run_id": {"help": "override the computed experiment directory name"},
    "shuffle_domains": {
        "action": "store_true",
        "help": "DANN control: permute training corpus labels",
    },
}

# Smoke is a short loop to verify the plumbing, not an experiment. It forces
# its own values over whatever was resolved, and marks the run id so it cannot
# land on top of a real experiment directory.
_SMOKE = {"patience": 3, "min_select_epoch": 0, "max_steps": 20}


@dataclass(frozen=True)
class TrainConfig:
    """Every value one run needs, already resolved."""

    family: str = "handcrafted"
    level: str = "duration"
    protocol: str = "mixed"
    split: object = 0            # fold index (mixed) or corpus name (lodo)
    regime: str = "baseline"
    dataset: str = None
    phone_class: str = None
    tasks: list = None
    norm: str = "global"

    epochs: int = 30
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0
    patience: int = 8
    min_select_epoch: int = 5    # see the module docstring before lowering
    seed: int = 0
    samples_per_epoch: int = None
    num_workers: int = 0
    device: str = "auto"

    width: int = None            # None = 256 (mlp) or 128 (cnn)
    dropout: float = None        # None = 0.3 (mlp) or 0.0 (cnn)
    lambda_max: float = 1.0
    lambda_gamma: float = 10.0
    # Permute training corpus labels so the adversary regularises without
    # seeing which corpus a row came from. Val and test labels stay true.
    shuffle_domains: bool = False

    max_steps: int = None
    smoke: bool = False
    run_id: str = None           # explicit override; None means compute it

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_mapping(cls, values):
        """Build from a dict, rejecting keys this config does not have.

        Numbers are coerced here because yaml is the one untyped source: `lr:
        1e-3` parses as a string, and nothing downstream coerces any more.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise SystemExit(f"unknown config keys: {unknown}")
        values = dict(values)
        for spec in fields(cls):
            given = values.get(spec.name)
            if given is not None and spec.type in (int, float):
                values[spec.name] = spec.type(given)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path=DEFAULTS):
        with open(path, encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})

    @classmethod
    def from_run_dir(cls, path):
        """Read back the config a finished run under `experiments/` wrote.

        The counterpart of `resolved_run_id()`: every run directory already
        carries its own metadata, so nothing downstream has to re-spell the
        run-id grammar as an f-string to know what a directory is.

        Tolerant in both directions. `loop.py` writes provenance and shape keys
        that are not fields (`commit`, `n_in`, `n_mels`, `domain_names`,
        `target_stats`), and files written before a field existed are missing
        it (`max_steps`, `smoke`). `from_mapping` rejects unknown keys on
        purpose -- that is what keeps a typo in the yaml from being silently
        ignored -- so the filtering happens here rather than there.
        """
        path = Path(path)
        values = json.loads(
            (path / "config.json").read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        values = {name: value for name, value in values.items()
                  if name in known}
        # The 30 `*_single_*` runs recorded `regime: baseline` -- the value the
        # model needs (`regime_for_model`), not the one they were launched
        # with. A baseline restricted to one corpus *is* `single`, and no real
        # baseline run sets `dataset`, so this recovers the launched regime and
        # the recomputed run id agrees with the directory name again.
        if values.get("regime") == "baseline" and values.get("dataset"):
            values["regime"] = "single"
        # The stored name is dropped rather than carried: an override here
        # would survive a `replace(cfg, split=i)` and hand back a stale name,
        # which is exactly what `resolved_run_id` computes on demand to avoid.
        values.pop("run_id", None)
        return cls.from_mapping(values)

    @classmethod
    def from_args(cls, args, path=DEFAULTS):
        """yaml profile, then flags, then the smoke overrides."""
        base = cls.from_yaml(path)
        given = {name: getattr(args, name) for name in _FLAGS
                 if getattr(args, name, None) is not None}
        cfg = replace(base, **given)
        return cfg.as_smoke() if getattr(args, "smoke", False) else cfg

    def as_smoke(self):
        return replace(
            self, smoke=True,
            epochs=min(self.epochs, 3),
            batch_size=min(self.batch_size, 32),
            samples_per_epoch=int(self.samples_per_epoch or 512),
            **_SMOKE)

    # ---- derived values ---------------------------------------------------

    def resolved_run_id(self):
        """The experiment directory name.

        Computed on demand rather than stored, so that overriding one field --
        `--all-configs` sweeping `split`, say -- cannot leave a stale name
        behind. The `_smoke` marker is applied here for the same reason.
        """
        if self.run_id:
            return self.run_id
        parts = [self.protocol, str(self.split), self.family, self.level,
                 self.regime]
        if self.dataset:
            parts.append(str(self.dataset))
        if self.phone_class:
            parts.append(str(self.phone_class))
        # `classical.py` has always spelled the scope for its gbt runs,
        # including the default one. Those directories are finished and must
        # not be renamed, so the encoder reproduces `_norm-global_` for gbt
        # instead of the neural grammar's silence.
        if self.norm and (self.norm != "global" or self.regime == "gbt"):
            parts.append(f"norm-{self.norm}")
        if self.shuffle_domains:
            parts.append("shuffle-domains")
        parts.append(f"s{self.seed}")
        name = "_".join(parts)
        return f"{name}_smoke" if self.smoke else name

    @property
    def regime_for_model(self):
        """`single` is a baseline restricted to one corpus, so no domain head.

        `regime` itself keeps the value the caller asked for, so the written
        config and the directory name agree.
        """
        return "baseline" if self.regime == "single" else self.regime

    @property
    def target_stats(self):
        """Which rows the target corpus was normalised against, for the record."""
        if self.norm != "corpus":
            return None
        return "global_fallback" if self.protocol == "lodo" else "same_corpus_train"

    def to_dict(self):
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["run_id"] = self.resolved_run_id()
        out["target_stats"] = self.target_stats
        return out


def find_runs(**match):
    """Configs of the finished runs whose fields all equal `match`, by split.

        find_runs(protocol="mixed", family="handcrafted", regime="dann")

    Directories with no `config.json` are skipped: the `*_gbt_*` runs are
    written by `classical.py`, and `_logs`, `_pre_minselect` and
    `corpus_norm_ablation` are not single runs at all.
    """
    found = []
    for path in sorted(EXPERIMENTS.iterdir()):
        if not (path / "config.json").exists():
            continue
        cfg = TrainConfig.from_run_dir(path)
        if all(getattr(cfg, name) == value for name, value in match.items()):
            found.append(cfg)
    return sorted(found, key=lambda cfg: str(cfg.split))


def add_arguments(parser):
    """Generate one flag per `_FLAGS` entry, defaulting to None.

    None means 'not given', so a flag never shadows the yaml with a default of
    its own -- that shadowing is how four layers of defaults appeared.
    """
    for name, spec in _FLAGS.items():
        spec = dict(spec)
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name,
                            default=None, **spec)
    parser.add_argument("--smoke", action="store_true",
                        help="short run to verify the loop, not an experiment")
    return parser
