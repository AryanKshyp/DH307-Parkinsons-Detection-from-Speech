"""Stage 7 training entry point.

    python -m src.training.run --dry-run
    python -m src.training.run --family handcrafted --level duration --split 0
    python -m src.training.run --regime dann --protocol lodo --split MDVR-KCL
    python -m src.training.run --all-configs --regime baseline
    python -m src.training.run --regime dann --shuffle-domains --split 0

Does not extract features or write splits. If those files are still being
rewritten, `--dry-run` is the thing to call: it joins whatever is on disk and
prints the resulting speaker counts, then exits.

Values come from `TrainConfig` (`src/training/config.py`), which resolves the
yaml profile and the flags once. This module does not default anything.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from src.training.config import DEFAULTS, TrainConfig, add_arguments
from src.training.data import Bundle, configs_for
from src.training.loop import train

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED = ("family", "level", "protocol", "split", "regime")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defaults", type=Path, default=DEFAULTS)
    add_arguments(ap)
    ap.add_argument("--all-configs", action="store_true",
                    help="run every configuration of the protocol in sequence")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the bundle, print counts, do not train")
    return ap.parse_args(argv)


def one_run(cfg, dry_run=False):
    if cfg.regime == "single" and not cfg.dataset:
        raise SystemExit("regime=single requires --dataset")
    if cfg.shuffle_domains and cfg.regime != "dann":
        raise SystemExit(
            "--shuffle-domains is the DANN control; regime must be dann")

    print(f"\n=== {cfg.protocol} split={cfg.split} "
          f"{cfg.family} {cfg.level} {cfg.regime} ===")
    bundle = Bundle(
        family=cfg.family,
        level=cfg.level,
        protocol=cfg.protocol,
        config=cfg.split,
        scope=cfg.norm,
        phone_class=cfg.phone_class,
        dataset=cfg.dataset,
        tasks=cfg.tasks,
        seed=cfg.seed,
        shuffle_domains=cfg.shuffle_domains,
    )
    print(bundle.summary().to_string(index=False))
    print(f"domains in train: {bundle.domain_names}")
    if cfg.shuffle_domains:
        print(f"shuffled train domain labels agree with the corpus on "
              f"{bundle.domain_agreement:.3f} of items "
              f"(chance {bundle.domain_chance:.3f})")
    if dry_run:
        return None

    out_dir = REPO_ROOT / "experiments" / cfg.resolved_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict() | {
            "commit": _git_commit(),
            "n_in": bundle.n_in,
            "n_mels": bundle.n_mels,
            "domain_names": bundle.domain_names,
        }, indent=2, default=str), encoding="utf-8")
    (out_dir / "commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    print(f"writing {out_dir.relative_to(REPO_ROOT)}")
    return train(bundle, cfg, out_dir)


def main(argv=None):
    args = parse_args(argv)
    cfg = TrainConfig.from_args(args, args.defaults)

    if args.all_configs:
        missing = [k for k in REQUIRED if k != "split"
                   and getattr(cfg, k) is None]
        if missing:
            raise SystemExit(f"--all-configs needs {missing} set in yaml or flags")
        values = configs_for(cfg.protocol)
        print(f"{len(values)} configurations: {values}")
        for value in values:
            # run_id is cleared so each configuration computes its own name;
            # an explicit --run-id would otherwise put every fold in one
            # directory. The smoke marker survives, because it is applied
            # when the name is resolved rather than baked into it.
            one_run(replace(cfg, split=value, run_id=None), dry_run=args.dry_run)
        return

    missing = [k for k in REQUIRED if getattr(cfg, k) is None]
    if missing:
        raise SystemExit(f"missing {missing}; set them in {args.defaults} or flags")
    one_run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
