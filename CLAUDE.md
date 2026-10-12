# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo (IIT Bombay DH307, Subtopic 4), not a product. It reproduces and
extends Ibarra et al. (2023) — domain adversarial training (DANN) for
Parkinson's disease screening from speech across corpora that differ in
language, demographics, and recording channel. Three corpora: NeuroVoz
(Spanish), ItalianPD (Italian), MDVR-KCL (English, smartphone).

`README.md` is the project overview. `PIPELINE.md` is the working
specification — nine numbered stages, each with what it consumes, what it
produces, decisions taken and their measured justification. **Read the relevant
`PIPELINE.md` stage and `reports/stage<N>_*.md` before changing anything in
that stage.** Nearly every non-obvious choice in the code was decided by
measurement and written up there; re-deciding it from first principles will
usually re-introduce a bug that was already found and fixed.

Python only, no build step, no linter config, no test suite (`tests/` is
empty — verification lives in `scripts/check_*.py` instead).

## Commands

Environment (`.venv/` is a Windows venv already in the tree):

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

Pipeline stages, in dependency order. All are module entry points run from the
repo root; `scripts/*.py` are thin CLI wrappers over the same code.

```bash
python -m src.data.build_manifest --dry-run    # always dry-run first
python -m src.data.build_manifest
python -m src.data.preprocess
python -m src.data.transcripts
python -m src.data.align --dataset NeuroVoz --resume
python -m src.data.asr                         # MDVR-KCL spontaneous only (Whisper)
python -m src.data.segments                    # --levels word phoneme
python -m src.features.extract                 # --levels word --family mel
python -m src.data.splits --protocol mixed --k 10
python -m src.data.splits --protocol lodo
```

Verification (run these after regenerating the thing they check):

```bash
python scripts/check_splits.py                 # re-reads written split files, file-level leak check
python scripts/check_features.py --level word  # recomputes sample segments, compares to cache
python scripts/check_train.py                  # GRL/model sanity; --bundle needs features on disk
python scripts/benchmark_gpu.py
```

Training — defaults live on `TrainConfig` (`src/training/config.py`);
`configs/train.yaml` is a profile layered over them, and flags override both:

```bash
python -m src.training.run --dry-run           # joins what's on disk, prints speaker counts, exits
python -m src.training.run --family handcrafted --level duration --split 0
python -m src.training.run --all-configs --regime baseline
python -m src.training.run --all-configs --regime dann
python -m src.training.run --protocol lodo --split MDVR-KCL --regime dann
```

Evaluation and probes:

```bash
python scripts/evaluate.py --family handcrafted --level duration
python scripts/summarise_stage7.py
python scripts/probe_normalisation.py --levels word phoneme
python scripts/probe_feature_importance.py
python scripts/eda.py
```

## Architecture

Data flows one direction through immutable stages, each keyed by a CSV that
downstream stages join against — no stage walks raw directories.

```
configs/datasets.yaml → manifest.csv → processed/*.wav + manifest_processed.csv
    → alignments/*.json → segments.csv → features/*.npy + *_index.csv
    → splits_{mixed,lodo}.csv → experiments/<run-id>/ → reports/stage8_*
```

- `src/data/` — one module per stage. `configs/datasets.yaml` is a dataset
  registry (root, language, channel, glob, ordered named-capture regex `rules`,
  `task_map`); adding a corpus means editing YAML, not code.
- `src/features/` — `extract.py` computes each *file* once and slices segments
  out of it, writing `<family>_<level>.npy` plus a joinable index CSV.
  `store.py` applies the normalisation scope **at load time**, with
  `fit_stats`/`apply` deliberately separable so statistics are fitted on
  training rows only.
- `src/models/` — `backbones.py` (`mlp` for the 40 handcrafted features, `cnn`
  with masked attention pooling for Mel), `grl.py` (gradient reversal +
  Ganin λ schedule), `classifier.py` (trunk + PD head + optional domain head;
  baseline passes `n_domain=None` so the head is not allocated at all).
- `src/training/` — `data.py` joins splits to features with speaker-balanced,
  `dataset × label`-stratified sampling; `loop.py` trains one run and writes an
  experiment directory; `run.py` is the CLI and builds the run id.
  `config.py` holds `TrainConfig`, which resolves `configs/train.yaml` and the
  flags into one value. **Every default lives there and nowhere else** — no
  module may re-default a key with `cfg.get(key, fallback)`. `min_select_epoch`
  was 5 in the yaml and 0 in `loop.py`, and that gap is what saved five of ten
  mixed folds at λ = 0 (see `reports/stage7_training.md`).
- `src/evaluation/` — `pool.py` concatenates the test role across the 10 mixed
  configurations (out-of-fold pooling), `geometry.py` does KL / covariance
  trace / t-SNE, `tables.py` formats.

One run = one directory `experiments/<protocol>_<split>_<family>_<level>_<regime>[_<extras>]_s<seed>/`
containing `config.json`, `commit.txt`, `predictions.csv`, `embeddings.csv`,
`metrics.csv`, `summary.json`, `checkpoints/`. Outputs and checkpoints are
git-ignored; the configs are tracked.

## Invariants

Violating any of these invalidates results (see `PIPELINE.md` § Invariants):

1. **Splits are speaker-level.** Never file- or segment-level. A speaker
   contributes thousands of correlated segments; `splits.py` asserts this and
   `check_splits.py` re-checks the written files independently.
2. **No per-corpus (or per-speaker, or per-utterance) statistics in the feature
   path.** Per-speaker z-scoring was measured to take PD/HC to chance — the
   label lives in the between-speaker spread of feature means. `norm=corpus`
   exists only as a documented ablation.
3. **Raw data is read-only.** Every stage writes to a new directory.
4. **Metrics are per test corpus and per segment level, never pooled into one
   number.** Mixed per-corpus scores must come from pooling out-of-fold
   predictions across all 10 configurations — MDVR-KCL contributes 1–2 patients
   to any single fold. The three LODO configurations are not comparable to each
   other and must not be averaged.
5. **Seeds are set and recorded** in the run config, along with the git commit.

## Gotchas that bite

- `data/` is git-ignored in full and contains speaker identifiers. Never commit
  audio, alignments, manifests, or split files.
- Corpus coverage is uneven by level and this must be carried into any claim:
  phoneme level is NeuroVoz + ItalianPD only (MMS_FA emits characters, not
  phones, and English orthography is not shallow); MDVR-KCL reaches word level
  only through its 35 accepted read-passage files, so a word-level LODO result
  on it is a read-passage result, not a dialogue one.
- Phone-class composition differs sharply by corpus, so **phoneme-level runs
  must set `--phone-class`** — a model trained across pooled classes can
  identify the corpus from the class mix alone.
- Language and channel are perfectly confounded (one recording setup per
  language), so no claim about *language* invariance stands without that caveat.
- ItalianPD's young-control cohort is a known age confound, deliberately kept
  and documented; `cohort` is carried through the manifest and both split files
  so an elderly-only number can be reported without regenerating anything.
- `notebooks/` is for looking at things. Nothing in the pipeline may depend on it.
- `tmp_deck/`, `tmp_week2/`, `tmp_week3*/`, `Presentations/` are slide-building
  scratch, unrelated to the pipeline.

## Working style

`AGENTS.md` carries the behavioural guidelines for this repo: state assumptions
before implementing, minimum code that solves the problem, surgical changes that
trace to the request, and verifiable success criteria per step.
