# Corpus- and Language-Invariant Parkinson's Detection from Speech

Domain adversarial training for Parkinson's disease screening across corpora
that differ in language, demographics, and recording channel.

**DH307 · Subtopic 4** — Aryan Kashyap (23B0741)
Supervisor: Prof. Nirmal Punjabi · Guide: Mayurakshi Mukherji

Stages 0–8 are complete. **[`RESULTS.md`](RESULTS.md) is the standalone summary
of experiments and findings**; [`PIPELINE.md`](PIPELINE.md) is the working
specification; `reports/stage<N>_*.md` records what each stage decided and
measured.

---

## Problem

Parkinson's speech corpora are small (typically under 100 speakers), so models
are trained on pooled datasets. When corpora are pooled, a network can reduce
its training loss by learning **which dataset** a recording came from —
language, microphone, room acoustics — rather than learning the pathology.
The learned embedding then clusters by corpus rather than by class, and
cross-dataset precision drops by more than 20 absolute points.

The problem is real on this data before any training: corpus identity is
predictable from 20 mean MFCCs at **0.892** balanced accuracy under a
speaker-disjoint split, while PD/HC reaches only **0.683**.

This project applies **domain adversarial training (DANN)**: a shared feature
extractor feeds a PD classifier (trained to succeed) and a corpus classifier
(trained, via a gradient reversal layer, to fail). At equilibrium the
representation retains PD-discriminative information while discarding corpus
identity. The headline question is whether DANN closes the gap under
**leave-one-dataset-out (LODO)** — a corpus, language and microphone the model
has never seen.

Reference: Ibarra, Arias-Londoño, Zañartu & Godino-Llorente (2023),
*Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease
from Voice and Speech through Domain Adaptation*, **Bioengineering** 10, 1316.
[doi:10.3390/bioengineering10111316](https://doi.org/10.3390/bioengineering10111316)

### How this project differs from the reference

| | Reference paper | This project |
|---|---|---|
| Corpora | PD-GITA, PD-Neurovoz, PD-Czech, PD-German | NeuroVoz, ItalianPD, MDVR-KCL |
| Tasks | Sustained vowel, DDK | + reading passages, words, phrases, spontaneous |
| Segmentation | Blind 400 ms windows | **Forced alignment** (`torchaudio` MMS_FA) |
| Segment levels | One | **Phoneme, word, duration, utterance** |
| Features | Mel spectrograms | + 40 handcrafted (MFCC, jitter, shimmer, ZCR, STE, F0, F1–F3, intensity) |
| Domain label | Corpus | Corpus (3-way; language and channel are confounded) |
| Evaluation | 10-fold CV, mixed corpora | Mixed **and** leave-one-dataset-out |

---

## Headline result

**DANN does not reliably help.** It does not improve pooled (mixed-protocol)
accuracy at any segment level or in either feature family. Under LODO, the sign
of its effect flips with the backbone: it improves cross-corpus AUC on the
44k-parameter MLP (handcrafted features) and degrades it on the 224k-parameter
1D-CNN (Mel), so the effect is a backbone interaction rather than a method
result. The Mel CNN baseline is the strongest configuration (0.838 pooled BA,
0.915 AUC).

Two further findings carry most of the analysis:

- **Gradient reversal inflates the embedding norm** — 3.8× on the MLP, 14.9× on
  the CNN — and the damage tracks the inflation. After L2-normalising, DANN
  does reduce intra-class corpus KL and tighten classes, so both of Ibarra's
  geometric claims (Tables 5–6) reproduce.
- **A fixed 0.5 threshold is invalid under corpus shift.** Under LODO the model
  calls 83–96% of the held-out corpus PD. A threshold tuned on source
  validation speakers transfers *worse* than 0.5, the signature of prior shift.
  AUC is therefore the headline LODO metric.

Full tables, controls and caveats are in [`RESULTS.md`](RESULTS.md).

---

## Repository layout

```
.
├── configs/
│   ├── datasets.yaml           # dataset registry: roots, parsing rules, task maps
│   ├── prompts.yaml            # published prompt text for scripted tasks
│   ├── noise_blacklist.txt     # files excluded during preprocessing
│   └── train.yaml              # training profile layered over TrainConfig defaults
├── data/                       # not in the repo; created locally
│   ├── raw/                    # corpora (symlinks are fine)
│   ├── interim/                # alignments, segment tables, reject lists
│   ├── processed/              # 16 kHz mono normalised audio + features/
│   └── manifests/              # manifest.csv, splits_*.csv
├── src/
│   ├── data/                   # manifest, preprocessing, transcripts, alignment, ASR, segments, splits
│   ├── features/               # Mel + handcrafted extraction; load-time normalisation
│   ├── models/                 # MLP / CNN backbones, gradient reversal, PD + domain heads
│   ├── training/               # TrainConfig, data joining, training loop, CLI
│   └── evaluation/             # out-of-fold pooling, KL / TCM / t-SNE, tables
├── scripts/                    # thin CLI wrappers, verification checks, probes, plots
├── reports/                    # one report per pipeline stage + stage 8 CSVs
├── experiments/                # one directory per run; corpus_norm_ablation/
├── notebooks/                  # exploration notes and figures only
├── Literature Review/          # reference paper and the assigned pipeline spec
├── PIPELINE.md                 # working specification, stages 0–8
└── RESULTS.md                  # standalone summary of findings
```

The rule of thumb: `src/` holds importable logic, `scripts/` holds entry
points, `notebooks/` holds nothing that anything else depends on.

---

## Setup

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

The default PyPI `torch` wheels are CPU-only on Windows; GPU alignment and
training need the CUDA build from the PyTorch index. `openai-whisper` is only
needed for the optional MDVR-KCL ASR pass.

**The data is not included in this repository.** Download the three corpora
from the links under *Data availability* below and place them (or symlinks to
them) under `data/raw/`:

```bash
ln -s /path/to/NeuroVoz   data/raw/NeuroVoz
ln -s /path/to/ItalianPD  data/raw/ItalianPD
ln -s /path/to/MDVR-KCL   data/raw/MDVR-KCL
```

Every other directory under `data/` is regenerated by the pipeline below.

---

## Pipeline

Data flows one direction through immutable stages, each keyed by a CSV that the
next stage joins against — no stage walks raw directories. All stages are
module entry points run from the repo root. Stage numbers match `PIPELINE.md`.

```
configs/datasets.yaml → manifest.csv → processed/*.wav + manifest_processed.csv
    → alignments/*.json → segments.csv → features/*.npy + *_index.csv
    → splits_{mixed,lodo}.csv → experiments/<run-id>/ → reports/stage8_*
```

### 0. Build the manifest

Adding a corpus means adding a block to `configs/datasets.yaml`, not editing
code.

```bash
python -m src.data.build_manifest --dry-run   # inspect matches, write nothing
python -m src.data.build_manifest             # write data/manifests/manifest.csv
```

Always run `--dry-run` first. It reports how many files matched, which files
matched no rule, and which task tokens are unmapped. Each dataset block has:

- `root`, `language`, `channel` — constants attached to every file
- `glob` — which files to consider
- `rules` — ordered regexes with named capture groups; first match wins.
  Fields interpolate captures, e.g. `speaker_id: "neurovoz_{spk}"`
- `task_map` — maps raw filename tokens onto the shared task vocabulary

Manifest columns: `path, dataset, language, channel, speaker_id, label, task,
cohort, sample_rate, duration, n_channels`.

### 1. Preprocess

Downmix to mono, resample to 16 kHz, peak-normalise, drop blacklisted files.
Raw audio is never modified; outputs go to `data/processed/`.

```bash
python -m src.data.preprocess
```

Loudness (LUFS) normalisation is deliberately *not* applied — reduced vocal
loudness is a genuine PD symptom. Add noisy files to
`configs/noise_blacklist.txt` (one repo-relative path per line) and re-run, so
the manual noise-removal decision lives in version control.

### 2. Exploratory analysis

```bash
python scripts/eda.py
```

Durations, class/domain balance, task availability, and the baseline t-SNE of
mean MFCCs coloured by corpus that establishes the corpus-clustering problem
before any training. Output in `notebooks/eda_notes.md` and `notebooks/figures/`.

### 3. Force-align the transcribed material

```bash
python -m src.data.transcripts                       # collect prompt text per file
python -m src.data.align --dataset NeuroVoz --resume
python -m src.data.asr                               # optional: MDVR-KCL spontaneous (Whisper)
```

One aligner for all languages (`torchaudio.pipelines.MMS_FA`), so alignment
quality is not itself a corpus variable. NeuroVoz ships a verbatim transcript
per recording; ItalianPD and MDVR-KCL scripted tasks use published prompts
(`configs/prompts.yaml`). 2101 files are aligned; 38 (1.8%) are rejected on
aligner confidence and listed in `data/interim/alignment_rejects.csv` rather
than silently dropped.

Whisper ASR was tried on the 36 MDVR-KCL spontaneous files and **rejected on
measurement** — mean alignment score 0.381, and the transcript contains the
interviewer. MDVR-KCL spontaneous speech is therefore duration/utterance level
only.

### 4. Cut segments

```bash
python -m src.data.segments                    # --levels word phoneme
```

Four levels — `utterance`, fixed-`duration` windows, `word`, `phoneme` —
1,106,469 segments in total, all carrying speaker, label and corpus provenance.
The aligner's character spans are CTC spikes (95% exactly 20 ms) rather than
phone extents, so they are used as boundary anchors; see
`reports/stage4_segmentation.md`.

### 5. Extract features

```bash
python -m src.features.extract                 # --levels word --family mel
python scripts/check_features.py --level word  # recompute samples, compare to cache
```

Mel spectrograms (65 bands, 25 ms / 10 ms) and 40 handcrafted descriptors.
Each file is computed once and segments are sliced out of it. **Features are
cached unnormalised**; the normalisation scope is applied at load time by
`src/features/store.py`, with statistics fitted on training rows only.

Normalisation is global. Per-speaker z-scoring was measured to take PD/HC to
chance (0.679 → 0.510 at word level): the label lives in the between-speaker
spread of feature means, and per-speaker normalisation sets that spread to zero.

### 6. Generate splits

```bash
python -m src.data.splits --protocol mixed --k 10   # pooled, stratified
python -m src.data.splits --protocol lodo           # leave-one-dataset-out
python scripts/check_splits.py                      # verify the written files
```

**Splits are assigned per speaker, never per file.** A single speaker
contributes thousands of highly correlated segments; a speaker on both sides of
a split inflates every metric downstream. `splits.py` asserts this and
`check_splits.py` re-checks the written files independently.

`mixed` stratifies on `dataset × label` jointly, so folds preserve both class
balance and corpus balance. `lodo` holds out one entire corpus — it appears
nowhere in training, not even as unlabelled audio — which makes it domain
*generalisation* rather than domain adaptation.

### 7. Train

Defaults live on `TrainConfig` (`src/training/config.py`); `configs/train.yaml`
is a profile layered over them, and flags override both.

```bash
python -m src.training.run --dry-run           # join what's on disk, print speaker counts, exit
python -m src.training.run --family handcrafted --level duration --split 0
python -m src.training.run --all-configs --regime baseline
python -m src.training.run --all-configs --regime dann
python -m src.training.run --protocol lodo --split MDVR-KCL --regime dann
python scripts/check_train.py                  # GRL / model sanity checks
```

Backbones: an MLP (44k parameters) for the 40 handcrafted features and a 1D-CNN
with masked attention pooling (224k) for Mel. The GRL is identity forward and
gradient × (−λ) backward, with the Ganin λ ramp (γ = 10). Checkpoint selection
opens only at `min_select_epoch` (default 5, λ ≈ 0.68), for baseline and DANN
alike — without it, half the mixed DANN folds selected a pre-adversarial
epoch-0 checkpoint (see `reports/stage7_training.md`).

Phoneme-level runs must set `--phone-class` (e.g. `vowel`): phone-class
composition differs sharply by corpus, so a model trained across pooled classes
can identify the corpus from the class mix alone.

Each run writes `experiments/<protocol>_<split>_<family>_<level>_<regime>[_<extras>]_s<seed>/`
containing `config.json`, `commit.txt`, `metrics.csv`, `summary.json`,
speaker-level `predictions.csv`, `embeddings.csv` and `checkpoints/`.

### 8. Evaluate

```bash
python scripts/evaluate.py --family handcrafted --level duration
python scripts/summarise_stage7.py
python scripts/plot_stage8.py
```

Accuracy, precision, recall, ROC-AUC; intra-class KL and covariance trace (TCM)
between corpora; t-SNE. Per-corpus mixed scores pool out-of-fold predictions
across all 10 configurations — MDVR-KCL contributes only 1–2 patients to any
single fold. The three LODO configurations are separate experiments and are
never averaged.

Probes and ablations behind specific decisions:
`scripts/probe_normalisation.py`, `probe_aggregation.py`,
`probe_feature_importance.py`, `probe_failure_modes.py`, and the
corpus-normalisation ablation in `experiments/corpus_norm_ablation/`.

---

## Datasets

| | NeuroVoz | ItalianPD | MDVR-KCL |
|---|---|---|---|
| Language | Castilian Spanish | Italian | English |
| Speakers with audio | 114 (56 PD / 58 HC) | 65 (28 PD / 37 HC) | 37 (16 PD / 21 HC) |
| Files | 2968 | 830 | 73 |
| Tasks | Vowels, DDK, listen-and-repeat, monologue | Read passages, vowels, words | Read passages, spontaneous dialogue |
| Recording | Clinical, patients in ON state | 44.1 kHz/16-bit, echo-free room, mic 15–25 cm | 44.1 kHz, Motorola Moto G4, phone-call posture |
| Labels | GRBAS, perceptual | Hoehn & Yahr | H&Y, UPDRS II-5, UPDRS III-18 |

3871 files, 12.26 h, 216 speakers after preprocessing. A fourth corpus
(FredPrior) is out of scope: sustained /a/ only, and its 8 kHz bandwidth was
the only bandwidth cue in the set.

### Known issues

**Channel shift is severe.** Two clinical corpora against one smartphone corpus
recorded in a room with roughly 500 ms reverberation. Expect the domain
discriminator to separate MDVR-KCL trivially in early training.

**Language and channel are perfectly confounded.** Each language appears in
exactly one recording setup, so removing "domain" cannot be attributed to
removing language rather than removing channel. Any claim about *language*
invariance needs this caveat stated. This is also why the domain head is a
single 3-way corpus head rather than separate dataset and language heads.

**Coverage is uneven by level.** Phoneme level is NeuroVoz and ItalianPD only —
MMS_FA emits characters, not phones, and English orthography is not shallow.
MDVR-KCL reaches word level only through its 35 accepted read-passage files, so
a word-level LODO result on MDVR-KCL is a read-passage result, not a dialogue
one. Mel features were trained at `duration` level only.

**ItalianPD has an age confound.** The control group is 15 speakers aged 19–29
plus 22 aged 60–77, while patients are 40–80 — every young control is below the
youngest patient. The young cohort is **kept** and documented: excluding it was
measured to change nothing at the headline level. `cohort` is carried through
the manifest and both split files, so an elderly-only number can be reported
without regenerating anything.

**ItalianPD has a recording-protocol shortcut.** PD/HC is 0.883 AUC predictable
from total recording seconds alone, which partly explains why ItalianPD sits at
ceiling in every configuration. NeuroVoz is the corpus whose scores move.

**MDVR-KCL ships no age metadata**, so its age balance cannot be checked.

**NeuroVoz speaker counts.** 114 speakers have audio, while the clinical
metadata tables describe 108; the published figures of 112 and 108 refer to the
same material counted differently. Counts here are speakers *with audio*.

### Data availability

The corpora are **not** included in this repository; `data/` is git-ignored.
Each is distributed by its authors under its own terms:

| Corpus | Source | Notes |
|---|---|---|
| NeuroVoz | [Zenodo 10777657](https://zenodo.org/records/10777657) · [GitHub](https://github.com/BYO-UPM/Neurovoz_Dababase) | Mendes-Laureano et al., *Scientific Data* 11:1367 (2024). CC BY-NC-ND 4.0 |
| ItalianPD | [IEEE DataPort](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech) · [doi:10.21227/aw6b-tg17](https://doi.org/10.21227/aw6b-tg17) | Dimauro & Girardi; recordings plus `.xlsx` metadata in one zip |
| MDVR-KCL | [Zenodo 2867216](https://zenodo.org/records/2867216) | Mobile Device Voice Recordings at King's College London, 2019 |

Do not commit audio, alignments, manifests or split files. The per-run
`predictions.csv` and `embeddings.csv` that are committed carry the corpora's
own pseudonymous speaker identifiers.

---

## Open questions

1. Every result is single-seed (`s0`), and each LODO cell is one run with no
   error bar.
2. Gradient reversal inflates the embedding norm and the damage tracks the
   inflation. Constraining the norm (L2 + temperature, or LayerNorm before the
   heads) is the obvious fix and is **untested**.
3. λ is pinned at λ_max = 1.0 and was never swept; the Mel CNN clearly
   over-adapts.
4. Vanilla DANN aligns marginals, not class-conditionals; CDAN or MMD-based
   adaptation is untried.
5. A fixed 0.5 threshold is invalid under corpus shift. AUC and a
   prior-matched threshold are reported instead; the latter is transductive.

---

## Conventions

- Splits are speaker-level. Always.
- No per-corpus, per-speaker or per-utterance statistics in the feature path;
  `norm=corpus` exists only as a documented ablation.
- Raw data is read-only; every stage writes to a new directory.
- Metrics are reported per test corpus and per segment level, never pooled into
  one number.
- One experiment = one directory under `experiments/`, containing the resolved
  config, git commit hash, and outputs.
- Seeds are set explicitly and recorded in the run config.
- `notebooks/` is for looking at things, not for producing artefacts anything
  else depends on.
