# Pipeline

Working specification for **Subtopic 4: Domain Adversarial Learning for
Dataset and Language Invariant Parkinson's Disease Detection**. It expands
the assigned pipeline (`pipeline.pdf`) into stages. Each stage lists what it
consumes, what it produces, what has to be decided, and when it can be called
done.

Stages are sequential in dependency but not in effort — Stage 3 (alignment) is
the largest single piece of work and the one with an unresolved design
question.

**Paper:** Ibarra et al. (2023), *Bioengineering* 10:1316 —
https://www.mdpi.com/2306-5354/10/11/1316

**Assigned corpora:** ItalianPD, NeuroVoz, MDVR-KCL (SharePoint folder
linked from `pipeline.pdf`). A fourth corpus, FredPrior, was bundled in
`PD.zip`; it is **out of scope** — sustained /a/ only, so none of the four
assigned tasks, and its 8 kHz bandwidth was the only bandwidth cue in the set.

**Assigned tasks:** spontaneous speech, reading passage, words, and phrases.

```
raw audio
   │
   ▼
[0] dataset registry ──────────► manifest.csv
   │
   ▼
[1] preprocessing ─────────────► processed/*.wav + manifest_processed.csv
   │
   ├──────────────────────────► [2] exploratory analysis  (baseline t-SNE)
   ▼
[3] alignment ─────────────────► interim/alignments/*.TextGrid | *.json
   │
   ▼
[4] segmentation ──────────────► segments.csv   (phoneme | word | duration | utterance)
   │
   ▼
[5] feature extraction ────────► features/*.npy  (spectrogram | handcrafted)
   │
   ▼
[6] splits ────────────────────► splits_mixed.csv | splits_lodo.csv
   │
   ▼
[7] training ──────────────────► baseline  →  DANN
   │
   ▼
[8] evaluation ────────────────► metrics + KL / TCM + t-SNE
```

---

## Stage 0 — Dataset registry and manifest

**In:** raw corpora under `data/raw/`
**Out:** `data/manifests/manifest.csv`
**Code:** `configs/datasets.yaml`, `src/data/build_manifest.py`

Produce a single index of every audio file across all three corpora, with
dataset, language, channel, speaker, label, and task attached.

**To do**

- [x] Symlink or copy each corpus into `data/raw/`
- [x] Inspect the actual directory layout of each corpus (`tree -L 3 | head -50`)
- [x] Update `rules` regexes and `task_map` in `configs/datasets.yaml` to match
- [x] Run `python -m src.data.build_manifest --dry-run` and iterate until:
      no unmatched files, no `task == unknown`
- [x] Cross-check speaker counts against published figures
      (NeuroVoz 112 or 108, ItalianPD 65, MDVR-KCL 37)
- [x] Resolve the NeuroVoz version question — 114 speakers have audio, but the
      metadata tables describe 108 (55 HC + 53 PD). Both published figures
      refer to the same copy, counted differently. Still to record in README.
- [ ] Commit the final `datasets.yaml`

**Decisions**

- Task vocabulary is shared across corpora. Where a corpus has a task no other
  corpus has, keep it in the manifest but expect it to be unusable for
  cross-corpus training.
- **FredPrior is excluded.** It contains none of the four assigned tasks, and
  dropping it leaves all three remaining corpora at an identical 8 kHz
  bandwidth after the Stage 1 resample — which retires the open question of
  whether every corpus needed low-passing to 4 kHz so that bandwidth was not a
  free cue for the domain discriminator. The cost is that language and channel
  are once again almost perfectly confounded. Its verified config block is kept
  commented out in `configs/datasets.yaml`.

**Done when** the summary table prints the expected speaker counts per
dataset × label, and every file has a known task.

---

## Stage 1 — Preprocessing

**In:** `manifest.csv`
**Out:** `data/processed/<dataset>/*.wav`, `manifest_processed.csv`
**Code:** `src/data/preprocess.py`, `configs/noise_blacklist.txt`

Assigned steps: resample to 16 kHz, normalise amplitude per file, remove
noise files manually. Implementation: downmix to mono, resample to 16 kHz,
peak-normalise. Raw audio is never modified.

**To do**

- [x] Run preprocessing on one corpus first (MDVR-KCL), verify the outputs
- [x] Spot-check outputs by ear — done by the user; no further exclusions
- [x] Add unusable files to `configs/noise_blacklist.txt` with a reason.
      Nine excluded, all on objective grounds: one duplicate file and eight
      NeuroVoz recordings too short to contain the prompted utterance. The
      blacklist also lists borderline files worth listening to. Stage 3 later
      found that alignment confidence detects this same defect better than
      duration does, and flagged more truncated recordings
- [x] Run on all three; confirm total hours per corpus look sane
      (3871 files, 12.26 h)
- [x] Check the duration distribution for outliers (near-zero, or far longer
      than the task should be) — these usually indicate segmentation or export
      errors in the source corpus

**Decisions**

- 16 kHz is fixed by the Wav2Vec2 aligner and SSL feature extractors. It
  discards content above 8 kHz, which matters little for articulation but does
  reduce fricative detail. Note this as a limitation rather than revisiting it.
- Peak normalisation only. Loudness normalisation (LUFS) would remove a real
  PD symptom — reduced vocal loudness — so it is deliberately not applied.

**Done when** every non-blacklisted file has a 16 kHz mono counterpart and
`manifest_processed.csv` is written.

---

## Stage 2 — Exploratory analysis

**In:** `manifest_processed.csv`
**Out:** figures + a short notes file in `notebooks/`

This stage exists to establish the problem on our own data before any modelling.

**To do**

- [x] Utterance duration distribution per dataset and per task
- [x] Class balance (PD/HC) per dataset; domain balance across datasets
- [x] Task availability matrix — **no raw task code exists in more than one
      corpus** except `read_text` and the vowels. Grouped into the four
      *assigned* tasks, however, each multi-corpus family spans exactly two
      corpora: reading passage (ItalianPD, MDVR-KCL), phrases (NeuroVoz,
      ItalianPD), spontaneous (NeuroVoz, MDVR-KCL), words (ItalianPD only).
      So a task-matched experiment is inherently 2-domain
- [x] Age distribution per dataset where metadata allows. Among the corpora in
      scope the young-control problem is confined to ItalianPD (15 of 32
      controls under 50, youngest patient 40); NeuroVoz is age-matched.
      **MDVR-KCL ships no age metadata at all**, so its age balance cannot be
      checked — a caveat on any LODO result that holds it out
- [x] **Baseline t-SNE of mean MFCCs coloured by dataset** —
      `notebooks/figures/tsne_mean_mfcc.png`. Corpus is predictable from the
      20 mean MFCCs at 0.892 accuracy (majority class 0.767) under a
      speaker-disjoint split; PD/HC only reaches 0.683

Code: `scripts/eda.py`. Output: `notebooks/eda_notes.md` and
`notebooks/figures/`.

The last item is the key deliverable. If raw acoustic features already cluster
by corpus — and given the channel differences they almost certainly will —
that reproduces the reference paper's motivating result on our data, before
training anything. It becomes the reference point that the DANN result is
measured against.

**Done when** the t-SNE plot exists and the task-availability matrix is
settled, since the latter constrains everything downstream.

---

## Stage 3 — Alignment

**In:** `manifest_processed.csv`
**Out:** `data/interim/alignments/` (TextGrid or JSON, per utterance)

Obtain phoneme- and word-level timestamps. The assigned pipeline is to pass
transcripts and audio through **MFA or a Wav2Vec2 aligner**, then cut
phoneme and word clips from those timestamps. This is the largest piece of
work and the main methodological departure from the reference paper.

**To do**

- [x] Decide the transcript strategy (see below) — **scripted tasks use
      published prompts; ASR only for MDVR-KCL spontaneous**
- [x] For scripted tasks, collect the canonical text of each read passage per
      corpus. ItalianPD `read_words` from the corpus; `read_text` / `read_phrases`
      from Dimauro et al. (IEEE Access 2017); MDVR-KCL `read_text` from the
      Zenodo appendix (North Wind and the Sun)
- [x] Set up MFA or a Wav2Vec2 aligner and run it on one corpus end to end —
      `torchaudio.pipelines.MMS_FA`, one aligner for all languages
- [x] Manually inspect ~20 alignments against the audio; measure boundary error
      qualitatively before trusting the output at scale
- [x] Record alignment failures in a rejection list rather than silently
      dropping them — `data/interim/alignment_rejects.csv`, 38 files (1.8%),
      rejected on low aligner confidence alone. Two other criteria were tried
      and discarded because they flagged properties of the prompt rather than
      defects in the recording; see the Stage 3 report
- [x] Run across all corpora with a transcript — **2101 aligned files**. First
      pass 1867 NeuroVoz + ItalianPD `read_words` in 3.3 min; follow-up added
      ItalianPD `read_text`/`read_phrases` and all of MDVR-KCL
- [x] Whisper `small.en` on the 36 MDVR-KCL spontaneous files (~7 min, local,
      MIT). Costantini et al. (Sensors 2024) diarized this corpus in MATLAB and
      did not release timestamps; pyannote is gated. **Diarization skipped.**
      ASR spontaneous mean score 0.381 — 19 of 27 aligned files rejected, 9
      more failed CTC. Not used at word/phoneme level. NeuroVoz low-coverage
      files were left as they are.

**The open decision — resolved, then narrowed.** Forced alignment needs a
transcript. Scripted tasks now all have one. Spontaneous speech still does not,
except for the cheap MDVR-KCL ASR pass, which confirmed the original objection
in numbers.

| Option | Cost |
|---|---|
| **Restrict alignment to what has text** ← default | First pass lost 243 files / 5.67 h, including all of MDVR-KCL |
| **Source published prompts** ← done | ItalianPD passages + MDVR-KCL North Wind: 207 files, no ASR |
| ASR first (Whisper), then align | Tried only on 36 MDVR-KCL dialogues. Alignment error compounds ASR error; interviewer speech is in the transcript; mean score 0.381 |
| Wav2Vec2 CTC alignment throughout | Language-agnostic and consistent across corpora; less accurate than MFA on scripted material |

The second-order problem still holds for any *language-dependent* ASR: if error
rate differs by language, the boundaries carry a domain confound. English-only
Whisper on one English corpus does not create that confound, but it *does*
include the interviewer, and the alignments are too poor to use.

**The consequence must be carried forward.** Phoneme level is still NeuroVoz and
ItalianPD only (English characters are not a phone set). Word level now includes
MDVR-KCL **read passages** (35 files after rejects). MDVR-KCL spontaneous is
duration/utterance only. A word-level LODO result on MDVR-KCL is a read-passage
result, not a dialogue result, and must not be reported as one.

**Done when** every retained utterance has phoneme and word boundaries, and
the failure rate per corpus is documented. **Done** — with the caveat that
MMS_FA emits characters rather than phones, so true phoneme labelling is
deferred to Stage 4. See `reports/stage3_alignment.md`.

---

## Stage 4 — Segmentation

**In:** alignments + processed audio
**Out:** `data/interim/segments.csv`

The assigned pipeline uses two segmentation methods, then extracts features
at five levels (four assigned, plus `duration_100` as a second window size).
One row per segment, carrying its parent utterance and speaker.

Methods:

- **Simple (fixed duration)** — fixed-length windows for sentence-level
  segmentation, and for very short clips that have no word or phoneme
  timestamps.
- **Alignment-based** — phoneme and word clips cut from MFA / Wav2Vec2
  timestamps (Stage 3).

Levels:

- **phoneme** — one segment per aligned phone (Spanish/Italian only)
- **word** — one segment per aligned word (includes MDVR-KCL read passages)
- **duration** — 250 ms windows, 125 ms hop (paper-equivalent baseline)
- **duration_100** — 100 ms windows, 50 ms hop (second size on the same axis)
- **utterance** — whole utterance

**To do**

- [x] Implement segment extraction from alignments. **The Stage 3 character
      spans could not be used as they stood**: they are CTC spikes at the
      model's 20 ms stride, not phone extents — 95% are exactly 20 ms and they
      cover only 39% of their parent word. Spikes are now used as boundary
      anchors, giving a median phone of 80 ms
- [x] Energy-trim phone tails to the same per-file −40 dB floor as `duration`,
      then apply a **200 ms safety cap**. The cap still binds at the 95th
      percentile; truncation remains the safer error
- [x] Decide minimum segment duration — **50 ms, uniform across levels**, three
      frames at the 25 ms window Stage 5 uses. Biases the phoneme level against
      short plosives; Stage 5 answers that with a 15 ms analysis window for
      that class only, not a lower floor
- [x] Decide a maximum too — **2 s for words**, dropping spans where the
      aligner bridged a pause between list items
- [x] MDVR-KCL windowing is silence-aware: windows more than 40 dB below their
      own file's peak are dropped, which removes **40.7% of its 250 ms windows**
      against 11.2% for NeuroVoz. Threshold is per file, never per corpus.
      Silence filter ≠ speaker filter: MDVR duration/utterance still mix in
      interviewer speech
- [x] Decide phoneme label handling across languages — **broad phonetic class**
      (vowel, plosive, fricative, affricate, nasal, liquid, glide) from
      per-language context rules with digraph merging and silent-letter
      dropping. The raw character is kept alongside it. English is word-only
- [x] Record segment counts per level per corpus — **1,106,469 segments** after
      the recut: `duration` 274,459 / `duration_100` 651,690 / `phoneme` 132,532
      / `word` 43,917 / `utterance` 3871. Word now covers 210 speakers (MDVR
      read included); phoneme still 175 (no English)

**Findings that change Stage 7.** Phone-class composition differs sharply by
corpus — Italian has 7x the affricate share, Spanish 2.2x the nasal share, and
glides occur only in Spanish — so **a phoneme-level model trained across pooled
classes could identify the corpus from the class mix alone**. Phoneme-level runs
must be trained *within* a phone class, which is what the assigned pipeline
already implies with "one DANN for the vowel-phoneme category". Note this is a
prompt difference as much as a language one and the two cannot be separated
with these corpora.

**Done when** `segments.csv` exists for all levels with speaker and
utterance provenance preserved on every row. **Done** — see
`reports/stage4_segmentation.md`. `word` / `phoneme` / `duration_100` feature
caches were rebuilt after the recut; `duration` and `utterance` windows did
not change.

---

## Stage 5 — Feature extraction

**In:** `segments.csv`
**Out:** `data/processed/features/` (4.15 GB)
**Code:** `src/features/extract.py`, `src/features/store.py`

```bash
python -m src.features.extract
python scripts/check_features.py --level word
python scripts/probe_normalisation.py --levels word phoneme
```

Two families:

- **Spectrogram** — Mel spectrogram per segment. Note the reference paper used
  40 ms analysis windows for sustained vowels and 15 ms for DDK, 10 ms hop,
  65 Mel bands, z-score normalised. Connected speech is closer to DDK in
  character, so start near 15–25 ms.
- **Handcrafted** — MFCC, jitter, shimmer, zero-crossing rate, short-term
  energy, F0, F1–F3, intensity. 40 features per segment, via Praat
  (`praat-parselmouth`) for everything except MFCC, ZCR and energy.

**To do**

- [x] Fix analysis window and hop per task; document the choice — **25 ms /
      10 ms, 65 bands, uniform across every level and task**, not varied per task
      as the paper did. One exception: **phoneme-level plosives use a 15 ms FFT
      (`PLOSIVE_N_FFT = 240`) at the same 10 ms hop**, so the frame grid is
      shared. A plosive DANN is not starved of frames; a table that pools
      plosives with vowels is partly a comparison of analysis settings.
      Handcrafted features stay on the 25 ms spectrogram. Also keeps faith
      with Stage 4's 50 ms floor
- [x] Extract both families at every segment level — all eight combinations
      cached, ~2 h on CPU, no GPU needed. Computed once per file and sliced per
      segment: 3871 spectrogram computations, not 374,195
- [x] Decide normalisation scope — **features are cached unnormalised and the
      scope is applied at load time**, so it is a Stage 7 knob rather than an
      irreversible commitment. Default **global, fitted on training rows only**.
      Per-speaker was tried first and *measured*: it takes PD/HC to chance at
      every level (0.679 → 0.510 at word level). PD/HC is constant within a
      speaker, so the label lives entirely in the between-speaker spread of
      feature means, and per-speaker z-scoring sets that spread to zero — 13.08
      to 1.9e-16. It removes the channel and the target together. Per-utterance
      fails identically, since utterances nest inside speakers. Removing channel
      is the adversary's job, not the normaliser's
- [x] Handle variable segment lengths — ragged contiguous storage with an index,
      so nothing is padded to a common length, plus **fixed 4 s chunks at a 2 s
      hop for the utterance level**. That last one is a leak fix, not a memory
      fix: utterance length is strongly corpus-dependent (3.5 s NeuroVoz against
      141 s MDVR-KCL), so a variable-length tensor would leak the domain label
      through its shape. 20,397 chunks from 3871 utterances, 2.81% padding, mask
      recoverable from `n_valid_frames`
- [x] Cache features to disk; do not recompute per epoch — `.npy` plus an index
      CSV per (family, level), joinable to `segments.csv` on `segment_id`
- [x] Record where voice quality is undefined — jitter, shimmer and formants need
      voiced signal with several pitch periods. NaN plus a `voice_quality_valid`
      flag rather than whatever Praat returns. Valid for 99.9% of utterances,
      99.0% of words, 94.4% of `duration` windows, 88.5% of phonemes

**Measured domain signal.** From the 40 handcrafted features, balanced accuracy
under speaker-grouped CV: corpus identity is easier to predict than pathology at
every level. `utterance` 0.938 corpus / 0.761 PD/HC, `duration` 0.757 / 0.738,
`word` 0.869 / 0.679, `phoneme` 0.804 / 0.652 (chance 0.333 for corpus at the
first two levels, 0.500 at the last two, which lack MDVR-KCL). The `duration`
level has the weakest corpus signal alongside nearly the strongest pathology
signal, making it the best level to attack first; the utterance level is the
hardest test of the adversary.

**Open items**

- [x] Stage 7 must fit the global statistics on **training-fold rows only**. The
      `fit_stats`/`apply` split exists for this; using all rows would leak.
      **Verified in code**: `src/training/data.py` fits on `X[train_mask]`,
      `_mel_stats` takes the same mask, NaN imputation uses the training
      median, and val/test speakers are asserted disjoint from train before
      any statistic is computed
- [x] Probe the Mel family for domain signal too — **measured**: 65 per-band
      means over each 250 ms `duration` window, same tree and folds, give corpus
      0.774 and PD/HC 0.744 against the 40 handcrafted features' 0.757 and
      0.738. Mel is more corpus-heavy, as expected, but only by 0.017 at the
      same pathology signal. This is a mean-pool result and says nothing about
      what the CNN sees on full spectrograms.
      `python scripts/probe_normalisation.py --levels duration --family mel-meanpool`
- [ ] `duration` features come from 50%-overlapping windows, so adjacent items
      share half their audio and are not independent samples

**Done when** every (feature family × segment level) combination is extracted
and cached. **Done** — see `reports/stage5_features.md`.

Per-corpus z-score and dropping corpus-heavy columns were tried as a
non-adversarial control. They do not replace DANN: see
`reports/ablation_corpus_norm.md`. Stage 7 stays on `norm=global` and all
40 handcrafted features.

The follow-up in the same report closes the shallow track. Talk figures:
`notebooks/figures/ablation_{moment_ladder,coral_corr,methods,tsne}.png`
(`python scripts/visualise_ablation.py`). A Gaussian-surrogate
moment ladder puts 79% of the corpus cue within reach of first and second
moments, yet **CORAL raises corpus identity to 0.855** and per-corpus z-score to
0.991 — both need the corpus label at apply time, so neither is LODO-valid or
comparable to `global`. The one method that is a single global map, iterative
null-space projection of the linearly corpus-predictive directions, leaves
corpus identity at **0.701** after deleting 9 of 40 dimensions. The cue is
non-linear, so DANN has no shallow baseline to beat on this data.

Two framing consequences carry into Stages 7 and 8. With **task and channel both
held fixed** — the five vowels, ItalianPD against NeuroVoz, both `clinical` —
corpus identity is still 0.821 against a 0.500 chance level, so the invariance
target is the corpus as a whole (language, site, speaker population, protocol)
and **not the recording channel**. And ItalianPD's young controls recorded only
`read_text` and `read_words`, never vowels, so an age-controlled vowel probe does
not exist; dropping them from `read_text` moves corpus identity from 0.807 to
0.836, i.e. the age confound is real but is not what separates the corpora.

---

## Stage 6 — Splits

**In:** `manifest_processed.csv`
**Out:** `data/manifests/splits_mixed.csv`, `data/manifests/splits_lodo.csv`
**Code:** `src/data/splits.py`

```bash
python -m src.data.splits --protocol mixed --k 10
python -m src.data.splits --protocol lodo
python scripts/check_splits.py
```

Splits are speaker-level and asserted leak-free. The assigned pipeline asks
for speaker-separated **train, validation, and test** splits:

- **Train** — PD + HC from multiple datasets, each sample tagged with its
  dataset and language.
- **Test** — PD + HC from each dataset kept separate (never pooled into one
  test number).

`mixed` is that protocol, stratified on `dataset × label` jointly. `lodo`
holds out one entire corpus and is the stricter generalisation check.

**To do**

- [x] Generate both, including a speaker-disjoint validation split — `mixed` has
      10 configurations over one stratified 10-fold partition, where fold *i* is
      test and fold *i+1* is validation, so validation never overlaps the test
      fold it selects against. `lodo` has 3, with 15% of the remaining speakers
      held for validation
- [x] Confirm fold composition is balanced across dataset × label — every test
      fold holds both classes of all three corpora: NeuroVoz 5–6 per class,
      ItalianPD 3–4 HC / 2–3 PD, MDVR-KCL 2–3 HC / 1–2 PD. The smallest stratum
      (MDVR-KCL PD, 16 speakers) is the ceiling on `k`, and `splits.py` refuses a
      larger one rather than emitting folds with no MDVR-KCL patients
- [x] Decide whether the ItalianPD young-control cohort is excluded —
      **decided: kept, with the confound documented**. The 15 young controls
      average 20.8 years against patients at 67, and every one is below the
      youngest patient, so age is an available shortcut (young controls vs
      patients separates at 0.905). Excluding them was measured to cost nothing
      (0.951 → 0.955 at utterance level), so inclusion buys no accuracy. `cohort`
      is carried in both split files so an elderly-only number can be reported
      without regenerating splits
- [x] Verify leak-freedom independently of the generator — `scripts/check_splits.py`
      re-checks the written files at **file** level, not just speaker level

**Two things later stages must respect.** Per-corpus test metrics cannot be
computed per fold: MDVR-KCL contributes 1–2 PD speakers to any single test fold.
Every speaker is in the test set of exactly one configuration, so Stage 8 must
pool out-of-fold predictions to get a per-corpus score over all 37 of its
speakers. And the three LODO configurations are not comparable experiments —
holding out NeuroVoz trains on 747 files and tests on 2968, while holding out
MDVR-KCL trains on 3190 and tests on 73 — so they must be reported separately
rather than averaged.

**LODO is strict**, meaning the held-out corpus appears nowhere in training, not
even as unlabelled audio. That makes it domain *generalisation*, not domain
adaptation; the transductive variant would score better and answer a different
question. Stage 8 should say which is being claimed.

**LODO is only fully meaningful at `utterance` and `duration` / `duration_100`.**
At `phoneme`, MDVR-KCL has no segments (English is word-only), so holding it
out is a no-op and holding out NeuroVoz leaves a single training corpus — a
domain adversary with one training domain has nothing to be adversarial about.
At `word`, MDVR-KCL now contributes the 35 accepted North Wind readings.
Holding it out is a real test, but it is a **read-passage** test, not a
spontaneous-dialogue one; `source=asr` rows are not in the word table.

**Done when** both split files exist and the leakage assertion passes.
**Done** — see `reports/stage6_splits.md`.

---

## Stage 7 — Training

**In:** features + splits
**Out:** `experiments/<run-id>/`

Two phases. Do not skip the first. A **separate model for every
feature × segmentation-type combination** (assigned pipeline). Example from
the spec: one DANN for the vowel-phoneme category using spectrogram features.

Input to models: handcrafted features or spectrograms. Training samples are
HC and PD from multiple datasets, each tagged with dataset and language.

**Which levels, and why.** The Scope note asks for one or two levels and
names utterance and phoneme. Three are run, and the third is the reason:

| level | why it is here |
|---|---|
| `duration` | Stage 5 measured it as the level with the **weakest corpus signal** (0.757) beside nearly the strongest pathology signal (0.738). It is where the adversary has the least free lunch, so it is the fair spine — but it is neither level the Scope note names, which is why it cannot be the only one |
| `utterance` | The Scope note's sane baseline, and by Stage 5's numbers the **hardest test of the adversary**: corpus identity is 0.938 predictable there. If DANN helps anywhere it should help most here |
| `phoneme`, vowel only | The assigned spec's own worked example — "one DANN for the vowel-phoneme category using spectrogram features" |

`word` and `duration_100` are cached but not trained: `word` adds no level the
three above do not already bracket, and `duration_100` is a window-size
ablation on an axis `duration` already occupies.

**Coverage is not symmetric across families, by decision.** All three levels
run on handcrafted (92 runs). Mel runs `duration` only: Mel `utterance` and Mel
`phoneme`/vowel were dropped after the handcrafted phase, at ~13 h of GPU for
levels the 40-feature family already covers end to end. So every *level*
claim rests on handcrafted, and the family comparison rests on `duration`.
Any claim that needs both at once is out of scope for this round.

**Phoneme is mixed-protocol only, and that is a property of the data, not an
omission.** Stage 6 states it: MDVR-KCL has no phoneme segments, so holding it
out is a no-op, and holding out NeuroVoz leaves a single training corpus — an
adversary with one domain has nothing to be adversarial about. Phoneme-vowel
is also a **2-domain** problem (ItalianPD + NeuroVoz, 175 speakers), so its
domain head is 2-way and its chance line is 0.50, not 0.33.

### 7a. Baseline (no domain adaptation)

Train the classifier alone, once per (feature × segment level) combination.
This establishes what the adversarial term has to improve on. Without it, any
DANN result is uninterpretable.

- [x] Backbone choice: MLP / CNN / transformer per the assigned pipeline. The
      reference paper found 1D-CNN with attention generalised best — a
      reasonable place to start. Spine is MLP (handcrafted) and 1D-CNN with
      masked attention (Mel). Transformer is implemented in the assigned
      menu, not in the spine.
- [x] Train on mixed corpora, evaluate per test corpus — `duration`,
      `utterance` and `phoneme`/vowel, both families
- [x] Also train single-corpus baselines, to reproduce the drop that occurs
      when corpora are pooled — `--regime single` over all ten mixed
      configurations for each of the three corpora. **Handcrafted only**: the
      same sweep on Mel is ~10 h of GPU to answer a question the 40-feature
      model already answers, and the drop being measured is a property of the
      corpora, not of the backbone

### 7b. DANN

Assigned architecture: a two-branch DANN. One branch increases PD vs. HC
accuracy; the other is the dataset/language classifier, trained so that
shared features become uninformative about domain (gradient reversal).
Either or both branches may be MLP, CNN, or transformer.

Add the domain branch and gradient reversal layer.

- [x] Implement GRL: identity forward, gradient × (−λ) backward
- [x] λ schedule — ramp from 0 rather than starting at full strength, or the
      adversary destabilises the shared trunk early in training
- [x] Weighted cross-entropy on **both** heads: corpora differ in size, and an
      imbalanced domain head biases the adversary toward the largest corpus
- [x] Log domain-head accuracy every epoch. It should start high (corpora are
      easily separable, especially MDVR-KCL) and fall toward chance. If it
      stays high, λ is too small; if PD accuracy collapses alongside it, λ is
      too large. Handcrafted DANN: ~0.64 → ~0.37 (chance 0.33). Five of ten
      mixed folds still saved a λ=0 checkpoint because val BA on 22 speakers
      never beat epoch 0
- [x] Domain label configuration: 3-way corpus head (language and channel are
      confounded; a separate language head would not be an independent signal).
      At `phoneme` it is necessarily 2-way — MDVR-KCL has no phonemes
- [x] **Checkpoint selection must not predate the adversary.** Early stopping
      on 21–22 validation speakers originally saved epoch 0 — where λ is still
      0 — in five of ten mixed DANN folds, so half the "DANN" numbers were
      scoring a pre-adversarial trunk. `min_select_epoch` (default 5, λ ≈ 0.68)
      opens checkpoint selection only once λ has ramped, and patience is
      counted from the same epoch. **It is applied to baseline as well**, which
      had the identical epoch-0 lottery, so both regimes are selected under one
      rule and stay comparable. Runs made before this rule are kept under
      `experiments/_pre_minselect/`

**Aggregation.** Segment-level predictions must combine into one decision per
speaker. Mean pooling, attention pooling, or majority vote — treat the choice
as a design axis, not a default.

**Chosen: mean-of-probabilities, then argmax** (`src/training/metrics.py`).
That is Ibarra's joint-probability aggregation in linear space. Scoring at
segment level instead would inflate every number, because one speaker
contributes hundreds of highly correlated windows.

**Treated as an axis, not assumed.** `scripts/probe_aggregation.py` measures
five alternatives against it on the finished runs — geometric mean of
probabilities (joint probability in log space), median, 10% trimmed mean,
majority vote, and a confidence-weighted mean over the quarter of segments
furthest from the boundary. The probe is **inference-only**: Stage 7 saves
`predictions.csv` already pooled, so it reloads `best.pt` and re-runs the test
split rather than retraining, and pools out-of-fold so each speaker is scored
once by a model that never trained on them. Results:
`reports/stage7_aggregation_<family>_<level>.csv`.

Attention pooling over segments is *not* compared. The Mel backbone already
carries masked attention pooling **within** a segment; a second attention layer
across a speaker's segments would have to be trained, which makes it a
different model rather than a different read-out of this one.

**Done when** baseline and DANN runs exist for the chosen feature × level
combinations under both split protocols. **Done** — 118 runs (92 handcrafted
across three levels, 26 Mel at `duration`), all under the corrected selection
rule, finished 2026-09-06. See `reports/stage7_training.md`.

The grid is one sequential GPU job, `scripts/run_stage7_grid.sh`, ordered so
that every level is covered at handcrafted cost (~1.5 h) before Mel is
started. Do not run a second job against the same GPU.

---

## Stage 8 — Evaluation

**In:** trained models
**Out:** metrics tables + figures

- [x] Classification: precision, recall, accuracy, ROC-AUC — reported
      **per test corpus and per segmentation level**, never pooled into one
      number. Both families at `duration`, three levels on handcrafted, in
      `reports/stage8_evaluation.md`
- [x] KL divergence between intra-class domain feature distributions, for each
      pair of corpora — directly comparable to the paper's Table 5
- [x] Trace of the covariance matrix of intra-class features — comparable to
      Table 6
- [x] t-SNE of the final-layer features, coloured by corpus and by class, for
      baseline vs. DANN. Compare against the Stage 2 baseline t-SNE
      (`notebooks/figures/tsne_handcrafted_duration_baseline_dann.png`)
- [x] **Decision rule for cross-corpus scoring.** A fixed 0.5 threshold calls
      83–96% of the held-out corpus PD when the true rate is 43–49%, costing
      0.089 mean balanced accuracy. A threshold tuned on *source* validation
      speakers transfers **worse than 0.5**, which identifies the problem as
      prior shift rather than a bad constant. LODO is therefore reported on
      **AUC** (threshold-free), with a prior-matched balanced accuracy beside
      it, declared transductive. See `scripts/probe_failure_modes.py`
- [x] Report mean and standard deviation across folds. Standard deviations in
      this literature are large (often 10–20 points); a mean difference smaller
      than the spread is not a result. Handcrafted mixed 0.820 ± 0.061
      (baseline) vs 0.812 ± 0.054 (DANN); Mel 0.839 ± 0.092 vs 0.765 ± 0.093 —
      neither gap clears its own spread. **LODO has no spread at all**: one run
      per held-out corpus, so those deltas carry no error bar

**The headline comparison** is not "DANN beats baseline on mixed corpora" — it
is whether DANN closes the gap under **leave-one-dataset-out**, on a corpus and
language the model has never seen.

**Answered, and the answer is backbone-dependent.** DANN improves all three
LODO hold-outs on the handcrafted MLP (+0.018 to +0.049) and degrades all three
on the Mel CNN (−0.023 to −0.175). KL falls on both families as Ibarra report,
but TCM inflates 7–280× rather than tightening: the adversary buys corpus
overlap by smearing the embedding, and the bigger trunk smears more. NeuroVoz
stays at 0.51–0.58 under every configuration. See
`reports/stage8_evaluation.md`.

---

## Invariants

These hold at every stage; violating any of them invalidates results.

1. **Splits are speaker-level.** Never file-level, never segment-level.
2. **No per-corpus statistics anywhere in the feature path.** Normalising by
   corpus statistics leaks the exact information the adversary is removing.
3. **Raw data is read-only.** Every stage writes to a new directory.
4. **Metrics are reported per test corpus and per segmentation level.** A
   pooled number hides the failure mode the project exists to study.
5. **Seeds are set and recorded.** Fold assignment, weight init, shuffling.
6. **One run = one directory** under `experiments/`, containing the resolved
   config and the git commit hash.

---

## Scope note

The full grid — 3 corpora × 4 tasks × 4 segment levels × 2 feature families ×
3 backbones × 2 training regimes × 2 split protocols — is far too large to run.
Pick a spine and go deep:

- one or two segment levels (utterance as the sane baseline, phoneme as the
  novel one)
- one feature family plus one alternative
- one backbone
- baseline vs. DANN
- mixed vs. LODO

Depth on a defensible question beats breadth across a grid that cannot be
interpreted.
