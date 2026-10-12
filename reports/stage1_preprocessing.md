# Stage 1 — Preprocessing

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

Every manifest entry was downmixed to mono, resampled to 16 kHz and
peak-normalised, leaving **3871 files (12.26 hours)** in `data/processed/`.
Nine files are excluded on documented grounds. Every output was verified as
16 kHz mono PCM_16 with no missing file, no orphan file and no name collision.
The stage also surfaced a recording-setup confound inside ItalianPD that is
aligned with the class label.

**In:** `data/manifests/manifest.csv`
**Out:** `data/processed/<dataset>/*.wav`, `data/manifests/manifest_processed.csv`
**Code:** `src/data/preprocess.py`, `configs/noise_blacklist.txt`

---

## What was done

`librosa.load(sr=16000, mono=True)` for resampling and downmix, then peak
normalisation to 0.99 and a 16-bit PCM write. Raw audio is never touched.

Two implementation decisions are worth stating:

**Output naming is `<speaker_id>__<task>__<source stem>.wav`.** The task has to
be in the name because MDVR-KCL reuses the same basename in its `ReadText` and
`SpontaneousDialogue` directories; a naive flatten would have silently
overwritten half that corpus. The manifest keeps `source_path` alongside `path`
so provenance survives the rename.

**Peak normalisation only — no loudness (LUFS) normalisation.** Reduced vocal
loudness is a symptom of Parkinson's disease, not a recording artefact.
Normalising it away would remove signal the classifier is supposed to use.

**16 kHz for all three corpora.** The rate is fixed by the Wav2Vec2 aligner and
the SSL feature extractors downstream. It discards content above 8 kHz, which
matters little for articulation but does reduce fricative detail — a limitation
to note rather than revisit. With FredPrior excluded, every corpus now carries
an identical 8 kHz bandwidth, so bandwidth is no longer available to a domain
discriminator as a free cue. What remains is the resampling signature near
Nyquist, which differs between a natively-16 kHz file and one downsampled from
44.1 kHz.

### Quality measurement instead of random sampling

The pipeline asks for a listening pass to find unusable files. Rather than
sample at random, the preprocessor computes three measures on the *source*
signal and stores them in the manifest, so the files worth listening to can be
picked out by measurement:

| column | meaning |
|---|---|
| `peak_dbfs` | peak level of the source |
| `clipped_frac` | fraction of samples at or above 0.999 full scale |
| `silence_frac` | fraction of 20 ms frames whose RMS is more than 50 dB below the file's own peak |

---

## Results

| | files in | blacklisted | files out | hours |
|---|---|---|---|---|
| NeuroVoz | 2976 | 8 | 2968 | 3.86 |
| ItalianPD | 831 | 1 | 830 | 5.54 |
| MDVR-KCL | 73 | 0 | 73 | 2.87 |
| **total** | **3880** | **9** | **3871** | **12.26** |

Verification gate, all three corpora:

| check | result |
|---|---|
| manifest rows with a file on disk | 3871 / 3871 |
| orphan files on disk | 0 |
| duplicate output paths | 0 |
| distinct sample rates | {16000} |
| distinct channel counts | {1} |

Utterance duration after processing:

| | shortest | median | longest |
|---|---|---|---|
| NeuroVoz | 0.90 s | 3.47 s | 80.3 s |
| ItalianPD | 3.38 s | 11.56 s | 250.3 s |
| MDVR-KCL | 73.0 s | 141.3 s | 220.6 s |

---

## The blacklist

Nine files, each with a reason recorded in `configs/noise_blacklist.txt`.

**One duplicate.** ItalianPD speaker Giulia P's `B2` is byte-identical to her
`B1`; the second reading of the text was never recorded, so retaining it would
duplicate one speaker's utterance.

**Eight truncated NeuroVoz recordings**, too short to contain the utterance
they are labelled with, against a sentence-task median near 2.9 s:

| file | duration | prompt |
|---|---|---|
| HC_CARMEN_0082 | 0.08 s | *carmen baila el mambo* |
| HC_CARMEN_0083 | 0.15 s | as above |
| HC_ABLANDADA_0082 | 0.17 s | *la patata no está bien ablandada* |
| HC_PETACA_BLANCA_0063 | 0.20 s | *la petaca blanca es mía* |
| HC_PIDIO_0065 | 0.20 s | *no pidas a quien pidió…* |
| HC_TOMAS_0068 | 0.45 s | *tomás tira de la manta* |
| HC_A2_0056 | 0.53 s | sustained vowel |
| HC_DIABLO_0080 | 0.66 s | *cuando el diablo no sabe qué hacer…* |

**Deliberately retained** and listed in the blacklist file as listening
candidates rather than excluded:

- Two very short NeuroVoz files — a 0.90 s vowel and a 2.39 s DDK — short but
  not physically impossible.
- ItalianPD sustained vowels running 27–35 s. Their silence fraction is only
  1–5%, so they are continuous signal rather than one phonation followed by
  dead air; trimming at Stage 4 is the appropriate remedy, not exclusion.
- Two ItalianPD patients reading at three to four times the median duration.
  Slow, effortful reading is a genuine PD symptom, so unusual length is not by
  itself evidence of an export error.

---

## Findings

**ItalianPD has a sample-rate confound aligned with the class label.** 160
files at 44.1 kHz belong to 10 PD speakers. Every control — 22 elderly, 15
young — and the remaining 18 patients are at 16 kHz. No speaker mixes rates, so
this is a per-session recording difference:

| cohort | speakers at 16 kHz | speakers at 44.1 kHz |
|---|---|---|
| elderly_control | 22 | 0 |
| young_control | 15 | 0 |
| patient | 18 | 10 |

For that subset of patients, class is readable straight off the spectrum.
Resampling everything to 16 kHz removes the obvious cue but not the resampling
signature near Nyquist. This sits *inside* the corpus that was supposed to be
the clean clinical one, and I would rank it above the documented age confound.

**Clipping is negligible but entirely in controls.** Thirteen files exceed 0.1%
clipped samples: 11 ItalianPD and 2 NeuroVoz, worst case 1.6%. All 13 are HC.
The direction is the opposite of the sample-rate confound, and at 13 of 3871
files the effect is too small to matter, but it is worth knowing that the
residual clipping in this dataset is not class-neutral.

**MDVR-KCL silence fractions reach 68%**, mean 25%, consistent with the
documented phone-call recording posture and long conversational pauses. No
MDVR-KCL file clips. This is not a defect but it means utterance-level features
over these files average across a great deal of non-speech, and it is the main
argument for windowed rather than whole-file features on this corpus.

**Peak levels differ systematically by corpus.** NeuroVoz sources average
−19.1 dBFS against −12.5 for ItalianPD and −12.8 for MDVR-KCL. Peak
normalisation removes the level difference itself, but a corpus-level
difference in recording gain usually travels with a difference in noise floor,
which normalisation does not remove.

---

## Open items

- The listening pass has been completed by the user. The blacklist was not
  amended as a result, so the retained candidates above stand.
- Re-running after a blacklist change requires clearing `data/processed/`
  first, otherwise the outputs of newly excluded files remain on disk as
  orphans. This stage was last re-run in full after FredPrior was dropped, and
  the orphan count was verified at zero afterwards.

## Reproduction

```bash
python -m src.data.preprocess --dataset MDVR-KCL   # one corpus
python -m src.data.preprocess                      # all three
```
