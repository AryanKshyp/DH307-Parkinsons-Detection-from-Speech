# Stage 5 — Feature Extraction

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

All eight **(feature family × segment level)** combinations are extracted and
cached, 4.15 GB in total. The stage's main technical decision is to cache
features **unnormalised** and apply normalisation at load time, which turned the
normalisation scope from an irreversible commitment into a measurable one.

Measuring it changed the answer. Per-speaker normalisation, the scope chosen at
the start of the stage, **drives PD/HC classification to chance**. PD/HC status
is constant within a speaker, so the whole label sits in how one speaker's
feature means differ from another's; subtracting each speaker's own mean sets
that spread to zero and takes the target with the channel. The default is now
global, fitted on training rows only.

The stage's main analytical result is a direct measurement of the problem the
project exists to solve: from 40 handcrafted features, **corpus identity is
easier to predict than pathology at every level**, most severely at the utterance
level, where corpus reaches 0.938 against 0.761 for PD/HC.

**In:** `data/interim/segments.csv`
**Out:** `data/processed/features/` (4.15 GB)
**Code:** `src/features/extract.py`, `src/features/store.py`
**Checks:** `scripts/check_features.py`, `scripts/probe_normalisation.py`

---

## Results

Mel spectrograms, 65 bands, 25 ms window, 10 ms hop:

| level | items | frames | size |
|---|---|---|---|
| `duration` | 274,459 | 6,862,994 | 1702 MB |
| `utterance` | 20,397 chunks | 8,158,800 | 2023 MB |
| `phoneme` | 75,311 | 758,290 | 188 MB |
| `word` | 20,554 | 801,899 | 199 MB |

Handcrafted, 40 features per segment — 13 MFCC means and standard deviations,
zero-crossing rate, RMS energy, F0, jitter, shimmer, F1–F3 and intensity:

| level | segments | voice-quality valid |
|---|---|---|
| `utterance` | 3871 | 99.9% |
| `word` | 20,554 | 99.0% |
| `duration` | 274,459 | 94.4% |
| `phoneme` | 75,311 | 88.5% |

The whole extraction runs on CPU in roughly two hours, of which the utterance
level is 18 minutes. No GPU is needed at this stage. Everything is computed once
per file and sliced per segment rather than recomputed per segment, which is
where the tractability comes from: 3871 spectrogram computations instead of
374,195. It also means neighbouring segments share a framing, so their features
are directly comparable.

---

## Normalisation: the scope that deletes the target

Because the cache is unnormalised, each scope could be tested rather than
argued about. `scripts/probe_normalisation.py` predicts corpus identity and
PD/HC from the same features under each scope, with a gradient-boosted tree
ensemble under speaker-grouped 5-fold cross-validation, and statistics fitted on
the training fold only.

The tree ensemble is not incidental. A linear model would need a scaler, and the
scaler would impose global normalisation on top of whatever scope was being
tested, collapsing the very comparison being made. Trees are invariant to
per-feature scaling, so `none` and `global` are expected to agree exactly — and
they do, at every level, which is itself a check that the harness works.

Balanced accuracy, `none` and `global` (identical to three decimals):

| level | corpus | corpus chance | PD/HC | PD/HC chance |
|---|---|---|---|---|
| `utterance` | 0.938 | 0.333 | 0.761 | 0.500 |
| `word` | 0.869 | 0.500 | 0.679 | 0.500 |
| `phoneme` | 0.804 | 0.500 | 0.652 | 0.500 |
| `duration` | 0.757 | 0.333 | 0.738 | 0.500 |

Under per-speaker and per-utterance normalisation, both targets collapse:

| level | corpus, per-speaker | PD/HC, per-speaker | PD/HC, per-utterance |
|---|---|---|---|
| `utterance` | 0.316 | 0.471 | 0.459 |
| `word` | 0.500 | 0.510 | 0.510 |
| `phoneme` | 0.403 | 0.490 | 0.487 |
| `duration` | 0.327 | 0.500 | 0.469 |

Every one of those PD/HC figures is at or below chance. The mechanism is
arithmetic, not statistical: the between-speaker spread of feature means goes
from 14.56 to 7.0e-16 at utterance level, 13.46 to 1.3e-16 at `duration`, and
13.08 to 1.9e-16 at `word`. A label that is constant within a speaker lives
entirely in that spread, so setting it to zero removes the label. Per-utterance
normalisation behaves the same way because utterances nest inside speakers.

This was worth checking carefully, because both targets falling to chance is
*also* the signature of a row-shuffling bug in the normaliser. `verify()` in the
probe now asserts against that on every run: within-speaker correlation between
raw and normalised columns is 1.000000, so row order survives, and per-speaker
means come out at 0 with standard deviations at 1, so the transform is the one
intended.

The conclusion is that removing the recording channel is the adversary's job, not
the normaliser's. `global` is information-preserving — an affine map applied
equally to every row — and leak-free provided its statistics come from training
rows only, which is why `fit_stats` and `apply` are separate functions rather
than one call. `speaker` and `utterance` remain implemented, as ablations that
document the collapse.

---

## How much domain signal there is

The table above is the clearest statement so far of the project's premise. At the
utterance level, corpus identity is predicted at 0.938 against a 0.333 chance
level, while pathology reaches 0.761 against 0.500. A model trained without a
domain penalty has a much easier corpus problem available to it than the
pathology problem it is meant to solve.

Two things follow for later stages.

The ranking of levels by *usable* signal is not the ranking by raw accuracy. The
`duration` level has the weakest corpus signal, 0.757, and almost the strongest
pathology signal, 0.738 — nearly equal. The utterance level has the strongest
pathology signal but a far stronger corpus signal alongside it. Fixed 250 ms
windows appear to carry proportionally more pathology than corpus information
than whole utterances do, which makes `duration` the most promising level to
attack first and the utterance level the hardest test of the adversary.

The `word` and `phoneme` rows are not comparable to the other two. Stage 3
aligned only material with a transcript, so MDVR-KCL is absent from both and
their corpus task is two-class with a 0.500 chance level rather than three-class
at 0.333.

For continuity with Stage 2: file-level 20-dimensional mean MFCCs gave 0.892
corpus and 0.683 PD/HC. The 40 handcrafted features raise both, to 0.938 and
0.761. Richer features help pathology, and help corpus identification slightly
more.

This also settles an open question Stage 2 could not answer. Stage 2 suspected
part of its 0.892 corpus separability was an averaging artefact rather than a
channel effect, since a mean MFCC over a 141 s MDVR-KCL call is a far smoother
estimate than one over a 3.5 s NeuroVoz sentence. The `duration` level is fixed
at 250 ms and so is duration-matched by construction: corpus separability there
is 0.757 against 0.938 at the utterance level. Part of the utterance-level corpus
signal was indeed an artefact of unequal averaging — but 0.757 on length-matched
windows, against a 0.333 chance level, confirms a large genuine channel effect
underneath it.

---

## Analysis window

25 ms window, 10 ms hop, 65 Mel bands, uniform across every level and task.

The reference paper used 40 ms for sustained vowels and 15 ms for DDK. A uniform
window was chosen instead because this project's central comparison is *across*
corpora and tasks, and a window that varies by task would make those comparisons
partly a comparison of analysis settings. Uniformity also keeps faith with
Stage 4, whose 50 ms minimum segment length was set on the assumption of a 25 ms
window: at three frames per shortest segment, raising the window would have
required revisiting that floor and re-cutting every segment.

The cost is accepted rather than hidden. Sustained vowels get less frequency
resolution than the reference design would give them, and the shortest phonemes
get only three frames.

---

## Utterance chunking

Utterance-level spectrograms are stored as fixed 4 s chunks at a 2 s hop, giving
20,397 chunks from 3871 utterances, rather than as whole utterances.

The reason is a leak, not memory. Utterance length is strongly corpus-dependent —
a 3.5 s NeuroVoz sentence against a 141 s MDVR-KCL phone call — so a
variable-length tensor would hand the model the domain label through its shape
before it examined any audio. A fixed chunk removes that channel entirely. It
also makes batching possible: whole utterances span 0.9 s to 250 s, a 250-fold
range in item size.

Utterances shorter than one chunk are zero-padded, and the index records
`n_valid_frames` so a mask is recoverable. Padding is 2.81% of all stored frames.
The final chunk of each utterance is placed at `duration - 4 s` rather than at a
multiple of the hop, so the tail of every utterance is covered.

This does mean the utterance level is no longer one item per utterance. Chunks
carry their parent `segment_id`, so predictions can be pooled back to the
utterance, and that pooling is a Stage 7 choice.

---

## Where voice quality is undefined

Jitter, shimmer and formants need voiced signal with several pitch periods, which
much short or unvoiced material does not provide. Rather than let Praat return
whatever it returns, undefined values are written as NaN and every segment
carries a `voice_quality_valid` flag.

Validity tracks segment length and voicing as expected: 99.9% at utterance level,
99.0% at word, 94.4% at `duration`, and 88.5% at `phoneme`. At the phoneme level
the missing columns are shimmer at 11.3%, jitter at 7.3% and F0 at 6.9% —
unsurprising on a level that is 29% plosives, fricatives and affricates.

`store.load_handcrafted` fills NaNs with the column median so rows stay
usable, and leaves the flag alongside so a model can exclude them instead if that
turns out to matter.

Physically implausible values were also checked, since a Praat analysis that fails
without raising is more dangerous than one that returns NaN. They are rare enough
to need no special handling: shimmer above 1.0, which means period detection
failed rather than that a voice is rough, affects 0.01% of phonemes and 0.00%
elsewhere; negative intensity, meaning a window held effectively nothing, affects
0.26% of utterances and 0.01% of `duration` windows.

---

## Storage layout and verification

Each (family, level) pair writes a `.npy` of float32 and an index CSV. Mel data
is time-major, shape `(total_frames, 65)`, so one item is
`data[row_start : row_start + n_frames]`. Storing frames ragged and contiguous
rather than padding every item to a common length is what keeps the phoneme level
at 188 MB. Handcrafted data is one fixed-length row per segment. Both indexes
carry `segment_id`, so either family joins back to `segments.csv` and from there
to speaker, label and corpus.

Because each file is computed once and sliced, an off-by-one in the frame index
would silently shift every segment while still producing plausible-looking
output. `scripts/check_features.py` therefore recomputes sampled segments from
their audio in isolation and compares. The worst disagreement across all four
levels is **3.8e-06 dB**, which is float32 rounding. It also confirms the memmap
was fully written — an unwritten row is exactly zero across all 65 bands, and
observed all-zero frames match the expected padding share, 2.31% against 2.81% at
utterance level and 0.00% everywhere else.

One quiet trap is worth recording: `librosa.power_to_db` defaults to referencing
each input's own maximum, which would be a per-segment normalisation applied by
accident, silently defeating the decision to cache raw values. A fixed reference
is used instead.

---

## Limitations and open items

- **Phoneme-level plosives use a 15 ms FFT at the same 10 ms hop** (`PLOSIVE_N_FFT
  = 240`). Every other class and level stays on 25 ms. The 50 ms floor is
  unchanged; this is the alternative that Stage 4 flagged. A dedicated plosive
  DANN is not starved of frames, but any table that pools plosives with vowels
  is partly a comparison of analysis settings. Handcrafted features stay on the
  25 ms spectrogram.
- **`word` / `phoneme` / `duration_100` were re-extracted after the Stage 4
  recut** (~32 min). Voice-quality valid: word 97.9%, phoneme 88.8%,
  `duration_100` 85.4%. Isolated-audio check: worst disagreement 3.8e-06 dB.
  `duration` (250 ms) and `utterance` windows were not recut.
- **Global normalisation statistics are not yet fitted per split.** The
  `fit_stats`/`apply` split makes this correct to do, and the probe does it, but
  Stage 7 must actually pass training-fold rows. Fitting on everything would leak
  test statistics.
- **`duration`-level features are cached from 50%-overlapping windows.** Adjacent
  items share half their audio, so they are not independent samples. This matters
  for any confidence interval computed over segments rather than speakers.
- **The 40 handcrafted features are summary statistics**, so they discard temporal
  trajectory within a segment. Tremor and rhythm cues that live in how F0 moves,
  rather than in its mean and spread, are not represented; the Mel family carries
  those.
- **Only the handcrafted family has been probed for domain signal.** The Mel
  family is very likely worse, being closer to the raw channel, but that is an
  expectation rather than a measurement.
- The utterance level is 20,397 chunks against 3871 utterances, so **chunk counts
  are not utterance counts** and per-corpus chunk totals inherit the corpus
  duration imbalance directly.

## Reproduction

```bash
python -m src.features.extract                      # all families, all levels
python -m src.features.extract --levels word --family mel
python scripts/check_features.py --level word       # verify against fresh audio
python scripts/probe_normalisation.py --levels word phoneme
```
