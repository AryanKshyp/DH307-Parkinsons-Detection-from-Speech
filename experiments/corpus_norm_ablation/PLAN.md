# Where this ablation goes next

The mean-only run is finished. It answered its question: **z-scoring each
corpus does not remove the shortcut**, and on duration it makes corpus identity
easier (0.757 → 0.991) while PD/HC stays put. More copies of that idea
(per-language z-score, per-task z-score, a second GBT fold) will not change
the conclusion.

What it *did* change is the hypothesis. The leftover cue is not a mean shift.
It is second-order structure, a subset of the 40 features, or something
non-linear that only a learned adversary can touch. The experiments below
test those three in order, then stop. Architectures belong on the Stage 7
track, not in this folder.

Do not start a neural grid from this file. The mainline is still
`norm=global` → 7a baseline → 7b DANN on duration + handcrafted.

---

## What 0.991 actually means

After each dataset is centred and scaled, a tree can name the corpus almost
perfectly. Three non-exclusive explanations:

1. **Covariance.** ItalianPD, NeuroVoz, and MDVR-KCL still have different
   feature–feature correlations (jitter with F0, MFCC cloud shape). Mean
   removal made those shapes comparable in scale, so easier to split on.
2. **A few features are almost pure domain.** Sample-rate / resampling
   signature near Nyquist, language-specific formants, task mix inside a
   250 ms window. Z-scoring them per corpus does not delete the *pattern*,
   only the offset.
3. **Non-linear channel.** Something a linear map cannot equalise. If (1)
   and (2) both fail to bring corpus BA down, this is the residue DANN is
   for.

The follow-ups map onto those three. Each has a stop rule.

---

## Track A — finish this ablation (trees, same 40-d cache)

All three stay on **duration + handcrafted**, the same probe harness, train
stats only, speaker-grouped evaluation. No new audio, no new nets.

### A1. Which features carry corpus vs PD  (do this first)

**Why.** If 0.991 is “three columns,” we should not invent CORAL or a new
head. We should name the columns.

**How.** On the mixed GroupKFold probe (same 40k sample, seed 0):

- Fit one GBT for `dataset`, one for `label`, under `norm=global`.
- Record permutation importance (or gain) per feature, both targets.
- Rank into four bins: corpus-only, PD-only, both, neither.
- Drop the corpus-only bin, re-score both targets.
- Optional: drop the “both” bin as a stress test (ItalianPD 44.1 kHz
  patients and age live here if anywhere).

**Reads as**

| outcome | next |
|---|---|
| Corpus BA falls a lot, PD holds | the shortcut is a feature subset; Stage 7 can mask or ignore those columns; skip A2 |
| Both fall | domain and target share directions (rate confound, age). Do not drop them. DANN has to keep PD while forgetting corpus — A2/A3 will not magically separate them |
| Corpus BA stays ~0.75 | the cue is spread across the vector. Go to A2 |

**Cost.** One script. No training loop. Write
`experiments/corpus_norm_ablation/feature_importance.csv`.

### A2. CORAL on the 40-d vectors  (only if A1 did not already kill corpus)

**Why.** RESULTS.md said skip CORAL because first-order moved the *wrong*
way. That is still right as “do not CORAL to rescue mixed PD.” It is no
longer right as “second-order cannot be the leftover cue.” 0.991 is exactly
the signature of a covariance shift that became visible after means died.

**How.** Sun & Saenko 2016, regularised inverse (`Σ + γI`). Reference = pooled
**training** covariance of that configuration, not NeuroVoz. Fit on train
speakers only.

- Mixed: each corpus whitened by its train `Σ_d`, recoloured by `Σ_train`.
  Test speakers of corpus *d* use `Σ_d` from **train**, not from test.
- Strict LODO: do **not** estimate `Σ` on the held-out corpus. Whiten sources
  toward the source barycenter; apply that same map’s `__all__` fallback to
  the target (same claim as `target_stats=global_fallback`).
- Re-run the mixed probe and `--lodo-probe` after CORAL, then one GBT on
  mixed split 0 if the probe moved.

**Reads as**

| outcome | next |
|---|---|
| Corpus BA drops toward chance, PD holds | first-order failed because the cue was covariance. CORAL is the shallow baseline DANN has to beat. Optional: A3 |
| Corpus still high, PD holds | leftover is non-linear or discrete (task mix). Shallow alignment is done. Stage 7 |
| PD collapses | covariance carried pathology (vowel-duration / jitter structure that differs by corpus because the *prompts* differ). Stop. Do not put CORAL in the main path |

**Cost.** A `fit_coral` / `apply_coral` next to `store.py`, same tests as
`__all__` fallback. Do not CORAL Mel frames.

### A3. Project out a linear corpus subspace  (optional, after A2)

**Why.** DANN’s domain head is a learned non-linear version of “make features
uninformative about corpus.” The linear version is: fit a corpus classifier
(multinomial logistic or LDA) on train, take the domain-discriminative
directions, remove them (null-space projection / INLP, one or two
iterations), then probe PD and corpus again.

This is still not a new architecture. It is the question “how much of 0.991
is linearly available?” Stage 2 already showed corpus is linearly available
from mean MFCCs. A3 asks whether *removing* that linear piece leaves PD
intact.

**Reads as**

| outcome | next |
|---|---|
| Corpus → chance, PD holds | a linear adversary would have sufficed; DANN must beat this on LODO to be worth the GRL |
| Both → chance | linear domain directions *are* the PD directions. Gradient reversal will fight itself. Report that; do not crank λ |
| Corpus stays high | 0.991 is non-linear. DANN is justified |

**Cost.** Small. Same GBT probe after a projection fitted on train folds
only. Do not use PD labels in the projection.

### Stop Track A when

- A1 names a small corpus-only feature set, **or**
- A2/A3 fail to cut corpus BA without cutting PD, **or**
- you are about to add MixStyle, ComBat, Deep CORAL, or a transformer
  “to see.” Those are different questions (Track C or skip).

---

## Track B — one measurement, not a method

### B1. Repeat the mixed probe at `duration_100` and `utterance`

Same script: `python scripts/probe_normalisation.py --levels duration_100 utterance`

**Why.** 0.991 might be an artefact of 250 ms windows (task fragments look
corpus-specific). If utterance (already 0.938 corpus under `global`) also
jumps toward 1.0 under `corpus`, the finding is about the *features*, not
the window. If `duration_100` does not jump, the 250 ms scale is special.

Mel is still un-probed (`reports/stage5_features.md`). A mean-pooled 65-d
Mel → the same GBT is the honest analogue; do **not** train the CNN for
this. If you only have time for one extra level after A1, prefer this Mel
mean-pool probe over a third window size.

---

## Track C — architectures (Stage 7, not this folder)

New nets will not rescue mean-only batch correction. They belong on the
**global-norm** spine the pipeline already specifies.

Assigned choice is MLP / CNN / transformer. This repo already has the two
that matter:

| family | trunk | when |
|---|---|---|
| handcrafted | `MlpTrunk` | first. Same 40-d space as this ablation, so 7a is comparable to the GBT |
| Mel | `CnnTrunk` (1D + attention pool, Ibarra’s best) | second, once 7a/7b exist on handcrafted |
| transformer | not implemented | only if CNN+DANN still fails LODO *and* you have a reason the failure is long-range context |

**Stage 7 order (do not permute):**

1. `duration` + handcrafted + mixed + **7a baseline** (`norm=global`).
2. Same + **7b DANN** (3-way corpus head, λ ramp, weighted CE on both heads).
3. Same pair under **LODO**, three configs reported separately.
4. Single-corpus 7a (pipeline: show the drop when you pool).
5. Then Mel CNN 7a/7b on duration, if compute remains.

**Optional DA variants — after vanilla DANN works, at most one:**

- **Deep CORAL loss** on the embedding, no GRL. Answers “covariance matching
  in the net vs adversarial.” Natural sibling of A2.
- **CDAN** (domain head conditioned on PD softmax). Use if vanilla DANN
  aligns domains but mixes PD/HC.
- **MixStyle / feature-stat mixing** at train time. Domain *generalisation*,
  so it matches strict LODO better than transductive CORAL. Only if LODO is
  the number that will not move.

Do not run these as a bake-off of five methods. Pick the one that matches
the failure mode of vanilla DANN.

Aggregation (mean pool vs attention vs vote) is a Stage 7 axis on *one*
already-trained setup, not a reason to multiply the grid.

---

## Explicitly do not do

- Another mean-based scope (`language`, `task`, `cohort`). Same method, more
  labels, same 0.991 geometry.
- ComBat. Batch is confounded with language, channel, and task.
- Per-speaker / per-utterance. Already at chance for PD.
- Class-conditional CORAL (align PD-to-PD across corpora). Uses the target
  label inside the transform.
- Transductive LODO (fit stats on the held-out corpus) as the headline.
  Allowed only as a named extra after strict LODO exists.
- A transformer, wav2vec frozen trunk, or extra DANN head “for the ablation.”
- 10 mixed configs × 2 norms × 2 regimes of the MLP until Track A is done
  and 7a global exists to match.

---

## Suggested sequence

```
now     A1  feature importance (global GBT, duration)
        B1  probe duration_100 + utterance under corpus scope
then    A2  CORAL, only if A1 left corpus BA high
maybe   A3  linear corpus projection, only if A2 left corpus BA high
then    Stage 7 spine (Track C), norm=global
later   one DA variant if DANN’s failure mode is clear
```

Minimum that changes what you believe: **A1**. If you only run one more
thing in this folder, run that.

Deliverable for Track A: append tables to `RESULTS.md` (or a short
`reports/ablation_corpus_norm.md` once 7a/7b exist). Do not rewrite the
Stage 5 cache.
