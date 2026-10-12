# Corpus-norm ablation — probe and GBT results

Duration + handcrafted. Trees only. No MLP, no DANN. 2026-09-03.

## What was run

```bash
python scripts/check_train.py
python scripts/probe_normalisation.py --levels duration --lodo-probe
python -m src.training.classical --level duration --split 0 --norm global
python -m src.training.classical --level duration --split 0 --norm corpus
```

Unseen-corpus fallback uses `__all__`, not identity. `held_out_in_stats` is
False on every LODO row.

## Mixed-style probe (GroupKFold, 40k segments, 216 speakers)

| scope | corpus BA | PD/HC BA |
|---|---|---|
| none / global | 0.757 | 0.738 |
| speaker | 0.327 | 0.500 |
| utterance | 0.332 | 0.469 |
| **corpus** | **0.991** | **0.738** |

Per-corpus z-score **did not hide dataset identity**. It made the shortcut
easier (0.757 → 0.991) while leaving PD/HC unchanged. Mean-only batch
correction equalises first-order shifts; what remains — covariance, task mix,
language, channel — becomes easier for a tree to use once the means are gone.

CORAL was not justified *as a rescue of mixed PD* — the first-order control
already moved the corpus probe the wrong way. It was later run anyway as
the second-order increment, and failed the same way: corpus 0.757 → 0.855,
PD 0.738 → 0.726. See `reports/ablation_corpus_norm.md` follow-up.

## Strict LODO probe (fit on the other two corpora)

| held out | global PD/HC | corpus PD/HC |
|---|---|---|
| ItalianPD | 0.591 | **0.504** |
| MDVR-KCL | 0.599 | 0.616 |
| NeuroVoz | 0.521 | 0.510 |

`target_in_stats=False` on all six rows. Train rows are centred *per source
corpus*; the target is scaled with pooled source `__all__`. That is a
transform mismatch, and it takes ItalianPD to chance. This is not a bug in
the fallback; it is what strict LODO *is* for this method.

## Stage 6 mixed fold 0, speaker-pooled GBT

22 test speakers. Per-corpus numbers on one fold are noisy (Stage 6); the
comparison is whether corpus-norm changed anything.

| | val BA | test BA | test AUC | IT | MDVR | NV |
|---|---|---|---|---|---|---|
| `norm=global` | 0.727 | **0.818** | 0.826 | 1.00 | 0.75 | 0.73 |
| `norm=corpus` | 0.773 | **0.818** | 0.826 | 1.00 | 0.75 | 0.73 |

Test is identical. Validation is a little higher under corpus-norm and did
not transfer. ItalianPD at 1.00 on a handful of speakers is the age confound
showing up in a thin fold, not a method effect — it is the same under both
scopes.

## Decision (mean-only run)

- Do **not** treat `norm=corpus` as a cleaned dataset for Stage 7.
- Mainline stays `norm=global`. Removing channel is the adversary's job.
- CORAL was not justified as a rescue of mixed PD. It is justified as the
  next test of why corpus BA went to 0.991 (covariance made visible after
  means died). That is Track A2 in `PLAN.md`, and only if A1 does not
  already explain the jump.

Interpretation row from the original brief: mixed PD did not rise; the
corpus probe got worse (easier shortcut); strict LODO did not improve and
ItalianPD collapsed. Shallow mean correction is not a substitute for DANN.
Whether shallow covariance or feature dropping is, is the next measurement,
not a new architecture.

## A1 — feature importance (duration, global norm)

Same 40k-row, seed-0, speaker-grouped 5-fold probe. Importance is the drop
in balanced accuracy when that column is shuffled on the held-out fold
(permutation, 5 repeats), then averaged across folds.

```bash
python scripts/probe_feature_importance.py --level duration
```

| bin | n | rule |
|---|---|---|
| corpus-only | 4 | above-median corpus, at-or-below-median PD |
| PD-only | 4 | the reverse |
| both | 16 | above median on both |
| neither | 16 | below median on both |

**Corpus-only:** ZCR std, MFCC-1 std, RMS std, MFCC-13 mean. Spectral
spread and a high-order MFCC — channel / language colouring, not
loudness-as-symptom.

**PD-only:** MFCC-4 mean, F1, MFCC-9 mean, F3. Formants and a mid MFCC.

**Both (the ones that matter):** MFCC-1/2/8 mean, ZCR mean, RMS mean,
intensity, shimmer. The strongest corpus features are also among the
strongest PD features.

Re-score after dropping:

| subset | n | corpus BA | PD/HC BA |
|---|---|---|---|
| all 40 | 40 | 0.757 | 0.738 |
| drop corpus-only | 36 | 0.750 | 0.735 |
| drop both + corpus-only | 20 | 0.531 | 0.600 |
| PD-only only | 4 | 0.416 | 0.563 |

Dropping the four corpus-only columns does almost nothing. The shortcut is
not “three bad features.” It lives in the **shared** 16, and cutting those
takes PD from 0.738 to 0.600 with it. PLAN row: *both fall* if you strip
the overlap; *corpus stays ~0.75* if you only strip the orange points.
A2 (CORAL) ran and failed (0.757 → 0.855). Do not mask columns in Stage 7.

Figures (also copied to `notebooks/figures/`). First wave:

1. `featimp_scatter.png` — one point per feature. Dashed lines are the
   two medians. Orange = corpus shortcut, teal = PD without domain,
   purple = shared (DANN’s problem). **Talking point:** almost everything
   useful for PD sits in the purple quadrant.
2. `featimp_bars.png` — same 40 features, two jobs, same colouring, sorted
   by corpus importance. **Talking point:** ZCR mean and shimmer are
   mid-pack for corpus and top-pack for PD; MFCC-2 mean is the opposite.
3. `featimp_drop.png` — what the tree can still do after those bins are
   removed. Dotted lines are chance (0.33 corpus, 0.50 PD). **Talking
   point:** first pair of bars barely moves; the third pair is the cost
   of deleting the shared subspace.

Follow-up (redraw: `python scripts/visualise_ablation.py`):

4. `ablation_moment_ladder.png` — L0–L3 corpus BA. Variances add nothing;
   correlations add 0.098; 0.088 sits above second order.
5. `ablation_coral_corr.png` — 40×40 correlation, three corpora, before
   vs after CORAL. The method matched the covariances (fitted-set gap
   0.003) and corpus identity still rose.
6. `ablation_methods.png` — four methods on the corpus-vs-PD plane.
   Nothing moves up and left.
7. `ablation_tsne.png` — held-out speakers after each transform, coloured
   by corpus. The clouds do not mix.

