# Stage 7 — Training

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

The spine is `duration` × handcrafted MLP, then Mel 1D-CNN with masked
attention; mixed protocol then LODO; baseline then DANN. Gradient-boosted trees
on the same splits are a thermometer, not the model under test. Sampling is
speaker-balanced, stratified on `dataset × label`. The domain head is 3-way
corpus and is absent on baseline runs, not a dead head.

**The whole grid is being re-run as of 2026-09-06** under a corrected
checkpoint-selection rule (below), and extended to the two segment levels the
assigned spec names. Runs made before the rule change are preserved under
`experiments/_pre_minselect/` — they are the evidence for the bug, not
results to quote.

**In:** `data/processed/features/`, `data/manifests/splits_mixed.csv`,
`data/manifests/splits_lodo.csv`
**Out:** `experiments/<protocol>_<split>_<family>_<level>_<regime>_s0/`
**Code:** `src/models/`, `src/training/`, `configs/train.yaml`
**Checks:** `scripts/check_train.py`, `scripts/summarise_stage7.py`

---

## What actually ran

| family | level | protocol | GBT | baseline | DANN | single |
|---|---|---|---|---|---|---|
| handcrafted | duration | mixed 10-fold | done | **done** | **done** | **done** |
| handcrafted | duration | LODO × 3 | done | **done** | **done** | n/a |
| handcrafted | utterance | mixed 10-fold + LODO × 3 | — | **done** | **done** | — |
| handcrafted | phoneme/vowel | mixed 10-fold | — | **done** | **done** | — |
| mel | duration | mixed 10-fold + LODO × 3 | — | **done** | **done** | not run |
| mel | utterance | — | — | *dropped* | *dropped* | — |
| mel | phoneme/vowel | — | — | *dropped* | *dropped* | — |

**Stage 7 is complete.** 92 handcrafted runs plus 26 Mel `duration` runs, all
under the corrected selection rule; 142 experiment directories in total. The
Mel grid finished 2026-09-06 22:08. Mel `utterance` and Mel `phoneme`/vowel
were **dropped by decision** — ~13 h of GPU for levels the handcrafted family
already covers end to end.

GBT rows were not re-run: trees have no epochs, so the selection rule cannot
touch them.

CLI for one run:

```
python -m src.training.run --family handcrafted --level duration --split 0 --regime baseline
```

`--all-configs` walks every mixed fold or every LODO corpus. Each run writes
`config.json`, `metrics.csv`, speaker-level `predictions.csv` and
`embeddings.csv` (val+test), and `checkpoints/best.pt`.

**Three levels, not one.** `duration` has the weakest corpus signal beside
nearly the strongest pathology signal (Stage 5), so it is the fair spine, but
it is neither level the Scope note names. `utterance` is the Scope note's sane
baseline and, at 0.938 corpus predictability, the hardest test of the
adversary. `phoneme`/vowel is the assigned spec's own worked example. Phoneme
is **mixed-protocol only** and its domain head is **2-way**: MDVR-KCL has no
phoneme segments, so LODO there is either a no-op or a single-domain training
set with nothing for an adversary to do.

Not in this round: transformer, HuBERT, `word`, `duration_100`, single-corpus
on Mel (~10 h of GPU for a corpus property the 40-feature model already
measures), corpus-norm as the main path (the ablation lives under
`experiments/corpus_norm_ablation/`).

---

## The selection rule, corrected

Early stopping selected on validation BA over 21–22 speakers with no floor, so
`best.pt` could be — and in five of ten mixed DANN folds was — **epoch 0,
where λ is still 0**. Half the previous "DANN" numbers were scoring a
pre-adversarial trunk, which makes the headline DANN-vs-baseline comparison
partly baseline-vs-baseline.

`min_select_epoch` (default 5) opens checkpoint selection only once λ has
ramped, and counts patience from the same epoch:

| epoch | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| λ | 0.000 | 0.165 | 0.322 | 0.462 | 0.583 | **0.682** | 0.762 | 0.870 |

It is applied to **baseline as well**. Baseline had the identical epoch-0
lottery — five folds also picked epoch 0 — so gating only DANN would have
changed one arm of the comparison and not the other. Both regimes are now
selected under one rule.

First folds under the new rule select epochs 5, 9, 5 rather than 0, and the
saved checkpoint holds a trunk trained at λ ≈ 0.68 or above.

---

## Aggregation

Segment-level outputs are combined by **mean-of-probabilities, then argmax**
(`src/training/metrics.py`) — Ibarra's joint-probability aggregation in linear
space. Scoring at segment level instead would inflate every number, since one
speaker contributes hundreds of correlated windows.

The assigned pipeline asks for this to be a design axis rather than a default,
so `scripts/probe_aggregation.py` measures five alternatives against it:
geometric mean (joint probability in log space), median, 10% trimmed mean,
majority vote, and a mean over the quarter of segments furthest from the
decision boundary. It is inference-only — Stage 7 saves `predictions.csv`
already pooled, so the probe reloads `best.pt` and re-runs the test split
rather than retraining — and pools out-of-fold so each speaker is scored once.

Attention pooling *across* a speaker's segments is deliberately not compared:
the Mel backbone already attends within a segment, and a second attention layer
over segments would have to be trained, making it a different model rather than
a different read-out of this one.

**Measured.** Pooled out-of-fold balanced accuracy over all 216 speakers,
handcrafted `duration`, every rule applied to the same saved checkpoints:

| rule | baseline | DANN |
|---|---|---|
| **mean_prob** (spine) | **0.817** | 0.810 |
| median_prob | 0.817 | **0.819** |
| trimmed_mean | 0.817 | 0.810 |
| majority_vote | 0.817 | **0.819** |
| mean_logit | 0.813 | 0.802 |
| confident_mean | 0.800 | 0.807 |

The whole spread across six rules is **0.017**, against a fold-to-fold standard
deviation of 0.054–0.061. Aggregation is not a sensitive axis on this spine,
and the default is not doing hidden work: median and majority vote tie or beat
it on DANN by 0.009, which is a quarter of the noise. Two rules are reliably
worse and both for the same reason — `mean_logit` (geometric mean) and
`confident_mean` weight confident segments more heavily, and on this data the
confident segments are not the correct ones.

---

## Design that is locked in


- **Backbones.** Handcrafted: MLP trunk, 44k parameters. Mel: 1D-CNN + masked
  attention, 224k parameters. The assigned pipeline also lists a transformer;
  it is not the spine.
- **Sampling unit is the speaker**, via `WeightedRandomSampler` on
  `dataset × label`. Uniform-over-segments would make NeuroVoz ~77% of every
  batch.
- **Normalisation** is global, fitted on training rows only. Per-speaker and
  per-utterance norm delete PD (Stage 5). Corpus-norm is an ablation, not
  default.
- **GRL** is identity forward, gradient × (−λ) backward. λ follows Ganin:
  `2 / (1 + exp(−γ p)) − 1`, γ = 10, ramp over the run.
- **Loss.** Weighted CE on the PD head always; weighted CE on the domain head
  only when the head exists. Domain labels are the three corpus names.
  Language and channel are confounded, so a 3-way corpus head is the honest
  adversary.
- **Selection.** Early-stop on *speaker-level* PD balanced accuracy on the
  21–22 validation speakers, patience 8, max 30 epochs. Item-level accuracy is
  logged and never used for selection.
- **`domain nan` on baseline** is expected: there is no domain head, so there
  is no domain accuracy. `lam 0.000` on baseline is the same fact.

Ibarra's mixed-corpus CV is not this project's strict LODO. Mixed numbers here
are pooled out-of-fold over 216 speakers. LODO is three separate numbers, never
averaged.

---

## Results — handcrafted, all three levels

Mixed protocol, pooled out-of-fold, speaker-level balanced accuracy. Per-corpus
scores are over the whole held-out population, never a single 21–22 speaker
fold.

| level | regime | pooled | ItalianPD | MDVR-KCL | NeuroVoz |
|---|---|---|---|---|---|
| duration | GBT | 0.777 | 0.982 | 0.750 | 0.666 |
| duration | baseline | 0.817 | 0.982 | 0.765 | 0.737 |
| duration | DANN | 0.810 | 0.982 | 0.765 | 0.720 |
| utterance | baseline | 0.818 | 0.982 | 0.686 | **0.764** |
| utterance | DANN | 0.809 | 0.964 | 0.662 | 0.764 |
| phoneme/vowel | baseline | 0.834 | 1.000 | — | 0.735 |
| phoneme/vowel | DANN | 0.796 | 1.000 | — | 0.674 |

Fold spread is 0.054–0.061, so no baseline/DANN difference in this table is a
result. What the levels say about each other is:

- **`utterance` is the best level for NeuroVoz** (0.764 against duration's
  0.737) and the worst for MDVR-KCL (0.686 against 0.765). MDVR-KCL utterances
  are 141 s dialogues chunked to 4 s including interviewer speech; NeuroVoz
  utterances are 3.5 s prompted phrases. The level rewards whichever corpus
  the chunking suits.
- **`phoneme`/vowel reaches 1.000 on ItalianPD.** That is a ceiling artefact on
  64 speakers across 2 corpora, not a detection result, and it is why the
  pooled 0.834 must not be quoted as the project's best number.
- ItalianPD sits at 0.98–1.00 at every level for every regime. It carries no
  information about which spine is better; NeuroVoz is the corpus that moves.

## LODO — the headline, and it changed

Balanced accuracy at a fixed 0.5 threshold, and AUC. **AUC is the number to
read** — Stage 8 shows the 0.5 threshold is broken under corpus shift, so the
BA columns understate every row.

| level | held out | n | BA base | BA DANN | **AUC base** | **AUC DANN** |
|---|---|---|---|---|---|---|
| duration | ItalianPD | 65 | 0.776 | 0.816 | 0.850 | **0.872** |
| duration | MDVR-KCL | 37 | 0.595 | 0.644 | 0.685 | **0.762** |
| duration | NeuroVoz | 114 | 0.508 | 0.526 | **0.700** | 0.692 |
| utterance | ItalianPD | 65 | 0.703 | 0.703 | **0.776** | 0.708 |
| utterance | MDVR-KCL | 37 | 0.576 | 0.591 | 0.628 | **0.667** |
| utterance | NeuroVoz | 114 | 0.517 | 0.500 | 0.543 | 0.504 |

**At `duration`, DANN beats baseline on two of three hold-outs by AUC** (and
all three by BA). Under the old λ=0 selection it lost on all three. The change
is the selection rule: half the old DANN checkpoints were pre-adversarial
trunks, so the old table was partly baseline against baseline.

Read it conservatively. Single seed, margins of 0.02–0.08 AUC, and the three
hold-outs are separate experiments that must not be averaged. The claim that
survives is the weak one: **at the level with the weakest corpus signal, the
adversary helps rather than hurts under leave-one-dataset-out**, the direction
Ibarra et al. report. It does not replicate on the Mel CNN (Stage 8).

**A correction to an earlier version of this report.** It said NeuroVoz "stays
at chance for every regime". That was an artefact of the 0.5 threshold, not a
property of the representation. At `duration` NeuroVoz reaches **0.700 AUC**
and 0.675 prior-matched BA — well above chance. At `utterance` it genuinely is
at chance (0.543 / 0.504 AUC), so the two levels differ and the original
sentence was right for the wrong one. See `reports/stage8_evaluation.md` and
`scripts/probe_failure_modes.py`.

## Single-corpus baselines — the pooling drop does not appear

Training on one corpus alone, ten mixed configurations each, against the same
corpus's rows in the pooled baseline:

| corpus | n | single-corpus | pooled baseline | pooled − single |
|---|---|---|---|---|
| ItalianPD | 65 | 0.982 | 0.982 | 0.000 |
| MDVR-KCL | 37 | 0.741 | 0.765 | +0.024 |
| NeuroVoz | 114 | 0.744 | 0.737 | −0.007 |

The expected drop from pooling heterogeneous corpora is **not there** at
`duration`. Pooling costs ItalianPD nothing, helps MDVR-KCL slightly, and
costs NeuroVoz 0.007 — all inside fold noise. That is a finding, not a null:
it says the corpus shift at this level is not severe enough to damage a
pooled classifier, which is consistent with `duration` having the weakest
measured corpus signal (Stage 5) and with DANN's small effect on mixed.

Handcrafted only, by decision — the same sweep on Mel is ~10 h of GPU to
measure a property of the corpora rather than of the backbone.

---

## Mel

`duration` only. Mel `utterance` and Mel `phoneme`/vowel were dropped after
Phase 1: ~13 h of GPU for levels the handcrafted family already covers end to
end. Mel `duration` is the second feature family on the spine and the one
Stage 8 needs against the MLP.

**Mel baseline is the best configuration in the project**: pooled 0.838
against the MLP's 0.817, NeuroVoz 0.788 against 0.737, AUC 0.915. That
survives the selection fix.

**Mel DANN is the worst.** The adversary costs the CNN 0.073 on mixed
(0.838 → 0.765) where it costs the MLP 0.007, and it loses on all three LODO
hold-outs where the MLP won all three. The sign of the DANN effect flips with
the backbone — that is the central Stage 8 finding, and the geometry explains
it: the CNN's intra-class TCM inflates ~250× under the adversary against the
MLP's ~10×. See `reports/stage8_evaluation.md`.

Every Mel DANN checkpoint was saved at λ between 0.682 and 0.997, none at 0.

---

## What this stage does not yet support as a claim

1. **A pooled mixed number.** ItalianPD is at ceiling at every level;
   NeuroVoz is the actual detection problem; MDVR-KCL is 37 speakers.
2. **Averaged LODO.** The three hold-outs are different experiments.
3. **DANN helps on mixed.** It does not, at any level; the gap is inside the
   fold spread in both directions.
4. **A DANN win at `utterance` or `phoneme`.** The LODO reversal is a
   `duration` result. At `utterance` the two regimes tie or trade places.
5. **Anything from a second seed.** Everything here is `s0`.
