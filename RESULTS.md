# Domain Adversarial Training for Corpus-Independent Parkinson's Screening

**DH307 · Subtopic 4** — Aryan Kashyap (23B0741)
Supervisor: Prof. Nirmal Punjabi · Guide: Mayurakshi Mukherji
Status as of 2026-09-07. Stages 0–8 complete, 142 training runs.

Reproduction and extension of Ibarra, Arias-Londoño, Zañartu &
Godino-Llorente (2023), *Towards a Corpus (and Language)-Independent Screening
of Parkinson's Disease from Voice and Speech through Domain Adaptation*,
Bioengineering 10:1316.

---

## 1. The question

Parkinson's speech corpora are small — typically under 100 speakers — so models
are trained on pooled datasets. When corpora are pooled, a network can lower
its training loss by learning **which dataset** a recording came from (language,
microphone, room) instead of the pathology. Domain adversarial training (DANN)
attaches a corpus classifier through a gradient reversal layer, so the shared
trunk is pushed to become uninformative about corpus while staying informative
about PD.

**The assigned headline question is not "does DANN beat the baseline on pooled
data".** It is whether DANN closes the gap under **leave-one-dataset-out
(LODO)** — a corpus, language and microphone the model has never seen.

The problem is real on this data, established before any training: **corpus
identity is predictable from 20 mean MFCCs at 0.892 balanced accuracy under a
speaker-disjoint split, while PD/HC reaches only 0.683.** The nuisance variable
is more learnable than the target.

---

## 2. Data

| corpus | language | channel | speakers (HC / PD) | files |
|---|---|---|---|---|
| NeuroVoz | Spanish | clinical mic | 114 (58 / 56) | 2968 |
| ItalianPD | Italian | clinical mic | 65 (37 / 28) | 830 |
| MDVR-KCL | English | smartphone | 37 (21 / 16) | 73 |

3871 files, 12.26 h, resampled to 16 kHz mono and peak-normalised. **216
speakers total.**

Loudness (LUFS) normalisation was deliberately *not* applied — reduced vocal
loudness is a genuine PD symptom and normalising it away would remove signal.

A fourth corpus bundled with the assignment (FredPrior) was excluded: sustained
/a/ only, so none of the four assigned tasks, and its 8 kHz bandwidth was the
only bandwidth cue in the set.

### Confounds carried forward, not hidden

- **Language and channel are perfectly confounded** — one recording setup per
  language. No claim about *language* invariance can be separated from channel.
  This is why the domain head is a 3-way corpus head and not a language head.
- **ItalianPD has 15 young controls** (mean 20.8 y against patients at 67).
  Kept and documented; an elderly-only number is reportable without
  regenerating anything, and it changes nothing (0.982 either way).
- **MDVR-KCL ships no age metadata at all**, so its age balance cannot be
  checked — a caveat on any result holding it out.
- **Phoneme level covers 2 corpora only.** MMS_FA emits characters, and English
  orthography is not shallow, so MDVR-KCL has no phoneme segments.

---

## 3. Pipeline

Nine stages, each keyed by a CSV the next stage joins against; no stage walks
raw directories.

```
registry → manifest → 16 kHz audio → forced alignment → segments
        → features → speaker-level splits → training → evaluation
```

**Alignment.** `torchaudio` MMS_FA, one aligner for all three languages.
Scripted tasks use published prompts. Whisper ASR was tried on the 36 MDVR-KCL
spontaneous files and **rejected on measurement**: mean alignment score 0.381,
19 of 27 files rejected, and the transcript contains the interviewer. 38 files
(1.8%) rejected on aligner confidence.

**Segmentation.** 1,106,469 segments across five levels. The Stage 3 character
spans could not be used as-is — 95% were exactly 20 ms, the CTC stride, not
phone extents — so they are used as boundary anchors instead, giving a median
phone of 80 ms.

**Features.** Two families, cached unnormalised so the normalisation scope is a
training-time knob: 65-band Mel spectrograms, and 40 handcrafted features
(MFCC, jitter, shimmer, ZCR, energy, F0, F1–F3, intensity).

**A measured decision that shaped everything downstream:** per-speaker
z-scoring takes PD/HC **to chance** (0.679 → 0.510 at word level). PD/HC is
constant within a speaker, so the label lives entirely in the *between-speaker*
spread of feature means, and per-speaker normalisation sets that spread to zero
(13.08 → 1.9e-16). It removes the channel and the target together. Removing
channel is the adversary's job, not the normaliser's. Normalisation is global,
fitted on training rows only.

**Splits are speaker-level and independently re-verified** at file level by
`scripts/check_splits.py`. Two protocols: `mixed` (10 configurations, stratified
on dataset × label) and `lodo` (3 configurations, one corpus entirely held out).

---

## 4. Experiments run

**142 runs**, one directory each with resolved config, seed, metrics,
speaker-level predictions, embeddings and checkpoint.

| group | runs |
|---|---|
| handcrafted MLP — duration, utterance, phoneme/vowel × baseline/DANN × mixed/LODO | 72 |
| Mel 1D-CNN — duration × baseline/DANN × mixed/LODO | 26 |
| single-corpus baselines — 3 corpora × 10 folds | 30 |
| gradient-boosted trees (non-neural reference) | 14 |
| corpus-normalisation ablation | 8 |
| archived pre-fix runs (kept as evidence, not results) | 36 |

**Backbones.** MLP (44k parameters) for the 40 handcrafted features; 1D-CNN
with masked attention pooling (224k) for Mel. GRL is identity forward,
gradient × (−λ) backward, with the Ganin λ ramp (γ=10).

**Three levels, chosen with reasons rather than convenience.** `duration` has
the weakest corpus signal (0.757) beside nearly the strongest pathology signal
(0.738), so it is the fair spine. `utterance` is the hardest test of the
adversary — corpus is 0.938 predictable there. `phoneme`/vowel is the assigned
spec's own worked example, and is trained *within* a phone class because
phone-class composition differs sharply by corpus (Italian has 7× the affricate
share) — a model trained across pooled classes could identify the corpus from
the class mix alone.

Mel covers `duration` only. Mel `utterance` and `phoneme` were dropped at ~13 h
of GPU for levels the handcrafted family already covers. **So every *level*
claim rests on handcrafted, and the family comparison rests on `duration`.**

---

## 5. A methodological bug found and fixed mid-project

Early stopping selected checkpoints on validation balanced accuracy over 21–22
speakers with **no floor on λ**. In five of ten mixed DANN folds the selected
checkpoint was **epoch 0, where λ is still 0** — a pre-adversarial trunk.
Half the "DANN" numbers were therefore baseline numbers, and the headline
comparison was partly baseline-against-baseline.

`min_select_epoch` (default 5, λ ≈ 0.68) opens checkpoint selection only once
the adversary has ramped, and counts patience from the same epoch. **It is
applied to the baseline as well** — the baseline had the identical epoch-0
lottery, so gating only DANN would have moved one arm of the comparison and not
the other.

The entire grid was re-run. After the fix:

| family | λ at checkpoint (min / median / max) | folds with λ=0 |
|---|---|---|
| handcrafted | 0.682 / 0.969 / 0.996 | **0 of 10** |
| Mel | 0.682 / 0.870 / 0.997 | **0 of 10** |

**The fix reversed the sign of the headline result on the MLP** — DANN went
from losing all three LODO hold-outs to winning them. This is the single most
important process point in the project: the original conclusion was an artefact
of model selection, not a property of the method.

---

## 6. Results

### 6.1 Mixed protocol — pooled out-of-fold, 216 speakers

Every speaker is in the test set of exactly one configuration, so per-corpus
scores are pooled across all ten (MDVR-KCL contributes 1–2 patients to any
single fold).

| family | level | regime | BA | AUC | fold BA ± std |
|---|---|---|---|---|---|
| handcrafted | duration | GBT | 0.777 | 0.883 | 0.780 ± 0.062 |
| handcrafted | duration | baseline | 0.817 | 0.902 | 0.820 ± 0.061 |
| handcrafted | duration | DANN | 0.810 | 0.890 | 0.812 ± 0.054 |
| handcrafted | utterance | baseline | 0.818 | 0.875 | 0.820 ± 0.082 |
| handcrafted | utterance | DANN | 0.809 | 0.882 | 0.810 ± 0.088 |
| handcrafted | phoneme/vowel | baseline | 0.834 | 0.891 | 0.834 ± 0.052 |
| handcrafted | phoneme/vowel | DANN | 0.796 | 0.881 | 0.795 ± 0.075 |
| **Mel** | duration | **baseline** | **0.838** | **0.915** | 0.839 ± 0.092 |
| Mel | duration | DANN | 0.765 | 0.863 | 0.765 ± 0.093 |

**Mel baseline is the best configuration in the project.** DANN does not
improve mixed-protocol accuracy at any level, in either family. Every
baseline−DANN gap is smaller than the fold-to-fold spread, so on mixed the
honest statement is "no difference" — except on Mel, where DANN's 0.073 loss
approaches the spread.

Per corpus (handcrafted duration baseline): ItalianPD 0.982, MDVR-KCL 0.765,
NeuroVoz 0.737. **ItalianPD is at ceiling in every configuration and carries no
information about which method is better.** NeuroVoz — 114 speakers — is the
corpus that moves and the one to read.

### 6.2 LODO — the assigned headline

Reported on **AUC**, because a fixed 0.5 threshold is invalid here (§6.3).

| family | held out | n | AUC baseline | AUC DANN | Δ |
|---|---|---|---|---|---|
| handcrafted | ItalianPD | 65 | 0.850 | **0.872** | **+0.021** |
| handcrafted | MDVR-KCL | 37 | 0.685 | **0.762** | **+0.077** |
| handcrafted | NeuroVoz | 114 | **0.700** | 0.692 | −0.007 |
| Mel | ItalianPD | 65 | **0.903** | 0.681 | **−0.221** |
| Mel | MDVR-KCL | 37 | 0.807 | 0.807 | 0.000 |
| Mel | NeuroVoz | 114 | **0.700** | 0.655 | −0.046 |

**The central finding, and it is a negative one: the sign of the DANN effect
flips with the backbone.** Same objective, same λ schedule, same splits, same
selection rule — only the trunk differs (44k MLP vs 224k CNN) — and the
direction reverses. The flip holds on balanced accuracy as well as AUC, so it
is not an artefact of the metric.

Reporting "DANN improves cross-corpus transfer" from the handcrafted row alone
would be reporting a backbone interaction as a method result.

The three hold-outs are **separate experiments and must not be averaged**:
holding out NeuroVoz trains on 747 files and tests on 2968, while holding out
MDVR-KCL trains on 3190 and tests on 73.

### 6.3 A second, separable failure: the decision threshold

Under LODO the model calls **83–96% of the held-out corpus PD** when the true
rate is 43–49%. Mean balanced accuracy over the twelve neural LODO
configurations:

| decision rule | mean BA | uses target features | uses target labels |
|---|---|---|---|
| fixed 0.5 | 0.623 | no | no |
| tuned on **source** validation speakers | **0.609** | no | no |
| **prior-matched** to design balance | **0.712** | yes | no |
| tuned on target labels (ceiling) | 0.743 | yes | yes |

**The decisive row is the second.** A threshold fitted on held-out speakers *of
the training corpora* transfers **worse than the naive 0.5**. That is the
signature of prior shift — the probability scale itself moves between corpora,
so no source-side calibration can repair it.

**Consequence: NeuroVoz LODO was never at chance.** At BA 0.508 it read as
total failure; its AUC is 0.700 and its prior-matched BA is 0.675. The
representation transfers considerably better than the reported accuracy
implied. What fails to transfer is the decision threshold, which is a different
and more tractable problem.

Prior matching reads the held-out corpus's own predictions (never its labels),
so it is **transductive** and is labelled as such throughout. AUC needs no such
caveat, which is why it is the headline.

### 6.4 Embedding geometry — and a correction

Ibarra's evidence for invariance is Table 5 (intra-class KL between corpora
falls) and Table 6 (intra-class TCM falls — classes tighten).

**KL reproduces on both backbones**: 11 of 12 corpus pairs drop, some sharply
(Mel HC ItalianPD/MDVR-KCL 36.84 → 1.47). The adversary does the job it is
given.

**TCM appeared to contradict the paper — and that reading was wrong.** Raw
intra-class TCM rises 7–13× (MLP) and 210–280× (CNN). But gradient reversal
places no constraint on the norm of the trunk output, and the cheapest way to
confuse a domain head is to grow it:

| family | mean ‖z‖ baseline → DANN |
|---|---|
| handcrafted (MLP) | 4.58 → 17.61 (**3.8×**) |
| Mel (CNN) | 6.97 → 103.78 (**14.9×**) |

TCM is a trace of covariance and scales with the square of the norm. After
L2-normalising each embedding, intra-class TCM **falls** under DANN — to 0.6×
(MLP) and 0.4–0.5× (CNN). **In direction, the classes do tighten; Table 6
reproduces after all.** What the adversary adds on top is an unconstrained
radial blow-up carrying no class information.

That blow-up is the mechanism behind the backbone sign flip: the MLP inflates
3.8× and loses nothing on mixed; the CNN inflates 14.9× and loses 0.073 on
mixed and up to 0.221 AUC on LODO. **The damage follows the inflation, not the
class geometry.** The bigger trunk simply has more room to inflate.

### 6.5 Controls

**Single-corpus baselines — the expected pooling penalty does not appear.**

| corpus | single-corpus | pooled | pooled − single |
|---|---|---|---|
| ItalianPD | 0.982 | 0.982 | 0.000 |
| MDVR-KCL | 0.741 | 0.765 | +0.024 |
| NeuroVoz | 0.744 | 0.737 | −0.007 |

Pooling three heterogeneous corpora costs essentially nothing at `duration`.
Consistent with `duration` having the weakest measured corpus signal, and with
DANN's small effect on mixed.

**Speaker aggregation is not a sensitive axis.** Mean-of-probabilities against
five alternatives, same checkpoints:

| rule | baseline | DANN |
|---|---|---|
| mean_prob (used) | 0.817 | 0.810 |
| median_prob | 0.817 | 0.819 |
| majority_vote | 0.817 | 0.819 |
| trimmed_mean | 0.817 | 0.810 |
| mean_logit | 0.813 | 0.802 |
| confident_mean | 0.800 | 0.807 |

Spread 0.017 against a fold std of 0.054–0.061. The two consistently worse
rules both up-weight confident segments — on this data the confident segments
are not the correct ones.

**A recording-protocol shortcut in ItalianPD.** PD/HC is **0.883 AUC
predictable from total recording seconds alone**, no speech content involved
(PD speakers record 1.3× longer). NeuroVoz is 0.775 in the *opposite*
direction. This partly explains ItalianPD's ceiling. It does **not** explain
the LODO failure — I tested whether predictions invert on transfer and they do
not (signs agree in 11 of 12 configurations).

---

## 7. What is and is not supported

**Supported**

1. Corpus identity is more learnable than pathology in raw features
   (0.892 vs 0.683) — the problem the paper describes is real on this data.
2. The Mel 1D-CNN baseline is the strongest model: 0.838 pooled BA, 0.915 AUC.
3. DANN reduces intra-class corpus KL on both backbones, and tightens classes
   directionally — both of Ibarra's geometric claims reproduce.
4. DANN does **not** improve pooled accuracy at any level or family.
5. The DANN effect on cross-corpus transfer **changes sign with the backbone**.
6. A fixed 0.5 threshold is invalid under corpus shift; source-side calibration
   is worse than none.
7. Speaker aggregation and corpus pooling are both non-sensitive axes here.

**Not supported**

- "DANN improves cross-corpus PD detection" — the sign depends on the backbone,
  single seed each.
- Any single pooled headline number — the ItalianPD ceiling inflates all of
  them.
- Averaged LODO — three different experiments.
- Any level claim on Mel — Mel is `duration` only.
- That constraining the embedding norm fixes DANN — the mechanism is measured,
  the fix is untested.

**Known limitations**

1. **Single seed** (`s0`) throughout. Each LODO cell is one run with no error
   bar; the effects driving the headline are 0.02–0.22 AUC.
2. Language and channel perfectly confounded — no clean language claim.
3. `duration` windows overlap 50%, so adjacent segments are not independent.
4. LODO is strict *generalisation*, not adaptation; a transductive variant
   would score better and answer a different question.
5. The repository has no git history, so runs record commit `unknown`.

---

## 8. Next steps, in order of value

| # | Action | Cost | Rationale |
|---|---|---|---|
| 1 | **Second seed** on handcrafted | ~3 h GPU | Nothing here has an error bar; the headline rests on 0.02–0.08 AUC margins |
| 2 | **Constrain the embedding norm** (L2 + learned temperature, or LayerNorm before the heads) | ~2 h GPU | Aimed at a measured mechanism; would separate "DANN aligns" from "DANN inflates" |
| 3 | **λ sweep** (currently pinned at λ_max = 1.0) | ~1 h GPU | Mel clearly over-adapts; λ was never tuned |
| 4 | Class-conditional adaptation (CDAN / MMD) | new code | Vanilla DANN aligns marginals, not class-conditionals — exactly the observed failure |
| 5 | Initialise git and commit | minutes | Restores reproducibility of every run |

Items 1–3 together are under a day of GPU on the existing laptop and would
convert most of §7's "not supported" list into either a result or a clean
negative.

---

## 9. Where to look

| what | where |
|---|---|
| Working specification, all nine stages | `PIPELINE.md` |
| Per-stage decisions and measurements | `reports/stage<N>_*.md` |
| Training details, selection rule, aggregation | `reports/stage7_training.md` |
| Full evaluation, geometry, threshold analysis | `reports/stage8_evaluation.md` |
| Figures (220 dpi PNG + vector PDF) | `notebooks/figures/` |
| Failure-mode diagnostics | `scripts/probe_failure_modes.py` |
| Aggregation comparison | `scripts/probe_aggregation.py` |
| Per-run outputs | `experiments/<run-id>/` |

Key figures: `stage8_ba_lodo.png` (the assigned headline),
`stage8_lodo_threshold.png` (the threshold failure),
`stage8_kl.png` / `stage8_tcm.png` (Ibarra Tables 5–6),
`tsne_handcrafted_duration_baseline_dann.png` (embeddings, corpus vs class).
