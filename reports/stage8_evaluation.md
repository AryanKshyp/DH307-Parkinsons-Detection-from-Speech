# Stage 8 — Evaluation

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Disease Detection from Speech

Scored on the Stage 7 grid re-run under the corrected checkpoint-selection rule
(`min_select_epoch`, see `reports/stage7_training.md`). **Nothing from the
previous version of this report is carried forward** — every DANN number here
comes from a checkpoint with λ ≥ 0.68, where half the old ones had λ = 0.

Two families at `duration` (handcrafted MLP, Mel 1D-CNN), plus three levels on
handcrafted. The headline is not "DANN vs baseline on mixed". It is whether
DANN closes **LODO**, and whether the embedding geometry matches Ibarra et al.
Tables 5–6: lower intra-class KL between corpora, tighter class clouds (lower
TCM).

**In:** `experiments/{mixed,lodo}_*_{handcrafted,mel}_duration_*`
**Out:** `reports/stage8_*.csv`, `notebooks/figures/stage8_*`
**Code:** `scripts/evaluate.py`, `scripts/plot_stage8.py`, `src/evaluation/`

```
python scripts/evaluate.py --family handcrafted --level duration
python scripts/evaluate.py --family mel --level duration
python scripts/plot_stage8.py
python scripts/plot_stage8.py --family mel --level duration
```

Handcrafted `duration` keeps the original unsuffixed figure stems; Mel figures
append `_mel_duration`.

---

## The result in one table

Speaker-level balanced accuracy. Mixed is pooled out-of-fold over all 216
speakers; LODO is one number per held-out corpus, and the three are **separate
experiments that must not be averaged**.

| | hc baseline | hc DANN | mel baseline | mel DANN |
|---|---|---|---|---|
| **mixed pooled** | 0.817 | 0.810 | **0.838** | 0.765 |
| mixed NeuroVoz | 0.737 | 0.720 | **0.788** | 0.675 |
| LODO ItalianPD — BA@0.5 | 0.776 | 0.816 | 0.789 | 0.614 |
| **LODO ItalianPD — AUC** | 0.850 | **0.872** | **0.903** | 0.681 |
| LODO MDVR-KCL — BA@0.5 | 0.595 | 0.644 | 0.571 | 0.548 |
| **LODO MDVR-KCL — AUC** | 0.685 | **0.762** | **0.807** | **0.807** |
| LODO NeuroVoz — BA@0.5 | 0.508 | 0.526 | 0.576 | 0.516 |
| **LODO NeuroVoz — AUC** | **0.700** | 0.692 | **0.700** | 0.655 |

Fold spread on mixed: handcrafted 0.061 (baseline) / 0.054 (DANN); Mel 0.092 /
0.093. LODO has no spread — one run per held-out corpus.

**LODO is reported on AUC.** Balanced accuracy at a fixed 0.5 threshold is not
a fair measure of a cross-corpus representation, for a reason that is measured
rather than argued in the next section: under domain shift the model calls
83–96% of the held-out corpus PD when the true rate is 43–49%. AUC is
threshold-free and unaffected. `balanced_acc` is kept in the CSVs for
continuity and `balanced_acc_prior` beside it.

**The DANN effect on LODO flips sign with the backbone.**

On AUC, the headline metric:

| held out | handcrafted DANN − baseline | mel DANN − baseline |
|---|---|---|
| ItalianPD | **+0.021** | **−0.221** |
| MDVR-KCL | **+0.077** | 0.000 |
| NeuroVoz | −0.007 | −0.046 |

and on balanced accuracy at 0.5, for comparison:

| held out | handcrafted DANN − baseline | mel DANN − baseline |
|---|---|---|
| ItalianPD | +0.040 | −0.175 |
| MDVR-KCL | +0.049 | −0.023 |
| NeuroVoz | +0.018 | −0.060 |

**The sign flip survives the change of metric**, which is the point of showing
both — it is not an artefact of the broken threshold. One seed each. **This is the central finding of
the stage, and it is a negative one:** the adversary's benefit on
leave-one-dataset-out is not a property of domain-adversarial training on these
corpora, it is a property of the 44k-parameter MLP. Swap in a 224k-parameter
CNN — same objective, same λ schedule, same splits, same selection rule — and
the sign reverses on all three hold-outs.

Reporting "DANN improves cross-corpus transfer" from the handcrafted column
alone would be reporting a backbone interaction as a method result.

---

## The fixed threshold is a second, separable failure

`scripts/probe_failure_modes.py --check threshold` isolates the decision rule
from the representation. Under LODO the predicted PD rate runs far above the
true one:

| held out | true PD rate | predicted PD rate (hc baseline) |
|---|---|---|
| ItalianPD | 0.431 | 0.508 |
| MDVR-KCL | 0.432 | 0.892 |
| NeuroVoz | 0.491 | 0.956 |

Mean balanced accuracy over the twelve neural LODO configurations, under four
decision rules:

| rule | mean BA | uses target features | uses target labels |
|---|---|---|---|
| fixed 0.5 | 0.623 | no | no |
| tuned on **source** validation speakers | **0.609** | no | no |
| **prior-matched** to the design balance | **0.712** | yes | no |
| tuned on target labels (ceiling) | 0.743 | yes | yes |

The decisive row is the second. A threshold fitted on held-out speakers *of
the training corpora* transfers **worse than the naive 0.5**. That is the
signature of prior shift: the probability scale itself moves between corpora,
so no source-side calibration can fix it. Prior matching recovers +0.089 of
the 0.120 available.

**Prior matching is transductive.** It reads the held-out corpus's own
predictions — never its labels — to place the threshold. Stage 6 requires
this to be declared, and it is: any `*_prior` column is a transductive number
and is not comparable to the strict inductive protocol. AUC needs no such
caveat, which is why it is the headline above and prior-matched BA is a
secondary column.

The practical reading: **NeuroVoz LODO was never at chance.** At 0.508 it
looked like total failure; its AUC is 0.700 and its prior-matched BA is 0.675.
The representation transfers considerably better than the reported accuracy
implied. What does not transfer is the decision threshold.

---

## Why the geometry explains it

Ibarra's evidence for language invariance is Table 5 (intra-class KL between
corpora falls) and Table 6 (intra-class TCM falls — classes get tighter).
Diagonal Gaussians in 16-d PCA of the speaker embeddings for KL; unreduced
trace of covariance for TCM.

**KL behaves exactly as the paper says, on both families.**

| class | pair | hc base | hc DANN | mel base | mel DANN |
|---|---|---|---|---|---|
| HC | IT / MDVR | 16.68 | **5.26** | 36.84 | **1.47** |
| HC | IT / NV | 16.44 | **3.27** | 12.43 | **2.65** |
| HC | MDVR / NV | 3.90 | **1.39** | 7.35 | **3.51** |
| PD | IT / MDVR | 4.55 | **2.49** | 8.09 | **2.53** |
| PD | IT / NV | 1.98 | **1.34** | 2.96 | 3.47 |
| PD | MDVR / NV | 3.54 | **1.24** | 9.12 | **7.10** |

Eleven of twelve pairs drop. The adversary does the job it was given.

**Raw TCM rises sharply — but that is embedding scale, not class geometry.**

| class | hc baseline | hc DANN | mel baseline | mel DANN |
|---|---|---|---|---|
| HC | 13.0 | 96.2 (**7×**) | 25.3 | 5318.9 (**210×**) |
| PD | 9.3 | 121.3 (**13×**) | 19.8 | 5535.6 (**280×**) |

Taken at face value this says DANN smears the classes and contradicts Ibarra
Table 6. **That reading is wrong**, and `scripts/probe_failure_modes.py
--check norm` shows why. Gradient reversal puts no constraint on the norm of
the trunk output, and the cheapest way to confuse a domain head is to grow it:

| family | mean ‖z‖ baseline → DANN |
|---|---|
| handcrafted (MLP) | 4.58 → 17.61 (**3.8×**) |
| mel (CNN) | 6.97 → 103.78 (**14.9×**) |

TCM is a trace of covariance, so it scales with the square of the embedding
norm. Remove the scale by L2-normalising each speaker embedding and the same
quantity **falls** under DANN:

| class | hc baseline | hc DANN | mel baseline | mel DANN |
|---|---|---|---|---|
| HC | 0.453 | **0.290** (0.6×) | 0.426 | **0.171** (0.4×) |
| PD | 0.421 | **0.273** (0.6×) | 0.382 | **0.187** (0.5×) |

So **in direction, the classes do tighten — Ibarra Table 6 reproduces after
all.** What the adversary adds on top is an unconstrained radial blow-up that
carries no class information and destabilises the PD head downstream.

That blow-up is the mechanism behind the backbone sign flip. The MLP inflates
3.8× and loses nothing on mixed (−0.007) while gaining on LODO; the CNN
inflates 14.9× and loses 0.073 on mixed and up to 0.221 AUC on LODO. **The
bigger trunk has more room to inflate, and the damage follows the inflation,
not the geometry.**

This is a concrete, testable fix rather than a dead end: constrain the
embedding — L2-normalise with a learned temperature, or a LayerNorm before the
heads — and the adversary keeps the alignment it earns while losing the
degenerate escape route. It is **not yet validated**; a cheap probe refitting
only the PD head on frozen embeddings was too underpowered to decide (LODO
validation sets are 16–27 speakers for 128–256-d weights). It needs a retrain.

Raw TCM is a trace over unreduced dimensions and the two families have
different embedding widths — 128-d handcrafted, 256-d Mel — so compare
baseline-to-DANN *within* a family, never magnitudes across families.

---

## Mixed protocol, pooled OOF, 216 speakers

Precision and recall are for PD.

| family | regime | dataset | n | acc | BA | prec | rec | AUC |
|---|---|---|---|---|---|---|---|---|
| hc | GBT | all | 216 | 0.782 | 0.777 | 0.805 | 0.700 | 0.883 |
| hc | baseline | all | 216 | 0.819 | 0.817 | 0.814 | 0.790 | 0.902 |
| hc | DANN | all | 216 | 0.810 | 0.810 | 0.786 | 0.810 | 0.890 |
| mel | baseline | all | 216 | 0.843 | **0.838** | 0.875 | 0.770 | **0.915** |
| mel | DANN | all | 216 | 0.769 | 0.765 | 0.766 | 0.720 | 0.863 |
| hc | baseline | ItalianPD | 65 | 0.985 | 0.982 | 1.000 | 0.964 | 0.999 |
| hc | baseline | MDVR-KCL | 37 | 0.784 | 0.765 | 0.833 | 0.625 | 0.824 |
| hc | baseline | NeuroVoz | 114 | 0.737 | 0.737 | 0.724 | 0.750 | 0.816 |
| mel | baseline | NeuroVoz | 114 | 0.789 | **0.788** | 0.833 | 0.714 | 0.847 |

**Mel baseline is the best configuration in the project** — 0.838 pooled,
NeuroVoz 0.788, AUC 0.915. It survives the selection-rule correction, so it is
a real result and not an artefact of the archived checkpoints.

ItalianPD sits at 0.98 for every configuration and carries no information about
which spine is better. NeuroVoz, 114 speakers, is the corpus that moves and the
one to quote. MDVR-KCL is 37 speakers and its recall swings on two or three
patients.

---

## Levels — handcrafted only

Mel `utterance` and Mel `phoneme`/vowel were dropped by decision (~13 h GPU), so
every *level* claim below rests on the handcrafted family alone.

| level | regime | pooled | ItalianPD | MDVR-KCL | NeuroVoz |
|---|---|---|---|---|---|
| duration | baseline | 0.817 | 0.982 | 0.765 | 0.737 |
| duration | DANN | 0.810 | 0.982 | 0.765 | 0.720 |
| utterance | baseline | 0.818 | 0.982 | 0.686 | **0.764** |
| utterance | DANN | 0.809 | 0.964 | 0.662 | 0.764 |
| phoneme/vowel | baseline | 0.834 | 1.000 | — | 0.735 |
| phoneme/vowel | DANN | 0.796 | 1.000 | — | 0.674 |

`utterance` is the best level for NeuroVoz and the worst for MDVR-KCL — MDVR
utterances are 141 s dialogues chunked to 4 s with interviewer speech in them,
NeuroVoz utterances are 3.5 s prompted phrases. `phoneme`/vowel reaching 1.000
on ItalianPD is a ceiling artefact on 64 speakers across 2 corpora; the pooled
0.834 must not be quoted as a headline.

DANN does not help at any level on mixed.

---

## Aggregation

Speaker decisions are mean-of-probabilities then argmax. Measured against five
alternatives on the same checkpoints (`scripts/probe_aggregation.py`), the
whole spread across six rules is **0.017**, against a fold-to-fold standard
deviation of 0.054–0.061. Aggregation is not a sensitive axis on this spine.
Full table in `reports/stage7_training.md`.

---

## Selection integrity

Every DANN checkpoint in this report was saved with the adversary engaged:

| family | n folds | λ at checkpoint (min / median / max) | folds with λ=0 |
|---|---|---|---|
| handcrafted | 10 | 0.682 / 0.969 / 0.996 | **0** |
| mel | 10 | 0.682 / 0.870 / 0.997 | **0** |

Best epochs range 5–19 (handcrafted) and 5–20 (Mel). Under the old rule, five
of ten handcrafted folds saved λ = 0. Baseline is selected under the identical
floor, so the comparison is like-for-like.

---

## Against Ibarra et al.

1. **KL behaves like the paper.** Intra-class corpus distributions converge, on
   both backbones, on 11 of 12 pairs.
2. **TCM also behaves like the paper, once scale is removed.** Unit-norm
   intra-class TCM falls to 0.4–0.6× under DANN. The raw 7–280× rise is an
   artefact of unconstrained embedding norm, not evidence against Table 6.
3. **LODO, the assigned headline, still does not close.** DANN improves the MLP
   (+0.021 to +0.077 AUC on two of three hold-outs) and degrades the CNN (0.000
   to −0.221). NeuroVoz sits at 0.655–0.700 AUC for every configuration.
4. **Mixed accuracy does not improve** on any family or level.
5. **A fixed 0.5 threshold is not a valid cross-corpus measure here.** It costs
   0.089 mean balanced accuracy against a prior-matched rule, and source-side
   calibration transfers worse than not calibrating at all.

Ibarra report mixed-corpus CV, not strict LODO, and this project's LODO is
*generalisation* (the held-out corpus appears nowhere in training), not
adaptation. A transductive variant would score better and answer a different
question.

---

## What this stage does not support

- **"DANN improves cross-corpus PD detection."** The sign depends on the
  backbone, on both AUC and balanced accuracy. One seed each.
- **"DANN smears the class clouds."** Withdrawn — that was a scale artefact.
  What is supported is that DANN inflates the embedding norm 3.8–14.9×, and
  that the inflation, not the class geometry, tracks the damage.
- **Any pooled single number.** The ItalianPD ceiling inflates all of them.
- **Averaged LODO.** Three different experiments.
- **A level claim on Mel.** Mel is `duration` only.
- **Uncertainty on LODO.** Mixed has ten folds; each LODO cell is a single run
  with no spread. The LODO deltas have no error bar, and a second seed is the
  cheapest thing that would give them one.
- **Any inductive claim from a `*_prior` column.** Those are transductive.
- **That constraining the embedding norm would fix DANN.** The mechanism is
  measured; the fix is untested.
