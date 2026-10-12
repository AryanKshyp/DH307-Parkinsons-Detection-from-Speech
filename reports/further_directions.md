# Further directions

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

Aryan Kashyap (23B0741) · Supervisor: Prof. Nirmal Punjabi · Guide: Mayurakshi Mukherji

Rewritten 2026-09-07, after the Stage 7 grid was re-run under the corrected
selection rule and Stage 8 was rescored. **The previous version of this note is
superseded**: its results table used checkpoints selected at λ = 0, and four of
its seven proposed tracks have since been executed. Read `RESULTS.md` first.

---

## 1. Where the project actually stands

118 neural runs plus controls, all under one selection rule. Three findings
frame everything below.

**F1 — The DANN effect changes sign with the backbone.** On LODO AUC, DANN is
+0.021 / +0.077 / −0.007 on the 44k-parameter MLP and −0.221 / 0.000 / −0.046
on the 224k-parameter CNN. Same objective, schedule, splits and selection rule.
This is the central result and it is a negative one.

**F2 — Gradient reversal inflates the embedding norm rather than reshaping
geometry.** Mean ‖z‖ goes 4.58 → 17.61 (MLP, 3.8×) and 6.97 → 103.78 (CNN,
14.9×). Raw intra-class TCM therefore rises 7–280×, which looked like Ibarra
Table 6 failing to replicate. After L2 normalisation the same quantity *falls*
to 0.4–0.6×: the classes do tighten directionally. **The damage tracks the
inflation, not the class geometry**, and the bigger trunk inflates more — which
is a mechanism for F1.

**F3 — A fixed 0.5 threshold is invalid under corpus shift.** The model calls
83–96% of the held-out corpus PD when the true rate is 43–49%. Prior-matching
recovers +0.089 mean BA. A threshold tuned on *source* validation speakers is
**worse than not calibrating at all** (0.609 vs 0.623), so the probability
scale itself moves between corpora.

Note what F3 is *not*. The true PD rate is roughly constant across the three
corpora (0.431 / 0.432 / 0.491), so this is **not label shift** — it is
covariate shift breaking calibration. That distinction picks the right
literature in §3.4 and rules out the obvious wrong fix.

Two supporting results worth keeping in view: the Mel CNN **baseline** is the
strongest model in the project (0.838 pooled BA, 0.915 AUC), and pooling three
corpora costs essentially nothing versus single-corpus training (−0.007 to
+0.024), which weakens the motivating premise on this data.

---

## 2. What the field looks like now

The 2023 reference paper is no longer the frontier. Three shifts matter.

**2.1 Self-supervised backbones have replaced handcrafted features and Mel
CNNs.** Current cross-lingual PD/dysarthria work builds on HuBERT-Large,
WavLM-Large and XLS-R-300M, usually with a low-capacity probe on a selected
layer rather than a trained CNN. `ParkMAE` (2025) goes further and pretrains a
multilingual PD-specific masked autoencoder on ~750 h across four languages,
with ~100 h of evaluation data, explicitly as a benchmark. **This project's
backbones are two generations behind the literature**, which is a fair critique
to pre-empt rather than discover in a viva.

**2.2 The most useful recent results are negative, and they rhyme with F1.**
*Motor, Cognitive, or Corpus?* (2026) probes nine SSL backbones layer-wise
across corpora and reports two things this project should take seriously:
optimal layer choice is **corpus-dependent rather than architecture-dependent**,
and — more damaging — the transferred classifier **lacks pathological
specificity**, assigning similarly high PD probability to *dementia* speech.
That is the same genre of finding as F1: what looks like a PD detector
generalising is often a corpus artefact or a generic "disordered speech"
detector.

**2.3 Simple representation alignment is competitive with adversarial
training.** *Adapting Self-Supervised Speech Representations for Cross-lingual
Dysarthria Detection in PD* (2026) proposes a representation-level **language
shift**: a centroid-based vector offset aligning source-language embeddings to
the target, **estimated from healthy-control speech only**. It improves
cross-lingual sensitivity and F1 and measurably reduces language identity in
the embedding space — with no adversary, no GRL, and no λ to tune. A separate
line uses instance-based adaptation via **kernel mean matching** for
multilingual PD.

The HC-anchored detail matters enormously here. Stage 5 established that
per-corpus and per-speaker normalisation destroy PD because the label lives in
the between-speaker spread. **A centroid estimated from controls only does not
have that failure mode**: it removes a corpus offset without collapsing the
PD direction. It is the one "shallow" alignment this project has not tried and
the one that does not violate Invariant 2.

---

## 3. Can we have better DANNs? Yes — four specific upgrades

Ranked by how directly each addresses a *measured* failure in this project, not
by novelty.

### 3.1 Constrain the feature norm — BSP or an explicit normalisation

**Targets F2 directly.** Chen et al., *Transferability vs. Discriminability:
Batch Spectral Penalization* (ICML 2019), show that adversarial adaptation
makes features transferable at the cost of discriminability, and that the
damage concentrates in the **top singular values** of the feature matrix —
a few directions grow and dominate, crowding out class-discriminative ones.
BSP adds a penalty on those top singular values.

This is F2 stated in the vocabulary of the DA literature. Two variants, in
increasing order of faithfulness to the diagnosis:

- **Cheap:** L2-normalise the trunk output with a learned temperature, or a
  LayerNorm before the heads. Removes the radial escape route entirely.
- **Faithful:** BSP — penalise the top-k singular values of the batch feature
  matrix, so the adversary cannot buy alignment by inflating a handful of
  directions.

**Prediction, worth writing down before running:** if F2 is the mechanism, the
CNN should recover most of its 0.221 AUC loss on ItalianPD, and the MLP should
change little (it inflates only 3.8×). If the CNN does *not* recover, F2 is a
correlate rather than a cause and the capacity story in §4.1 is wrong.

**Cost:** ~2 h handcrafted, ~10 h with Mel.

### 3.2 Condition the adversary on the label — CDAN

Vanilla DANN aligns the **marginal** p(z) across corpora. It has no mechanism
to align p(z | PD) and p(z | HC) separately, which is exactly the failure the
geometry shows: KL between corpora falls (marginals align) while class
structure does not improve. **CDAN** conditions the domain discriminator on the
classifier's predictions via multilinear conditioning, plus entropy
conditioning to down-weight uncertain examples.

This is the principled answer to "the adversary aligned the wrong thing", and
it is a small change to `src/models/classifier.py` — the domain head takes the
outer product of features and softmax predictions instead of features alone.

**Caveat specific to this data:** entropy conditioning down-weights uncertain
predictions, and under F3 the predictions are badly miscalibrated on the target.
CDAN and calibration (§3.4) interact; run §3.4 first or expect to confound them.

### 3.3 Smooth the loss landscape — SDAT

**SDAT** (Rangwani et al., ICML 2022) shows that adversarial DA converges to
sharp minima of the *task* loss, and that selectively smoothing the
classification component (sharpness-aware minimisation on the label head only,
not the adversary) improves every DANN-family method it is applied to. It is a
generic wrapper, roughly ten lines, and is the cheapest thing on this list that
is likely to help both backbones.

Relevant here because the CNN's collapse under DANN has the signature of an
unstable optimisation rather than a representational impossibility — the same
architecture is the project's best model without the adversary.

### 3.4 Fix the decision rule properly — calibration under covariate shift

F3 is currently patched with a prior-matched threshold, which works (+0.089)
but is ad hoc and transductive. The principled versions:

- **CPCS** (Park et al., AISTATS 2020) and **TransCal** — importance-weighted
  calibration, where a source-vs-target discriminator supplies density-ratio
  weights and the source calibration loss is reweighted onto the target.
- **PseudoCal** — source-free, avoids needing source data at calibration time.

Do **not** reach for BBSE, the Saerens EM procedure, or LaSCal here. Those
correct **label shift** and assume p(Y) changes with p(X|Y) fixed. In this data
p(Y) is stable at 0.43–0.49 and p(X|Y) is what moves. Applying a label-shift
estimator to a covariate-shift problem would be a methodological error, and it
is the sort of thing an examiner will ask about.

There is also an honest cheap option: because the corpora are ~50/50 by design,
the prior does not need estimating at all — it is known. Keep the prior-matched
number, label it transductive, and report AUC as the headline. That is what is
implemented now and it is defensible.

---

## 4. Ablations that would change what we believe

The project's weakness is not a shortage of methods; it is that **every number
is a single seed** and the central finding rests on a two-point capacity
comparison. These are ordered by inferential value per GPU-hour.

### 4.1 The capacity ladder — the direct test of F1 ★ highest value

F1 currently rests on exactly two architectures that differ in *both* capacity
and input representation (40 handcrafted features vs 65-band Mel). That is a
confounded comparison, and it is the headline.

Train the **same** Mel CNN at four widths (e.g. 32 / 64 / 128 / 256 channels,
spanning ~30k to ~500k parameters), baseline and DANN, mixed and LODO. Plot
Δ(DANN − baseline) and mean ‖z‖ inflation against parameter count.

**This converts an anecdote into a curve.** If the DANN benefit declines
monotonically with capacity and inflation rises with it, F1 and F2 become a
single coherent claim with a mechanism. If the two backbones simply differ and
there is no trend, F1 is about handcrafted-vs-Mel, not about capacity, and the
write-up must say so.

**Cost:** ~6 h. **Do this before any new method.**

### 4.2 Shuffled-domain control — is the adversary doing anything? ★

Run DANN with **randomly permuted corpus labels**. The GRL then regularises
without carrying any domain information.

The MLP's LODO gain is +0.021 to +0.077 AUC on one seed. If random-label DANN
produces a similar gain, that gain is stochastic regularisation, not domain
alignment, and the project's one positive result evaporates. **No DANN claim in
this thesis is safe until this control exists.** It is one grid, ~1 h
handcrafted, and it is the single cheapest way to be wrong less often.

### 4.3 Seeds ★

Three seeds minimum on handcrafted. Every LODO cell is currently one run, and
the effects under discussion are 0.02–0.08 AUC. ~3 h. Non-negotiable before any
claim is written down.

### 4.4 The missing skyline — how much transfer is even available?

There is no **target-trained upper bound** for LODO. Train and test within the
held-out corpus (speaker-disjoint) to get the ceiling any adaptation method
could reach, and compare against a **source-only** floor. Without those two
numbers, "LODO AUC 0.700 on NeuroVoz" is uninterpretable — it could be 90% of
what is achievable or 60%.

Cheap (`--regime single` already exists and the within-corpus runs are done for
mixed) and it reframes every LODO row in the thesis.

### 4.5 λ sweep

λ_max has been pinned at 1.0 throughout and never tuned. The CNN clearly
over-adapts. Sweep λ_max ∈ {0.1, 0.3, 1.0} on both backbones. If the CNN's
collapse disappears at λ = 0.1, then F1 is partly "λ was too large for this
capacity" — which is a weaker and more accurate claim than "DANN fails on
CNNs". ~2 h handcrafted.

### 4.6 Number of source domains

DANN can only align the corpora it sees. With three corpora, each LODO
configuration gives the adversary **two** source domains — the bare minimum for
domain generalisation, and arguably too few for it to work at all. Compare
2-source LODO against a 1-source variant (adversary disabled, no domains to
align) to quantify what the second domain buys. This is a design-limit
argument, and it may be the honest explanation for why LODO does not close.

### 4.7 A specificity control

Following *Motor, Cognitive, or Corpus?*: the model may be detecting
"disordered or elderly speech" rather than Parkinsonian speech. A cheap proxy
with data already on disk — check whether PD probability tracks age (where
metadata exists) and recording length. We know ItalianPD PD/HC is **0.883 AUC
from recording seconds alone**, so at least one corpus has a non-speech route
to the label. A stronger version needs a non-PD disordered-speech corpus, which
is out of scope this semester but is the right next dataset acquisition.

### 4.8 HC-anchored centroid shift — the non-adversarial competitor

From §2.3. Compute the healthy-control centroid per corpus on training data,
shift target embeddings by the source−target centroid offset, re-score. No
training. It respects Invariant 2 because the offset is estimated from controls
only and so does not collapse the between-speaker PD spread.

**If this matches or beats DANN on LODO, it is the most publishable result the
project can produce** — a negative result on adversarial training with a simple,
strictly cheaper alternative that works. It is roughly an afternoon of work on
saved embeddings.

---

## 5. Larger directions, if the scope can grow

**5.1 SSL backbone (highest expected accuracy).** Frozen WavLM-Large or
HuBERT-Large features with a linear probe, on the existing splits and reporting
rules. Two warnings from §2.2: choose the layer on **source validation only**
(layer choice is corpus-dependent, so tuning it on the target is leakage), and
expect the corpus signal to be strong — SSL embeddings encode channel and
language heavily, which is precisely the setting DANN is meant for. This is the
single change most likely to raise absolute numbers, and it makes the DANN
question more interesting, not less.

**5.2 Task-matched two-domain.** Language, channel and task are perfectly
confounded across the three corpora. ItalianPD vs MDVR-KCL on the **reading
passage** is a matched-task, two-domain experiment that isolates **channel**
(clinic mic vs smartphone). It supports a defensible "channel-invariant" claim
where the current three-corpus design cannot support a language claim at all.

**5.3 Severity rather than detection.** MDVR-KCL carries Hoehn & Yahr and two
UPDRS items in the filename; NeuroVoz has GRBAS. An ordinal head uses labels
that binary PD/HC discards, and severity may transfer where detection does not.
Different project, but the data is already there.

**5.4 Transductive UDA, declared as such.** Let unlabelled target audio into the
domain head only. Most of the DANN literature reports this setting. It will
score better. It is **not** the assigned strict LODO and must not silently
replace it.

---

## 6. What not to do

- **Any new method before §4.2 and §4.3.** A shuffled-domain control and three
  seeds cost ~4 h combined and could invalidate the project's only positive
  result. Adding CDAN on top of an unverified +0.02 is building on sand.
- **A five-method bake-off.** MixStyle, ComBat, Deep CORAL, transformer and
  HuBERT in one semester produces five underpowered numbers, not a result.
- **Label-shift correction** (BBSE / EM / LaSCal) for the F3 threshold problem.
  Wrong shift type; see §3.4.
- **Averaging the three LODO configurations.** They train on 747 to 3190 files
  and test on 73 to 2968. They are different experiments.
- **Treating ItalianPD 0.98 as a detection result.** 0.883 AUC of it is
  available from recording duration alone.
- **Tuning the SSL layer on target data.** Corpus-dependent layer choice makes
  this a very easy leak to commit by accident.
- **Reinstating FredPrior** to unconfound language from channel. It reintroduces
  the 8 kHz bandwidth cue and contains none of the assigned tasks.

---

## 7. A concrete two-week plan

Assumes the existing laptop GPU (~25 min per Mel run, ~2 min per handcrafted
run) and no new data.

| week | work | GPU | what it buys |
|---|---|---|---|
| 1 | §4.2 shuffled-domain control, §4.3 three seeds, §4.4 skyline/floor | ~5 h | Establishes whether the MLP's DANN gain is real and what the achievable range is. **Any of these can falsify the current headline.** |
| 1 | §4.8 HC-anchored centroid shift | ~0 h (saved embeddings) | A non-adversarial competitor, free |
| 2 | §4.1 capacity ladder | ~6 h | Turns F1 from a two-point anecdote into a curve with a mechanism |
| 2 | §3.1 norm constraint on the two extremes of that ladder | ~3 h | Tests F2 causally |

That is ~14 h of compute and it addresses every "not supported" item in
`RESULTS.md` §7. It deliberately contains **no new architecture**: the
project's problem is inferential strength, not method count.

If more time exists after that, §5.1 (SSL backbone) is the highest-value
addition, and §3.3 (SDAT) is the cheapest method upgrade.

---

## 8. Questions for the supervisor

1. **Is a negative result acceptable as the thesis headline?** The strongest
   honest statement available is "DANN's cross-corpus benefit is a backbone
   interaction, and the geometric claims of the reference paper replicate only
   after controlling for embedding norm." That is a real contribution and it is
   defensible — but it is not "we improved PD detection."
2. **Inferential depth or method breadth?** §4 spends two weeks making the
   existing result trustworthy. An alternative spends it adding CDAN/SDAT/SSL
   for higher numbers with the same single-seed weakness. I recommend the
   former, and §4.2 is the reason.
3. **Should the SSL gap be closed?** The field has moved to WavLM/HuBERT/XLS-R
   and PD-specific pretraining (ParkMAE). Handcrafted features and a Mel CNN are
   two generations behind. Is matching the current literature in scope, or is
   the methodological contribution (forced alignment, segment levels, honest
   protocol) the deliverable?
4. **Is the title claim revised?** Language, channel and task are perfectly
   confounded. "Corpus-invariant" is supportable; "language-invariant" is not,
   without §5.2 or a fourth corpus.

---

## Sources

- Ibarra et al. (2023), *Towards a Corpus (and Language)-Independent Screening
  of Parkinson's Disease from Voice and Speech through Domain Adaptation* —
  https://doi.org/10.3390/bioengineering10111316
- *Motor, Cognitive, or Corpus? What Survives Cross-Lingual Transfer in
  Speech-Based Parkinson's Disease Detection* — https://arxiv.org/abs/2608.13425
- *Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria
  Detection in Parkinson's Disease* — https://arxiv.org/abs/2603.22225
- *ParkMAE: a cross-linguistic masked autoencoder framework for robust
  Parkinson's disease detection from speech* —
  https://pubmed.ncbi.nlm.nih.gov/41372338/
- *Multilingual voice signal analysis for Parkinson's disease detection using
  instance-based domain adaptation through kernel mean matching* —
  https://www.sciencedirect.com/science/article/abs/pii/S0885230826000732
- Chen et al., *Transferability vs. Discriminability: Batch Spectral
  Penalization for Adversarial Domain Adaptation* (ICML 2019) —
  https://proceedings.mlr.press/v97/chen19i.html
- Rangwani et al., *A Closer Look at Smoothness in Domain Adversarial Training*
  (SDAT) — https://openreview.net/pdf?id=Fj1Tpym9KxH
- Park et al., *Calibrated Prediction with Covariate Shift via Unsupervised
  Domain Adaptation* (CPCS, AISTATS 2020) — https://arxiv.org/abs/2003.00343
- *PseudoCal: A Source-Free Approach to Unsupervised Uncertainty Calibration in
  Domain Adaptation* — https://arxiv.org/abs/2307.07489
- Lipton et al., *Detecting and Correcting for Label Shift with Black Box
  Predictors* (BBSE) — cited as **not applicable** here; see §3.4
