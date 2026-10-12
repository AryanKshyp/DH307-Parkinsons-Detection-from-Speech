# Follow-up: is the residual corpus cue second-order, and can anything shallow remove it?

> **Status: complete.** All of R0–R6 ran on the matrix below; results and the
> write-up are in `reports/ablation_corpus_norm.md`. Outcomes against the
> pre-registered kill criteria:
>
> | run | result | criterion hit |
> |---|---|---|
> | R0 | corpus 0.757 [0.730, 0.786], PD 0.738 [0.704, 0.771] | PD half-width 0.034 > 0.03, so PD differences under ~0.035 were not read as findings |
> | R1 | L0 0.569, L1 0.570, L2 0.669, L3 0.757, L2z 0.591 | L2 − L1 = +0.098 and L2 = 0.79 of the span, so **R4 authorised** on both criteria |
> | R2 | task-matched 0.821 / 0.807 vs all-task 0.840 / 0.799; age-controlled 0.836 | task mix, channel and age are each **not** the cue |
> | R3 | corpus 0.701 at k=3 (9 of 40 dims removed), PD 0.723; LODO flat | corpus ≥ 0.70 after 3 rounds → **the cue is non-linear**; shallow track retired |
> | R4 | corpus **0.855** (up from 0.757), PD 0.726; LODO 1 of 3 improved | fails the promotion bar on the numbers and by construction |
> | R5 | `duration_100` corpus 1.000 under `corpus` scope | ≥ 0.97 → the window is not the story; no third size, no `utterance` run |
> | R6 | mean-pooled Mel corpus 0.774, PD 0.744 | closes the Stage 5 Mel open item |
>
> The prediction recorded for R1 (L2 at 0.55–0.68 of the span) was low: it came
> out at 0.79. Correlations carry more of the cue than expected — and matching
> them still made corpus identity easier, which is the finding.

Successor to `PLAN.md`. A1 (feature importance) and the corpus normalisation
scope are done; `reports/ablation_corpus_norm.md` is the write-up. This file
plans what comes after, and it reorders `PLAN.md`'s Track A on the strength of
two facts measured while planning it.

Scope: duration + handcrafted, `handcrafted_duration.npy`, Stage 6 splits, GBT
thermometer, 40k seed-0 sample, speaker-grouped 5-fold, train-only statistics.
No new feature cache, no neural runs, no change to the mainline feature path or
to Stage 7 defaults (`norm=global`, all 40 columns).

---

## Two corrections to the premise, before spending an afternoon

**1. `norm=corpus` is not mean-only.** `fs.fit_stats(..., "corpus")` stores
`(mean, std)` per dataset and `apply` divides by the std, so the completed run
already equalised per-corpus **means and marginal variances**. "Equalising
first-order shifts" understates what was done, and it narrows the residual:
marginal variance is already gone.

Correlation is invariant to a per-column affine map with positive scale, so each
corpus's **correlation matrix is exactly unchanged** by `norm=corpus`. The
remaining second-order residual is therefore precisely the three correlation
matrices — which makes CORAL a much narrower experiment than `PLAN.md` implies.
CORAL's only increment over the run already completed is the off-diagonal.

Measured on the same 40k sample, after `norm=corpus`:

| pair | off-diagonal Frobenius distance | max abs difference |
|---|---|---|
| ItalianPD vs MDVR-KCL | 8.45 | 0.61 |
| ItalianPD vs NeuroVoz | 4.21 | 0.38 |
| MDVR-KCL vs NeuroVoz | 10.76 | 0.87 |

Mean absolute off-diagonal correlation: ItalianPD 0.213, MDVR-KCL 0.138,
NeuroVoz 0.271. Higher marginal moments differ too — mean absolute excess
kurtosis 5.80 / 2.58 / 2.94, mean absolute skew 1.12 / 0.85 / 0.97. So **both**
the second-order channel and the higher-order channel are live. Neither can be
assumed to be the whole story, which is why R1 apportions them before R4 is
authorised.

**2. Per-corpus z-score and CORAL both need the corpus label at inference.**
They are domain-*conditional* maps, not domain removal. DANN and a linear
projection are one global map applied identically to every row. Those are
different classes of method, and their numbers are not on the same axis:

| method | test-time corpus label? | corpus BA comparable to 0.757? | valid under strict LODO? |
|---|---|---|---|
| `global` | no | yes (reference) | yes |
| `corpus` z-score | **yes** | no | no — pooled fallback only |
| CORAL | **yes** | no | no — pooled fallback only |
| linear projection (INLP) | no | yes | yes |
| DANN | no | yes | yes |

This is why `norm=corpus` strict LODO took ItalianPD to 0.504: the method has no
defined behaviour on a corpus it has not seen. CORAL inherits that exactly. So
**R3 (linear projection) is promoted ahead of CORAL** — it is the only candidate
in `PLAN.md` that is like-for-like with DANN.

---

## Promotion bar — what "deserves to sit next to DANN" means

A shallow method qualifies as a control DANN must beat only if **all** hold:

1. one global map, no corpus label at inference;
2. fitted on training speakers only;
3. computable under strict LODO with zero information from the held-out corpus;
4. mixed **corpus BA at or below 0.55** with mixed **PD BA at or above 0.70**; and
5. strict-LODO PD BA above the `global` reference (ItalianPD 0.591, MDVR-KCL
   0.599, NeuroVoz 0.521) on **at least two of three** held-out corpora.

Only R3 can clear 1–3. R4 fails 1 and 3 by construction and can only ever be
reported as a transductive extra. Nothing is reported as corpus BA alone; every
method is a **(corpus BA, PD BA)** pair.

References to beat or hold: corpus 0.757 (chance 0.333), PD/HC 0.738 (chance
0.500). PD 0.600 is the floor from `feature_drop.csv` — the cost of deleting the
shared 16 columns. A method that lands there has bought nothing.

---

## Ordered matrix

Cost is measured, not estimated: loading `handcrafted_duration.npy` plus
`segments.csv` takes 93 s; one 3-way GBT fold on 32k x 40 takes 24 s, so one
pooled 5-fold score is about 2 min (3-way) or 1 min (binary). Nothing here is
compute-limited, so the order is by information gain, not by cost.

### R0 — reference re-run with per-fold scores and a speaker bootstrap CI

- **Consumes** `handcrafted_duration.npy`, `segments.csv`, 40k seed-0 sample.
- **Command**
  `python scripts/probe_normalisation.py --levels duration --per-fold --bootstrap 1000`
- **Produces** 0.757 and 0.738 with a 95% CI, plus 5 per-fold values each.
- **Settles** what counts as a real move. Every kill threshold below is stated in
  BA points and is meaningless without this.
- **Kill** if the CI half-width exceeds 0.03, single-run comparisons are
  uninterpretable — widen every threshold below and say so in the write-up rather
  than reading 0.02 differences as findings.
- **Code** new: per-fold return plus a bootstrap over **speakers, not segments**
  (about 25 lines in `probe_normalisation.py`). Resampling segments would be
  wrong twice over: duration windows are 50%-overlapping and are correlated
  within speaker.

### R1 — moment ladder by Gaussian surrogate (the decisive run)

How much of the corpus cue is reachable from moments alone, apportioned by order.

Per speaker-grouped fold: fit per-corpus mean and covariance on **training
speakers only**; sample surrogate rows matching the train per-corpus counts;
train the GBT on the surrogate; **test on the real held-out fold**. Test sets are
identical to R0's, so the numbers are directly comparable and no held-out speaker
enters any statistic.

| rung | surrogate | what it contains |
|---|---|---|
| L0 | per-corpus mean, pooled covariance | first order only |
| L1 | per-corpus mean, diagonal of per-corpus covariance | + marginal variance — what `norm=corpus` already removes |
| L2 | per-corpus mean and full covariance | + correlations — what CORAL removes |
| L3 | real data | all orders = 0.757, already in hand |
| L2z | zero mean, unit variance, per-corpus correlation, tested on real corpus-z'd rows | correlations alone, against 0.991 |

- **Consumes** same cache, same sample, same folds.
- **Command**
  `python scripts/probe_moment_ladder.py --level duration --rungs L0 L1 L2 L2z --bootstrap 1000`
- **Produces** corpus BA at each rung, the share `(L2 - 0.333) / (0.757 - 0.333)`,
  and a surrogate-trained/surrogate-tested upper bound as tie-breaker.
- **Settles** whether CORAL can possibly work — without writing CORAL.
- **Kill / promote**
  - `L2 - L1` at or below 0.03 → correlations add nothing beyond what
    `norm=corpus` already equalised. **CORAL is dead**; skip R4 entirely. Its
    only increment over a completed experiment is empty.
  - L2 covers 0.70 or more of the chance-to-0.757 span → moments carry most of
    the cue; **R4 authorised**.
  - L2 low on the real test set but the surrogate/surrogate bound high → the gap
    is Gaussian mismatch, not moment content. Inconclusive on this axis; fall
    back on R4 as the direct test.
  - L2 low in **both** variants → the cue is higher-order or discrete. Skip
    CORAL; go to R3 and then DANN.
- **Prediction, recorded before running:** given a Frobenius distance up to 10.76
  *and* excess kurtosis 5.80 against 2.58, both channels are live, so L2 lands
  mid-ladder (roughly 0.55–0.68 of the span) and CORAL removes some but not all.
  If L2 comes out at either extreme, one of the two measurements above is being
  misread — check that before believing the ladder.
- **Code** new: `scripts/probe_moment_ladder.py`, about 90 lines. Verification —
  assert sampled moments match the fitted moments to tolerance, and assert the
  fit touched no held-out speaker.

### R2 — task-matched and cohort-matched pairwise probes (missing from `PLAN.md`)

Task mix is on the hypothesis list and nothing in `PLAN.md` tests it. **No task
spans all three corpora**, so a task-matched 3-way probe is impossible. Two
pairwise probes are, and ItalianPD is the pivot:

| run | subset | corpora | speakers | rows | matched | differs |
|---|---|---|---|---|---|---|
| R2a | the five `vowel_*` tasks | ItalianPD vs NeuroVoz | 50 / 111 | 73,321 | task **and channel** (both `clinical`) | language, site, cohort |
| R2b | `read_text` | ItalianPD vs MDVR-KCL | 64 / 37 | 81,681 | task | language **and** channel |
| R2c | all tasks | ItalianPD vs NeuroVoz | 65 / 114 | — | — | reference for R2a |
| R2d | all tasks | ItalianPD vs MDVR-KCL | 65 / 37 | — | — | reference for R2b |
| R2e | vowels, `young_control` dropped | ItalianPD vs NeuroVoz | — | — | task, channel, **age band** | language, site |

Chance is **0.500** here, not 0.333, so R2a is read against R2c and R2b against
R2d — never against 0.757.

- **Command**
  `python scripts/probe_normalisation.py --levels duration --corpora ItalianPD NeuroVoz --tasks vowel_a vowel_e vowel_i vowel_o vowel_u --per-fold`
  and the four siblings; add `--drop-cohorts young_control` for R2e.
- **Produces** corpus BA and PD BA per subset, under `global` and `corpus`.
- **Settles** which of task mix / channel / language-and-site / age the residual
  actually is — the decomposition the four `PLAN.md` candidates cannot deliver at
  any price.
- **Reads as**
  - R2a about equal to R2c → task mix contributes nothing; the cue is site,
    language, or speaker population.
  - R2a well below R2c → a large share of the 3-way cue is **task composition**.
    No feature-space method removes that, and DANN will learn it as a shortcut
    too. The honest fix is task-matched evaluation, not a domain loss — and that
    finding outranks every method in this file.
  - R2a high **with channel matched** → channel is not necessary for the cue;
    language and site suffice, and "channel invariance" is too narrow a framing
    for the write-up.
  - R2e well below R2a → the ItalianPD cue is substantially the documented age
    confound, and any corpus head is partly an age head. Must be stated.
- **Kill** if R2a and R2b both land at or above 0.95, subsetting isolates nothing
  and the residual is everywhere. Stop after these two; do not enumerate more
  subsets.
- **Code** new: `--corpora`, `--tasks` and `--drop-cohorts` filters on
  `probe_normalisation.py`, about 12 lines total. `Bundle` already takes `tasks`;
  the probe does not, and `classical.py` exposes neither.

### R3 — linear corpus-subspace removal (INLP), promoted ahead of CORAL

Fit a multinomial logistic corpus classifier on training speakers, project onto
the null space of its weight rows, iterate. One global linear map, no corpus
label at inference, valid under strict LODO. **PD labels never enter the fit.**

- **Consumes** same cache and splits.
- **Command**
  `python scripts/probe_normalisation.py --levels duration --project inlp --project-iters 0 1 2 3 --lodo-probe`
- **Produces** corpus BA and PD BA against iteration count k = 0..3, plus
  strict-LODO PD BA per held-out corpus.
- **Settles** whether a *linear* adversary already does DANN's job, and whether
  the corpus directions and the PD directions are the same directions.
- **Kill / promote**
  - corpus at or below 0.45 with PD at or above 0.70, and LODO clears bar item 5
    → **this is the shallow control DANN must beat**, and DANN then has to win on
    strict LODO, not on mixed.
  - both fall together (PD at or below 0.63, back at the 0.600 `drop both` floor)
    → the linear domain directions *are* the PD directions. Report it and do not
    crank lambda; gradient reversal will fight itself.
  - corpus still at or above 0.70 after 3 iterations → the cue is non-linear.
    This is the strongest justification for the GRL the ablation can produce, and
    the single best sentence available for the Stage 7 write-up. Stop the shallow
    track here.
- **Code** new: `src/features/project.py` with `fit_projection` / `apply`
  mirroring `fit_stats` / `apply`, about 70 lines, plus a probe flag.
  Verification — assert the projection is idempotent, that its rank deficit
  equals the number of removed directions, and that `label` is absent from the
  fit's inputs.

### R4 — CORAL, **conditional on R1**, reported as transductive only

- **Consumes** same cache and splits. Regularised covariance with a ridge term;
  reference is the pooled **training** covariance of the configuration, not
  NeuroVoz.
- **Command**
  `python scripts/probe_normalisation.py --levels duration --align coral --lodo-probe`
- **Produces** mixed corpus BA and PD BA; strict-LODO PD BA for all three.
- **Settles** whether the correlation residual R1 sized is actually removable in
  practice.
- **Kill** PD below 0.70 → stop, and keep CORAL out of the main path. Corpus
  stays at or above 0.90 → shallow alignment is finished; go to Stage 7. Corpus
  falls to chance with PD held → report as a **transductive** baseline, stating
  explicitly that it cannot serve strict LODO and so does not clear the promotion
  bar.
- **Code** new: `src/features/coral.py`, about 60 lines. Verification — assert
  per-corpus covariance after the transform matches the reference within
  tolerance on the fitted set. Do not CORAL Mel frames.

### R5 — `duration_100` under `corpus` scope (cheap, confirmatory)

- **Command** `python scripts/probe_normalisation.py --levels duration_100`
- **Produces** corpus BA under `global` and `corpus` at a 100 ms window.
- **Settles** only whether 0.991 is a 250 ms artefact.
- **Kill** if `duration_100` also lands at or above 0.97, the window is not the
  story. Do not run a third window size, and do not add `utterance` unless this
  comes out ambiguous.

### R6 — mean-pooled Mel to GBT (relabel it)

Worth running, but it does **not** bear on the second-order question. It closes
the standing Stage 5 open item "probe the Mel family for domain signal", which is
a different claim. Run it last here, or move it to Stage 5's open items; either
way do not present it as part of this ablation's argument.

- **Command**
  `python scripts/probe_normalisation.py --levels duration --family mel-meanpool`
- **Code** new: a mean-pool helper over `fs.load_mel` / `fs.mel_item`, about 20
  lines.

---

## Redundant — do not run

- **Per-language z-score.** Arithmetically **identical** to `norm=corpus` in this
  data: each dataset has exactly one language (`it`, `en`, `es`). Not a judgement
  call, an identity.
- **Per-channel z-score.** Distinct — it groups ItalianPD with NeuroVoz, 2 groups
  not 3 — but answers nothing R2a does not answer better.
- **`utterance` under `corpus` scope.** Subsumed by R5's logic. Conditional only.
- **Class-conditional CORAL.** Uses the target label inside the transform.
- **A2 before A3, as `PLAN.md` ordered them.** Reversed, for the reason in the
  premise-corrections section.
- Everything on `PLAN.md`'s existing do-not list still stands.

---

## Resources

| run | wall clock (CPU) | new code |
|---|---|---|
| R0 | ~8 min | ~25 lines |
| R1 | ~25 min | ~90 lines, 1 new script |
| R2 (a–e) | ~20 min | ~12 lines of flags |
| R3 | ~15 min | ~70 lines, 1 new module |
| R4 | ~15 min | ~60 lines, 1 new module |
| R5 | ~10 min | none |
| R6 | ~20 min | ~20 lines |

Total about 2 h on one CPU, single process, **no GPU**, under 5 MB of CSVs, no
checkpoints. Three new files, roughly 240 lines, plus roughly 55 lines of flags
on existing scripts. Nothing touches `src/features/extract.py`, the cache, or
Stage 7 defaults.

## Analysis plan

- **Primary** balanced accuracy for corpus and for PD/HC, pooled out-of-fold
  across the 5 speaker-grouped folds — same construction as the existing probe,
  so every number lands in the same table as 0.757 / 0.738.
- **Secondary** per-fold BA and a 95% CI from bootstrapping **speakers**.
- **Never** corpus BA without PD BA beside it.
- **Figure** one method-level scatter: corpus BA on x, PD BA on y, one point per
  method, chance lines and the `global` reference marked, an arrow from `global`
  to each method. Same visual grammar as `featimp_scatter.png`, one level up.
- **Failed runs** any run tripping a `verify()` assertion is discarded, not
  patched around.
- **Seeds** `SEED = 0` throughout for comparability; R1 additionally records its
  surrogate sampling seed.
- **Deliverable** append to `reports/ablation_corpus_norm.md`. Do not rewrite the
  Stage 5 cache, and do not touch the mainline Stage 7/8 runs.

## Minimum that changes what you believe

**R1 and R2a.** R1 tells you whether any moment-matching method has a target at
all; R2a tells you whether the thing you would be matching is task composition.
If you run only two things from this file, run those.
