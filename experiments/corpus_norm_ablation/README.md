# Optional ablation — corpus-wise batch correction

**Status.** Closed. A1 (z-score + drop-columns), A2 (CORAL), A3 (INLP),
and the moment ladder all ran. None of them replaces DANN. See
`RESULTS.md` for the first wave and `reports/ablation_corpus_norm.md` for
the full argument and talk figures. Mainline remains `norm=global` →
Stage 7.

This is a **named control**, not a replacement dataset. It answers one
question: *does any shallow, non-adversarial correction already remove
the corpus shortcut that Stage 5 measured, and does that help
Parkinson’s detection?* It does not. Matching the covariances (CORAL)
makes the shortcut *easier* (0.757 → 0.855).

---

## 1. Why this exists

Stage 5’s gradient-boosted probe (`scripts/probe_normalisation.py`) showed that
from the 40 handcrafted features, **corpus identity is easier to predict than
PD vs HC** at every segment level:

| level | corpus | chance | PD/HC | chance |
|---|---|---|---|---|
| utterance | 0.938 | 0.333 | 0.761 | 0.500 |
| duration | 0.757 | 0.333 | 0.738 | 0.500 |
| word | 0.869 | 0.500 | 0.679 | 0.500 |
| phoneme | 0.804 | 0.500 | 0.652 | 0.500 |

A pooled PD model therefore has an easier wrong problem available than the one
it is meant to solve. DANN is supposed to remove that shortcut during training.
This ablation asks whether a **preprocessing** step that centres each corpus
does the same job more simply.

It does **not** follow that we should normalise the main cache and proceed.
Global z-scoring was already measured: `none` and `global` are identical for
both targets (trees are scale-invariant; global is an affine map applied to
every row equally). Global scaling equalises units. It does **not** make
NeuroVoz, ItalianPD, and MDVR-KCL look like each other.

Per-speaker and per-utterance z-scoring were also measured: they take PD/HC to
chance, because diagnosis is constant inside a speaker. Those scopes stay
ablations that document a collapse. **Do not use them here.**

The operation that actually attacks the probe finding is **per-corpus**
centering: a different mean and std per dataset. That is batch correction.
The main pipeline forbids it on the feature path
(`PIPELINE.md` invariant 2) because it leaks the variable the adversary is
supposed to remove. This experiment is the declared exception: a named run
that uses corpus identity on purpose, so we can see what it buys and what it
destroys.

---

## 2. What this is, and what it is not

| This is | This is not |
|---|---|
| A Stage 7 **run config**: same `.npy` cache, `norm=corpus`, PD head only | A second 4 GB feature extract |
| A **shallow DA / DG baseline** next to 7a and 7b | A replacement for DANN |
| Allowed to use the dataset tag as an input to the transform | Allowed to use PD/HC to fit the transform |
| Reported as an ablation in Stage 8 | The headline result |

Features stay cached unnormalised. `src/features/store.py` already applies
scope at load time via `fit_stats` / `apply`. Add a `corpus` scope there.
Do not write a parallel `data/processed/features/` tree.

---

## 3. Literature and what to copy

DANN papers (Ganin et al., 2016) compare the adversary against **source-only**,
**pooled no-DA**, and at least one **statistical alignment** method. Without
that third arm, “DANN beats baseline” cannot distinguish “the adversary works”
from “any attempt to equalise corpora would have worked.”

The reference PD paper (Ibarra et al., 2023, *Bioengineering* 10:1316) compares
each architecture to its non-adversarial twin. It does **not** include CORAL.
Adding a statistical baseline is therefore stronger than the reference, not a
deviation from it.

Methods, in the order they should be considered:

### 3.1 Per-corpus z-score (do this)

First-order batch correction. For each dataset \(d\), on **training speakers
only**:

\[
x' = (x - \mu_d) / \sigma_d
\]

Unsupervised with respect to PD/HC: \(\mu_d\) and \(\sigma_d\) are computed
over both classes of that corpus. Class-conditional corpus stats would leak
the target into the transform. Do not do that.

This is the minimum honest version of “normalise so the classifier cannot see
which dataset it is.” Implement this first. If it does nothing interesting,
stop. Do not pile on heavier methods to rescue a null.

### 3.2 CORAL (optional, only if 3.1 moves mixed PD or the corpus probe)

Sun & Saenko, “Return of Frustratingly Easy Domain Adaptation” / Deep CORAL
(ECCV 2016): align **second-order** statistics with a linear map. After
centering, whiten by \(\Sigma_d^{-1/2}\) and recolour by a reference
covariance \(\Sigma_R^{1/2}\). Add \(\gamma I\) to each covariance so the
inverse is stable (the original paper’s regulariser).

Use it only on the **40-d handcrafted** vectors. Do not CORAL Mel spectrograms
frame-wise; the covariance is huge and the ablation would stop being a simple
control.

Pick one reference \(R\) and keep it: the **pooled training set** of that
configuration, not a particular corpus. Aligning everything to NeuroVoz would
make the transform itself a domain choice.

Deep CORAL (a training loss on batch covariances) is a **competitor to DANN**,
not this ablation. Do not implement it here.

### 3.3 ComBat — skip

Johnson et al. (2007) empirical-Bayes batch correction, common in genomics and
sometimes in multi-site neuroimaging. It needs a batch id at train and
typically at test, and it is designed to strip site effects. Here “site” is
perfectly confounded with language, channel, and task mix
(`README.md`, `reports/stage2_exploratory_analysis.md`). ComBat can therefore
remove PD-relevant structure that happens to differ by corpus (ItalianPD’s
44.1 kHz patients, phonetic inventory, age). It is more machinery than this
control needs. Skip it.

### 3.4 What “best practice” actually constrains

1. **Fit on training speakers only.** Same rule as global. A speaker in val or
   test must never contribute to \(\mu_d\) or \(\Sigma_d\).
2. **Do not use the PD label** when fitting the transform.
3. **Name the scientific claim.** Mixed + per-corpus stats is *site-known
   batch correction* (you know which clinic recorded the test file). LODO +
   target-corpus stats is *transductive UDA* (you peeked at the held-out
   distribution). LODO + train-only fallback is *domain generalisation*, which
   is what this project’s LODO means (`reports/stage6_splits.md`). Mixing those
   three without saying which one you ran invalidates the comparison to DANN.
4. **Same model, same splits, same seed as 7a.** The only changed knob is the
   normalisation scope. If you also change the backbone or the window, you
   cannot attribute the delta to batch correction.
5. **Metrics stay per test corpus**, speaker-aggregated. A pooled number hides
   the failure mode. Mixed per-corpus scores are pooled **out-of-fold across
   all 10 configurations**, not averaged per fold (MDVR-KCL has 1–2 PD
   speakers in any single test fold).
6. **Do not average the three LODO configurations** into one “LODO score.”

---

## 4. Mixed vs LODO — the part that is easy to get wrong

### Mixed (`splits_mixed.csv`)

Every configuration’s train, val, and test sets contain all three corpora.
Corpus identity of a test file is known, because evaluation is already
per-corpus.

**Allowed:** apply NeuroVoz **training-speaker** stats to NeuroVoz **test
speakers** (and likewise for the other two). That is not speaker leakage.
It *is* using the domain label at test time. Say so in the run config:
`norm=corpus`, `target_stats=same_corpus_train`.

This is the setting in which batch correction is expected to help, if it
helps at all: a clinic that knows its own recording chain.

### LODO (`splits_lodo.csv`) — strict, matching Stage 6

The held-out corpus appears **nowhere** in training or validation, not even
as unlabelled audio.

**Forbidden unless the run is explicitly named transductive:** computing
\(\mu\) / \(\sigma\) / \(\Sigma\) on the held-out corpus, even without using
PD labels.

**Required for the run that compares to DANN:** the held-out dataset key is
absent from `stats`. `apply` must fall back to the **global training**
mean and std of that configuration (pooled source corpora), **not** to
`(0, 1)`. A zero/one fallback would smash the target features and look like
a failed method for the wrong reason.

Under that rule, corpus-norm on the target is almost global-norm. The
honest expectation is: **mixed may move; strict LODO should look like 7a.**
If someone then “cheats” and z-scores the target with its own unlabelled
features, that is a **separately named** run (`target_stats=unlabelled_target`)
and answers a different question (transductive UDA). Do not lead with it.
Implement it only if the strict run is done and you still want the extra
number.

`apply` fallback today in `store.py` is `(zeros, ones)` for unseen speaker or
path keys. That is wrong for an unseen **corpus**. The corpus scope needs its
own fallback: the `__all__` global train stats stored alongside the per-dataset
dicts.

---

## 5. What to implement

Touch little. Match existing style.

### 5.1 `src/features/store.py`

Add `"corpus"` to `SCOPES`.

`fit_stats(..., "corpus")`:

- Group training rows by `meta.dataset`.
- Store `{dataset: (mean, std)}` plus `{"__all__": (global_mean, global_std)}`
  from the same training rows, for the LODO fallback.
- Guard `std < 1e-8` the same way `apply` already does.

`apply(..., "corpus")`:

- Key rows by `meta.dataset`.
- Missing key → `__all__`, never `(0, 1)`.

Do not change the behaviour of `none`, `global`, `speaker`, or `utterance`.

Optional, same file or a sibling function, only if §3.2 is in scope:
`fit_coral` / `apply_coral` on handcrafted matrices. Keep it out of the
default path.

### 5.2 Probe before any training

Extend `scripts/probe_normalisation.py` so it iterates the new scope
automatically via `fs.SCOPES`.

Run:

```bash
python scripts/probe_normalisation.py --levels duration
```

(and `utterance` if the main experiment also trained utterance).

**GroupKFold probe (current harness).** Each fold still contains all corpora,
so this is the mixed-style test of corpus z-score. Expected:

- Corpus balanced accuracy should **fall** relative to `global` (between-corpus
  mean shift is what the trees were using).
- PD/HC may stay, rise, or fall. All three are results. If it falls toward
  chance, stop: the corpus means were carrying the target (task mix, the
  ItalianPD rate confound, age). Do not train a Stage 7 model on a transform
  that deletes PD.

Because trees ignore a *global* affine map, `none` vs `global` staying equal
is still required. `corpus` applies a **different** affine map per dataset, so
trees **will** see it. That is the point.

**LODO-style probe (add this, keep it small).** For each held-out corpus: fit
`corpus` stats on the other two only, apply with `__all__` fallback to the
held-out rows, score PD/HC on the held-out corpus under speaker-grouped
evaluation. This is the number that should resemble 7a. Put it in the same
script behind a flag, e.g. `--lodo-probe`, rather than a new file.

Write the probe table into the run directory, not only stdout.

### 5.3 Training runs

The Stage 7 trainer already exists. This ablation is “same 7a, different
scope,” not a new loop.

```bash
python -m src.training.run --norm corpus --regime baseline --all-configs
python -m src.training.run --norm corpus --regime baseline --protocol lodo --all-configs
python -m src.training.classical --level duration --split 0 --norm corpus
```

`--norm corpus` is refused for Mel. Run ids include `norm-corpus` so they
cannot overwrite a `global` run. `config.json` records `target_stats`:
`same_corpus_train` on mixed, `global_fallback` on strict LODO.

Required runs (mirror the main spine; default spine is **duration +
handcrafted**):

| run-id pattern | protocol | scope | head |
|---|---|---|---|
| `..._7a_global_*` | already exists | `global` | PD only |
| `..._7b_dann_*` | already exists | `global` | PD + domain GRL |
| `corpusnorm_7a_mixed_duration_hc` | mixed, all 10 configs | `corpus` | PD only |
| `corpusnorm_7a_lodo_duration_hc` | lodo, 3 configs | `corpus`, strict fallback | PD only |

Optional, only after those two exist and the mixed probe was interesting:

- `corpusnorm_coral_mixed_duration_hc`
- `corpusnorm_7a_lodo_transductive_duration_hc` (target unlabelled stats;
  label the claim in `config.yaml`)

**Match the main 7a run exactly** for: backbone, seed, batch size, epochs,
loss weights (class imbalance), aggregation (speaker-level mean pool unless
7a used something else), feature family, segment level, split files.

`experiments/<run-id>/` must contain the resolved config, git commit, seed,
scope, and `target_stats` (`same_corpus_train` | `global_fallback` |
`unlabelled_target`).

No domain head. No GRL. Weighted PD loss still applies (corpora differ in
size; that is a class-sampling issue, not a reason to add an adversary).

### 5.4 Checks the agent must write or run

- Stats for a configuration are fitted on `role==train` speakers only.
  Assert that no val/test `speaker_id` appears in the rows passed to
  `fit_stats`.
- For mixed `corpus` apply: every test row’s dataset is a key in `stats`.
- For strict LODO apply: the held-out dataset is **not** a key (except
  `__all__`); those rows used `__all__`. Log the count.
- `scripts/check_splits.py` still passes; this experiment must not rewrite
  split files.
- Do not re-run feature extraction.

---

## 6. How to read the outcome

Compare three numbers, per test corpus, speaker-aggregated, same metric as
Stage 8 (balanced accuracy and ROC-AUC at least):

**7a global** · **this ablation** · **7b DANN**

| If you see | It means |
|---|---|
| Mixed PD up, strict LODO ≈ 7a | Site-known centering helps when the clinic is known; it is not a substitute for DANN on an unseen corpus. This is the outcome the design predicts. |
| Mixed PD up **and** strict LODO up | First-order corpus shift was most of the problem. DANN may be unnecessary for this feature family; say that plainly. Still report DANN’s domain-head curves, because the claim is then “shallow correction suffices,” not “skip evaluation.” |
| Mixed PD down (or probe PD → chance) | Corpus means carried PD-relevant structure. Do not use `norm=corpus` anywhere else. The GBT finding still stands; the shortcut and the target were not orthogonal. |
| Corpus probe still high after z-score | The shortcut is not a mean shift (covariances, higher-order, Mel channel). CORAL is the next shallow step; DANN is still the learned one. |
| Ablation beats DANN on mixed but loses on LODO | Expected if DANN is doing DG and corpus-norm is doing site-known correction. Do not declare a winner from mixed alone. |

`PIPELINE.md`: the headline is whether the method closes the gap under
**leave-one-dataset-out**, not whether it beats 7a on the pooled mix.

A higher pooled PD number after corpus-normalising is easy to get and easy
to misread. The interesting plot is the Stage 5-style probe on the
**transformed features** (or on 7b embeddings): can a tree still name the
dataset, and did PD/HC hold.

---

## 7. Constraints (fail the run if violated)

From `PIPELINE.md` and the stage reports, plus this ablation’s exception:

1. Splits are speaker-level. Never file-level, never segment-level.
2. The **main** feature path still has no per-corpus statistics. Only these
   named runs may use them, and the config must say so.
3. Raw audio and the Stage 5 cache are read-only.
4. Metrics per test corpus and per level; mixed scores from pooled OOF
   predictions; LODO configurations reported separately.
5. Seeds recorded. One run = one directory under `experiments/`.
6. Do not fit normalisation on all rows, then split.
7. Do not drop ItalianPD `young_control` unless 7a also did (Stage 6 kept
   them; `cohort` is on the split files for an elderly-only **report**, not a
   different split).
8. Duration windows overlap 50%. Do not treat segment counts as independent
   samples when writing confidence intervals; aggregate to speakers.
9. `word` / `phoneme` lack MDVR-KCL. Do not run this ablation at those levels
   unless the main experiment did, and then LODO holding out MDVR-KCL is a
   no-op — skip that configuration.

---

## 8. Suggested order of work

The probe and a GBT on Stage 6 splits do **not** need 7a/7b. The MLP/DANN
comparison does.

```
1. python scripts/check_train.py
   → verify: "corpus scope: unseen dataset uses __all__"
2. python scripts/probe_normalisation.py --levels duration --lodo-probe
   → verify: probe_mixed.csv and probe_lodo.csv; if PD/HC collapsed, stop
3. python -m src.training.classical --level duration --split 0 --norm global
   python -m src.training.classical --level duration --split 0 --norm corpus
   → verify: two experiment dirs; speaker BA per corpus
4. Confirm 7a global and 7b DANN exist for duration + handcrafted
   → verify: experiments/ dirs with config + metrics
5. python -m src.training.run --all-configs --regime baseline --norm corpus
   python -m src.training.run --protocol lodo --all-configs --regime baseline --norm corpus
   → verify: run ids contain norm-corpus; per-corpus metrics
6. Write reports/ablation_corpus_norm.md
   → verify: three-way table 7a / corpus-norm / DANN; interpretation from §6
7. CORAL or transductive LODO only if step 2–6 warrant it
```

Minimum code that solves the problem. No new feature family, no second
window size, no ComBat, no Deep CORAL, no new split protocol.

---

## 9. Deliverable

`reports/ablation_corpus_norm.md`, in the same voice as the stage reports:

- Probe table (`global` vs `corpus`, mixed-style and LODO-style).
- Training table vs 7a and 7b, per corpus, mixed OOF and each LODO hold-out.
- Which `target_stats` was used.
- The interpretation row from §6 that actually happened.
- Confirmation that the Stage 5 cache was not rewritten.

Reproduction commands at the bottom, including the exact 7a run-id this was
matched against.

---

## 10. Pointers in this repo

| | |
|---|---|
| Probe and the original finding | `reports/stage5_features.md`, `scripts/probe_normalisation.py` |
| Load-time scopes | `src/features/store.py` |
| Why speaker-norm is forbidden | same report; `store.py` module docstring |
| Splits, mixed OOF, strict LODO | `reports/stage6_splits.md`, `data/manifests/splits_*.csv` |
| Invariant 2 and Stage 7 | `PIPELINE.md` |
| Confounds that corpus-norm can erase by accident | ItalianPD sample-rate vs label (`reports/stage1_preprocessing.md`); age (`reports/stage2_exploratory_analysis.md`, Stage 6) |

References to cite in the report if a method is used:

- Ganin et al. (2016), JMLR — DANN; compare against a statistical baseline.
- Sun, Feng & Saenko (2016), AAAI — CORAL; Sun & Saenko (2016), ECCV — Deep CORAL (loss; out of scope).
- Ibarra et al. (2023), Bioengineering 10:1316 — PD DANN reference; no CORAL arm.
- Johnson et al. (2007), *Biostatistics* — ComBat (do not use).
