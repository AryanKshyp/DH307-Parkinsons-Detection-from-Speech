# Stage 6 — Splits

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

Both split protocols exist and are asserted leak-free, at file level as well as
speaker level, by a checker independent of the code that wrote them. `mixed`
gives 10 speaker-disjoint configurations stratified on `dataset × label`
jointly; `lodo` gives 3, each holding out one entire corpus.

Two findings shape how later stages must use these files. First, **per-corpus
test metrics cannot be computed per fold**: MDVR-KCL contributes only 1–2 PD
speakers to any single test fold, so a per-corpus score is only meaningful when
out-of-fold predictions are pooled across all 10 configurations. Second, **LODO
is very unbalanced by construction** — holding out NeuroVoz leaves 747 training
files, while holding out MDVR-KCL leaves 3190 but tests on only 73.

**In:** `data/manifests/manifest_processed.csv`
**Out:** `data/manifests/splits_mixed.csv` (2160 rows),
`data/manifests/splits_lodo.csv` (648 rows)
**Code:** `src/data/splits.py`
**Checks:** `scripts/check_splits.py`

---

## The speaker population being split

216 speakers, 3871 files, 12.26 hours.

| corpus | HC | PD | total | files | median files/speaker |
|---|---|---|---|---|---|
| NeuroVoz | 58 | 56 | 114 | 2968 | 27 |
| ItalianPD | 37 | 28 | 65 | 830 | 16 |
| MDVR-KCL | 21 | 16 | 37 | 73 | 2 |
| **all** | **116** | **100** | **216** | **3871** | |

The corpora are unequal in every dimension that matters to splitting. NeuroVoz
holds 53% of speakers and 77% of files; MDVR-KCL holds 17% of speakers but 1.9%
of files. Any metric averaged over files is therefore close to a NeuroVoz-only
metric, and any metric averaged over speakers weights MDVR-KCL nine times more
heavily than its audio warrants.

The smallest `dataset × label` cell is MDVR-KCL PD at 16 speakers. That is the
hard ceiling on fold count: `splits.py` refuses a `k` above it rather than
silently producing folds with no MDVR-KCL patients in them.

---

## Mixed protocol

10 configurations over the same stratified 10-fold partition of speakers. In
configuration *i*, fold *i* is test, fold *i+1* is validation, and the remaining
eight folds are training. Every configuration therefore has a speaker-disjoint
train, validation and test set, which is what the assigned pipeline asks for,
and validation never overlaps the test fold it is used to select against.

Roles hold 21–22 test speakers, 21–22 validation, and 172–174 training. Test
fold composition by corpus and class:

| corpus | HC per fold | PD per fold |
|---|---|---|
| NeuroVoz | 5–6 | 5–6 |
| ItalianPD | 3–4 | 2–3 |
| MDVR-KCL | 2–3 | 1–2 |

Every fold contains both classes of all three corpora, with the thinnest cell
holding a single speaker. That is the point worth carrying forward. A per-fold
per-corpus score for MDVR-KCL would rest on one or two patients, so its variance
would swamp any effect the DANN has.

The protocol that does work is pooling: every speaker is in the test set of
exactly one configuration — asserted, not assumed — so concatenating out-of-fold
predictions yields one prediction per speaker and a per-corpus score computed
over all 37 MDVR-KCL speakers rather than over two. Stage 8 should report that,
not a mean of per-fold numbers.

File counts per role vary between configurations (test ranges 335–425 files)
because speakers carry very different numbers of recordings. Stratification
equalises speakers, not files, and equalising both is not possible here.

---

## LODO protocol

Three configurations, each holding out one corpus entirely as test. Training and
validation are drawn from the remaining corpora, stratified on `dataset × label`,
with 15% of the remaining speakers held for validation.

| held out | test speakers | train | val | train files | test files |
|---|---|---|---|---|---|
| NeuroVoz | 114 | 86 | 16 | 747 | 2968 |
| ItalianPD | 65 | 128 | 23 | 2600 | 830 |
| MDVR-KCL | 37 | 152 | 27 | 3190 | 73 |

These three configurations are not three comparable experiments, and averaging
them into a single "LODO score" would be misleading. Holding out NeuroVoz trains
on 747 files and tests on 2968 — the training set is a quarter the size of the
test set. Holding out MDVR-KCL trains on 3190 files but tests on 73, so its test
score moves in steps of roughly 1.4 percentage points per speaker. Each
configuration should be reported and discussed separately.

Two further asymmetries are structural rather than incidental. Holding out
NeuroVoz removes the only Spanish corpus, so that configuration tests
cross-language transfer as much as cross-corpus transfer, and the two cannot be
separated with these three corpora. Holding out MDVR-KCL removes the only
mobile-recorded corpus.

At the `phoneme` level the picture narrows: English is word-only, so MDVR-KCL
has no phoneme segments. Holding it out is a no-op there, and holding out
NeuroVoz leaves ItalianPD as the sole training corpus — a domain adversary with
one training domain has nothing to be adversarial about. At `word`, MDVR-KCL
now contributes the 35 accepted North Wind readings; holding it out is a real
test, but a **read-passage** test, not a spontaneous-dialogue one. **LODO is
fully meaningful at `utterance` and `duration` / `duration_100`; at `word` it
is read-passage-only for MDVR-KCL; at `phoneme` it is two-corpus.**

---

## Strict LODO, and what that makes it

The held-out corpus appears nowhere in training or validation, not even as
unlabelled audio. This is asserted directly by the checker.

That choice makes LODO a **domain generalisation** experiment rather than domain
adaptation. The adversary can only make features invariant across the corpora it
sees during training, and the hope is that this invariance extends to a corpus it
has never encountered. The alternative — letting unlabelled held-out audio into
training so the domain branch can align to it — is transductive domain adaptation
and is the setting much of the DANN literature actually reports. It would very
likely score better, and it would answer a different question.

The stricter reading is used because it matches what the assigned pipeline means
by holding out a corpus. It is worth stating plainly in Stage 8 which of the two
is being claimed, since they are easy to conflate and differ by a large margin.

---

## Leakage checks

`splits.py` asserts while building; `scripts/check_splits.py` re-checks the files
as written, which is the version that matters because a split file is read many
times after it is produced and a leak in it surfaces only as an implausibly good
score much later.

The independent checker verifies that every speaker maps to exactly one corpus
and one label in the manifest, that the corpus and label recorded in the split
file agree with the manifest, that expanding speakers to their 3871 files leaves
no file in two roles in any configuration, and that the three role sets are
pairwise disjoint. For `mixed` it additionally confirms each speaker is tested
exactly once and no test fold is missing a corpus or class. For `lodo` it
confirms each configuration tests exactly its held-out corpus and that the corpus
appears in neither training nor validation.

Checking at file level rather than speaker level is deliberate: it would catch a
speaker whose recordings disagreed about which corpus they belong to, which a
speaker-level check cannot see.

---

## Limitations and open items

- **The ItalianPD young controls are included, and they are an age confound.**
  15 controls averaging 20.8 years (range 19–29) against patients averaging 67
  (range 40–80): every young control is below the youngest patient. Measured at
  utterance level, young controls versus patients is separable at 0.905, so age
  is an available shortcut, and it is a shortcut that cannot transfer to NeuroVoz
  or MDVR-KCL where controls are age-matched. Excluding them was measured to cost
  nothing — ItalianPD PD/HC goes from 0.951 to 0.955 at utterance level and 0.918
  to 0.920 at `duration` — so the inclusion buys no accuracy. Per the Stage 6
  decision they are kept; `cohort` is carried in both split files so Stage 8 can
  report an elderly-only robustness number without regenerating anything.
- **NeuroVoz age asymmetry is untouched.** Its HC mean is 64 against 71.1 for PD,
  a 7-year gap in the same direction as the confound above, and it is not
  correctable by excluding a cohort.
- **MDVR-KCL has no age metadata at all**, so no age-matched analysis is possible
  for the one corpus whose channel differs most.
- **Validation folds are single folds, not repeated.** With 21–22 validation
  speakers, any hyperparameter chosen on them carries real variance. Repeated
  k-fold would reduce it at 10× the compute.
- **Splits are speaker-level and level-agnostic.** A speaker with no `word`
  segments simply contributes nothing at that level, so effective training set
  size varies by segment level in a way the split files do not record.
- **Stage 7 must not compute normalisation statistics before splitting.** Stage 5
  caches features unnormalised precisely so that the global statistics can be
  fitted on training rows only; these split files are what defines those rows.

## Reproduction

```bash
python -m src.data.splits --protocol mixed --k 10
python -m src.data.splits --protocol lodo
python scripts/check_splits.py               # re-checks the written files
```
