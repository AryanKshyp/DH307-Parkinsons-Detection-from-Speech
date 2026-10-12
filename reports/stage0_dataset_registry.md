# Stage 0 — Dataset registry and manifest

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

Three corpora were extracted, their directory layouts and filename grammars
reverse-engineered, and a single index built over all of them. The manifest
covers **3880 files, 216 speakers and 12.28 hours** with every file matched by
an explicit rule and assigned a task from a shared vocabulary. No file is
unmatched, no task is `unknown`, and no speaker carries conflicting labels.

**In:** raw corpora under `data/raw/`
**Out:** `data/manifests/manifest.csv`
**Code:** `configs/datasets.yaml`, `src/data/build_manifest.py`

---

## What was done

The workspace initially held three zip archives and two PDFs. The repository
scaffold described in `README.md` did not exist, so it was created: `git init`,
`.gitignore`, `requirements.txt`, and the `configs/ src/ scripts/ notebooks/
experiments/ tests/ data/` tree.

### The archives were not what their names suggested

`PD.zip` is a 1.96 GB bundle rather than a corpus. It contained the ItalianPD
archive, a FredPrior archive set, and a complete second copy of MDVR-KCL. All
73 MDVR-KCL files were verified identical across the two archives by CRC32 and
byte length before `PD.zip` was deleted. `NeuroVoz.zip` was a wrapper around
`neurovoz_v3.zip`.

`PD.zip` also contained an undocumented fourth corpus, **FredPrior**: 81
sustained /a/ recordings at 8 kHz float32, one per speaker, with a demographics
spreadsheet and sheets of precomputed LPC/LAR/cepstral/MFCC features. It was
briefly registered and then **excluded by decision** — see below. Its raw audio
is left in place under `data/raw/`, and its verified config block is preserved
commented out in `configs/datasets.yaml`, so reinstating it is a matter of
uncommenting.

### Registry design

`configs/datasets.yaml` holds one block per corpus: `root`, `language`,
`channel`, a `glob`, and an ordered list of rules. Each rule is a regex matched
against the file's POSIX path *relative to the dataset root* — not just the
filename, because for two corpora the speaker is identified by the directory.
Named capture groups interpolate into field templates, and `label_map`,
`cohort_map` and `task_map` translate raw tokens into the shared vocabulary:

```
vowel_a vowel_e vowel_i vowel_o vowel_u
ddk_pataka ddk_pa ddk_ta
sentence read_text read_words read_phrases
monologue spontaneous
```

Adding a corpus means adding a config block, not editing code.
`build_manifest.py --dry-run` reports unmatched files, unmapped task tokens and
speakers whose files disagree on label; the writer refuses to run while any of
those counts is non-zero.

---

## Results

| | NeuroVoz | ItalianPD | MDVR-KCL |
|---|---|---|---|
| Language | Spanish | Italian | English |
| Channel | clinical | clinical | smartphone |
| Files | 2976 | 831 | 73 |
| Speakers | 114 (58 HC / 56 PD) | 65 (37 / 28) | 37 (21 / 16) |
| Files by label | 1509 / 1467 | 394 / 437 | 42 / 31 |
| Audio | 3.86 h | 5.55 h | 2.87 h |
| Sample rate | 44.1 kHz | **16 kHz and 44.1 kHz** | 44.1 kHz, 24-bit |

Speaker counts match the published figures for ItalianPD (65) and MDVR-KCL
(37). Files per task:

| task | NeuroVoz | ItalianPD | MDVR-KCL |
|---|---|---|---|
| vowel_a / e / i / o / u | 250 / 204 / 237 / 240 / 135 | 99 each | – |
| ddk_pataka | 101 | – | – |
| ddk_pa / ddk_ta | – | 50 / 50 | – |
| sentence | 1733 | – | – |
| monologue | 76 | – | – |
| read_text | – | 124 | 37 |
| read_words | – | 65 | – |
| read_phrases | – | 47 | – |
| spontaneous | – | – | 36 |

---

## Findings and data issues

**ItalianPD speaker identity cannot be taken from filenames.** The grammar is
`{TASK}{speakercode}{birthyear}{sex}{date}{time}.wav`, but in the young-control
group 15 speaker folders share only 4 distinct codes — the string
`LBULCAAS94M` is copy-pasted across most of them. Keying speakers off the
filename would have collapsed 15 people into 4, and speaker-level splits would
then have scattered one "speaker" across folds while believing they were
leak-free. Identity is therefore taken from the directory, which is one folder
per person. The PD group additionally nests speakers under bin folders (`1-5`,
`6-10`, `11-16`, `17-28`) and four leaf folder names repeat across bins, so the
bin forms part of the speaker id.

**NeuroVoz contains 663 duplicate wav files.** They sit in
`data/transcriptions/` and are byte-identical to files in `data/audios/`. The
glob targets `data/audios/*.wav` only; a naive `**/*.wav` would have inflated
the corpus by 22% and placed the same recording in two folds.

**The NeuroVoz version ambiguity is resolved.** 114 speakers have audio (58 HC
/ 56 PD), but the clinical metadata tables describe 108 (55 HC + 53 PD), of
which 106 carry an age. The published figures of 112 and 108 refer to the same
material counted differently. This should be recorded in `README.md`.

**ItalianPD task coverage is uneven.** Only 50 of 65 speakers have DDK and
vowel recordings; the 15 young controls performed the reading tasks only. Any
vowel-only or DDK-only experiment therefore silently drops that cohort.

**ItalianPD task codes** come from the corpus's own `FILE CODES.xlsx`: B1/B2
are the first and second reading of the phonemically balanced text *Il ramarro
della zia*; D1 and D2 are the syllables /pa/ and /ta/ held for 5 s; FB1 is a
set of phonemically balanced phrases; VA/VE/VI/VO/VU 1 and 2 are two phonations
of each vowel. The documentation lists FB2 for balanced words, but no file uses
that prefix — the word-reading files are prefixed `PR1` instead.

**MDVR-KCL encodes clinical scores in filenames**:
`ID##_{hc|pd}_{HoehnYahr}_{UPDRS-II-5}_{UPDRS-III-18}.wav`, with the task given
by the parent directory. Two quirks: one file, `ID22hc_0_0_0.wav`, omits the
underscore after the speaker id, so the rule makes that separator optional; and
basenames are *not* unique across the two task directories, which constrains any
flattened output naming downstream. One PD speaker has a read passage but no
spontaneous recording, giving 37 ReadText against 36 SpontaneousDialogue.

**One duplicate audio pair exists.** A corpus-wide MD5 scan of ItalianPD found
that speaker Giulia P's B1 and B2 are byte-identical, so the second reading was
never recorded. It is excluded in Stage 1.

---

## Decisions taken

**FredPrior is excluded.** Two reasons. It contains none of the four assigned
tasks — sustained /a/ only, where the assignment is spontaneous speech, reading
passage, words and phrases — so it cannot participate in any cross-corpus task
comparison. And its 8 kHz bandwidth was the only bandwidth difference in the
set, which made it a trivially separable domain for the wrong reason. Excluding
it leaves all three remaining corpora at an identical 8 kHz bandwidth after the
Stage 1 resample, which retires the open question of whether every corpus
needed low-passing to 4 kHz so that bandwidth was not a free cue for the domain
discriminator.

The cost of excluding it is real and should be stated: FredPrior contributed 81
speakers, 27% of all speakers in the four-corpus set, for 0.6% of the audio,
and it was the only case of one language appearing in two very different
channels. Without it, language and channel are once again almost perfectly
confounded — Spanish and Italian are clinical, English is smartphone — which is
the limitation `README.md` already flags.

**Speaker ids are namespaced per corpus** and, for ItalianPD, per cohort and
bin, so that no identifier can collide across datasets.

**Task vocabulary is shared across corpora**, with tasks that exist in only one
corpus retained in the manifest. Stage 2 quantifies the overlap, both for raw
task codes and for the four assigned task families.

---

## Open items

- ~~The NeuroVoz 114-versus-108 finding is not yet recorded in `README.md`.~~
  Recorded in the *Known issues* section, along with the convention that speaker
  counts in this project mean speakers *with audio*.
- `configs/datasets.yaml` is not committed; the repository has no commits yet.
  Still true, and now larger: nothing in the repository is committed, including
  every stage of pipeline code and all six stage reports.
- Four ItalianPD leaf folder names repeat across PD bins. They are treated as
  distinct speakers. If any pair is in fact the same person, that is leakage;
  it can be checked against the 28 named rows in `TAB 5.xlsx`.

## Reproduction

```bash
python -m src.data.build_manifest --dry-run   # must report no problems
python -m src.data.build_manifest             # writes data/manifests/manifest.csv
```
