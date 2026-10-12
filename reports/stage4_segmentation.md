# Stage 4 — Segmentation

DH307 Subtopic 4 · Corpus- and Language-Invariant Parkinson's Detection from Speech

**1,106,469 segments** were cut across five levels, each carrying its parent
utterance, speaker and label. The first cut was 374,195 rows on four levels;
the recut adds `duration_100`, energy-trims phone tails, and brings MDVR-KCL
read passages into the word table. The stage's main technical result is that
the Stage 3 alignment could not be used as it stood — its character spans are
CTC spikes, not phone extents — and the fix, treating spikes as boundary
anchors then energy-trimming to the next spike, yields a phoneme level with a
median duration of 80 ms. Its main analytical result is that **phone-class
composition differs sharply between the two aligned corpora**, so the class mix
is itself a domain cue and phoneme-level models must be trained within a class.

**In:** `data/manifests/manifest_processed.csv`, `data/interim/alignments/`
**Out:** `data/interim/segments.csv`
**Code:** `src/data/segments.py`

---

## Results

The table below is the **first cut** (four levels, NeuroVoz + ItalianPD
`read_words` only). The recut numbers — five levels, energy-trimmed phones,
MDVR-KCL words — are in the follow-up at the bottom of this report.

| level | segments | files | speakers | median duration |
|---|---|---|---|---|
| `duration` | 274,459 | 3871 | 216 | 250 ms (fixed) |
| `phoneme` | 75,311 | 1851 | 175 | 80 ms |
| `word` | 20,554 | 1851 | 175 | 362 ms |
| `utterance` | 3871 | 3871 | 216 | 3.90 s |

| level | NeuroVoz | ItalianPD | MDVR-KCL |
|---|---|---|---|
| `duration` | 98,753 | 126,731 | 48,975 |
| `phoneme` | 58,749 | 16,562 | – |
| `word` | 17,390 | 3,164 | – |
| `utterance` | 2968 | 830 | 73 |

Segments per speaker, which is what governs batch construction and epoch size:
`duration` a median of 1088, `phoneme` 370, `word` 109, `utterance` 20.

The `duration` level totals 19.06 hours of segment time from 12.26 hours of
audio, because the windows overlap by 50%. That is expected, but it means
segment counts and segment *hours* are not interchangeable when reporting.

---

## Turning CTC spikes into boundaries

Stage 3 delivered per-character spans, and the obvious reading of them is
wrong. MMS_FA is a CTC model with a 20 ms stride, so each character span marks
the frame where the model recognised that character, not the sound's extent.
Measured across the accepted alignments: 95% of character spans are exactly
20 ms, and all the spans in a word together cover only **39% of that word**.
Cutting phones from them directly would have produced single-frame segments —
useless for a spectrogram, and quietly so, since they would still have looked
like a plausible number of rows in a CSV.

Each spike is therefore used as a boundary *anchor*: a phone runs from its own
spike to the next spike, capped at `MAX_PHONE_S = 200 ms` so an inter-word pause
is not absorbed into the phone before it. On this data that gives a median phone
of 80 ms with 80% between 50 and 200 ms, which is where real phones live.

Two consequences worth recording.

**Word spans had to be rebuilt from phones as well.** The aligner's own word
`end` stops at the last spike, so it truncates the final phone of every word.
Taking word boundaries from the rebuilt phones fixes this consistently.

**The final phone of an utterance has no next anchor**, and the aligner's word
end cannot supply one. It is given the median length of the other phones in the
same utterance. Without this it collapses to 20 ms, fails the minimum-length
filter, and takes the last word of every utterance with it — 1851 words, 9% of
the word level.

The 200 ms cap binds on 11.8% of phones, and 78% of those are vowels. Those
segments are truncated rather than wrong: a long vowel is cut short, but no
silence is admitted. Truncation is the safer of the two errors, though vowel
duration is itself a PD-relevant cue, so this is worth revisiting with
energy-based trimming rather than a fixed cap.

---

## Phone labelling across languages

Characters are not comparable across languages — Italian `c` before a front
vowel is an affricate where Spanish `c` is a fricative — so a character label
would make the phoneme level look cross-lingual while actually encoding
orthography. Each unit is therefore labelled with a **broad phonetic class**:
vowel, plosive, fricative, affricate, nasal, liquid, glide.

The mapping uses per-language context rules and digraph merging, both of which
matter:

- **Digraphs are one phone, not two.** Spanish `ch` (affricate), `ll`, `rr`;
  Italian `gn`, `gli`, `sc` before a front vowel, `ch` and `gh` (both plosives).
- **Silent letters get no segment.** `h` is silent in both languages, and a
  segment cut there would contain whatever follows it.
- **Silent vowels in digraphs are absorbed**: Spanish `qu`/`gu` before `e`/`i`,
  Italian `ci`/`gi` before a back vowel.
- **Geminates are merged in Italian only.** A doubled Italian consonant is one
  long phone, but Spanish `cc` in *acción* is two different phones, so merging
  there would be wrong.
- **Context decides class**: `c` and `g` before `e`/`i` are fricatives in
  Spanish but affricates in Italian; `z` is a fricative in Spanish and an
  affricate in Italian; `v` patterns with `b` in Spanish and is a true fricative
  in Italian.

Both the class and the raw character are kept on every row, so nothing is lost
to the interpretation.

### The class mix is a domain cue

| class | Spanish | Italian | Spanish % | Italian % |
|---|---|---|---|---|
| vowel | 29,312 | 7677 | 49.9 | 46.4 |
| plosive | 11,024 | 3290 | 18.8 | 19.9 |
| nasal | 7316 | 940 | **12.5** | **5.7** |
| liquid | 5682 | 2085 | 9.7 | 12.6 |
| fricative | 4493 | 1995 | 7.6 | 12.0 |
| affricate | 299 | 575 | **0.5** | **3.5** |
| glide | 623 | 0 | 1.1 | 0.0 |

Italian has seven times the affricate share, Spanish more than twice the nasal
share, and glides occur only in Spanish. **A phoneme-level model trained across
pooled classes could identify the corpus from the class mix alone**, without
learning anything acoustic — which is precisely the failure mode this project
exists to prevent. Phoneme-level DANN runs should therefore be trained *within*
a phone class, which is what the assigned pipeline already asks for when it
gives "one DANN for the vowel-phoneme category" as its example.

One caveat on interpreting the table: this is a difference between the *prompts*
as much as between the languages. NeuroVoz's material is short sentences,
ItalianPD's is a 50-word list, and their phonetic content differs by design. The
two causes cannot be separated with the corpora available, which is another
reason to condition on class rather than to model the mixture.

---

## Minimum and maximum segment length

**Minimum 50 ms, applied uniformly to every level except `utterance`.** At the
25 ms window and 10 ms hop Stage 5 plans, 50 ms is three frames — the least that
can produce a spectrogram worth pooling. 10,080 segments fall below it. The
`utterance` level is exempt because a short whole file is still the unit of
interest rather than a fragment of one.

The cost is not neutral and should be stated: short plosive bursts are the
segments most likely to be dropped, so the phoneme level is biased slightly
toward vowels and continuants relative to what was actually spoken.

**Maximum 2 s for words.** 50 words exceed it, almost all in the ItalianPD
word-reading task, where the aligner bridged the pause between two list items
and produced spans up to 9.8 s. They are dropped rather than trimmed, because
nothing in the alignment indicates where inside the span the word actually sits.
Their alignment scores were near zero, so a confidence filter would have caught
most of the same files, but a physical bound on word length is the more direct
statement of why they are wrong.

No maximum is imposed on `duration` (fixed by construction) or `utterance`
(where the length *is* the datum).

---

## Silence-aware windowing

The `duration` level is the blind-windowing baseline: **250 ms windows at a
125 ms hop**, chosen to match the median aligned word so that the comparison
against the word level is direct rather than confounded by segment length.

Windows whose RMS falls more than 40 dB below their own file's peak are dropped.
The threshold is per file and never per corpus — a per-corpus statistic would
leak domain information into which segments exist at all, violating the
project's own invariant.

The filter matters most exactly where it was predicted to:

| | windows before filter | kept | dropped as silent |
|---|---|---|---|
| NeuroVoz | 111,203 | 98,753 | 11.2% |
| ItalianPD | 159,409 | 126,731 | 20.5% |
| MDVR-KCL | 82,548 | 48,975 | **40.7%** |

Two of every five MDVR-KCL windows contain no speech, consistent with the
phone-call recording posture and the 25% mean silence fraction measured in
Stage 1. Without the filter, 40% of that corpus's segments would have been
silence labelled with a diagnosis.

---

## Coverage and its limits

`phoneme` covers NeuroVoz and ItalianPD only: English is word-only because MMS
characters are not a phone set. `word` now includes MDVR-KCL **read passages**
(35 files after rejects). MDVR-KCL spontaneous ASR alignments were almost all
rejected, so that task stays at `duration` / `utterance`. Silence filtering
those windows is not diarization — interviewer speech remains in the mix.

| level | corpora | languages | channels |
|---|---|---|---|
| `duration`, `duration_100`, `utterance` | 3 | es, it, en | clinical, smartphone |
| `word` | 3 | es, it, en | clinical + smartphone read |
| `phoneme` | 2 | es, it | clinical only |

**The `utterance` level is not length-comparable across corpora** — a median of
3.90 s overall but 250 s at the maximum, since an MDVR-KCL "utterance" is a
two-minute phone call and a NeuroVoz one is a four-word sentence. It is kept
whole by decision, on the grounds that the file *is* the recording unit, but any
utterance-level result has to be read with that 40-fold spread in mind. The
`duration` level exists partly to give a length-controlled alternative.

---

## Follow-up — energy trim, 100 ms windows, plosives, English words

Phone tails are now energy-trimmed with the same per-file −40 dB floor as
`duration`, then capped at 200 ms. Median phone is still 80 ms; the 95th
percentile sits on the cap (0.200 s), so the cap still binds on the long
tail. ItalianPD `read_text` / `read_phrases` and MDVR-KCL `read_text` are in
the word table. English is word-only: MMS characters are not a phone set.

**1,106,469 segments** after the recut (was 374,195). The jump is
`duration_100` (651,690 windows at 100 ms / 50 ms hop).

| level | NeuroVoz | ItalianPD | MDVR-KCL |
|---|---|---|---|
| `duration` | 98,753 | 126,731 | 48,975 |
| `duration_100` | 244,841 | 295,394 | 111,455 |
| `phoneme` | 57,620 | 74,912 | – |
| `word` | 17,369 | 20,930 | 5,618 |
| `utterance` | 2968 | 830 | 73 |

The 50 ms floor stands. Plosive *analysis* uses a 15 ms FFT (`PLOSIVE_N_FFT
= 240`) at the same 10 ms hop, so the frame grid is shared with every other
class. That is a per-class exception to the uniform 25 ms window: a plosive
DANN is not starved of frames, but a table that pools plosives with vowels
is partly a comparison of analysis settings. Handcrafted features stay on
the 25 ms spectrogram.

Stage 5 caches for `word`, `phoneme`, and `duration_100` were rebuilt after
this recut. `duration` (250 ms) and `utterance` windows did not change.

## Open items

- Energy trim did not retire the 200 ms cap; it still binds at the 95th
  percentile. Worth measuring how many phones hit it after trim, and whether
  PD/HC vowel-duration gap moved.
- `duration_100` features are extracted; they have not been probed for corpus
  vs PD/HC the same way as the 250 ms level. Do that before picking a Stage 7
  default window size.
- MDVR-KCL word-level is read-passage only (35 files after rejects).
  Spontaneous ASR alignments were almost all rejected.

## Reproduction

```bash
python -m src.data.segments                        # all five levels
python -m src.data.segments --levels word phoneme  # still reads audio (energy trim)
```
