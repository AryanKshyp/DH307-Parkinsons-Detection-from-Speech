"""Stage 4 segmentation: cut every utterance into segments at four levels.

    python -m src.data.segments
    python -m src.data.segments --levels word phoneme

Writes one row per segment to data/interim/segments.csv, carrying the parent
utterance, speaker and label so that provenance survives into Stage 5.

Levels:

    utterance  the whole file
    duration   fixed windows, the blind-windowing baseline
    word       one segment per aligned word      (needs Stage 3)
    phoneme    one segment per aligned phone unit (needs Stage 3)

The alignment-based levels cover every corpus that has a transcript. English
(MDVR-KCL) contributes at word level only: MMS_FA emits characters, and
English orthography is not shallow enough to map those onto phones the way
Spanish and Italian do.

Turning CTC spikes into boundaries
----------------------------------
MMS_FA is a CTC model with a 20 ms stride, so the per-character spans in the
Stage 3 output are *spikes* marking where a character was recognised, not the
acoustic extent of a phone: 95% of them are exactly 20 ms long and together they
cover only 39% of their parent word. Cutting phones from them directly would
yield single-frame segments.

Each spike is therefore treated as a boundary anchor: a phone runs from its own
spike to the next spike. Trailing silence is trimmed with the same per-file
energy floor used for `duration` windows, then a 200 ms cap is applied only to
phones that never drop below that floor (a long vowel with no pause after it).
Word spans are rebuilt from their phones for the same reason -- the aligner's
own word `end` stops at the last spike and so truncates the final phone of
every word.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "manifests" / "manifest_processed.csv"
INDEX = REPO_ROOT / "data" / "interim" / "alignment_index.csv"
REJECTS = REPO_ROOT / "data" / "interim" / "alignment_rejects.csv"
OUT = REPO_ROOT / "data" / "interim" / "segments.csv"

SAMPLE_RATE = 16000

# One floor for every level, so segment length never depends on how a segment
# was produced. 50 ms is three frames at the 25 ms window / 10 ms hop Stage 5
# plans to use; anything shorter cannot yield a spectrogram worth pooling.
MIN_SEGMENT_S = 0.05

# Upper bound on a single phone, applied *after* energy trimming. A long vowel
# that never falls silent is truncated rather than allowed to swallow the
# following pause; 200 ms is past the 95th percentile of spike-to-spike gaps.
MAX_PHONE_S = 0.20
ENERGY_FRAME_S = 0.02   # 20 ms RMS frames, same grid as the CTC stride

# No word is plausibly longer than this, even at a dysarthric reading rate. A
# longer span means the aligner bridged a pause between two list items, which
# happens in the ItalianPD word-reading task. Such words are dropped rather
# than trimmed, because nothing in the alignment says where the word really is.
MAX_WORD_S = 2.0

# Two duration windows: 250 ms matches the median aligned word; 100 ms is
# closer to the median phone, so the two can be compared without a length
# confound. Hop is always half the window.
DURATION_WINDOWS = {
    "duration":     (0.25, 0.125),
    "duration_100": (0.10, 0.05),
}
SILENCE_DB = -40.0       # window RMS below this, relative to the file's own
                         # peak, counts as silence and is dropped

# --------------------------------------------------------------------------
# phonology
#
# MMS_FA emits characters. Spanish and Italian orthography is shallow enough
# that characters map onto phones with a small set of context rules, which is
# what makes a cross-language phoneme level meaningful: the label is a broad
# phonetic class, not a letter, so an Italian /tS/ and a Spanish /tS/ are the
# same category even though the spelling differs.
# --------------------------------------------------------------------------

VOWELS = set("aeiou")
FRONT = set("ei")

# (sequence, required following vowels or None, class). Tried in order, so
# longer or more specific sequences come first.
DIGRAPHS = {
    "es": [
        ("ch", None, "affricate"),
        ("ll", None, "liquid"),       # /J/, one phone, not two l's
        ("rr", None, "liquid"),       # trill
        ("qu", FRONT, "plosive"),     # u is silent: "que", "qui"
        ("gu", FRONT, "plosive"),     # u is silent: "guerra"
    ],
    "it": [
        ("gn", None, "nasal"),        # /J/, one phone
        ("gli", None, "liquid"),      # /L/
        ("sc", FRONT, "fricative"),   # /S/ before e or i
        ("ch", None, "plosive"),      # /k/
        ("gh", None, "plosive"),      # /g/
        ("ci", set("aou"), "affricate"),   # i is silent: "ciao"
        ("gi", set("aou"), "affricate"),
    ],
}

# Italian doubles a consonant to lengthen one phone. Spanish does not: "cc" in
# "accion" is two different phones, so merging would be wrong there.
GEMINATES = {"it"}


def single_class(char, nxt, language):
    """Broad phonetic class of one character, given the character after it."""
    if char in VOWELS:
        return "vowel"
    if char in "mn":
        return "nasal"
    if char in "lr":
        return "liquid"
    if char in "fs":
        return "fricative"
    if char in "pbtdkq":
        return "plosive"
    if char in "wy":
        return "glide"
    if char == "x":
        return "fricative"
    if char in "cg":
        if nxt in FRONT:
            return "fricative" if language == "es" else "affricate"
        return "plosive"
    if char == "z":
        return "fricative" if language == "es" else "affricate"
    if char == "j":
        return "fricative" if language == "es" else "glide"
    if char == "v":
        # Spanish b and v are the same phoneme; Italian v is a true fricative.
        return "plosive" if language == "es" else "fricative"
    return "other"


def word_units(entry, language):
    """Merge a word's character spikes into labelled phone units.

    Silent letters are dropped rather than given a segment: an "h" in either
    language has no acoustic extent of its own, and a segment cut there would
    contain whatever sound happens to follow.
    """
    chars = [c for c in entry["chars"] if c["char"].isalpha()]
    text = "".join(c["char"] for c in chars)

    units, i = [], 0
    while i < len(text):
        match = None
        for seq, cond, klass in DIGRAPHS.get(language, []):
            if not text.startswith(seq, i):
                continue
            after = text[i + len(seq)] if i + len(seq) < len(text) else ""
            if cond is None or after in cond:
                match = (seq, klass, len(seq))
                break

        if match is None and language in GEMINATES and i + 1 < len(text) \
                and text[i] == text[i + 1] and text[i] not in VOWELS:
            nxt = text[i + 2] if i + 2 < len(text) else ""
            match = (text[i:i + 2], single_class(text[i], nxt, language), 2)

        if match is None and text[i] == "h":
            i += 1
            continue

        if match is None:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            match = (text[i], single_class(text[i], nxt, language), 1)

        seq, klass, n = match
        units.append({
            "unit": seq,
            "phone_class": klass,
            "start": chars[i]["start"],
            "score": min(c["score"] for c in chars[i:i + n]),
            "word": entry["word"],
        })
        i += n
    return units


def phone_units(words, language, utterance_end, y, sr, floor):
    """All phone units of an utterance, with boundaries set from the anchors."""
    units = []
    for entry in words:
        units.extend(word_units(entry, language))
    if not units:
        return []

    for this, nxt in zip(units, units[1:]):
        this["end"] = close_phone(y, sr, this["start"], nxt["start"], floor)
    last = units[-1]
    last["end"] = close_phone(y, sr, last["start"], utterance_end, floor)
    # The last phone has no next spike. Energy-trim-to-end-of-file is the
    # right extent, but a CTC spike sitting at the tail of the last phone
    # can still collapse it to 20 ms. In that case fall back to the median
    # length of the other phones, as before, so the last word is not lost.
    if len(units) > 1 and last["end"] - last["start"] < MIN_SEGMENT_S:
        typical = float(np.median([u["end"] - u["start"] for u in units[:-1]]))
        last["end"] = min(last["start"] + min(typical, MAX_PHONE_S),
                          utterance_end)
    return units


def close_phone(y, sr, start, raw_end, floor):
    """Energy-trim the tail, then cap. Truncation beats swallowing a pause."""
    trimmed = energy_trim(y, sr, start, raw_end, floor)
    return min(trimmed, start + MAX_PHONE_S)


def energy_trim(y, sr, start, raw_end, floor):
    """Walk the tail back while 20 ms frames sit below the file's silence floor."""
    n0 = int(round(start * sr))
    n1 = min(int(round(raw_end * sr)), len(y))
    if n1 < n0:
        n1 = n0
    frame = max(1, int(round(ENERGY_FRAME_S * sr)))
    if floor <= 0.0:
        return n1 / sr
    while n1 - n0 > frame:
        if float(np.sqrt(np.mean(y[n1 - frame:n1] ** 2))) >= floor:
            break
        n1 -= frame
    return n1 / sr


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def utterance_rows(row):
    return [{"start": 0.0, "end": row.duration, "unit": "", "phone_class": "",
             "score": np.nan}]


def duration_rows(y, sr, floor, window_s, hop_s):
    """Fixed windows, dropping those that are essentially silent.

    The silence threshold is relative to each file's own peak, never to a
    corpus statistic, which would leak domain information into which segments
    exist at all.
    """
    if floor <= 0.0:
        return []
    width, hop = int(window_s * sr), int(hop_s * sr)
    rows = []
    for start in range(0, max(1, y.size - width + 1), hop):
        window = y[start:start + width]
        if window.size < width:
            break
        if np.sqrt(np.mean(window ** 2)) < floor:
            continue
        rows.append({"start": start / sr, "end": (start + width) / sr,
                     "unit": "", "phone_class": "", "score": np.nan})
    return rows


def word_rows(units):
    """One segment per word, spanning that word's phone units."""
    rows, current = [], []
    for unit in units:
        if current and unit["word"] != current[-1]["word"]:
            rows.append(current)
            current = []
        current.append(unit)
    if current:
        rows.append(current)

    return [{"start": group[0]["start"], "end": group[-1]["end"],
             "unit": group[0]["word"], "phone_class": "",
             "score": min(u["score"] for u in group)} for group in rows]


def phoneme_rows(units):
    return [{"start": u["start"], "end": u["end"], "unit": u["unit"],
             "phone_class": u["phone_class"], "score": u["score"]}
            for u in units]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+",
                    default=["utterance", "duration", "duration_100",
                             "word", "phoneme"],
                    choices=["utterance", "duration", "duration_100",
                             "word", "phoneme"])
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST)
    alignments = {}
    if {"word", "phoneme"} & set(args.levels):
        index = pd.read_csv(INDEX)
        rejected = set(pd.read_csv(REJECTS).path) if REJECTS.exists() else set()
        ok = index[(index.status == "ok") & ~index.path.isin(rejected)]
        alignments = dict(zip(ok.path, ok.alignment))
        print(f"{len(alignments)} accepted alignments "
              f"({len(rejected)} rejected in Stage 3)")

    keep = ["path", "dataset", "language", "channel", "speaker_id", "label",
            "cohort", "task"]
    duration_levels = [lv for lv in args.levels if lv in DURATION_WINDOWS]
    need_audio = bool(duration_levels) or bool({"word", "phoneme"} & set(args.levels))
    rows, short, long_words = [], 0, 0
    for row in tqdm(list(manifest.itertuples()), unit="file"):
        y = floor = None
        if need_audio:
            y, sr = sf.read(REPO_ROOT / row.path, dtype="float32")
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            floor = peak * 10 ** (SILENCE_DB / 20) if peak > 0.0 else 0.0

        units = None
        if row.path in alignments and {"word", "phoneme"} & set(args.levels):
            words = json.loads((REPO_ROOT / alignments[row.path]).read_text(
                encoding="utf-8"))["words"]
            units = phone_units(words, row.language, row.duration,
                                y, SAMPLE_RATE, floor)

        produced = {}
        if "utterance" in args.levels:
            produced["utterance"] = utterance_rows(row)
        for level in duration_levels:
            window_s, hop_s = DURATION_WINDOWS[level]
            produced[level] = duration_rows(y, SAMPLE_RATE, floor,
                                            window_s, hop_s)
        if units:
            if "word" in args.levels:
                produced["word"] = word_rows(units)
            # English orthography is not shallow; character spikes are still
            # usable as word anchors, but not as a cross-language phoneme set.
            if "phoneme" in args.levels and row.language in DIGRAPHS:
                produced["phoneme"] = phoneme_rows(units)

        meta = {k: getattr(row, k) for k in keep}
        stem = Path(row.path).stem
        for level, segments in produced.items():
            index_in_level = 0
            for seg in segments:
                length = seg["end"] - seg["start"]
                # The utterance level is exempt: a whole file shorter than the
                # floor is still the unit of interest, not a fragment of one.
                if level != "utterance" and length < MIN_SEGMENT_S:
                    short += 1
                    continue
                if level == "word" and length > MAX_WORD_S:
                    long_words += 1
                    continue
                rows.append({**meta, "level": level,
                             "segment_id": f"{stem}__{level}__"
                                           f"{index_in_level:05d}",
                             "start": round(seg["start"], 4),
                             "end": round(seg["end"], 4),
                             "duration": round(length, 4),
                             "unit": seg["unit"],
                             "phone_class": seg["phone_class"],
                             "score": seg["score"]})
                index_in_level += 1

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\ndropped {short} segments under {MIN_SEGMENT_S*1000:.0f} ms and "
          f"{long_words} words over {MAX_WORD_S:.0f} s")
    print(f"wrote {len(out)} segments to {OUT.relative_to(REPO_ROOT)}\n")
    print(pd.crosstab(out.level, out.dataset).to_string())
    print("\nsegment duration by level (s):")
    print(out.groupby("level").duration.describe(
        percentiles=[.05, .5, .95]).round(3).to_string())
    print("\nspeakers and files retained per level:")
    print(out.groupby("level").agg(
        files=("path", "nunique"), speakers=("speaker_id", "nunique"),
        hours=("duration", lambda s: round(s.sum() / 3600, 2))).to_string())
    phones = out[out.level == "phoneme"]
    if len(phones):
        print("\nphone classes:")
        print(pd.crosstab(phones.phone_class, phones.language).to_string())


if __name__ == "__main__":
    main()
