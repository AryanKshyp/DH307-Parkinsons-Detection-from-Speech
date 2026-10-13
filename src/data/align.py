"""Forced alignment with the multilingual MMS_FA CTC model.

    python -m src.data.align --dataset NeuroVoz
    python -m src.data.align --dataset NeuroVoz --limit 20   # smoke test

One aligner is used for every language on purpose. A per-language aligner is
more accurate on any single corpus, but its error profile would then differ by
corpus, which reintroduces exactly the domain signal this project removes.

Writes one JSON per utterance to data/interim/alignments/<dataset>/ and an
index to data/interim/alignment_index.csv. Failures are recorded in the index
with a reason rather than dropped silently.

MMS_FA emits character tokens, not phones. For Spanish and Italian the
orthography is shallow enough that characters approximate phones closely; for
English it is not. Phone-level segmentation is revisited in Stage 4.
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = REPO_ROOT / "data" / "interim" / "transcripts.csv"
PROMPTS = REPO_ROOT / "configs" / "prompts.yaml"
ALIGN_DIR = REPO_ROOT / "data" / "interim" / "alignments"
INDEX = REPO_ROOT / "data" / "interim" / "alignment_index.csv"

SAMPLE_RATE = 16000


def romanize(text, vocabulary):
    """Fold to the model's character set the way uroman would for Latin script.

    Accents are stripped rather than dropped, so "ma\u00f1ana" becomes "manana"
    and keeps its five characters instead of losing one.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Apostrophes are in the MMS dictionary but occupy the CTC blank index,
    # so keeping them makes the aligner refuse the whole transcript
    # ("targets Tensor shouldn't contain blank index"). Drop them: "don't"
    # becomes "dont", "observer's" becomes "observers".
    text = text.replace("'", "")
    text = "".join(c if (c in vocabulary or c.isspace()) else " " for c in text)
    return [w for w in re.split(r"\s+", text) if w]


class Backend:
    """MMS_FA on the requested device, falling back to CPU when VRAM runs out.

    Self-attention over a long utterance is what exhausts a small card, so the
    fallback is per file rather than global: short utterances keep the GPU.
    """

    def __init__(self, device, gpu_max_seconds):
        bundle = torchaudio.pipelines.MMS_FA
        self.bundle = bundle
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.vocabulary = set(bundle.get_dict())
        self.device = torch.device(device)
        self.model = bundle.get_model().to(self.device).eval()
        self.gpu_max_seconds = gpu_max_seconds
        self._cpu_model = None
        self.long_on_cpu = 0
        self.oom_fallbacks = 0

    def _cpu(self):
        if self._cpu_model is None:
            if self.device.type == "cuda":
                self._cpu_model = self.bundle.get_model().to("cpu").eval()
            else:
                self._cpu_model = self.model
        return self._cpu_model

    def emit(self, waveform):
        seconds = waveform.size(1) / SAMPLE_RATE
        # Self-attention cost grows with the square of the utterance length, so
        # a couple of long monologues will exhaust a small card and leave the
        # allocator fragmented for everything that follows. Send them to the
        # CPU before that happens.
        if self.device.type == "cuda" and seconds <= self.gpu_max_seconds:
            try:
                with torch.inference_mode():
                    return self.model(waveform.to(self.device))[0].cpu(), "cuda"
            except Exception as exc:                   # noqa: BLE001
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                self.oom_fallbacks += 1
        elif self.device.type == "cuda":
            self.long_on_cpu += 1

        with torch.inference_mode():
            return self._cpu()(waveform)[0], "cpu"


def align_one(path, text, backend):
    # soundfile rather than torchaudio.load: torchaudio 2.11 routes loading
    # through TorchCodec, which is an extra native dependency for no gain on
    # 16 kHz mono PCM.
    samples, sr = sf.read(REPO_ROOT / path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    waveform = torch.from_numpy(samples.T)

    words = romanize(text, backend.vocabulary)
    if not words:
        raise ValueError("no alignable characters in transcript")

    emission, used = backend.emit(waveform)
    spans = backend.aligner(emission[0], backend.tokenizer(words))

    ratio = waveform.size(1) / emission.size(1) / SAMPLE_RATE
    out = []
    for word, span in zip(words, spans):
        out.append({
            "word": word,
            "start": round(span[0].start * ratio, 4),
            "end": round(span[-1].end * ratio, 4),
            "score": round(sum(s.score * len(s) for s in span)
                           / sum(len(s) for s in span), 4),
            "chars": [{"char": word[i],
                       "start": round(s.start * ratio, 4),
                       "end": round(s.end * ratio, 4),
                       "score": round(s.score, 4)}
                      for i, s in enumerate(span)],
        })
    return out, waveform.size(1) / SAMPLE_RATE, used


def prompt_variants(dataset, task, default_text, prompts):
    """Texts to try for one utterance.

    Scripted tasks with a `text_extended` entry (MDVR-KCL read_text) are
    aligned against both the short and long prompt; the higher-scoring
    alignment is kept. Everyone else has a single transcript.
    """
    texts = [default_text]
    extra = (prompts.get(dataset, {}).get(task) or {}).get("text_extended")
    if extra:
        long = " ".join(extra.split())
        if long and long != default_text:
            texts.append(long)
    return texts


def mean_score(words):
    return sum(w["score"] for w in words) / len(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, help="align only the first N files")
    ap.add_argument("--per-task", type=int,
                    help="align N files per task, spread over speakers "
                         "(for smoke tests, which head -n would not spread)")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already aligned successfully; failures "
                         "are retried")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--gpu-max-seconds", type=float, default=30.0,
                    help="utterances longer than this are aligned on the CPU")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    d = pd.read_csv(TRANSCRIPTS)
    prompts = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    todo = d[(d.dataset == args.dataset) & (d.source != "missing")]
    if todo.empty:
        raise SystemExit(f"no transcripts for {args.dataset!r}")
    total = len(todo)

    if args.per_task:
        todo = (todo.sample(frac=1, random_state=0)
                    .groupby("task").head(args.per_task).sort_values("path"))
    if args.limit:
        todo = todo.head(args.limit)

    previous = pd.read_csv(INDEX) if INDEX.exists() else None
    keep = pd.DataFrame()
    if args.resume and previous is not None:
        mine = previous[previous.dataset == args.dataset]
        done = mine[(mine.status == "ok") & mine.alignment.apply(
            lambda p: isinstance(p, str) and (REPO_ROOT / p).exists())]
        keep = done
        todo = todo[~todo.path.isin(set(done.path))]
        print(f"resuming: {len(done)} already aligned, {len(todo)} to go")

    if todo.empty:
        print(f"nothing left to align for {args.dataset!r}")
        return
    print(f"{len(todo)} of {total} transcribed {args.dataset} files")

    backend = Backend(args.device, args.gpu_max_seconds)
    if args.device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"device: {name} ({vram:.1f} GB)")
    else:
        print("device: cpu")
    print(f"MMS_FA vocabulary ({len(backend.vocabulary)}): "
          f"{''.join(sorted(c for c in backend.vocabulary if len(c) == 1))}")

    out_dir = ALIGN_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    audio_seconds = 0.0
    for i, row in enumerate(todo.itertuples(), 1):
        record = {"path": row.path, "dataset": row.dataset,
                  "speaker_id": row.speaker_id, "task": row.task}
        try:
            best, seconds, used, variant = None, 0.0, "", 0
            last_error = None
            for v, text in enumerate(prompt_variants(
                    row.dataset, row.task, row.text, prompts)):
                try:
                    words, seconds, used = align_one(row.path, text, backend)
                except Exception as exc:                   # noqa: BLE001
                    last_error = exc
                    continue
                if best is None or mean_score(words) > mean_score(best):
                    best, variant = words, v
            if best is None:
                raise last_error or ValueError("no prompt variant aligned")
            audio_seconds += seconds
            words = best
            dst = out_dir / (Path(row.path).stem + ".json")
            dst.write_text(json.dumps(
                {**record, "duration": seconds, "prompt_variant": variant,
                 "words": words}, indent=1),
                encoding="utf-8")
            scores = [w["score"] for w in words]
            record.update(status="ok", alignment=dst.relative_to(
                REPO_ROOT).as_posix(), n_words=len(words),
                mean_score=round(sum(scores) / len(scores), 4),
                min_score=round(min(scores), 4), device=used, error="")
        except Exception as exc:                       # noqa: BLE001
            record.update(status="failed", alignment="", n_words=0,
                          mean_score=float("nan"), min_score=float("nan"),
                          device="", error=f"{type(exc).__name__}: {exc}")
        rows.append(record)

        if i % 50 == 0 or i == len(todo):
            elapsed = time.time() - started
            rtf = audio_seconds / elapsed if elapsed else 0
            print(f"  {i}/{len(todo)}  {elapsed/60:.1f} min  "
                  f"{rtf:.1f}x realtime", flush=True)

    index = pd.DataFrame(rows)
    if len(keep):
        index = pd.concat([keep, index], ignore_index=True)
    if previous is not None:
        index = pd.concat([previous[previous.dataset != args.dataset], index],
                          ignore_index=True)
    index.to_csv(INDEX, index=False)

    done = index[index.dataset == args.dataset]
    ok = done[done.status == "ok"]
    print(f"\naligned {len(ok)}/{len(done)}  -> {out_dir.relative_to(REPO_ROOT)}")
    if len(ok):
        print(f"mean word score {ok.mean_score.mean():.3f}, "
              f"worst file {ok.mean_score.min():.3f}")
    if backend.long_on_cpu:
        print(f"routed to CPU as longer than {args.gpu_max_seconds:.0f} s: "
              f"{backend.long_on_cpu} file(s)")
    if backend.oom_fallbacks:
        print(f"fell back to CPU after running out of VRAM: "
              f"{backend.oom_fallbacks} file(s)")
    failed = done[done.status == "failed"]
    if len(failed):
        print(f"\nfailed ({len(failed)}):")
        print(failed.error.value_counts().to_string())
    print(f"index: {INDEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
