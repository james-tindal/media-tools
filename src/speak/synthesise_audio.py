"""Supertonic chunk synthesis, buffered playback, and Opus encoding."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import subprocess
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
from supertonic.utils import chunk_text

from audio_metadata import embed_metadata, synthesis_metadata


class IncompleteSynthesisError(RuntimeError):
    """Some chunks failed; successful chunks remain available for resume."""


def split_text(text, max_len):
    chunks = []
    for chunk in chunk_text(re.sub(r" {2,}", " ", text), max_len=max_len):
        if len(chunk) <= max_len:
            chunks.append(chunk)
            continue
        start = None
        end = None
        for word in re.finditer(r"\S+", chunk):
            if start is not None and word.end() - start > max_len:
                chunks.append(chunk[start:end])
                start = None
            if start is None:
                start = word.start()
            end = word.end()
        if start is not None:
            chunks.append(chunk[start:end])
    return chunks


def initialize_worker(args):
    from supertonic import TTS
    global worker_tts, worker_voice, worker_args
    worker_args = args
    worker_tts = TTS(intra_op_num_threads=args.threads)
    worker_voice = worker_tts.get_voice_style(args.voice)


def worker_description():
    return {"model_name": worker_tts.model_name, "sample_rate": worker_tts.sample_rate}


def cache_matches(path, tags, sample_rate):
    if not path.exists():
        return False
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format_tags",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    fields = json.loads(result.stdout).get("format", {}).get("tags", {})
    stored = json.loads(fields.get("comment", "{}"))
    info = sf.info(path)
    return (stored == tags and info.frames > 0
            and info.samplerate == sample_rate and info.channels == 1)


def save_chunk(chunk, path, tags, tts, voice, args):
    wav, _ = tts.synthesize(
        chunk, voice_style=voice, lang=args.lang, speed=args.speed,
        total_steps=args.steps, silence_duration=0,
    )
    samples = np.asarray(wav).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Synthesis returned empty or nonfinite audio")
    with tempfile.TemporaryDirectory(prefix=".chunk-", dir=path.parent) as directory:
        raw = Path(directory) / "raw.wav"
        tagged = Path(directory) / "tagged.wav"
        sf.write(raw, samples, tts.sample_rate, subtype="FLOAT")
        embed_metadata(raw, tagged, tags, False)
        os.replace(tagged, path)


def synthesize_worker_chunk(chunk, path, tags):
    save_chunk(chunk, path, tags, worker_tts, worker_voice, worker_args)


def synthesise_resumable(text, destination, tts, args, cache_root, conversion, pool=None):
    chunks = split_text(text, args.max_chunk_length)
    if not chunks:
        raise ValueError("No text chunks to synthesise")
    metadata = synthesis_metadata(text, conversion, args, None)
    metadata["tts_model"] = tts.model_name
    metadata["tts_sample_rate"] = str(tts.sample_rate)
    metadata["chunk_cache_version"] = "1"
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_root) / identity
    cache.mkdir(parents=True, exist_ok=True)
    voice = tts.get_voice_style(args.voice) if pool is None else None
    failed = []
    paths = []
    pending = {}
    completed = 0
    for index, chunk in enumerate(chunks, start=1):
        path = cache / f"{index:06d}.wav"
        tags = dict(metadata)
        tags["chunk_text_sha256"] = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        tags["chunk_index"] = str(index)
        tags["chunk_count"] = str(len(chunks))
        paths.append(path)
        try:
            reusable = cache_matches(path, tags, tts.sample_rate)
            if not reusable:
                if path.exists():
                    print(f"Chunk {index}: cached audio does not match; synthesising again", file=sys.stderr)
                if pool is not None:
                    pending[pool.submit(synthesize_worker_chunk, chunk, path, tags)] = index
                    continue
                save_chunk(chunk, path, tags, tts, voice, args)
        except Exception as error:
            failed.append(index)
            print(f"\nChunk {index}/{len(chunks)} failed: {type(error).__name__}: {error}", file=sys.stderr)
        completed += 1
        show_progress(completed, len(chunks))
    for future in as_completed(pending):
        index = pending[future]
        try:
            future.result()
        except Exception as error:
            failed.append(index)
            print(f"\nChunk {index}/{len(chunks)} failed: {type(error).__name__}: {error}", file=sys.stderr)
        completed += 1
        show_progress(completed, len(chunks))
    sys.stderr.write("\n")
    if failed:
        raise IncompleteSynthesisError(
            f"Failed chunks: {', '.join(map(str, failed))}. Saved chunks: {cache}. No final audio assembled."
        )
    with sf.SoundFile(destination, mode="w", samplerate=tts.sample_rate,
                      channels=1, subtype="FLOAT") as output:
        for index, path in enumerate(paths):
            if index and args.silence_duration:
                output.write(np.zeros(round(tts.sample_rate * args.silence_duration), dtype=np.float32))
            with sf.SoundFile(path) as source:
                for block in source.blocks(blocksize=65536, dtype="float32"):
                    output.write(block)


def show_progress(completed, total):
    width = 30
    ratio = completed / total
    filled = round(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\rSynthesising [{bar}] {completed}/{total} {ratio:>4.0%}")
    sys.stderr.flush()


def produce_chunks(chunks, audio, tts, voice, args):
    show_progress(0, len(chunks))
    try:
        for index, chunk in enumerate(chunks, start=1):
            wav, _ = tts.synthesize(
                chunk, voice_style=voice, lang=args.lang, speed=args.speed,
                total_steps=args.steps, silence_duration=0,
            )
            samples = np.asarray(wav).reshape(-1)
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError(f"Invalid audio from synthesis chunk {index}")
            audio.put(samples)
            show_progress(index, len(chunks))
    finally:
        sys.stderr.write("\n")
        audio.put(None)


def synthesise_audio(text, destination, tts, args):
    """Write a WAV, or play audio if destination is None."""
    chunks = split_text(text, args.max_chunk_length)
    if not chunks:
        raise ValueError("No text chunks to synthesise")
    voice = tts.get_voice_style(args.voice)
    audio = Queue(maxsize=args.buffer)
    output = None
    try:
        if destination is not None:
            output = sf.SoundFile(destination, mode="w", samplerate=tts.sample_rate, channels=1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(produce_chunks, chunks, audio, tts, voice, args)
            completed = 0
            # If playback or writing fails, keep draining the producer so its
            # bounded queue cannot deadlock the executor during shutdown.
            consumer_error = None
            while True:
                wav = audio.get()
                if wav is None:
                    break
                if consumer_error is not None:
                    continue
                try:
                    if completed and args.silence_duration:
                        if output is not None:
                            output.write(np.zeros(round(tts.sample_rate * args.silence_duration), dtype=np.float32))
                        else:
                            sd.sleep(int(args.silence_duration * 1000))
                    if output is not None:
                        output.write(wav)
                    else:
                        sd.play(wav, tts.sample_rate)
                        sd.wait()
                    completed += 1
                except Exception as error:
                    consumer_error = error
            if consumer_error is not None:
                # Retrieve any producer failure too, while keeping the original
                # consumer failure visible as the cause.
                try:
                    future.result()
                except Exception as producer_error:
                    raise producer_error from consumer_error
                raise consumer_error
            future.result()
    finally:
        if output is not None:
            output.close()
    if destination is not None and sf.info(destination).frames <= 0:
        raise ValueError("Synthesis produced an empty audio file")


def encode_opus(source, destination, bitrate):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(source), "-c:a", "libopus", "-b:a", f"{bitrate}k",
        "-vbr", "on", "-f", "opus", str(destination),
    ], check=True)
