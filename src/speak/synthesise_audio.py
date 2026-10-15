"""Supertonic chunk synthesis, buffered playback, and Opus encoding."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event
import hashlib
import json
import os
from pathlib import Path
import re
import select
import termios
import tty
import tempfile
import subprocess
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
from supertonic.utils import chunk_text

from audio_metadata import embed_metadata, synthesis_metadata
from playback_speed import PlaybackSpeed
from playback_progress import PlaybackProgress, predict_durations
from output_device import output_device_changes, refresh_output_device


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
        units = []
        if "|" in chunk:
            for cell in re.finditer(r"[^|]*\||[^|]+$", chunk):
                if len(cell.group().strip()) > max_len:
                    units.extend((cell.start() + word.start(), cell.start() + word.end())
                                 for word in re.finditer(r"\S+", cell.group()))
                elif cell.group().strip():
                    leading = len(cell.group()) - len(cell.group().lstrip())
                    units.append((cell.start() + leading,
                                  cell.start() + len(cell.group().rstrip())))
        else:
            units = [(word.start(), word.end()) for word in re.finditer(r"\S+", chunk)]
        for unit_start, unit_end in units:
            if start is not None and unit_end - start > max_len:
                chunks.append(chunk[start:end])
                start = None
            if start is None:
                start = unit_start
            end = unit_end
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


def progress_bar(completed, total, counts=True):
    width = 30
    ratio = completed / total
    filled = round(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    count = f" {completed}/{total}" if counts else ""
    return f"[{bar}]{count} {ratio:>4.0%}"


def show_progress(completed, total):
    sys.stderr.write("\r" + progress_bar(completed, total))
    sys.stderr.flush()


def produce_chunks(chunks, audio, tts, voice, args, progress=True, playback_progress=None, start_index=0):
    if progress:
        show_progress(0, len(chunks))
    try:
        for index, chunk in enumerate(chunks, start=1):
            if audio.cancelled:
                return
            wav, _ = tts.synthesize(
                chunk, voice_style=voice, lang=args.lang, speed=args.speed,
                total_steps=args.steps, silence_duration=0,
            )
            samples = np.asarray(wav).reshape(-1)
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError(f"Invalid audio from synthesis chunk {index}")
            if playback_progress is not None:
                playback_progress.durations[start_index + index - 1] = len(samples) / tts.sample_rate
            if not audio.put_chunk(samples):
                return
            if progress:
                show_progress(index, len(chunks))
    finally:
        if progress:
            sys.stderr.write("\n")
        audio.put_chunk(None)


@contextmanager
def playback_keyboard(enabled):
    if not enabled or not sys.stdin.isatty():
        yield None
        return
    keyboard = sys.stdin.fileno()
    original = termios.tcgetattr(keyboard)
    try:
        tty.setcbreak(keyboard)
        print("Space: pause/resume. Up/Down: playback speed. Ctrl+C: stop.", file=sys.stderr)
        yield keyboard
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()
        termios.tcsetattr(keyboard, termios.TCSANOW, original)


class AudioQueue(Queue):
    cancelled = False

    def put_chunk(self, samples):
        while not self.cancelled:
            try:
                self.put(samples, timeout=0.05)
                return True
            except Full:
                continue
        return False

    def buffered_frames(self):
        with self.mutex:
            return sum(len(samples) for samples in self.queue if samples is not None)


@dataclass
class PlaybackControls:
    paused: bool = False
    speed: float = 1.0
    keys: bytes = b""
    chunk: int = 0
    total_chunks: int = 1
    elapsed: float = 0.0
    duration: float | None = None
    displayed: bool = False
    output_changed: Event | None = None
    progress: PlaybackProgress | None = None


def chunk_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def update_playback(keyboard, controls, audio, sample_rate, remaining=0, buffering=False):
    if keyboard is None:
        return
    if select.select([keyboard], [], [], 0)[0]:
        controls.keys += os.read(keyboard, 64)
    while controls.keys:
        if controls.keys.startswith(b"\x1b"):
            if len(controls.keys) < 3:
                break
            key, controls.keys = controls.keys[:3], controls.keys[3:]
            if key in (b"\x1b[A", b"\x1bOA"):
                controls.speed = min(2.0, round(controls.speed + 0.1, 1))
            elif key in (b"\x1b[B", b"\x1bOB"):
                controls.speed = max(0.5, round(controls.speed - 0.1, 1))
        else:
            key, controls.keys = controls.keys[:1], controls.keys[1:]
            if key == b" ":
                controls.paused = not controls.paused
    state = "⏸ Paused" if controls.paused else "⏳ Buffering" if buffering else "▶ Playing"
    seconds = (audio.buffered_frames() + remaining) / sample_rate / controls.speed
    timing = chunk_time(controls.elapsed)
    if controls.duration is not None:
        timing += " / " + chunk_time(controls.duration)
    prefix = "\r\033[1A" if controls.displayed else "\r"
    played = max(0, controls.chunk - 1)
    if controls.duration:
        played += min(1.0, controls.elapsed / controls.duration)
    if controls.progress is not None:
        bar = progress_bar(controls.progress.fraction(), 1, counts=False)
    else:
        bar = progress_bar(played, controls.total_chunks, counts=False)
    status = (f"{state} · Chunk {controls.chunk}/{controls.total_chunks} · {timing}"
              f" · {controls.speed:.1f}× · Buffered: {seconds:.1f}s")
    print(f"{prefix}\033[2K{bar}\n\r\033[2K{status}", end="", file=sys.stderr, flush=True)
    controls.displayed = True


def play_audio(samples, sample_rate, keyboard, controls, audio, chunk=True):
    finished = Event()
    errors = []
    if chunk:
        controls.elapsed = 0.0
        controls.duration = len(samples) / sample_rate
    update_playback(keyboard, controls, audio, sample_rate, len(samples))

    with PlaybackSpeed(samples, sample_rate) as source:
        playback_complete = False
        submitted = deque()
        replay = deque()

        def callback(outdata, frames, time, status):
            nonlocal playback_complete
            outdata.fill(0)
            if controls.paused or (controls.output_changed is not None
                                   and controls.output_changed.is_set()):
                return
            try:
                while submitted and submitted[0][0] + len(submitted[0][1]) / sample_rate <= time.currentTime:
                    submitted.popleft()
                replaying = bool(replay)
                if replaying:
                    queued, start, end = replay.popleft()
                    count = min(frames, len(queued))
                    block = queued[:count]
                    position = start + (end - start) * count / len(queued)
                    if count < len(queued):
                        replay.appendleft((queued[count:], position, end))
                else:
                    start = (len(samples) - source.remaining_frames) / sample_rate
                    block = source.read(frames, controls.speed)
                    position = (len(samples) - source.remaining_frames) / sample_rate
                outdata[:len(block), 0] = block
                if len(block):
                    submitted.append((time.outputBufferDacTime, block.copy(), start, position))
                if controls.progress is not None:
                    controls.progress.elapsed = position
                if chunk:
                    controls.elapsed = position
            except Exception as error:
                errors.append(error)
                raise sd.CallbackAbort from error
            if len(block) < frames and not replaying:
                playback_complete = True
                if controls.progress is not None:
                    controls.progress.elapsed = len(samples) / sample_rate
                if chunk:
                    controls.elapsed = controls.duration
                raise sd.CallbackStop

        while True:
            if controls.output_changed is not None and controls.output_changed.is_set():
                controls.output_changed.clear()
                refresh_output_device()
            finished.clear()
            switched = False
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                 callback=callback, finished_callback=finished.set) as stream:
                while True:
                    done = finished.wait(0.05)
                    if controls.output_changed is not None and controls.output_changed.is_set():
                        stopped_at = stream.time
                        stream.abort()
                        retained = deque()
                        for dac_time, block, start, end in submitted:
                            offset = min(len(block), max(0, round((stopped_at - dac_time) * sample_rate)))
                            if offset < len(block):
                                position = start + (end - start) * offset / len(block)
                                retained.append((block[offset:], position, end))
                        retained.extend(replay)
                        replay.clear()
                        replay.extend(retained)
                        submitted.clear()
                        if replay:
                            if controls.progress is not None:
                                controls.progress.elapsed = replay[0][1]
                            if chunk:
                                controls.elapsed = replay[0][1]
                        playback_complete = False
                        switched = True
                        break
                    if done:
                        if not errors and not playback_complete:
                            raise RuntimeError("Audio output stopped before playback finished")
                        break
                    if not stream.active and not playback_complete:
                        raise RuntimeError("Audio output became inactive before playback finished")
                    update_playback(keyboard, controls, audio, sample_rate, source.remaining_frames)
            if not switched or errors:
                break
        if errors:
            raise errors[0]
        update_playback(keyboard, controls, audio, sample_rate, source.remaining_frames)


@contextmanager
def synthesis_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    wait = True
    try:
        yield executor
    except KeyboardInterrupt:
        wait = False
        raise
    finally:
        executor.shutdown(wait=wait, cancel_futures=True)


def synthesise_audio(text, destination, tts, args):
    """Write a WAV, or play audio if destination is None."""
    chunks = split_text(text, args.max_chunk_length)
    if not chunks:
        raise ValueError("No text chunks to synthesise")
    start_chunk = getattr(args, "start_chunk", None)
    if start_chunk is None:
        start_chunk = 1
    if not 1 <= start_chunk <= len(chunks):
        raise ValueError(f"Start chunk must be between 1 and {len(chunks)}")
    voice = tts.get_voice_style(args.voice)
    progress = None
    if destination is None:
        progress = PlaybackProgress(predict_durations(chunks, tts, voice, args),
                                    args.silence_duration, chunk_index=start_chunk - 1)
    audio = AudioQueue(maxsize=args.buffer)
    output = None
    try:
        if destination is not None:
            output = sf.SoundFile(destination, mode="w", samplerate=tts.sample_rate, channels=1)
        with output_device_changes(destination is None) as output_changed, \
                playback_keyboard(destination is None) as keyboard, \
                synthesis_executor() as executor:
            future = executor.submit(produce_chunks, chunks[start_chunk - 1:], audio, tts, voice, args,
                                     keyboard is None, progress, start_chunk - 1)
            completed = 0
            controls = PlaybackControls(chunk=start_chunk - 1, total_chunks=len(chunks),
                                        output_changed=output_changed, progress=progress)
            try:
                while True:
                    try:
                        wav = audio.get(timeout=0.05)
                    except Empty:
                        update_playback(keyboard, controls, audio, tts.sample_rate, buffering=True)
                        continue
                    if wav is None:
                        break
                    if progress is not None:
                        progress.chunk_index = completed + start_chunk - 1
                        progress.elapsed = 0.0
                        progress.in_silence = bool(completed and args.silence_duration)
                    if completed and args.silence_duration:
                        if output is not None:
                            output.write(np.zeros(round(tts.sample_rate * args.silence_duration), dtype=np.float32))
                        else:
                            play_audio(
                                np.zeros(round(tts.sample_rate * args.silence_duration), dtype=np.float32),
                                tts.sample_rate, keyboard, controls, audio, chunk=False,
                            )
                    if output is not None:
                        output.write(wav)
                    else:
                        progress.in_silence = False
                        progress.elapsed = 0.0
                        controls.chunk = completed + start_chunk
                        play_audio(wav, tts.sample_rate, keyboard, controls, audio)
                    completed += 1
                future.result()
            except KeyboardInterrupt:
                audio.cancelled = True
                raise
            except BaseException as error:
                audio.cancelled = True
                print(f"\nSpeech stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                raise
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
