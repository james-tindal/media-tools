"""Supertonic chunk synthesis, buffered playback, and Opus encoding."""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import subprocess
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
from supertonic.utils import chunk_text


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
    chunks = chunk_text(text, max_len=args.max_chunk_length)
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
