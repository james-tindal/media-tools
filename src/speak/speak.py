import argparse
import sys
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import sounddevice as sd
import soundfile as sf

from supertonic import TTS
from supertonic.utils import chunk_text


def parse_format(value):
  parts = value.split(':')
  if parts[0] != 'opus' or len(parts) > 2:
    raise argparse.ArgumentTypeError('Use opus or opus:N, where N is the bitrate in kbps')
  try:
    bitrate = int(parts[1]) if len(parts) == 2 else 24
  except ValueError:
    raise argparse.ArgumentTypeError('Opus bitrate must be an integer in kbps') from None
  if not 6 <= bitrate <= 510:
    raise argparse.ArgumentTypeError('Opus bitrate must be between 6 and 510 kbps')
  return bitrate


def parse_args():
  parser = argparse.ArgumentParser(description='Read a text file aloud with Supertonic')

  parser.add_argument('file', type=Path)
  parser.add_argument('-o', '--output', type=Path, help='Write audio to this file instead of playing it')
  parser.add_argument('--format', type=parse_format, metavar='opus[:KBPS]',
                      help='Encode Opus with FFmpeg using a target bitrate in kbps (default: 24). Requires --output')
  parser.add_argument('--voice', default='M1')
  parser.add_argument('--lang', default=None)
  parser.add_argument('--speed', type=float, default=1.05)
  parser.add_argument('--steps', type=int, default=8)
  parser.add_argument('--max-chunk-length', type=int, default=300)
  parser.add_argument('--silence-duration', type=float, default=0.3)
  parser.add_argument('--buffer', type=int, default=3, help='Number of synthesised chunks to buffer ahead')

  args = parser.parse_args()
  if args.format is not None and args.output is None:
    parser.error('--format requires --output')
  return args

def show_progress(completed, total):
  width = 30
  ratio = completed / total if total else 1
  filled = round(width * ratio)
  bar = '#' * filled + '-' * (width - filled)
  sys.stderr.write(f'\rSynthesising [{bar}] {completed}/{total} {ratio:>4.0%}')
  sys.stderr.flush()


def synthesise(chunks, audio, tts, args):
  voice = tts.get_voice_style(args.voice)
  show_progress(0, len(chunks))
  try:
    for index, chunk in enumerate(chunks, start=1):
      wav, _ = tts.synthesize(
        chunk,
        voice_style=voice,
        lang=args.lang,
        speed=args.speed,
        total_steps=args.steps,
        silence_duration=0,
      )

      show_progress(index, len(chunks))
      audio.put(wav.squeeze())

  finally:
    sys.stderr.write('\n')
    audio.put(None)

def main():
  args = parse_args()
  text = args.file.read_text(encoding='utf-8')
  tts = TTS()
  chunks = chunk_text(text, max_len=args.max_chunk_length)
  audio = Queue(maxsize=args.buffer)
  output = None
  temporary = None

  try:
    output_path = args.output
    if args.format is not None:
      temporary = tempfile.TemporaryDirectory(prefix='media-tools-speak-')
      output_path = Path(temporary.name) / 'speech.wav'
    if output_path:
      output = sf.SoundFile(output_path, mode='w', samplerate=tts.sample_rate, channels=1)

    with ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(synthesise, chunks, audio, tts, args)
      completed = 0

      while True:
        wav = audio.get()

        if wav is None:
          break

        if output:
          if completed and args.silence_duration:
            silence = np.zeros(round(tts.sample_rate * args.silence_duration), dtype=np.float32)
            output.write(silence)
          output.write(wav)
        else:
          sd.play(wav, tts.sample_rate)
          sd.wait()

          if completed < len(chunks) - 1 and args.silence_duration:
            sd.sleep(int(args.silence_duration * 1000))

        completed += 1

      future.result()
    if output:
      output.close()
      output = None
    if args.format is not None:
      subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-i', str(output_path), '-c:a', 'libopus', '-b:a', f'{args.format}k',
        '-vbr', 'on', '-f', 'opus', str(args.output),
      ], check=True)
  finally:
    if output:
      output.close()
    if temporary:
      temporary.cleanup()


if __name__ == '__main__': main()
