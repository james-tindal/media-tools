import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import sounddevice as sd

from supertonic import TTS
from supertonic.utils import chunk_text


def parse_args():
  parser = argparse.ArgumentParser(description='Read a text file aloud with Supertonic')

  parser.add_argument('file', type=Path)
  parser.add_argument('--voice', default='M1')
  parser.add_argument('--lang', default=None)
  parser.add_argument('--speed', type=float, default=1.05)
  parser.add_argument('--steps', type=int, default=8)
  parser.add_argument('--max-chunk-length', type=int, default=300)
  parser.add_argument('--silence-duration', type=float, default=0.3)
  parser.add_argument('--buffer', type=int, default=3, help='Number of synthesised chunks to buffer ahead')

  return parser.parse_args()

def synthesise(chunks, audio, tts, args):
  voice = tts.get_voice_style(args.voice)
  try:
    for chunk in chunks:
      wav, _ = tts.synthesize(
        chunk,
        voice_style=voice,
        lang=args.lang,
        speed=args.speed,
        total_steps=args.steps,
        silence_duration=0,
      )

      audio.put(wav.squeeze())

  finally:
    audio.put(None)

def main():
  args = parse_args()
  text = args.file.read_text(encoding='utf-8')
  tts = TTS()
  chunks = chunk_text(text, max_len=args.max_chunk_length)
  audio = Queue(maxsize=args.buffer)
  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(synthesise, chunks, audio, tts, args)

    while True:
      wav = audio.get()

      if wav is None:
        break

      sd.play(wav, tts.sample_rate)
      sd.wait()

      if args.silence_duration:
        sd.sleep(int(args.silence_duration * 1000))

    future.result()


if __name__ == '__main__': main()
