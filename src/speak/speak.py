import argparse
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from supertonic import TTS

from batch import prepare_jobs, run_batch
from synthesise_audio import initialize_worker, worker_description


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

  parser.add_argument('paths', nargs='+', type=Path, help='Markdown/text files or directories (recursive)')
  parser.add_argument('-o', '--output', type=Path, help='Write one input to a WAV or Opus file instead of playing it')
  parser.add_argument('--output-dir', type=Path, help='Write one Opus file per input to this directory')
  parser.add_argument('--overwrite', action='store_true', help='Replace existing audio after synthesis and verification succeed')
  parser.add_argument('--format', type=parse_format, metavar='opus[:KBPS]',
                      help='Encode Opus with FFmpeg using a target bitrate in kbps (default: 24). Requires --output')
  parser.add_argument('--voice', default='M1')
  parser.add_argument('--lang', default=None)
  parser.add_argument('--speed', type=float, default=1.05)
  parser.add_argument('--steps', type=int, default=8)
  parser.add_argument('--max-chunk-length', type=int, default=300)
  parser.add_argument('--silence-duration', type=float, default=0.3)
  parser.add_argument('--buffer', type=int, default=3, help='Number of synthesised chunks to buffer ahead')

  parser.add_argument('--workers', type=int, default=None, help='Synthesis workers (default: 2 for file output, 1 for playback)')
  parser.add_argument('--threads', type=int, default=2, help='Inference threads per worker (default: 2)')

  args = parser.parse_args()
  if args.workers is None:
    args.workers = 2 if args.output is not None or args.output_dir is not None else 1
  if args.workers < 1 or (args.threads is not None and args.threads < 1):
    parser.error('Workers and threads must be positive integers')
  if args.workers > 1 and args.output is None and args.output_dir is None:
    parser.error('Multiple workers require file output')
  if (not args.voice.strip() or args.steps < 1 or args.max_chunk_length < 1
      or args.buffer < 1 or not math.isfinite(args.speed) or args.speed <= 0
      or not math.isfinite(args.silence_duration) or args.silence_duration < 0):
    parser.error('Invalid synthesis parameters: voice must be nonempty; steps, speed, chunk length, and buffer must be positive; silence must be finite and nonnegative')
  return args

def main():
  args = parse_args()
  jobs = prepare_jobs(args)
  print('Loading Supertonic...', flush=True)
  if args.workers > 1:
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context('spawn'),
        initializer=initialize_worker, initargs=(args,),
    ) as pool:
      tts = SimpleNamespace(**pool.submit(worker_description).result())
      run_batch(jobs, args, tts, pool)
  else:
    tts = TTS(intra_op_num_threads=args.threads)
    # All input and destination checks happen before the model loads once.
    tts.get_voice_style(args.voice)
    run_batch(jobs, args, tts)


if __name__ == '__main__': main()
