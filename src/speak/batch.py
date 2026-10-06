"""Discover inputs, prepare destinations, and publish complete audio files."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

from audio_metadata import embed_metadata, synthesis_metadata
from markdown_conversion import CONVERSION, markdown_to_text
from synthesise_audio import IncompleteSynthesisError, encode_opus, synthesise_audio, synthesise_resumable


@dataclass(frozen=True)
class AudioJob:
    source: Path
    text: str
    conversion: str
    destination: Path | None
    bitrate: int | None


def prepare_jobs(args):
    """Validate the entire batch before initializing the expensive TTS model."""
    discovered = {}
    for input_path in args.paths:
        path = input_path.expanduser().resolve()
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in {".md", ".markdown", ".txt"}:
                    source = candidate.resolve()
                    discovered.setdefault(source, candidate.relative_to(path))
        elif path.is_file():
            discovered.setdefault(path, Path(path.name))
        else:
            raise FileNotFoundError(f"Expected a file or directory: {path}")
    if not discovered:
        raise ValueError("No Markdown or text files found")
    if args.output is not None and args.output_dir is not None:
        raise ValueError("Use --output or --output-dir, not both")
    if len(discovered) != 1 and args.output_dir is None:
        raise ValueError("Multiple inputs require --output-dir")
    if args.format is not None and args.output is None and args.output_dir is None:
        raise ValueError("--format requires --output or --output-dir")
    sources = set(discovered)
    destinations = set()
    jobs = []
    for source, relative in sorted(discovered.items()):
        destination = None
        bitrate = None
        if args.output_dir is not None:
            destination = args.output_dir.expanduser().absolute() / relative.with_suffix(".opus")
            bitrate = args.format if args.format is not None else 24
        elif args.output is not None:
            destination = args.output.expanduser().absolute()
            if destination.suffix.lower() == ".opus":
                bitrate = args.format if args.format is not None else 24
            elif destination.suffix.lower() != ".wav":
                raise ValueError("Output must have a .wav or .opus extension")
            elif args.format is not None:
                raise ValueError("--format opus requires a .opus output path")
        if destination is not None:
            resolved = destination.resolve()
            if resolved in sources:
                raise ValueError(f"Audio destination overlaps an input: {destination}")
            if resolved in destinations:
                raise ValueError(f"Multiple inputs map to the same destination: {destination}")
            destinations.add(resolved)
            if destination.exists() or destination.is_symlink():
                if not destination.is_file():
                    raise ValueError(f"Audio destination is not a file: {destination}")
                if not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite {destination}. Use --overwrite explicitly.")
            parent = destination.parent
            while not parent.exists():
                parent = parent.parent
            if not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
        raw = source.read_text(encoding="utf-8")
        markdown = source.suffix.lower() in {".md", ".markdown"}
        text = markdown_to_text(raw) if markdown else raw
        if not text.strip():
            raise ValueError(f"No spoken text in {source}")
        jobs.append(AudioJob(source, text, CONVERSION if markdown else "plain-text-utf8-v1", destination, bitrate))
    if any(job.destination is not None for job in jobs):
        for executable in ("ffmpeg", "ffprobe"):
            if shutil.which(executable) is None:
                raise FileNotFoundError(f"Missing system dependency: {executable}. Install FFmpeg before generating audio.")
    return jobs


def process_job(job, args, tts):
    if job.destination is None:
        synthesise_audio(job.text, None, tts, args)
        return
    output = job.destination
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem. Failed runs leave final audio alone.
    with tempfile.TemporaryDirectory(prefix=".media-tools-speak-", dir=output.parent) as directory:
        temporary = Path(directory)
        wav_path = temporary / "speech.wav"
        synthesise_resumable(
            job.text, wav_path, tts, args,
            output.parent / f".{output.name}.chunks", job.conversion,
        )
        raw = wav_path
        opus = job.bitrate is not None
        if opus:
            raw = temporary / "speech.opus"
            encode_opus(wav_path, raw, job.bitrate)
        tagged = temporary / ("tagged.opus" if opus else "tagged.wav")
        metadata = synthesis_metadata(job.text, job.conversion, args, job.bitrate)
        embed_metadata(raw, tagged, metadata, opus)
        if args.overwrite:
            os.replace(tagged, output)
        else:
            # Atomic create-only publication: a concurrent writer cannot be
            # overwritten between preflight and publication.
            os.link(tagged, output)
    print(f"Saved {output} (SHA-256 {metadata['source_text_sha256']})", flush=True)


def run_batch(jobs, args, tts):
    incomplete = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.source}", flush=True)
        try:
            process_job(job, args, tts)
        except IncompleteSynthesisError as error:
            incomplete.append(str(job.source))
            print(f"Incomplete input {job.source}: {error}", flush=True)
    if incomplete:
        raise IncompleteSynthesisError(
            "Incomplete inputs:\n" + "\n".join(incomplete)
        )
