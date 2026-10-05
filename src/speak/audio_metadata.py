"""Record and verify the exact synthesis input and parameters."""

import hashlib
from importlib.metadata import version as package_version
import json
import subprocess


def synthesis_metadata(text, conversion, args, bitrate):
    metadata = {
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_text_conversion": conversion,
        "source_hash_provenance": "synthesis-input",
        "tts_engine": "supertonic",
        "tts_engine_version": package_version("supertonic"),
        "tts_voice": args.voice,
        "tts_steps": str(args.steps),
        "tts_speed": str(args.speed),
        "tts_language": args.lang or "auto",
        "tts_max_chunk_length": str(args.max_chunk_length),
        "tts_silence_duration": str(args.silence_duration),
        "tts_parameters_provenance": "synthesis-invocation",
    }
    if bitrate is not None:
        metadata["opus_target_bitrate_kbps"] = str(bitrate)
    return metadata


def embed_metadata(source, destination, metadata, opus):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(source), "-map", "0:a:0", "-map_metadata", "-1", "-c:a", "copy",
    ]
    if opus:
        for key, value in metadata.items():
            command.extend(["-metadata:s:a:0", f"{key}={value}"])
    else:
        # RIFF/WAV does not support arbitrary tags. Its standard comment field
        # stores the same metadata as JSON, with verification after remuxing.
        command.extend(["-metadata", "comment=" + json.dumps(metadata, sort_keys=True)])
    subprocess.run([*command, str(destination)], check=True)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_name:stream_tags:format=duration:format_tags",
        "-of", "json", str(destination),
    ], check=True, capture_output=True, text=True)
    document = json.loads(result.stdout)
    streams = document.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected exactly one audio stream")
    if opus and streams[0]["codec_name"] != "opus":
        raise ValueError("Expected Opus audio")
    if not opus and not streams[0]["codec_name"].startswith("pcm_"):
        raise ValueError("Expected PCM WAV audio")
    if float(document.get("format", {}).get("duration", "0")) <= 0:
        raise ValueError("Encoded audio has no positive duration")
    if opus:
        tags = {key.lower(): value for key, value in streams[0].get("tags", {}).items()}
    else:
        fields = {key.lower(): value for key, value in document.get("format", {}).get("tags", {}).items()}
        tags = json.loads(fields.get("comment", "{}"))
    for key, value in metadata.items():
        if tags.get(key.lower()) != value:
            raise ValueError(f"Audio metadata did not preserve {key}")
