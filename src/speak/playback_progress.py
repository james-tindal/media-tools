"""Duration-weighted playback progress using Supertonic's own predictor."""

from dataclasses import dataclass
import math

from supertonic.pipeline import DEFAULT_MAX_CHUNK_LENGTH, DEFAULT_MAX_CHUNK_LENGTH_KO, UNKNOWN_LANGUAGE
from supertonic.utils import chunk_text


def predict_durations(chunks, tts, voice, args):
    lang = (args.lang or UNKNOWN_LANGUAGE) if tts.is_multilingual else None
    max_length = DEFAULT_MAX_CHUNK_LENGTH_KO if lang == "ko" else DEFAULT_MAX_CHUNK_LENGTH
    durations = []
    for chunk in chunks:
        duration = 0.0
        # Match TTS.synthesize's internal splitting and language handling.
        for text in chunk_text(chunk, max_length):
            ids, mask = tts.model.text_processor([text], lang)
            predicted = tts.model.dp_ort.run(
                None, {"text_ids": ids, "style_dp": voice.dp, "text_mask": mask},
            )[0]
            duration += float(predicted.reshape(-1)[0]) / args.speed
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Supertonic predicted an invalid chunk duration")
        durations.append(duration)
    return durations


@dataclass
class PlaybackProgress:
    durations: list[float]
    silence: float
    chunk_index: int = 0
    elapsed: float = 0.0
    in_silence: bool = False

    def fraction(self):
        total = sum(self.durations) + self.silence * (len(self.durations) - 1)
        preceding_gaps = max(0, self.chunk_index - int(self.in_silence))
        segment_duration = self.silence if self.in_silence else self.durations[self.chunk_index]
        played = (sum(self.durations[:self.chunk_index]) + preceding_gaps * self.silence
                  + min(segment_duration, max(0.0, self.elapsed)))
        return min(1.0, max(0.0, played / total))
