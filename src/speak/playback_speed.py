"""Pitch-preserving playback through the installed Rubber Band library."""

import ctypes as C
from ctypes.util import find_library
from functools import lru_cache

import numpy as np


_FLOAT = C.POINTER(C.c_float)
_CHANNELS = C.POINTER(_FLOAT)


@lru_cache(maxsize=1)
def rubberband():
    path = find_library("rubberband")
    if path is None:
        raise RuntimeError("Playback speed requires the Rubber Band library. Install it with: brew install rubberband")
    library = C.CDLL(path)
    signatures = {
        "new": ([C.c_uint, C.c_uint, C.c_int, C.c_double, C.c_double], C.c_void_p),
        "delete": ([C.c_void_p], None),
        "set_time_ratio": ([C.c_void_p, C.c_double], None),
        "get_preferred_start_pad": ([C.c_void_p], C.c_uint),
        "get_start_delay": ([C.c_void_p], C.c_uint),
        "set_max_process_size": ([C.c_void_p, C.c_uint], None),
        "process": ([C.c_void_p, _CHANNELS, C.c_uint, C.c_int], None),
        "available": ([C.c_void_p], C.c_int),
        "retrieve": ([C.c_void_p, _CHANNELS, C.c_uint], C.c_uint),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, "rubberband_" + name)
        function.argtypes = arguments
        function.restype = result
    return library


class PlaybackSpeed:
    def __init__(self, samples, sample_rate):
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        self.sample_rate = sample_rate
        self.position = 0
        self.played = 0.0
        self.state = None
        self.final = False
        self.delay = 0
        self.speed = 1.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.state is not None:
            self.library.rubberband_delete(self.state)
            self.state = None

    @property
    def remaining_frames(self):
        # Native tempo changes have a short transition, so this is an estimate.
        return max(0, len(self.samples) - self.played)

    def process(self, samples, final=False):
        self.library.rubberband_process(
            self.state, (_FLOAT * 1)(samples.ctypes.data_as(_FLOAT)), len(samples), int(final),
        )

    def start(self, speed):
        self.library = rubberband()
        # R3 preserves pitch; use real-time, single-threaded processing and a short window.
        self.state = self.library.rubberband_new(
            self.sample_rate, 1, 0x1 | 0x10000 | 0x100000 | 0x20000000, 1 / speed, 1.0,
        )
        if not self.state:
            self.state = None
            raise MemoryError("Rubber Band could not allocate a playback processor")
        self.speed = speed
        self.library.rubberband_set_max_process_size(self.state, 512)
        self.delay = self.library.rubberband_get_start_delay(self.state)
        padding = self.library.rubberband_get_preferred_start_pad(self.state)
        for offset in range(0, padding, 512):
            self.process(np.zeros(min(512, padding - offset), dtype=np.float32))

    def read(self, frames, speed):
        if self.state is None and (speed == 1.0 or self.position == len(self.samples)):
            result = self.samples[self.position:self.position + frames]
            self.position += len(result)
            self.played += len(result)
            return result
        if self.state is None:
            self.start(speed)
        elif speed != self.speed:
            self.library.rubberband_set_time_ratio(self.state, 1 / speed)
            self.speed = speed
        output = np.empty(frames, dtype=np.float32)
        written = 0
        while written < frames:
            available = self.library.rubberband_available(self.state)
            if available < 0:
                break
            if available:
                count = min(available, self.delay or frames - written)
                target = np.empty(count, dtype=np.float32) if self.delay else output[written:written + count]
                count = self.library.rubberband_retrieve(
                    self.state, (_FLOAT * 1)(target.ctypes.data_as(_FLOAT)), count,
                )
                if not count:
                    raise RuntimeError("Rubber Band did not return its available audio")
                if self.delay:
                    self.delay -= count
                else:
                    written += count
                continue
            if self.final:
                raise RuntimeError("Rubber Band did not finish the audio stream")
            samples = self.samples[self.position:self.position + 512]
            self.position += len(samples)
            self.final = self.position == len(self.samples)
            self.process(samples, self.final)
        self.played += written * speed
        return output[:written]
