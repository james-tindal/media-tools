"""Native DSP tests. Require the installed Rubber Band library, not speakers."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playback_speed import PlaybackSpeed, rubberband


SAMPLE_RATE = 24000


def tone(seconds=3):
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32)


def frequency(samples):
    spectrum = abs(np.fft.rfft(samples * np.hanning(len(samples))))
    peak = int(np.argmax(spectrum))
    left, center, right = np.log(spectrum[peak - 1:peak + 2] + 1e-20)
    offset = 0.5 * (left - right) / (left - 2 * center + right)
    return (peak + offset) * SAMPLE_RATE / len(samples)


class PlaybackSpeedTests(unittest.TestCase):
    def render(self, samples, speed):
        blocks = []
        with PlaybackSpeed(samples, SAMPLE_RATE) as source:
            for _ in range(2000):
                block = source.read(240, speed)
                blocks.append(block.copy())
                if len(block) < 240:
                    break
            else:
                self.fail("Processor did not finish")
        output = np.concatenate(blocks)
        self.assertTrue(np.isfinite(output).all())
        return output

    def test_unity_playback_is_sample_exact_without_native_library(self):
        samples = tone(0.1)
        with patch("playback_speed.rubberband", side_effect=AssertionError("Unneeded native dependency")):
            np.testing.assert_array_equal(self.render(samples, 1.0), samples)

    def test_speed_changes_duration_without_changing_pitch(self):
        samples = tone()
        for speed in (0.5, 1.5, 2.0):
            with self.subTest(speed=speed):
                output = self.render(samples, speed)
                self.assertAlmostEqual(len(output) / SAMPLE_RATE, 3 / speed, delta=0.1)
                self.assertAlmostEqual(frequency(output[2400:-2400]), 440, delta=2)

    def test_live_changes_preserve_pitch_and_apply_before_chunk_end(self):
        blocks = []
        count = 0
        observed = set()
        with PlaybackSpeed(tone(), SAMPLE_RATE) as source:
            for _ in range(2000):
                elapsed = count / SAMPLE_RATE
                speed = 1.0 if elapsed < 0.5 else 1.5 if elapsed < 1.2 else 0.75
                observed.add(speed)
                block = source.read(240, speed)
                blocks.append(block.copy())
                count += len(block)
                if len(block) < 240:
                    break
            else:
                self.fail("Processor did not finish after live changes")
        self.assertEqual(observed, {1.0, 1.5, 0.75})
        output = np.concatenate(blocks)
        expected = 0.5 + 0.7 + (3 - 0.5 - 0.7 * 1.5) / 0.75
        self.assertAlmostEqual(len(output) / SAMPLE_RATE, expected, delta=0.15)
        # Measure both sides of the changes, not just the whole recording.
        for seconds in (0.1, 0.7, 1.5, 2.2):
            start = round(seconds * SAMPLE_RATE)
            self.assertAlmostEqual(frequency(output[start:start + 4800]), 440, delta=3)

    def test_native_processor_is_released_on_exception(self):
        library = rubberband()
        with patch.object(library, "rubberband_delete", wraps=library.rubberband_delete) as delete:
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with PlaybackSpeed(tone(), SAMPLE_RATE) as source:
                    source.read(240, 1.5)
                    raise RuntimeError("consumer failed")
            delete.assert_called_once()

    def test_missing_library_is_an_explicit_error(self):
        rubberband.cache_clear()
        try:
            with patch("playback_speed.find_library", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "brew install rubberband"):
                    with PlaybackSpeed(tone(), SAMPLE_RATE) as source:
                        source.read(240, 1.5)
        finally:
            rubberband.cache_clear()


if __name__ == "__main__":
    unittest.main()
