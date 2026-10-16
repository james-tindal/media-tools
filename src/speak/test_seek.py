"""Chunk navigation keys and their playback effects."""

from contextlib import nullcontext, redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import synthesise_audio as playback
from test_playback import AudioDriver


class SeekTests(unittest.TestCase):
    def setUp(self):
        output = redirect_stderr(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def key(self, controls, key, now=1.0):
        with patch.object(playback.select, "select", return_value=([7], [], [])), \
                patch.object(playback.os, "read", return_value=key), \
                patch.object(playback.time, "monotonic", return_value=now):
            playback.update_playback(7, controls, playback.AudioQueue(), 100)

    def settle_right(self, controls, now):
        with patch.object(playback.select, "select", return_value=([], [], [])), \
                patch.object(playback.time, "monotonic", return_value=now):
            playback.update_playback(7, controls, playback.AudioQueue(), 100)

    def test_left_restarts_and_double_left_selects_previous_within_400ms(self):
        controls = playback.PlaybackControls(chunk=3, total_chunks=5, paused=True, speed=1.5)
        self.key(controls, b"\x1b[D", 1)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.2)
        self.assertIsNone(controls.seek_chunk)
        self.key(controls, b"\x1bOD", 1.3)
        self.assertIsNone(controls.seek_chunk)
        self.assertEqual(controls.pending_left, 2)
        self.settle_right(controls, 1.46)
        self.assertEqual(controls.seek_chunk, 2)
        self.assertTrue(controls.paused)
        self.assertEqual(controls.speed, 1.5)

    def test_slow_left_presses_both_restart_current_chunk(self):
        controls = playback.PlaybackControls(chunk=3, total_chunks=5)
        self.key(controls, b"\x1b[D", 1)
        self.settle_right(controls, 1.39)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.41)
        self.assertEqual(controls.seek_chunk, 3)
        controls.seek_chunk = None
        self.key(controls, b"\x1b[D", 1.5)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.91)
        self.assertEqual(controls.seek_chunk, 3)

    def test_double_left_can_arrive_in_one_read_and_clamps_at_first(self):
        controls = playback.PlaybackControls(chunk=1, total_chunks=5)
        self.key(controls, b"\x1b[D\x1b[D")
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.16)
        self.assertEqual(controls.seek_chunk, 1)
        controls.chunk = 4
        controls.seek_chunk = None
        self.key(controls, b"\x1b[D\x1b[D", 2)
        self.settle_right(controls, 2.16)
        self.assertEqual(controls.seek_chunk, 3)

    def test_right_advances_and_does_nothing_at_last_chunk(self):
        controls = playback.PlaybackControls(chunk=2, total_chunks=4)
        self.key(controls, b"\x1bOC\x1b[C")
        self.assertIsNone(controls.seek_chunk)
        self.assertEqual(controls.pending_right, 4)
        self.settle_right(controls, 1.16)
        self.assertEqual(controls.seek_chunk, 4)
        controls.chunk = 4
        controls.seek_chunk = None
        self.key(controls, b"\x1b[C")
        self.assertIsNone(controls.seek_chunk)

    def test_right_while_initially_buffering_selects_second_chunk(self):
        controls = playback.PlaybackControls(chunk=0, total_chunks=4)
        self.key(controls, b"\x1b[C")
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.16)
        self.assertEqual(controls.seek_chunk, 2)

    def test_right_repeat_delays_seek_until_quiet_and_preserves_pause(self):
        controls = playback.PlaybackControls(chunk=2, total_chunks=10, paused=True)
        self.key(controls, b"\x1b[C", 1)
        self.key(controls, b"\x1b[C", 1.1)
        self.key(controls, b"\x1b[C", 1.2)
        self.assertEqual(controls.pending_right, 5)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.3)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.36)
        self.assertEqual(controls.seek_chunk, 5)
        self.assertIsNone(controls.pending_right)
        self.assertTrue(controls.paused)

    def test_right_updates_visible_counter_for_each_repeat_before_seeking(self):
        controls = playback.PlaybackControls(chunk=2, total_chunks=10)
        with patch("builtins.print") as output:
            for now, expected in [(1.0, 3), (1.05, 4), (1.1, 5)]:
                self.key(controls, b"\x1b[C", now)
                self.assertIn(f"Chunk {expected}/10", output.call_args.args[0])
                self.assertIsNone(controls.seek_chunk)
                self.assertEqual(controls.chunk, 2)
            self.settle_right(controls, 1.26)
            self.assertIn("Chunk 5/10", output.call_args.args[0])
            self.assertEqual(controls.seek_chunk, 5)

    def test_left_cancels_pending_right_target(self):
        controls = playback.PlaybackControls(chunk=2, total_chunks=10)
        self.key(controls, b"\x1b[C", 1)
        self.key(controls, b"\x1b[D", 1.1)
        self.assertIsNone(controls.pending_right)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.4)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.51)
        self.assertEqual(controls.seek_chunk, 2)

    def test_left_double_tap_ends_400ms_window_and_repeats_only_debounce(self):
        controls = playback.PlaybackControls(chunk=8, total_chunks=10, paused=True, speed=1.5)
        with patch("builtins.print") as output:
            for now, expected in [(1.0, 8), (1.1, 7), (1.22, 6), (1.34, 5), (1.46, 4)]:
                self.key(controls, b"\x1b[D", now)
                self.assertEqual(controls.pending_left, expected)
                self.assertIsNone(controls.seek_chunk)
                self.assertIn(f"Chunk {expected}/10", output.call_args.args[0])
            self.settle_right(controls, 1.60)
            self.assertIsNone(controls.seek_chunk)
            self.settle_right(controls, 1.62)
            self.assertEqual(controls.seek_chunk, 4)
        self.assertIsNone(controls.last_left)
        self.assertFalse(controls.left_repeating)
        self.assertTrue(controls.paused)
        self.assertEqual(controls.speed, 1.5)
        controls.chunk = 4
        controls.seek_chunk = None
        self.key(controls, b"\x1b[D", 1.65)
        self.assertEqual(controls.pending_left, 4)
        self.settle_right(controls, 1.82)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 2.06)
        self.assertEqual(controls.seek_chunk, 4)

    def test_right_cancels_left_double_tap_window(self):
        controls = playback.PlaybackControls(chunk=3, total_chunks=10)
        self.key(controls, b"\x1b[D", 1)
        self.key(controls, b"\x1b[C", 1.1)
        self.assertIsNone(controls.pending_left)
        self.assertIsNone(controls.last_left)
        self.settle_right(controls, 1.26)
        self.assertEqual(controls.seek_chunk, 4)
        self.settle_right(controls, 1.5)
        self.assertEqual(controls.seek_chunk, 4)

    def test_held_left_starts_only_final_target_producer(self):
        args = SimpleNamespace(max_chunk_length=300, voice="M1", buffer=1, start_chunk=8,
                               lang=None, speed=1.05, steps=8, silence_duration=0)
        tts = SimpleNamespace(sample_rate=100, get_voice_style=lambda voice: voice,
                              synthesize=lambda text, **kw: (np.full(4, int(text)), None))
        starts = []
        played = []
        clock = [1.0]
        repeats = [0]
        original_producer = playback.produce_chunks
        original_update = playback.update_playback

        def producer(chunks, *args):
            starts.append(int(chunks[0]))
            return original_producer(chunks, *args)

        def update(keyboard, controls, audio, rate, *args, **kwargs):
            clock[0] += 0.05
            if controls.pending_left is not None and repeats[0] < 6:
                controls.keys += b"\x1b[D"
                repeats[0] += 1
            with patch.object(playback.select, "select", return_value=([], [], [])), \
                    patch.object(playback.time, "monotonic", return_value=clock[0]):
                original_update(7, controls, audio, rate, *args, **kwargs)

        def play(wav, rate, keyboard, controls, audio, **kwargs):
            played.append(controls.chunk)
            if len(played) == 1:
                controls.keys = b"\x1b[D"
                with patch.object(playback.select, "select", return_value=([], [], [])), \
                        patch.object(playback.time, "monotonic", return_value=1.0):
                    original_update(7, controls, audio, rate)

        with patch.object(playback, "split_text", return_value=list(map(str, range(1, 11)))), \
                patch.object(playback, "predict_durations", return_value=[1.0] * 10), \
                patch.object(playback, "playback_keyboard", return_value=nullcontext(None)), \
                patch.object(playback, "output_device_changes", return_value=nullcontext(None)), \
                patch.object(playback, "produce_chunks", side_effect=producer), \
                patch.object(playback, "play_audio", side_effect=play), \
                patch.object(playback, "update_playback", side_effect=update):
            playback.synthesise_audio("text", None, tts, args)
        self.assertEqual(starts, [8, 2])
        self.assertEqual(played, [8] + list(range(2, 11)))

    def test_held_right_starts_only_final_target_producer(self):
        args = SimpleNamespace(max_chunk_length=300, voice="M1", buffer=1,
                               lang=None, speed=1.05, steps=8, silence_duration=0)
        tts = SimpleNamespace(sample_rate=100, get_voice_style=lambda voice: voice,
                              synthesize=lambda text, **kw: (np.full(4, int(text)), None))
        starts = []
        played = []
        clock = [1.0]
        repeats = [0]
        original_producer = playback.produce_chunks
        original_update = playback.update_playback

        def producer(chunks, *args):
            starts.append(int(chunks[0]))
            return original_producer(chunks, *args)

        def update(keyboard, controls, audio, rate, *args, **kwargs):
            clock[0] += 0.05
            if controls.pending_right is not None and repeats[0] < 5:
                controls.keys += b"\x1b[C"
                repeats[0] += 1
            with patch.object(playback.select, "select", return_value=([], [], [])), \
                    patch.object(playback.time, "monotonic", return_value=clock[0]):
                original_update(7, controls, audio, rate, *args, **kwargs)

        def play(wav, rate, keyboard, controls, audio, **kwargs):
            played.append(controls.chunk)
            if len(played) == 1:
                controls.keys = b"\x1b[C\x1b[C"
                with patch.object(playback.select, "select", return_value=([], [], [])), \
                        patch.object(playback.time, "monotonic", return_value=1.0):
                    original_update(7, controls, audio, rate)

        with patch.object(playback, "split_text", return_value=list(map(str, range(1, 11)))), \
                patch.object(playback, "predict_durations", return_value=[1.0] * 10), \
                patch.object(playback, "playback_keyboard", return_value=nullcontext(None)), \
                patch.object(playback, "output_device_changes", return_value=nullcontext(None)), \
                patch.object(playback, "produce_chunks", side_effect=producer), \
                patch.object(playback, "play_audio", side_effect=play), \
                patch.object(playback, "update_playback", side_effect=update):
            playback.synthesise_audio("text", None, tts, args)
        self.assertEqual(starts, [1, 8])
        self.assertEqual(played, [1, 8, 9, 10])

    def test_pending_right_freezes_audio_without_changing_pause_state(self):
        driver = AudioDriver()
        controls = playback.PlaybackControls(chunk=2, total_chunks=4)
        original_wait = driver.wait

        def wait(timeout):
            if driver.waits == 0:
                controls.pending_right = 3
            else:
                controls.pending_right = None
                controls.seek_chunk = 3
            return original_wait(timeout)

        with driver.installed(), patch.object(driver, "wait", side_effect=wait):
            playback.play_audio(np.ones(12), 100, None, controls, playback.AudioQueue())
        self.assertFalse(controls.paused)
        self.assertEqual(controls.elapsed, 0)
        np.testing.assert_array_equal(np.concatenate(driver.blocks), np.zeros(8))

    def test_left_sequence_can_be_split_across_reads(self):
        controls = playback.PlaybackControls(chunk=3, total_chunks=5)
        self.key(controls, b"\x1b")
        self.assertIsNone(controls.seek_chunk)
        self.key(controls, b"[D")
        self.assertEqual(controls.pending_left, 3)
        self.assertIsNone(controls.seek_chunk)
        self.settle_right(controls, 1.41)
        self.assertEqual(controls.seek_chunk, 3)

    def test_seeking_aborts_current_stream_without_changing_pause_state(self):
        driver = AudioDriver()
        controls = playback.PlaybackControls(chunk=2, total_chunks=3, paused=True)
        original_wait = driver.wait

        def wait(timeout):
            controls.seek_chunk = 2
            return original_wait(timeout)

        with driver.installed(), patch.object(driver, "wait", side_effect=wait):
            playback.play_audio(np.arange(12, dtype=np.float32), 100, None,
                                controls, playback.AudioQueue())
        self.assertTrue(controls.paused)
        self.assertEqual(controls.elapsed, 0)
        np.testing.assert_array_equal(driver.blocks[0], np.zeros(4))

    def test_restart_forward_and_backward_preserve_audio_order_and_pause(self):
        args = SimpleNamespace(max_chunk_length=300, voice="M1", buffer=1,
                               lang=None, speed=1.05, steps=8, silence_duration=0)
        tts = SimpleNamespace(sample_rate=100, get_voice_style=lambda voice: voice,
                              synthesize=Mock(side_effect=lambda text, **kw: (np.full(4, int(text)), None)))
        actions = iter([(1, 1), (1, 2), (2, 1), (1, None), (2, None), (3, None)])
        seen = []
        first_audio = []

        def play(wav, rate, keyboard, controls, audio, **kwargs):
            expected, target = next(actions)
            self.assertEqual(controls.chunk, expected)
            np.testing.assert_array_equal(wav, np.full(4, expected))
            if not seen:
                controls.paused = True
                controls.speed = 1.5
                first_audio.append(wav)
            else:
                self.assertTrue(controls.paused)
                self.assertEqual(controls.speed, 1.5)
            if len(seen) == 1:
                self.assertIs(wav, first_audio[0])
            seen.append(expected)
            self.assertEqual(controls.progress.chunk_index, expected - 1)
            self.assertEqual(controls.progress.elapsed, 0)
            controls.seek_chunk = target

        with patch.object(playback, "split_text", return_value=["1", "2", "3"]), \
                patch.object(playback, "predict_durations", return_value=[1.0, 1.0, 1.0]), \
                patch.object(playback, "playback_keyboard", return_value=nullcontext(None)), \
                patch.object(playback, "output_device_changes", return_value=nullcontext(None)), \
                patch.object(playback, "play_audio", side_effect=play):
            playback.synthesise_audio("text", None, tts, args)
        self.assertEqual(seen, [1, 1, 2, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
