"""Run with: cd src/speak && .venv/bin/python -B -m unittest -v test_playback"""

from contextlib import contextmanager, nullcontext, redirect_stderr
import io
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesise_audio as playback
import output_device


class AudioDriver:
    """Drive real playback callbacks deterministically without an audio device."""

    def __init__(self):
        self.blocks = []
        self.finished = False
        self.waits = 0

    def clear(self):
        self.finished = False

    @property
    def active(self):
        return not self.finished

    def abort(self):
        self.finished_callback()

    def set(self):
        self.finished = True

    def wait(self, timeout):
        self.waits += 1
        if self.waits > 1000:
            raise AssertionError("Playback did not complete")
        block = np.empty((4, 1), dtype=np.float32)
        try:
            self.callback(block, len(block), None, None)
        except (playback.sd.CallbackStop, playback.sd.CallbackAbort):
            self.finished_callback()
        self.blocks.append(block[:, 0].copy())
        return self.finished

    @contextmanager
    def stream(self, **kwargs):
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs["finished_callback"]
        self.finished = False
        yield self

    @contextmanager
    def installed(self):
        with patch.object(playback, "Event", return_value=self), \
                patch.object(playback.sd, "OutputStream", side_effect=self.stream):
            yield self


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.stderr = io.StringIO()
        self.output = redirect_stderr(self.stderr)
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_start_chunk_skips_earlier_synthesis_and_preserves_numbering(self):
        args = SimpleNamespace(max_chunk_length=300, start_chunk=2, voice="F5",
                               buffer=3, silence_duration=0)
        tts = SimpleNamespace(sample_rate=100, get_voice_style=Mock(return_value="voice"))
        played = []

        def produce(chunks, audio, *unused):
            self.assertEqual(chunks, ["second", "third"])
            audio.put(np.ones(10))
            audio.put(np.ones(10))
            audio.put(None)

        with patch.object(playback, "split_text", return_value=["first", "second", "third"]), \
                patch.object(playback, "produce_chunks", side_effect=produce), \
                patch.object(playback, "playback_keyboard", return_value=nullcontext(None)), \
                patch.object(playback, "play_audio", side_effect=lambda wav, rate, keyboard, controls, audio: played.append((controls.chunk, controls.total_chunks))):
            playback.synthesise_audio("text", None, tts, args)
        self.assertEqual(played, [(2, 3), (3, 3)])

    def test_start_chunk_rejects_out_of_range_before_synthesis(self):
        for start in (0, -1, 4):
            args = SimpleNamespace(max_chunk_length=300, start_chunk=start)
            with self.subTest(start=start), \
                    patch.object(playback, "split_text", return_value=["first", "second", "third"]):
                with self.assertRaisesRegex(ValueError, "Start chunk must be between 1 and 3"):
                    playback.synthesise_audio("text", None, Mock(), args)

    def test_output_change_reopens_stream_without_restarting_audio(self):
        changed = Event()
        controls = playback.PlaybackControls(output_changed=changed)
        driver = AudioDriver()
        original_wait = driver.wait
        opens = []
        original_stream = driver.stream

        def wait(timeout):
            result = original_wait(timeout)
            if driver.waits == 1:
                changed.set()
            return result

        @contextmanager
        def stream(**kwargs):
            opens.append(True)
            with original_stream(**kwargs) as output:
                yield output

        with driver.installed(), \
                patch.object(driver, "wait", side_effect=wait), \
                patch.object(playback.sd, "OutputStream", side_effect=stream), \
                patch.object(playback, "refresh_output_device") as refresh:
            playback.play_audio(np.arange(12, dtype=np.float32), 100, None,
                                controls, playback.AudioQueue())
        self.assertEqual(len(opens), 2)
        refresh.assert_called_once_with()
        np.testing.assert_array_equal(np.concatenate(driver.blocks)[:12], np.arange(12))

    def test_device_refresh_reinitializes_before_reading_default(self):
        calls = Mock()
        with patch.object(output_device.sd, "_terminate", calls.terminate), \
                patch.object(output_device.sd, "_initialize", calls.initialize), \
                patch.object(output_device.sd, "query_devices", calls.query):
            calls.query.return_value = {"name": "New output"}
            self.assertEqual(output_device.refresh_output_device(), "New output")
        self.assertEqual([call[0] for call in calls.mock_calls], ["terminate", "initialize", "query"])
        calls.query.assert_called_once_with(kind="output")

    def test_inactive_stream_raises_instead_of_waiting_forever(self):
        controls = playback.PlaybackControls(paused=True)
        driver = AudioDriver()
        with driver.installed(), \
                patch.object(driver, "wait", return_value=False), \
                patch.object(AudioDriver, "active", False):
            with self.assertRaisesRegex(RuntimeError, "became inactive"):
                playback.play_audio(np.ones(12), 100, None, controls, playback.AudioQueue())

    def test_output_change_preserves_pause_and_speed(self):
        changed = Event()
        changed.set()
        controls = playback.PlaybackControls(paused=True, speed=1.5, output_changed=changed)
        driver = AudioDriver()
        original_wait = driver.wait

        def wait(timeout):
            self.assertEqual(controls.speed, 1.5)
            if driver.waits == 1:
                self.assertTrue(controls.paused)
                controls.paused = False
            return original_wait(timeout)

        with driver.installed(), patch.object(driver, "wait", side_effect=wait), \
                patch.object(playback, "refresh_output_device") as refresh:
            playback.play_audio(np.ones(12), 24000, None, controls, playback.AudioQueue())
        refresh.assert_called_once_with()

    def test_buffer_counts_samples_not_chunks_and_excludes_sentinel(self):
        audio = playback.AudioQueue(maxsize=3)
        audio.put(np.ones(100))
        audio.put(np.ones(250))
        audio.put(None)
        self.assertEqual(audio.buffered_frames(), 350)
        audio.get()
        self.assertEqual(audio.buffered_frames(), 250)
        audio.get()
        self.assertEqual(audio.buffered_frames(), 0)

    def test_status_replaces_two_lines_with_chunk_number_and_time(self):
        audio = playback.AudioQueue()
        audio.put(np.ones(200))
        controls = playback.PlaybackControls(chunk=51, total_chunks=470, elapsed=12, duration=28)
        with patch.object(playback.select, "select", return_value=([], [], [])):
            playback.update_playback(7, controls, audio, 100, remaining=150)
            controls.paused = True
            playback.update_playback(7, controls, audio, 100, remaining=50)
        self.assertEqual(
            self.stderr.getvalue(),
            "\r\033[2K[###---------------------------]  11%\n"
            "\r\033[2K▶ Playing · Chunk 51/470 · 0:12 / 0:28 · 1.0× · Buffered: 3.5s"
            "\r\033[1A\033[2K[###---------------------------]  11%\n"
            "\r\033[2K⏸ Paused · Chunk 51/470 · 0:12 / 0:28 · 1.0× · Buffered: 2.5s",
        )

    def test_space_toggles_pause_while_buffering(self):
        audio = playback.AudioQueue()
        controls = playback.PlaybackControls()
        with patch.object(playback.select, "select", return_value=([7], [], [])), \
                patch.object(playback.os, "read", return_value=b" "):
            playback.update_playback(7, controls, audio, 100, buffering=True)
            self.assertTrue(controls.paused)
            playback.update_playback(7, controls, audio, 100, buffering=True)
            self.assertFalse(controls.paused)
        self.assertIn("⏸ Paused · Chunk 0/1 · 0:00 · 1.0× · Buffered: 0.0s", self.stderr.getvalue())
        self.assertIn("⏳ Buffering · Chunk 0/1 · 0:00 · 1.0× · Buffered: 0.0s", self.stderr.getvalue())
        self.assertNotIn(" / ", self.stderr.getvalue())

    def test_mid_chunk_pause_preserves_every_sample(self):
        driver = AudioDriver()
        samples = np.arange(1, 25, dtype=np.float32)
        # Initial status, pause after one block, stay paused, then resume.
        keys = iter([False, True, False, True])
        positions = []

        def ready(*args):
            positions.append(controls.elapsed)
            return ([7] if next(keys, False) else [], [], [])

        with driver.installed(), \
                patch.object(playback.select, "select", side_effect=ready), \
                patch.object(playback.os, "read", return_value=b" "):
            controls = playback.PlaybackControls()
            playback.play_audio(samples, 100, 7, controls, playback.AudioQueue())
        self.assertFalse(controls.paused)
        np.testing.assert_array_equal(driver.blocks[0], samples[:4])
        np.testing.assert_array_equal(driver.blocks[1], np.zeros(4))
        np.testing.assert_array_equal(driver.blocks[2], np.zeros(4))
        heard = np.concatenate(driver.blocks)
        np.testing.assert_array_equal(heard[heard != 0], samples)
        self.assertIn("⏸ Paused", self.stderr.getvalue())
        self.assertIn("▶ Playing", self.stderr.getvalue())
        self.assertEqual(positions[:4], [0.0, 0.04, 0.04, 0.04])
        self.assertAlmostEqual(controls.elapsed, 0.24)
        self.assertAlmostEqual(controls.duration, 0.24)
        self.assertNotIn("Synthesising", self.stderr.getvalue())

    def test_controls_work_before_delayed_synthesis_delivers_audio(self):
        release = Event()
        self.addCleanup(release.set)
        samples = np.arange(1, 13, dtype=np.float32)

        def synthesize(*args, **kwargs):
            if not release.wait(2):
                raise AssertionError("Controls did not release delayed synthesis")
            return samples, None

        tts = SimpleNamespace(
            sample_rate=100, get_voice_style=lambda voice: voice,
            synthesize=synthesize,
        )
        args = SimpleNamespace(
            max_chunk_length=300, voice="M1", buffer=3, lang=None,
            speed=1.05, steps=8, silence_duration=0,
        )
        states = []
        update = playback.update_playback

        def observe(*args, **kwargs):
            update(*args, **kwargs)
            if kwargs.get("buffering"):
                states.append(args[1].paused)
                if len(states) == 2:
                    release.set()

        @contextmanager
        def keyboard(enabled):
            self.assertTrue(enabled)
            yield 7

        # Both spaces arrive while the producer is blocked, before any audio.
        keys = iter([True, True])
        driver = AudioDriver()
        with driver.installed(), \
                patch.object(playback, "playback_keyboard", side_effect=keyboard), \
                patch.object(playback, "split_text", return_value=["one chunk"]), \
                patch.object(playback, "update_playback", side_effect=observe), \
                patch.object(playback.select, "select", side_effect=lambda *args: (
                    [7] if next(keys, False) else [], [], [])), \
                patch.object(playback.os, "read", return_value=b" "):
            playback.synthesise_audio("text", None, tts, args)
        self.assertEqual(states[:2], [True, False])
        heard = np.concatenate(driver.blocks)
        np.testing.assert_array_equal(heard[heard != 0], samples)
        self.assertIn("⏸ Paused · Chunk 0/1 · 0:00 · 1.0× · Buffered: 0.0s", self.stderr.getvalue())
        self.assertIn("⏳ Buffering · Chunk 0/1 · 0:00 · 1.0× · Buffered: 0.0s", self.stderr.getvalue())
        self.assertIn("▶ Playing · Chunk 1/1 · 0:00 / 0:00", self.stderr.getvalue())

    def test_noninteractive_status_does_not_poll_or_print(self):
        controls = playback.PlaybackControls(paused=True)
        with patch.object(playback.select, "select") as poll:
            playback.update_playback(None, controls, playback.AudioQueue(), 100)
            self.assertTrue(controls.paused)
        poll.assert_not_called()
        self.assertEqual(self.stderr.getvalue(), "")

    def test_arrow_keys_handle_split_sequences_repeats_and_limits(self):
        controls = playback.PlaybackControls(paused=True)
        audio = playback.AudioQueue()
        audio.put(np.ones(200))
        with patch.object(playback.select, "select", return_value=([7], [], [])), \
                patch.object(playback.os, "read", side_effect=[b"\x1b", b"[", b"A", b"\x1b[B", b"\x1bOA", b"\x1bOB"]):
            for _ in range(3):
                playback.update_playback(7, controls, audio, 100)
            self.assertEqual(controls.speed, 1.1)
            for expected in (1.0, 1.1, 1.0):
                playback.update_playback(7, controls, audio, 100)
                self.assertEqual(controls.speed, expected)
        self.assertTrue(controls.paused)
        for key, expected in ((b"\x1b[A" * 30, 2.0), (b"\x1b[B" * 30, 0.5)):
            with patch.object(playback.select, "select", return_value=([7], [], [])), \
                    patch.object(playback.os, "read", return_value=key):
                playback.update_playback(7, controls, audio, 100)
            self.assertEqual(controls.speed, expected)
        self.assertIn("0.5× · Buffered: 4.0s", self.stderr.getvalue())

    def test_playback_rate_persists_across_chunks(self):
        controls = playback.PlaybackControls(speed=1.5)
        source = Mock()
        source.read.side_effect = [np.ones(2), np.ones(2)]
        source.remaining_frames = 0
        processor = Mock()
        processor.__enter__ = Mock(return_value=source)
        processor.__exit__ = Mock(return_value=False)
        driver = AudioDriver()
        with driver.installed(), patch.object(playback, "PlaybackSpeed", return_value=processor):
            for _ in range(2):
                playback.play_audio(np.ones(2), 24000, None, controls, playback.AudioQueue())
        self.assertEqual(source.read.call_count, 2)
        for call in source.read.call_args_list:
            self.assertEqual(call.args, (4, 1.5))
        self.assertEqual(controls.speed, 1.5)

    def test_chunk_timer_resets_but_inter_chunk_silence_does_not_reset_it(self):
        driver = AudioDriver()
        controls = playback.PlaybackControls()
        audio = playback.AudioQueue()
        with driver.installed():
            playback.play_audio(np.ones(12), 4, None, controls, audio)
            self.assertEqual((controls.elapsed, controls.duration), (3.0, 3.0))
            playback.play_audio(np.zeros(4), 4, None, controls, audio, chunk=False)
            self.assertEqual((controls.elapsed, controls.duration), (3.0, 3.0))
            playback.play_audio(np.ones(8), 4, None, controls, audio)
            self.assertEqual((controls.elapsed, controls.duration), (2.0, 2.0))

    def test_bar_includes_within_chunk_progress_and_matches_chunk_boundaries(self):
        audio = playback.AudioQueue()
        cases = [(0, 0, None, "0%"), (1, 0, 10, "0%"),
                 (2, 0, 10, "25%"), (2, 5, 10, "38%"),
                 (2, 10, 10, "50%"), (3, 0, 10, "50%"),
                 (4, 10, 10, "100%")]
        for chunk, elapsed, duration, percent in cases:
            with self.subTest(chunk=chunk, elapsed=elapsed), \
                    patch.object(playback.select, "select", return_value=([], [], [])):
                self.stderr.seek(0)
                self.stderr.truncate()
                controls = playback.PlaybackControls(chunk=chunk, total_chunks=4, elapsed=elapsed, duration=duration)
                playback.update_playback(7, controls, audio, 100)
                bar, status = self.stderr.getvalue().split("\n")
                self.assertTrue(bar.endswith(" " + percent), bar)
                self.assertNotIn("/", bar)
                self.assertIn(f"Chunk {chunk}/4", status)

    def test_paused_bar_does_not_advance(self):
        controls = playback.PlaybackControls(chunk=2, total_chunks=4, elapsed=5, duration=10, paused=True)
        audio = playback.AudioQueue()
        with patch.object(playback.select, "select", return_value=([], [], [])):
            playback.update_playback(7, controls, audio, 100)
            audio.put(np.ones(100))
            playback.update_playback(7, controls, audio, 100)
        lines = self.stderr.getvalue().split("\n")
        self.assertTrue(lines[0].endswith("38%"))
        self.assertIn("[###########-------------------]  38%", lines[1])
        self.assertEqual(controls.elapsed, 5)

    def test_last_chunk_renders_100_percent_after_playback(self):
        driver = AudioDriver()
        controls = playback.PlaybackControls(chunk=1, total_chunks=1)
        with driver.installed(), patch.object(playback.select, "select", return_value=([], [], [])):
            playback.play_audio(np.ones(12), 4, 7, controls, playback.AudioQueue())
        self.assertIn("[##############################] 100%", self.stderr.getvalue())
        self.assertIn("Chunk 1/1 · 0:03 / 0:03", self.stderr.getvalue())

    def test_file_generation_progress_keeps_chunk_counts(self):
        playback.show_progress(51, 470)
        self.assertEqual(self.stderr.getvalue(), "\r[###---------------------------] 51/470  11%")

    def test_chunk_time_formats_minutes_without_hour_rollover(self):
        self.assertEqual(playback.chunk_time(12.9), "0:12")
        self.assertEqual(playback.chunk_time(61), "1:01")
        self.assertEqual(playback.chunk_time(3601), "60:01")

    def test_processor_failure_surfaces_outside_audio_callback(self):
        driver = AudioDriver()
        with driver.installed(), patch.object(playback.PlaybackSpeed, "read", side_effect=RuntimeError("processor failed")):
            with self.assertRaisesRegex(RuntimeError, "processor failed"):
                playback.play_audio(np.ones(8), 24000, None, playback.PlaybackControls(), playback.AudioQueue())

    def test_terminal_restored_after_completion_and_exception(self):
        keyboard = Mock()
        keyboard.isatty.return_value = True
        keyboard.fileno.return_value = 7
        original = ["original terminal attributes"]
        for fail in (False, True):
            with self.subTest(exception=fail), \
                    patch.object(playback.sys, "stdin", keyboard), \
                    patch.object(playback.termios, "tcgetattr", return_value=original), \
                    patch.object(playback.tty, "setcbreak") as cbreak, \
                    patch.object(playback.termios, "tcsetattr") as restore:
                def run():
                    with playback.playback_keyboard(True) as fd:
                        self.assertEqual(fd, 7)
                        if fail:
                            raise RuntimeError("playback failed")
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "playback failed"):
                        run()
                else:
                    run()
                cbreak.assert_called_once_with(7)
                restore.assert_called_once_with(7, playback.termios.TCSANOW, original)

    def test_disabled_or_noninteractive_keyboard_leaves_terminal_unchanged(self):
        keyboard = Mock()
        keyboard.isatty.return_value = False
        with patch.object(playback.sys, "stdin", keyboard), \
                patch.object(playback.termios, "tcgetattr") as attributes:
            for enabled in (False, True):
                with playback.playback_keyboard(enabled) as fd:
                    self.assertIsNone(fd)
        attributes.assert_not_called()
        self.assertEqual(self.stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
