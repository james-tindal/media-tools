"""Run with: cd src/speak && .venv/bin/python -B -m unittest -v test_playback"""

from contextlib import contextmanager, redirect_stderr
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


class AudioDriver:
    """Drive real playback callbacks deterministically without an audio device."""

    def __init__(self):
        self.blocks = []
        self.finished = False
        self.waits = 0

    def set(self):
        self.finished = True

    def wait(self, timeout):
        self.waits += 1
        if self.waits > 100:
            raise AssertionError("Playback did not complete")
        block = np.empty((4, 1), dtype=np.float32)
        try:
            self.callback(block, len(block), None, None)
        except playback.sd.CallbackStop:
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

    def test_status_includes_current_chunk_and_replaces_one_line(self):
        audio = playback.AudioQueue()
        audio.put(np.ones(200))
        with patch.object(playback.select, "select", return_value=([], [], [])):
            playback.update_playback(7, False, audio, 100, remaining=150)
            playback.update_playback(7, True, audio, 100, remaining=50)
        self.assertEqual(
            self.stderr.getvalue(),
            "\r\033[2K▶ Playing · Buffered: 3.5s"
            "\r\033[2K⏸ Paused · Buffered: 2.5s",
        )

    def test_space_toggles_pause_while_buffering(self):
        audio = playback.AudioQueue()
        with patch.object(playback.select, "select", return_value=([7], [], [])), \
                patch.object(playback.os, "read", return_value=b" "):
            paused = playback.update_playback(7, False, audio, 100, buffering=True)
            self.assertTrue(paused)
            paused = playback.update_playback(7, paused, audio, 100, buffering=True)
            self.assertFalse(paused)
        self.assertEqual(
            self.stderr.getvalue(),
            "\r\033[2K⏸ Paused · Buffered: 0.0s"
            "\r\033[2K⏳ Buffering · Buffered: 0.0s",
        )

    def test_mid_chunk_pause_preserves_every_sample(self):
        driver = AudioDriver()
        samples = np.arange(1, 25, dtype=np.float32)
        # Initial status, pause after one block, stay paused, then resume.
        keys = iter([False, True, False, True])

        def ready(*args):
            return ([7] if next(keys, False) else [], [], [])

        with driver.installed(), \
                patch.object(playback.select, "select", side_effect=ready), \
                patch.object(playback.os, "read", return_value=b" "):
            paused = playback.play_audio(samples, 100, 7, False, playback.AudioQueue())
        self.assertFalse(paused)
        np.testing.assert_array_equal(driver.blocks[0], samples[:4])
        np.testing.assert_array_equal(driver.blocks[1], np.zeros(4))
        np.testing.assert_array_equal(driver.blocks[2], np.zeros(4))
        heard = np.concatenate(driver.blocks)
        np.testing.assert_array_equal(heard[heard != 0], samples)
        self.assertIn("⏸ Paused", self.stderr.getvalue())
        self.assertIn("▶ Playing", self.stderr.getvalue())
        self.assertNotIn("\n", self.stderr.getvalue())

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
            paused = update(*args, **kwargs)
            if kwargs.get("buffering"):
                states.append(paused)
                if len(states) == 2:
                    release.set()
            return paused

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
        np.testing.assert_array_equal(np.concatenate(driver.blocks), samples)
        self.assertIn("⏸ Paused · Buffered: 0.0s", self.stderr.getvalue())
        self.assertIn("⏳ Buffering · Buffered: 0.0s", self.stderr.getvalue())

    def test_noninteractive_status_does_not_poll_or_print(self):
        with patch.object(playback.select, "select") as poll:
            self.assertTrue(playback.update_playback(None, True, playback.AudioQueue(), 100))
        poll.assert_not_called()
        self.assertEqual(self.stderr.getvalue(), "")

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
