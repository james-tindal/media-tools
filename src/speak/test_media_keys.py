"""macOS media commands use the normal playback controls without a focused TTY."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import media_keys
import synthesise_audio as playback


class MediaKeyTests(unittest.TestCase):
    def setUp(self):
        self.handlers = {}
        self.commands = {}
        self.tokens = {}
        center = SimpleNamespace()
        for name in ("playCommand", "pauseCommand", "togglePlayPauseCommand",
                     "previousTrackCommand", "nextTrackCommand"):
            command = Mock()
            command.addTargetWithHandler_.side_effect = lambda handler, name=name: self.register(name, handler)
            self.commands[name] = command
            setattr(center, name, Mock(return_value=command))
        self.info = Mock()
        self.run_loop = Mock()
        self.framework = SimpleNamespace(
            MPRemoteCommandCenter=SimpleNamespace(sharedCommandCenter=lambda: center),
            MPNowPlayingInfoCenter=SimpleNamespace(defaultCenter=lambda: self.info),
            MPMediaItemPropertyTitle="title", MPNowPlayingPlaybackStatePlaying=1,
            MPNowPlayingPlaybackStatePaused=2, MPNowPlayingPlaybackStateStopped=3,
            MPRemoteCommandHandlerStatusSuccess=0,
        )
        modules = patch.dict(sys.modules, {
            "MediaPlayer": self.framework,
            "AppKit": SimpleNamespace(NSApplication=Mock(), NSApplicationActivationPolicyAccessory=1),
            "CoreFoundation": SimpleNamespace(CFRunLoopRunInMode=self.run_loop,
                                               kCFRunLoopDefaultMode="default"),
        })
        modules.start()
        self.addCleanup(modules.stop)
        platform = patch.object(media_keys.sys, "platform", "darwin")
        platform.start()
        self.addCleanup(platform.stop)
        self.controls = playback.PlaybackControls(chunk=3, total_chunks=6, speed=1.5)

    def register(self, name, handler):
        self.handlers[name] = handler
        self.tokens[name] = object()
        return self.tokens[name]

    def poll(self, now):
        with patch.object(playback.time, "monotonic", return_value=now), \
                patch.object(playback.select, "select") as select:
            playback.update_playback(None, self.controls, playback.AudioQueue(), 100)
        select.assert_not_called()

    def test_registers_title_and_commands_then_cleans_up(self):
        with media_keys.mac_media_keys(self.controls, "Friendship") as keys:
            self.info.setNowPlayingInfo_.assert_called_once_with({"title": "Friendship"})
            self.info.setPlaybackState_.assert_called_with(1)
            self.assertEqual(len(keys.targets), 5)
            for command in self.commands.values():
                command.setEnabled_.assert_called_once_with(True)
        for name, command in self.commands.items():
            command.removeTarget_.assert_called_once_with(self.tokens[name])
        self.info.setPlaybackState_.assert_called_with(3)
        self.info.setNowPlayingInfo_.assert_called_with(None)

    def test_global_play_pause_and_toggle_work_without_terminal(self):
        with media_keys.mac_media_keys(self.controls, "Friendship") as keys:
            self.controls.media_keys = keys
            self.run_loop.side_effect = lambda *args: self.handlers["togglePlayPauseCommand"](None)
            self.poll(1)
            self.assertTrue(self.controls.paused)
            self.info.setPlaybackState_.assert_called_with(2)
            self.run_loop.side_effect = None
            self.assertEqual(self.handlers["playCommand"](None), 0)
            self.poll(1.1)
            self.assertFalse(self.controls.paused)
            self.info.setPlaybackState_.assert_called_with(1)
            self.handlers["pauseCommand"](None)
            self.poll(1.2)
            self.assertTrue(self.controls.paused)
            self.info.setPlaybackState_.assert_called_with(2)
            self.assertEqual(self.controls.speed, 1.5)
            self.run_loop.assert_called_with("default", 0.001, True)

    def test_global_previous_uses_double_tap_then_debounce_without_terminal(self):
        self.controls.paused = True
        with media_keys.mac_media_keys(self.controls, "Friendship") as keys:
            self.controls.media_keys = keys
            self.handlers["previousTrackCommand"](None)
            self.poll(1)
            self.assertEqual(self.controls.pending_left, 3)
            self.assertIsNone(self.controls.seek_chunk)
            self.handlers["previousTrackCommand"](None)
            self.poll(1.3)
            self.assertEqual(self.controls.pending_left, 2)
            self.poll(1.46)
            self.assertEqual(self.controls.seek_chunk, 2)
            self.assertTrue(self.controls.paused)

    def test_global_next_repeats_accumulate_without_terminal(self):
        with media_keys.mac_media_keys(self.controls, "Friendship") as keys:
            self.controls.media_keys = keys
            self.handlers["nextTrackCommand"](None)
            self.poll(1)
            self.handlers["nextTrackCommand"](None)
            self.poll(1.1)
            self.assertIsNone(self.controls.seek_chunk)
            self.assertEqual(self.controls.pending_right, 5)
            self.poll(1.26)
            self.assertEqual(self.controls.seek_chunk, 5)

    def test_disabled_and_non_macos_do_not_load_native_support(self):
        with patch.object(media_keys, "MacMediaKeys") as native:
            with media_keys.mac_media_keys(self.controls, "unused", enabled=False) as keys:
                self.assertIsNone(keys)
            with patch.object(media_keys.sys, "platform", "linux"):
                with media_keys.mac_media_keys(self.controls, "unused") as keys:
                    self.assertIsNone(keys)
        native.assert_not_called()

    def test_exception_still_unregisters_commands(self):
        with self.assertRaisesRegex(RuntimeError, "playback failed"):
            with media_keys.mac_media_keys(self.controls, "Friendship"):
                raise RuntimeError("playback failed")
        for command in self.commands.values():
            command.removeTarget_.assert_called_once()
        self.info.setNowPlayingInfo_.assert_called_with(None)


if __name__ == "__main__":
    unittest.main()
