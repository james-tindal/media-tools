"""Register playback with macOS Now Playing, independent of terminal focus."""

from contextlib import contextmanager
import sys


class MacMediaKeys:
    def __init__(self, controls):
        import MediaPlayer
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode

        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.framework = MediaPlayer
        self.run_loop = CFRunLoopRunInMode
        self.run_loop_mode = kCFRunLoopDefaultMode
        self.controls = controls
        self.center = MediaPlayer.MPRemoteCommandCenter.sharedCommandCenter()
        self.info = MediaPlayer.MPNowPlayingInfoCenter.defaultCenter()
        self.targets = []
        self.paused = None

    def start(self, title):
        for name, action in (
            ("playCommand", "play"), ("pauseCommand", "pause"),
            ("togglePlayPauseCommand", "toggle"),
            ("previousTrackCommand", "previous"), ("nextTrackCommand", "next"),
        ):
            command = getattr(self.center, name)()
            handler = lambda event, action=action: self.handle(action)
            target = command.addTargetWithHandler_(handler)
            self.targets.append((command, target))
            command.setEnabled_(True)
        self.info.setNowPlayingInfo_({self.framework.MPMediaItemPropertyTitle: title})
        self.update_state()

    def handle(self, action):
        if action == "play":
            self.controls.paused = False
        elif action == "pause":
            self.controls.paused = True
        elif action == "toggle":
            self.controls.paused = not self.controls.paused
        elif action == "previous":
            self.controls.keys += b"\x1b[D"
        elif action == "next":
            self.controls.keys += b"\x1b[C"
        return self.framework.MPRemoteCommandHandlerStatusSuccess

    def poll(self):
        self.run_loop(self.run_loop_mode, 0.001, True)

    def update_state(self):
        if self.paused != self.controls.paused:
            state = (self.framework.MPNowPlayingPlaybackStatePaused if self.controls.paused
                     else self.framework.MPNowPlayingPlaybackStatePlaying)
            self.info.setPlaybackState_(state)
            self.paused = self.controls.paused

    def close(self):
        for command, target in self.targets:
            command.removeTarget_(target)
        self.targets.clear()
        self.info.setPlaybackState_(self.framework.MPNowPlayingPlaybackStateStopped)
        self.info.setNowPlayingInfo_(None)


@contextmanager
def mac_media_keys(controls, title, enabled=True):
    if not enabled or sys.platform != "darwin":
        yield None
        return
    keys = MacMediaKeys(controls)
    try:
        keys.start(title)
        yield keys
    finally:
        keys.close()
