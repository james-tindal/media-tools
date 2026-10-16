"""Ctrl+C must let the child finish cleanup before the CLI exits."""

import json
import os
from pathlib import Path
import runpy
import select
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class InterruptTests(unittest.TestCase):
    def test_python_cancellation_exits_130_without_traceback(self):
        with patch.object(sys, "argv", ["speak.py", "unused.md"]), \
                patch("batch.prepare_jobs", side_effect=KeyboardInterrupt), \
                patch("os._exit", side_effect=SystemExit(130)):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(ROOT / "src/speak/speak.py"), run_name="__main__")
        self.assertEqual(result.exception.code, 130)

    def test_ctrl_c_exits_silently_without_waiting_for_active_synthesis(self):
        script = '''
from contextlib import contextmanager, nullcontext
import runpy, sys, time
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import synthesise_audio as playback
started = Event()
calls = []
def synthesize(*args, **kwargs):
    calls.append(True)
    if len(calls) == 2:
        started.set()
        time.sleep(30)
    return np.ones(24), None
def play(*args, **kwargs):
    assert started.wait(2)
    print("READY", flush=True)
    time.sleep(30)
@contextmanager
def keyboard(enabled):
    try:
        yield None
    finally:
        print("TERMINAL RESTORED", flush=True)
tts = SimpleNamespace(sample_rate=24000, get_voice_style=lambda voice: voice, synthesize=synthesize)
def batch(jobs, args, model):
    playback.synthesise_audio("text", None, model, args)
sys.argv = ["speak.py", "unused.md"]
with patch("supertonic.TTS", return_value=tts), \\
     patch("batch.prepare_jobs", return_value=[]), \\
     patch("batch.run_batch", side_effect=batch), \\
     patch.object(playback, "split_text", return_value=["one", "two"]), \\
     patch.object(playback, "predict_durations", return_value=[1.0, 1.0]), \\
     patch.object(playback, "playback_keyboard", side_effect=keyboard), \\
     patch.object(playback, "mac_media_keys", side_effect=lambda *args: nullcontext(None)), \\
     patch.object(playback, "play_audio", side_effect=play):
    runpy.run_path("speak.py", run_name="__main__")
'''
        process = subprocess.Popen(
            [sys.executable, "-B", "-u", "-c", script], cwd=ROOT / "src/speak",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        output = b""
        try:
            deadline = time.monotonic() + 10
            while b"READY" not in output:
                self.assertLess(time.monotonic(), deadline, output.decode())
                if select.select([process.stdout], [], [], 0.1)[0]:
                    data = os.read(process.stdout.fileno(), 65536)
                    self.assertTrue(data, output.decode())
                    output += data
            os.kill(process.pid, signal.SIGINT)
            tail, _ = process.communicate(timeout=2)
            output += tail
            self.assertEqual(process.returncode, 130, output.decode())
            self.assertIn(b"TERMINAL RESTORED", output)
            self.assertNotIn(b"Speech stopped", output)
            self.assertNotIn(b"Traceback", output)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()

    def test_file_abort_confirmation(self):
        child = """import subprocess, sys, time
worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
print('CHILD READY', flush=True)
time.sleep(30)
"""
        script = """
import childProcess from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
const spawn = childProcess.spawn;
childProcess.spawn = (executable, args, options) => spawn(PYTHON, ['-u', '-c', CHILD], options);
syncBuiltinESMExports();
const {default: Speak} = await import('./src/speak/command.ts');
const command = new Speak([], {});
command.parse = async () => ({argv: ['unused.md'], flags: {
  output: 'unused.opus', voice: 'M1', speed: 1.05, steps: 8,
  'max-chunk-length': 300, 'silence-duration': 0.3, buffer: 3,
}});
command.exit = code => process.exit(code);
command.error = message => { throw new Error(message); };
await command.run();
""".replace("PYTHON", json.dumps(sys.executable)).replace("CHILD", json.dumps(child))
        for answer in ('y', 'ctrl-c', 'n', 'timeout'):
            with self.subTest(answer=answer):
                process = subprocess.Popen(
                    ['node', '--import', 'tsx', '--input-type=module', '-e', script],
                    cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                output = b''

                def wait_for(marker):
                    nonlocal output
                    deadline = time.monotonic() + 10
                    while marker not in output:
                        self.assertLess(time.monotonic(), deadline, output.decode())
                        if select.select([process.stdout], [], [], 0.1)[0]:
                            data = os.read(process.stdout.fileno(), 65536)
                            self.assertTrue(data, output.decode())
                            output += data

                try:
                    wait_for(b'CHILD READY')
                    os.killpg(process.pid, signal.SIGINT)
                    wait_for(b'Are you sure you want to abort?')
                    self.assertIsNone(process.poll())
                    if answer in ('y', 'n'):
                        process.stdin.write(answer.encode())
                        process.stdin.flush()
                    elif answer == 'ctrl-c':
                        os.killpg(process.pid, signal.SIGINT)
                    if answer in ('n', 'timeout'):
                        time.sleep(0.2 if answer == 'n' else 5.2)
                        self.assertIsNone(process.poll())
                        output = b''
                        os.killpg(process.pid, signal.SIGINT)
                        wait_for(b'Are you sure you want to abort?')
                        self.assertIsNone(process.poll())
                        os.killpg(process.pid, signal.SIGINT)
                    tail, _ = process.communicate(timeout=2)
                    output += tail
                    self.assertEqual(process.returncode, 130, output.decode())
                    self.assertNotIn(b'Traceback', output)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    process.stdin.close()
                    process.stdout.close()

    def test_cli_waits_for_interrupted_child_cleanup(self):
        child = """import sys, time
print('CHILD READY', flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    print('CLEANUP START', flush=True)
    time.sleep(0.5)
    print('CLEANUP DONE', flush=True)
    sys.exit(130)
"""
        script = """
import childProcess from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
const spawn = childProcess.spawn;
childProcess.spawn = () => spawn(PYTHON, ['-c', CHILD], {stdio: 'inherit'});
syncBuiltinESMExports();
const {default: Speak} = await import('./src/speak/command.ts');
const command = new Speak([], {});
command.parse = async () => ({argv: ['unused.md'], flags: {
  voice: 'M1', speed: 1.05, steps: 8, 'max-chunk-length': 300,
  'silence-duration': 0.3, buffer: 3,
}});
command.exit = code => process.exit(code);
command.error = message => { throw new Error(message); };
await command.run();
""".replace("PYTHON", json.dumps(sys.executable)).replace("CHILD", json.dumps(child))
        process = subprocess.Popen(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            self.assertTrue(select.select([process.stdout], [], [], 10)[0], "CLI did not start")
            self.assertEqual(process.stdout.readline().strip(), "CHILD READY")
            os.killpg(process.pid, signal.SIGINT)
            time.sleep(0.1)
            self.assertIsNone(process.poll(), "CLI exited before child cleanup")
            output, _ = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, output)
            self.assertIn("CLEANUP START", output)
            self.assertIn("CLEANUP DONE", output)
            self.assertNotIn("Traceback", output)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
