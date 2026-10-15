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
                patch("batch.prepare_jobs", side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(ROOT / "src/speak/speak.py"), run_name="__main__")
        self.assertEqual(result.exception.code, 130)

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
