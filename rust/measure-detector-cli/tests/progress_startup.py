"""Regression for a PDF metadata query blocking before any visible progress.

Run after cargo build: python3 rust/measure-detector-cli/tests/progress_startup.py
The fake-pdfinfo cases stall at the real CLI boundary before ONNX loading.
The remaining cases exercise actual embedded inference and native loading errors.
"""
import json
import os
import pty
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[1]
BINARY = CLI / 'target/debug/measure-detector-v2'
PDF = CLI.parents[1] / 'tests/fixtures/two-pages.pdf'


class StartupProgressTests(unittest.TestCase):
    def capture_blocked_metadata(self, quiet=False):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marker = directory / 'entered'
            fake = directory / 'pdfinfo'
            fake.write_text('#!/bin/sh\nprintf ready > ' + shlex.quote(str(marker)) + '\nsleep 30\n')
            fake.chmod(0o755)
            master, slave = pty.openpty()
            command = [str(BINARY), '--format', 'xfdf', str(PDF)]
            if quiet:
                command.append('--no-progress')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=slave,
                                       start_new_session=True, env={**os.environ,
                                       'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                                       'TERM': 'xterm'})
            os.close(slave)
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail('CLI exited before invoking pdfinfo')
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), 'CLI did not invoke pdfinfo')
                output = b''
                deadline = time.monotonic() + 0.3
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([master], [], [], 0.05)
                    if ready:
                        output += os.read(master, 65536)
                return output.decode()
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate()
                os.close(master)

    def test_embedded_runtime_and_model_process_pdf(self):
        result = subprocess.run([str(BINARY), '--no-progress', str(PDF)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([page['page'] for page in json.loads(result.stdout)['results']], [1, 2])

    def test_invalid_cached_library_fails_instead_of_deadlocking(self):
        target, name = ('macos-aarch64', 'libonnxruntime.1.27.0.dylib') if sys.platform == 'darwin' else (
            'linux-x86_64', 'libonnxruntime.so.1.27.0')
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / 'measure-detector-v2-cli/onnxruntime-1.27.0' / target / name
            cached.parent.mkdir(parents=True)
            # Same length bypasses extraction, exercising a native loading error.
            with cached.open('wb') as file:
                file.truncate((CLI / 'assets' / name).stat().st_size)
            result = subprocess.run([str(BINARY), '--no-progress', str(PDF)],
                                    env={**os.environ, 'TMPDIR': directory},
                                    capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('failed to load embedded ONNX Runtime', result.stderr)
            self.assertEqual(result.stdout, '')

    def test_geometry_stage_visible_while_pdfinfo_is_blocked(self):
        self.assertIn('Reading PDF geometry', self.capture_blocked_metadata())

    def test_no_progress_stays_quiet_during_startup(self):
        self.assertEqual(self.capture_blocked_metadata(quiet=True), '')


if __name__ == '__main__':
    unittest.main()
