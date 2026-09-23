"""Tests for ``python -m pqc_auth.transport serve|request``.

Real separate processes over loopback, no root needed. The CLI always uses
the real OqsDilithiumSigner, so these skip when liboqs isn't installed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pqc_auth.transport import main

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import oqs  # noqa: F401  # type: ignore[import-not-found]

    HAVE_OQS = True
except (ImportError, RuntimeError, SystemExit):
    HAVE_OQS = False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ArgumentValidationTests(unittest.TestCase):
    def _exit_code(self, argv):
        with self.assertRaises(SystemExit) as ctx, open(os.devnull, "w") as devnull:
            stderr, sys.stderr = sys.stderr, devnull
            try:
                main(argv)
            finally:
                sys.stderr = stderr
        return ctx.exception.code

    def test_request_needs_a_trust_mode(self):
        self.assertEqual(self._exit_code(["request", "--port", "1", "--slice-type", "URLLC"]), 2)

    def test_request_rejects_both_trust_modes(self):
        argv = ["request", "--port", "1", "--slice-type", "URLLC", "--expected-pubkey-file", "k",
                "--trust-store", "t", "--server-id", "s"]
        self.assertEqual(self._exit_code(argv), 2)

    def test_trust_store_requires_server_id(self):
        self.assertEqual(self._exit_code(["request", "--port", "1", "--slice-type", "URLLC", "--trust-store", "t"]), 2)

    def test_then_target_must_be_host_port(self):
        argv = ["request", "--port", "1", "--slice-type", "URLLC", "--expected-pubkey-file", "k", "--then", "nohost"]
        self.assertEqual(self._exit_code(argv), 2)


@unittest.skipUnless(HAVE_OQS, "requires liboqs-python (the CLI always uses the real OqsDilithiumSigner)")
class LoopbackProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.port = _free_port()
        cls.env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "pqc_auth.transport", "serve", "--port", str(cls.port),
             "--key-path", str(cls.dir / "keys"), "--audit-log", str(cls.dir / "served.jsonl")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT, env=cls.env,
        )
        for line in cls.server.stdout:
            if line.startswith("READY"):
                cls.ready = json.loads(line.split(" ", 1)[1])
                break
        else:
            raise RuntimeError("server exited before READY")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=10)
        cls.tmp.cleanup()

    def _request(self, *extra):
        result = subprocess.run(
            [sys.executable, "-m", "pqc_auth.transport", "request", "--port", str(self.port), *extra],
            capture_output=True, text=True, cwd=REPO_ROOT, env=self.env, timeout=60,
        )
        return result.returncode, [json.loads(l) for l in result.stdout.splitlines() if l.startswith("{")]

    def test_server_binds_loopback_by_default_and_persists_its_key(self):
        self.assertEqual(self.ready["host"], "127.0.0.1")
        self.assertTrue((self.dir / "keys" / "public_key.bin").exists())

    def test_explicit_pin_from_file_is_trusted_and_audit_log_verifies(self):
        log = self.dir / "pin_audit.jsonl"
        code, records = self._request("--slice-type", "URLLC", "--expected-pubkey-file", str(self.dir / "keys" / "public_key.bin"),
                                      "--audit-log", str(log), "--now", "1000", "--now-step", "1000", "--count", "2")
        self.assertEqual(code, 0)
        self.assertEqual([r["result"]["trusted"] for r in records], [True, True])
        self.assertTrue(all(r["rtt_ms"] > 0 for r in records))
        audit = subprocess.run([sys.executable, "-m", "pqc_auth.audit_verify", str(log)],
                               capture_output=True, text=True, cwd=REPO_ROOT, env=self.env)
        self.assertEqual(audit.returncode, 0, audit.stdout)

    def test_wrong_pinned_key_is_rejected(self):
        wrong = self.dir / "wrong.bin"
        wrong.write_bytes(b"\x00" * 32)
        code, (record,) = self._request("--slice-type", "eMBB", "--expected-pubkey-file", str(wrong), "--now", "5000")
        self.assertEqual(code, 0)
        self.assertFalse(record["result"]["trusted"])
        self.assertTrue(record["result"]["pinned_key_mismatch"])

    def test_tofu_learns_the_servers_key(self):
        store = self.dir / "trust.json"
        code, (record,) = self._request("--slice-type", "mMTC", "--trust-store", str(store), "--server-id", "core", "--now", "9000")
        self.assertTrue(record["result"]["trusted"])
        self.assertEqual(json.loads(store.read_text())["core"], (self.dir / "keys" / "public_key.bin").read_bytes().hex())

    def test_connection_refused_is_reported_per_request_with_nonzero_exit(self):
        code, (record,) = self._request("--slice-type", "URLLC", "--expected-pubkey-file", str(self.dir / "keys" / "public_key.bin"),
                                        "--then", f"127.0.0.1:{_free_port()}", "--now", "20000", "--count", "0")
        self.assertEqual(code, 1)
        self.assertEqual(record["error"]["type"], "ConnectionRefusedError")


if __name__ == "__main__":
    unittest.main()
