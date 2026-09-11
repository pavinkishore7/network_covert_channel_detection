"""Tests for the audit log (pqc_auth/audit_log.py) and its independent
checker (pqc_auth/audit_verify.py).

The FakeSigner-backed tests run without liboqs, matching the rest of this
project's "the non-oqs path is never allowed to skip" convention. A
separate, gated test proves real key persistence (pqc_auth/dilithium.py's
OqsDilithiumSigner key_path) actually reloads the same identity, not just
that some file got written.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pqc_auth.audit_verify import verify_log
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer

from tests.fake_signer import FakeSigner


class CleanAuditLogTests(unittest.TestCase):
    """(a) A clean end-to-end run: real socket, real client-side logging,
    correct hash chain, and the independent auditor reports all-clean."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "audit_log.jsonl"

        self.controller = DualTriggerReauthController(signer=FakeSigner())
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(
            self.server.host,
            self.server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            audit_log_path=str(self.log_path),
        )

    def tearDown(self):
        self.server.stop()
        self._tmpdir.cleanup()

    def test_clean_log_has_correct_record_count_and_chain_and_passes_independent_audit(self):
        self.client.request_reauth("URLLC", 0)
        self.client.request_reauth("eMBB", 0)
        self.client.request_reauth("URLLC", 40)  # second periodic reauth for URLLC

        lines = self.log_path.read_text().splitlines()
        self.assertEqual(len(lines), 3, "one audit record per due reauth")

        records = [json.loads(line) for line in lines]
        self.assertEqual([r["seq"] for r in records], [0, 1, 2])
        self.assertEqual(records[0]["prev_hash"], "0" * 64)
        # Each later record's prev_hash links to the exact previous raw
        # line (not merely the previous record_hash) -- verify_log() below
        # is the actual, independent check of that; asserting the formula
        # again here would just duplicate audit_verify.py's own logic.
        for r in records:
            self.assertEqual(r["backend"], "FakeSigner")
            self.assertTrue(r["trusted"])
            self.assertFalse(r["rejected_as_replay"])

        result = verify_log(self.log_path)
        self.assertTrue(result.all_clean, [r.reasons for r in result.records if not r.ok])
        self.assertEqual(len(result.records), 3)


class TamperDetectionTests(unittest.TestCase):
    """(b) and (c): the auditor must actually catch tampering, on a copy of
    a clean log it never saw get written."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "audit_log.jsonl"

        self.controller = DualTriggerReauthController(signer=FakeSigner())
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(
            self.server.host,
            self.server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            audit_log_path=str(self.log_path),
        )
        self.client.request_reauth("URLLC", 0)
        self.client.request_reauth("eMBB", 0)
        self.addCleanup(self.server.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def _tampered_copy(self, mutate) -> Path:
        lines = self.log_path.read_text().splitlines()
        record = json.loads(lines[0])
        mutate(record)
        lines[0] = json.dumps(record)  # NOT re-canonicalized -- simulates a raw tamper, record_hash left stale
        tampered_path = Path(self._tmpdir.name) / "tampered.jsonl"
        tampered_path.write_text("\n".join(lines) + "\n")
        return tampered_path

    def test_flipped_signature_byte_is_reported_as_a_signature_failure_on_that_record(self):
        def flip_signature_byte(record: dict) -> None:
            sig = bytearray(bytes.fromhex(record["signature"]))
            sig[0] ^= 0xFF
            record["signature"] = sig.hex()

        tampered_path = self._tampered_copy(flip_signature_byte)
        result = verify_log(tampered_path)

        self.assertFalse(result.all_clean)
        record0 = result.records[0]
        self.assertFalse(record0.ok)
        self.assertTrue(
            any("signature check mismatch" in reason for reason in record0.reasons),
            record0.reasons,
        )
        # Record #1 is also affected (its prev_hash no longer matches record
        # #0's now-different bytes) -- that's the chain doing its job, not a
        # false negative on record #0's own signature failure.

    def test_content_edit_without_recomputing_hash_is_reported_as_a_chain_break_not_a_signature_failure(self):
        def corrupt_slice_type_only(record: dict) -> None:
            record["slice_type"] = "TAMPERED_SLICE"
            # record_hash deliberately left as-is -- the attacker didn't
            # recompute it, which is exactly what this check must catch.

        tampered_path = self._tampered_copy(corrupt_slice_type_only)
        result = verify_log(tampered_path)

        self.assertFalse(result.all_clean)
        record0 = result.records[0]
        self.assertFalse(record0.ok)
        self.assertTrue(
            any("record_hash mismatch" in reason for reason in record0.reasons),
            record0.reasons,
        )
        self.assertFalse(
            any("signature check mismatch" in reason for reason in record0.reasons),
            "editing slice_type must not affect the independently-recomputed signature check",
        )


class OqsKeyPersistenceIdentityTests(unittest.TestCase):
    """(Step 4.2) Only meaningful with real liboqs -- proves key_path
    persistence reloads the SAME signing identity, not just that a file
    exists on disk."""

    @classmethod
    def setUpClass(cls):
        import pytest

        pytest.importorskip("oqs")

    def test_new_signer_instance_with_same_key_path_has_identical_public_key(self):
        from pqc_auth.dilithium import OqsDilithiumSigner

        with tempfile.TemporaryDirectory() as tmpdir:
            first = OqsDilithiumSigner(key_path=tmpdir)
            first_public_key = first.public_key
            message = b"key persistence identity probe"
            first_signature = first.sign(message)
            del first

            second = OqsDilithiumSigner(key_path=tmpdir)
            self.assertEqual(second.public_key, first_public_key)
            # Not just matching bytes -- the reloaded signer can verify a
            # signature made by the discarded instance, proving it holds
            # the same private key, not merely the same public key file.
            self.assertTrue(second.verify(message, first_signature))


if __name__ == "__main__":
    unittest.main()
