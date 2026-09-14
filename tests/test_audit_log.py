"""Tests for the audit log (pqc_auth/audit_log.py) and its independent
checker (pqc_auth/audit_verify.py).

The FakeSigner-backed tests run without liboqs, matching the rest of this
project's "the non-oqs path is never allowed to skip" convention. A
separate, gated test proves real key persistence (pqc_auth/dilithium.py's
OqsDilithiumSigner key_path) actually reloads the same identity, not just
that some file got written.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path

from pqc_auth.audit_verify import verify_log
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer
from pqc_auth.trust_store import TrustStore

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


class PinningAuditTests(unittest.TestCase):
    """(Task A.4/A.6) The server is backed by a DIFFERENT signer than the
    client is pinned to, so the logged record's signature is genuinely
    valid under its own recorded public_key while trusted=False and
    rejected_as_replay=False -- exactly the case audit_verify's signature
    check must not misjudge. Confirms the independent auditor reports
    this as a correct, internally-consistent rejection (PASS), not a
    false FAIL."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "audit_log.jsonl"

        self.pinned_signer = FakeSigner(key=b"pinned-identity-key-not-real-crypto")
        self.other_signer = FakeSigner(key=b"a-totally-different-identity-key")
        self.controller = DualTriggerReauthController(signer=self.other_signer)
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(
            self.server.host,
            self.server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            expected_public_key=self.pinned_signer.public_key,
            audit_log_path=str(self.log_path),
        )
        self.addCleanup(self.server.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def test_pinning_rejection_record_passes_independent_audit_as_a_correct_rejection(self):
        result = self.client.request_reauth("URLLC", 0)
        self.assertTrue(result.pinned_key_mismatch)

        lines = self.log_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertTrue(record["pinned_key_mismatch"])
        self.assertFalse(record["trusted"])
        self.assertFalse(record["rejected_as_replay"])
        # The recorded public_key really is the OTHER signer's -- and its
        # signature really would verify, which is exactly why a naive
        # "crypto_valid == trusted or rejected_as_replay" check would
        # wrongly FAIL this record.
        self.assertEqual(record["public_key"], self.other_signer.public_key.hex())

        verification = verify_log(self.log_path)
        self.assertTrue(verification.all_clean, [r.reasons for r in verification.records if not r.ok])
        self.assertEqual(len(verification.records), 1)

    def test_pinning_record_that_also_claims_trusted_is_still_caught_as_inconsistent(self):
        """The new check is not a rubber stamp: an impossible record
        (pinned_key_mismatch=True AND trusted=True) must still FAIL."""
        self.client.request_reauth("URLLC", 0)
        lines = self.log_path.read_text().splitlines()
        record = json.loads(lines[0])
        record["trusted"] = True  # contradicts pinned_key_mismatch=True; record_hash deliberately left stale
        tampered_path = Path(self._tmpdir.name) / "tampered.jsonl"
        tampered_path.write_text(json.dumps(record) + "\n")

        verification = verify_log(tampered_path)
        self.assertFalse(verification.all_clean)
        reasons = verification.records[0].reasons
        self.assertTrue(
            any("pinned_key_mismatch=True" in r and "but record also claims" in r for r in reasons),
            reasons,
        )


class TofuAuditTests(unittest.TestCase):
    """Mirrors PinningAuditTests above, for TOFU key-change rejections
    instead of explicit-pinning rejections. The server is backed by a
    DIFFERENT signer than the one already trusted in the trust store for
    this server_id, so the logged record's signature is genuinely valid
    under its own recorded public_key while trusted=False and
    rejected_as_replay=False -- the same shape audit_verify's signature
    check must not misjudge, now via trust_store_key_changed instead of
    pinned_key_mismatch."""

    SERVER_ID = "pqc-auth-test-tofu-server"

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "audit_log.jsonl"
        self.trust_store_path = Path(self._tmpdir.name) / "trust_store.json"

        self.original_signer = FakeSigner(key=b"tofu-original-identity-key")
        self.changed_signer = FakeSigner(key=b"tofu-changed-identity-key")

        # Pre-populate the trust store as if a prior run already learned
        # the original signer's key via first contact.
        TrustStore(self.trust_store_path).trust_first_contact(self.SERVER_ID, self.original_signer.public_key)

        self.controller = DualTriggerReauthController(signer=self.changed_signer)
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(
            self.server.host,
            self.server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            trust_store_path=str(self.trust_store_path),
            server_id=self.SERVER_ID,
            audit_log_path=str(self.log_path),
        )
        self.addCleanup(self.server.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def test_tofu_key_change_rejection_record_passes_independent_audit_as_a_correct_rejection(self):
        result = self.client.request_reauth("URLLC", 0)
        self.assertTrue(result.trust_store_key_changed)

        lines = self.log_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertTrue(record["trust_store_key_changed"])
        self.assertFalse(record["pinned_key_mismatch"])
        self.assertFalse(record["trusted"])
        self.assertFalse(record["rejected_as_replay"])
        # The recorded public_key really is the CHANGED signer's -- and its
        # signature really would verify, which is exactly why a naive
        # "crypto_valid == trusted or rejected_as_replay" check would
        # wrongly FAIL this record.
        self.assertEqual(record["public_key"], self.changed_signer.public_key.hex())

        verification = verify_log(self.log_path)
        self.assertTrue(verification.all_clean, [r.reasons for r in verification.records if not r.ok])
        self.assertEqual(len(verification.records), 1)

    def test_tofu_record_that_also_claims_trusted_is_still_caught_as_inconsistent(self):
        """Not a rubber stamp: an impossible record (trust_store_key_changed=True
        AND trusted=True) must still FAIL."""
        self.client.request_reauth("URLLC", 0)
        lines = self.log_path.read_text().splitlines()
        record = json.loads(lines[0])
        record["trusted"] = True  # contradicts trust_store_key_changed=True; record_hash deliberately left stale
        tampered_path = Path(self._tmpdir.name) / "tampered.jsonl"
        tampered_path.write_text(json.dumps(record) + "\n")

        verification = verify_log(tampered_path)
        self.assertFalse(verification.all_clean)
        reasons = verification.records[0].reasons
        self.assertTrue(
            any("trust_store_key_changed=True" in r and "but record also claims" in r for r in reasons),
            reasons,
        )


class HmacBackedBackendNameTests(unittest.TestCase):
    """Regression test for a pre-existing bug found independently twice,
    in two separate rounds of work, neither introduced by that round's
    own changes: once while actually running `python -m pqc_auth.live_loop`
    for real (on the live-loop-single-controller branch/PR #7), and once
    while running `python -m pqc_auth.demo` for real (on this branch/PR
    #6). audit_verify's HMAC dispatch only recognized
    backend == "FakeSigner", so any other HMAC-backed backend name
    (pqc_auth/demo.py's _DemoFakeSigner, pqc_auth/live_loop.py's
    _DemoLoopSigner -- both separate classes implementing the identical
    scheme) incorrectly fell through to the oqs-backed verification path
    and failed with a RuntimeError, turning every non-oqs demo/live_loop
    run's self-audit into a false FAIL."""

    def _hmac_signed_record(self, backend: str, key: bytes) -> dict:
        nonce = b"some-nonce-bytes"
        signature = hmac.new(key, nonce, hashlib.sha256).digest()
        fields = {
            "seq": 0, "timestamp": 0.0, "slice_type": "URLLC", "reason": "periodic",
            "nonce": nonce.hex(), "signature": signature.hex(), "public_key": key.hex(),
            "backend": backend, "trusted": True, "rejected_as_replay": False,
            "pinned_key_mismatch": False, "trust_store_key_changed": False,
            "prev_hash": "0" * 64,
        }
        record_hash = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {**fields, "record_hash": record_hash}

    def _assert_backend_name_recognized(self, backend: str) -> None:
        record = self._hmac_signed_record(backend, b"a-demo-key")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.jsonl"
            path.write_text(json.dumps(record) + "\n")
            result = verify_log(path)
        self.assertTrue(result.all_clean, [r.reasons for r in result.records if not r.ok])

    def test_demo_fake_signer_backend_name_is_recognized_as_hmac(self):
        self._assert_backend_name_recognized("_DemoFakeSigner")

    def test_live_loop_demo_signer_backend_name_is_recognized_as_hmac(self):
        self._assert_backend_name_recognized("_DemoLoopSigner")

    def test_tests_fake_signer_backend_name_still_recognized_as_hmac(self):
        self._assert_backend_name_recognized("FakeSigner")


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
