"""Tests for signed key rotation (pqc_auth/key_rotation.py) and how
ReauthClient, TrustStore and audit_verify apply it.

The FakeSigner tests use REAL local TCP sockets and must run without
liboqs (the non-oqs path is never allowed to skip). FakeSigner is HMAC, so
its "public key" is its secret -- fine for exercising the protocol logic,
which is identical for any Signer. The oqs-gated tests at the end run the
same flow with real ML-DSA-65 keys, including the persisted-key helper and
the CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pqc_auth import audit_verify, key_rotation
from pqc_auth.audit_verify import verify_log
from pqc_auth.key_rotation import (
    MAX_CHAIN_LINKS,
    RotationFormatError,
    RotationStatement,
    SignedRotation,
    decode_statement,
    encode_statement,
    issue_rotation,
    pubkey_hash,
    verify_rotation_chain,
)
from pqc_auth.outcomes import RequestOutcome
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import SIGNATURE_DOMAIN, ReauthClient, ReauthServer
from pqc_auth.trust_store import TrustStore, TrustStoreError

from tests.fake_signer import FakeSigner

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_ID = "rotating-server"
VERIFY = FakeSigner.verify_with_public_key

try:
    import oqs  # noqa: F401  # type: ignore[import-not-found]

    HAVE_OQS = True
except (ImportError, RuntimeError, SystemExit):
    HAVE_OQS = False


def _keys(n: int) -> list[FakeSigner]:
    return [FakeSigner(key=f"server-key-{i}".encode()) for i in range(n)]


def _chain(signers: list[FakeSigner], start_epoch: int = 1) -> list[SignedRotation]:
    """Statements rotating signers[0] -> signers[1] -> ... with consecutive epochs."""
    return [
        issue_rotation(old, new.public_key, SERVER_ID, epoch=start_epoch + i, issued_at=1_700_000_000 + i)
        for i, (old, new) in enumerate(zip(signers, signers[1:]))
    ]


class StatementEncodingTests(unittest.TestCase):
    def setUp(self):
        self.statement = RotationStatement(SERVER_ID, pubkey_hash(b"old"), b"new-public-key", 3, 1_700_000_000)

    def test_round_trips(self):
        self.assertEqual(decode_statement(encode_statement(self.statement)), self.statement)

    def test_layout_is_domain_then_length_prefixed_fields_then_fixed_width_ints(self):
        encoded = encode_statement(self.statement)
        self.assertTrue(encoded.startswith(key_rotation.ROTATION_DOMAIN))
        self.assertEqual(encoded[-16:], (3).to_bytes(8, "big") + (1_700_000_000).to_bytes(8, "big"))

    def test_decoder_rejects_anything_the_encoder_could_not_produce(self):
        encoded = encode_statement(self.statement)
        for label, bad in {
            "trailing byte": encoded + b"\x00",
            "truncated": encoded[:-1],
            "wrong domain": b"pqc_auth.reauth.v2\x00" + encoded[len(key_rotation.ROTATION_DOMAIN):],
            "epoch 0": encoded[:-16] + (0).to_bytes(8, "big") + encoded[-8:],
        }.items():
            with self.subTest(label), self.assertRaises(RotationFormatError):
                decode_statement(bad)

    def test_domain_literal_matches_the_auditors_copy_and_differs_from_the_reauth_domain(self):
        self.assertEqual(key_rotation.ROTATION_DOMAIN, audit_verify._ROTATION_DOMAIN)
        self.assertNotEqual(key_rotation.ROTATION_DOMAIN, SIGNATURE_DOMAIN)

    def test_auditor_decoder_agrees_with_the_writer(self):
        decoded = audit_verify._decode_rotation(encode_statement(self.statement))
        self.assertEqual(decoded["new_pubkey"], self.statement.new_pubkey)
        self.assertEqual(decoded["epoch"], self.statement.epoch)


class VerifyRotationChainTests(unittest.TestCase):
    """The pure acceptance rules, no store or socket involved."""

    def check(self, pinned, epoch, presented, rotations):
        return verify_rotation_chain(pinned_key=pinned.public_key, pinned_epoch=epoch,
                                     presented_key=presented.public_key, server_id=SERVER_ID,
                                     rotations=rotations, verify_fn=VERIFY)

    def test_single_valid_rotation_is_accepted(self):
        k0, k1 = _keys(2)
        result = self.check(k0, 0, k1, _chain([k0, k1]))
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.new_epoch, 1)

    def test_statement_signed_by_an_attacker_key_is_rejected(self):
        k0, _ = _keys(2)
        attacker = FakeSigner(key=b"attacker")
        # Names the pinned key as "old", but the signature is the attacker's own.
        statement = encode_statement(RotationStatement(SERVER_ID, pubkey_hash(k0.public_key), attacker.public_key, 1, 0))
        forged = SignedRotation(statement, attacker.sign(statement))
        result = self.check(k0, 0, attacker, [forged])
        self.assertFalse(result.accepted)
        self.assertIn("signature does not verify", result.reason)

    def test_statement_from_a_different_old_key_is_rejected(self):
        k0, k1, other = _keys(3)
        result = self.check(k0, 0, k1, _chain([other, k1]))
        self.assertFalse(result.accepted)
        self.assertIn("not issued by the currently trusted key", result.reason)

    def test_stale_epoch_is_rejected(self):
        k0, k1 = _keys(2)
        result = self.check(k0, 1, k1, _chain([k0, k1]))  # stored epoch already 1
        self.assertFalse(result.accepted)
        self.assertIn("stale epoch", result.reason)

    def test_statement_for_another_server_id_is_rejected(self):
        k0, k1 = _keys(2)
        other = issue_rotation(k0, k1.public_key, "someone-else", epoch=1)
        self.assertIn("server_id", self.check(k0, 0, k1, [other]).reason)

    def test_valid_statement_for_a_different_new_key_is_rejected(self):
        k0, k1, k2 = _keys(3)
        self.assertIn("does not end at", self.check(k0, 0, k2, _chain([k0, k1])).reason)

    def test_no_statement_is_rejected(self):
        k0, k1 = _keys(2)
        self.assertEqual(self.check(k0, 0, k1, []).reason, "no rotation statement")

    def test_multi_hop_chain_is_accepted_in_order(self):
        keys = _keys(4)
        result = self.check(keys[0], 0, keys[3], _chain(keys))
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual((result.new_epoch, len(result.links)), (3, 3))

    def test_multi_hop_skips_links_already_applied(self):
        keys = _keys(4)
        result = self.check(keys[1], 1, keys[3], _chain(keys))
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(len(result.links), 2)

    def test_chain_with_a_missing_link_is_rejected(self):
        keys = _keys(4)
        chain = _chain(keys)
        result = self.check(keys[0], 0, keys[3], [chain[0], chain[2]])
        self.assertFalse(result.accepted)
        self.assertIn("not issued by the currently trusted key", result.reason)

    def test_chain_out_of_order_is_rejected(self):
        keys = _keys(3)
        chain = _chain(keys)
        self.assertFalse(self.check(keys[0], 0, keys[2], list(reversed(chain))).accepted)

    def test_chain_longer_than_the_cap_is_rejected(self):
        keys = _keys(MAX_CHAIN_LINKS + 2)
        result = self.check(keys[0], 0, keys[-1], _chain(keys))
        self.assertFalse(result.accepted)
        self.assertIn("MAX_CHAIN_LINKS", result.reason)


class TrustStoreRotationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "trust.json"

    def test_legacy_bare_hex_entry_reads_as_epoch_zero(self):
        self.path.write_text(json.dumps({"s": b"key".hex()}))
        store = TrustStore(self.path)
        self.assertEqual((store.get_trusted_key("s"), store.get_epoch("s")), (b"key", 0))

    def test_accept_rotation_is_compare_and_set(self):
        store = TrustStore(self.path)
        store.trust_first_contact("s", b"k0")
        with self.assertRaises(TrustStoreError):
            store.accept_rotation("s", b"not-the-pin", b"k1", 1)
        with self.assertRaises(TrustStoreError):
            store.accept_rotation("s", b"k0", b"k1", 0)
        store.accept_rotation("s", b"k0", b"k1", 1)
        self.assertEqual((store.get_trusted_key("s"), store.get_epoch("s")), (b"k1", 1))
        with self.assertRaises(TrustStoreError):
            store.accept_rotation("s", b"k1", b"k0", 1)  # same epoch again: never backwards or sideways


class _SocketTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.store_path = self.dir / "trust.json"
        self.log_path = self.dir / "audit.jsonl"
        self.now = 0

    def serve(self, signer, rotations=None) -> ReauthServer:
        server = ReauthServer(DualTriggerReauthController(signer=signer), server_id=SERVER_ID, rotations=rotations)
        server.start()
        self.addCleanup(server.stop)
        return server

    def client(self, server, **kwargs) -> ReauthClient:
        kwargs.setdefault("trust_store_path", str(self.store_path))
        kwargs.setdefault("server_id", SERVER_ID)
        return ReauthClient(server.host, server.port, verify_fn=VERIFY, audit_log_path=str(self.log_path),
                            max_attempts=1, **kwargs)

    def request(self, server, **kwargs):
        self.now += 100
        return self.client(server, **kwargs).request_reauth("URLLC", self.now, detector_alert=True)

    def pinned(self):
        store = TrustStore(self.store_path)
        return store.get_trusted_key(SERVER_ID), store.get_epoch(SERVER_ID)


class ClientRotationTests(_SocketTestCase):
    def test_valid_rotation_is_accepted_and_the_pin_and_epoch_move(self):
        k0, k1 = _keys(2)
        self.assertTrue(self.request(self.serve(k0)).trusted)  # first contact pins k0 at epoch 0
        result = self.request(self.serve(k1, rotations=_chain([k0, k1])))
        self.assertEqual(result.outcome, RequestOutcome.VERIFIED)
        self.assertTrue(result.key_rotation_accepted)
        self.assertEqual(result.trusted_epoch, 1)
        self.assertEqual(self.pinned(), (k1.public_key, 1))
        # From now on k1 is simply the pinned key: no rotation re-applied.
        again = self.request(self.serve(k1, rotations=_chain([k0, k1])))
        self.assertTrue(again.trusted)
        self.assertFalse(again.key_rotation_accepted)

    def test_key_change_without_a_statement_is_the_unchanged_tofu_rejection(self):
        k0, k1 = _keys(2)
        self.request(self.serve(k0))
        result = self.request(self.serve(k1))
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertTrue(result.trust_store_key_changed)
        self.assertEqual(result.detail, "no rotation statement")
        self.assertEqual(self.pinned(), (k0.public_key, 0))

    def test_forged_statement_signed_by_an_attacker_key_is_rejected(self):
        k0, _ = _keys(2)
        attacker = FakeSigner(key=b"attacker")
        self.request(self.serve(k0))
        statement = encode_statement(RotationStatement(SERVER_ID, pubkey_hash(k0.public_key), attacker.public_key, 1, 0))
        forged = [SignedRotation(statement, attacker.sign(statement))]
        result = self.request(self.serve(attacker, rotations=forged))
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertIn("signature does not verify", result.detail)
        self.assertEqual(self.pinned(), (k0.public_key, 0))

    def test_replayed_old_statement_for_a_retired_key_is_rejected_by_epoch(self):
        """The case where ONLY the epoch saves the client: the server rotated
        k0 -> k1 (epoch 1) and later back to k0 (epoch 2, e.g. restored from
        backup). The client follows both and is pinned to k0 at epoch 2. An
        attacker who kept the retired k1 secret replays the genuine epoch-1
        statement: it IS signed by the currently pinned key k0 and names k0
        as its old key, so the epoch check is the only thing that rejects it."""
        k0, k1 = _keys(2)
        s1 = issue_rotation(k0, k1.public_key, SERVER_ID, epoch=1)
        s2 = issue_rotation(k1, k0.public_key, SERVER_ID, epoch=2)
        self.request(self.serve(k0))
        self.assertTrue(self.request(self.serve(k1, rotations=[s1])).key_rotation_accepted)
        self.assertTrue(self.request(self.serve(k0, rotations=[s1, s2])).key_rotation_accepted)
        self.assertEqual(self.pinned(), (k0.public_key, 2))

        result = self.request(self.serve(k1, rotations=[s1]))
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertIn("stale epoch", result.detail)
        self.assertEqual(self.pinned(), (k0.public_key, 2))

    def test_previously_retired_key_with_its_old_statement_is_rejected_by_epoch(self):
        k0, k1 = _keys(2)
        s1 = issue_rotation(k0, k1.public_key, SERVER_ID, epoch=1)
        self.request(self.serve(k0))
        self.request(self.serve(k1, rotations=[s1]))
        # Attacker holding the retired k0 secret presents k0 plus the statement that once endorsed k1.
        result = self.request(self.serve(k0, rotations=[s1]))
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertIn("stale epoch", result.detail)
        self.assertEqual(self.pinned(), (k1.public_key, 1))

    def test_rotation_is_not_persisted_when_the_response_itself_fails(self):
        k0, k1 = _keys(2)
        self.request(self.serve(k0))
        server = self.serve(k1, rotations=_chain([k0, k1]))
        client = self.client(server)
        captured, _ = client._send_request("URLLC", 500, detector_alert=True)
        result = client.process_response(captured, 500, expected_challenge=b"\x01" * 32)  # not this request's challenge
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_CHALLENGE_MISMATCH)
        self.assertEqual(self.pinned(), (k0.public_key, 0))

    def test_explicit_pin_is_not_overridden_by_a_valid_rotation(self):
        k0, k1 = _keys(2)
        self.request(self.serve(k0))
        result = self.request(self.serve(k1, rotations=_chain([k0, k1])), expected_public_key=k0.public_key)
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_PINNED_KEY)
        self.assertFalse(result.key_rotation_accepted)
        self.assertEqual(self.pinned(), (k0.public_key, 0))

    def test_client_that_missed_several_rotations_accepts_the_verified_chain(self):
        keys = _keys(4)
        self.request(self.serve(keys[0]))
        result = self.request(self.serve(keys[3], rotations=_chain(keys)))
        self.assertTrue(result.key_rotation_accepted)
        self.assertEqual(self.pinned(), (keys[3].public_key, 3))

    def test_client_further_behind_than_the_retained_chain_needs_manual_retrust(self):
        keys = _keys(MAX_CHAIN_LINKS + 2)
        self.request(self.serve(keys[0]))
        # The server retains (and sends) only the last MAX_CHAIN_LINKS statements.
        result = self.request(self.serve(keys[-1], rotations=_chain(keys)))
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertIn("not issued by the currently trusted key", result.detail)
        self.assertEqual(self.pinned(), (keys[0].public_key, 0))

    def test_first_contact_takes_the_epoch_of_the_statement_endorsing_the_key(self):
        keys = _keys(3)
        self.request(self.serve(keys[2], rotations=_chain(keys)))
        self.assertEqual(self.pinned(), (keys[2].public_key, 2))

    def test_malformed_rotations_field_is_a_tofu_rejection_not_a_crash(self):
        k0, k1 = _keys(2)
        self.request(self.serve(k0))
        server = self.serve(k1)
        server._rotations_wire = [{"statement": "zz", "signature": "00"}]
        result = self.request(server)
        self.assertEqual(result.outcome, RequestOutcome.REJECTED_TOFU_KEY_CHANGED)
        self.assertIn("malformed", result.detail)


def _rewrite(log_path: Path, index: int, mutate) -> None:
    """Edit one record and RE-CHAIN the whole log (recompute every
    record_hash / prev_hash), so the only thing left for the auditor to
    object to is the record's content, not a broken hash chain."""
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    mutate(records[index])
    prev, lines = "0" * 64, []
    for record in records:
        record.pop("record_hash", None)
        record["prev_hash"] = prev
        canonical = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))  # noqa: E731
        record["record_hash"] = hashlib.sha256(canonical(record).encode()).hexdigest()
        line = canonical(record)
        prev = hashlib.sha256(line.encode()).hexdigest()
        lines.append(line)
    log_path.write_text("\n".join(lines) + "\n")


class AuditRotationTests(_SocketTestCase):
    def setUp(self):
        super().setUp()
        self.k0, self.k1 = _keys(2)
        self.request(self.serve(self.k0))
        self.assertTrue(self.request(self.serve(self.k1, rotations=_chain([self.k0, self.k1]))).key_rotation_accepted)
        records = [json.loads(l) for l in self.log_path.read_text().splitlines()]
        self.rotation_index = next(i for i, r in enumerate(records) if r.get("key_rotation_accepted"))

    def failures(self):
        result = verify_log(self.log_path)
        return [reason for r in result.records for reason in r.reasons]

    def test_accepted_rotation_passes_independent_audit(self):
        self.assertEqual(self.failures(), [])

    def test_rejected_rotation_record_passes_audit(self):
        attacker = FakeSigner(key=b"attacker")
        statement = encode_statement(RotationStatement(SERVER_ID, pubkey_hash(self.k1.public_key), attacker.public_key, 2, 0))
        self.request(self.serve(attacker, rotations=[SignedRotation(statement, attacker.sign(statement))]))
        self.assertEqual(self.failures(), [])

    def assert_rotation_claim_fails(self, mutate, expected):
        _rewrite(self.log_path, self.rotation_index, mutate)
        failures = self.failures()
        self.assertTrue(any(expected in f for f in failures), failures)
        self.assertFalse(any("hash mismatch" in f for f in failures), failures)  # re-chained: only the claim is wrong

    def test_claim_without_any_statement_fails(self):
        self.assert_rotation_claim_fails(lambda r: r.update(rotation_statements=[]), "no rotation statement")

    def test_claim_on_a_record_with_no_rotation_fields_fails(self):
        def strip(record):
            for key in ("rotation_statements", "previous_public_key", "previous_epoch", "new_epoch"):
                del record[key]
        self.assert_rotation_claim_fails(strip, "no rotation statement")

    def test_statement_signed_by_another_key_fails(self):
        attacker = FakeSigner(key=b"attacker")

        def forge(record):
            statement = encode_statement(RotationStatement(SERVER_ID, pubkey_hash(self.k0.public_key), self.k1.public_key, 1, 0))
            record["rotation_statements"] = [SignedRotation(statement, attacker.sign(statement)).to_wire()]
        self.assert_rotation_claim_fails(forge, "does not verify under the previously trusted key")

    def test_statement_not_from_the_recorded_previous_key_fails(self):
        self.assert_rotation_claim_fails(lambda r: r.update(previous_public_key=b"some-other-key".hex()),
                                         "not issued by the previously trusted key")

    def test_stale_epoch_claim_fails(self):
        self.assert_rotation_claim_fails(lambda r: r.update(previous_epoch=1), "not above")

    def test_rotation_fields_without_the_claim_fail(self):
        self.assert_rotation_claim_fails(lambda r: r.update(key_rotation_accepted=False), "without key_rotation_accepted")


@unittest.skipUnless(HAVE_OQS, "requires liboqs-python")
class PersistedOqsRotationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.key_dir = self.dir / "keys"

    def test_rotate_replaces_the_key_signs_with_the_old_one_and_keeps_a_bounded_chain(self):
        from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

        old = OqsDilithiumSigner(key_path=str(self.key_dir))
        old_secret = (self.key_dir / "secret_key.bin").read_bytes()
        signed = key_rotation.rotate_persisted_key(self.key_dir, SERVER_ID)
        st = signed.statement
        self.assertEqual(st.old_pubkey_hash, pubkey_hash(old.public_key))
        self.assertEqual(st.new_pubkey, (self.key_dir / "public_key.bin").read_bytes())
        self.assertEqual(st.epoch, 1)
        self.assertTrue(verify_with_public_key(signed.encoded, signed.signature, old.public_key))
        self.assertNotEqual((self.key_dir / "secret_key.bin").read_bytes(), old_secret)
        self.assertFalse(any(old_secret in p.read_bytes() for p in self.key_dir.iterdir()))
        # The reloaded signer is the new key.
        self.assertEqual(OqsDilithiumSigner(key_path=str(self.key_dir)).public_key, st.new_pubkey)
        for _ in range(MAX_CHAIN_LINKS + 1):
            key_rotation.rotate_persisted_key(self.key_dir, SERVER_ID)
        retained = key_rotation.load_rotations(self.key_dir)
        self.assertEqual([r.statement.epoch for r in retained], list(range(3, 3 + MAX_CHAIN_LINKS)))

    def test_real_ml_dsa_rotation_over_a_socket_is_accepted_and_audits_clean(self):
        from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

        store, log = self.dir / "trust.json", self.dir / "audit.jsonl"

        def request(now):
            signer = OqsDilithiumSigner(key_path=str(self.key_dir))
            server = ReauthServer(DualTriggerReauthController(signer=signer), server_id=SERVER_ID,
                                  rotations=key_rotation.load_rotations(self.key_dir))
            server.start()
            try:
                return ReauthClient(server.host, server.port, verify_fn=verify_with_public_key,
                                    trust_store_path=str(store), server_id=SERVER_ID,
                                    audit_log_path=str(log)).request_reauth("URLLC", now, detector_alert=True)
            finally:
                server.stop()

        self.assertTrue(request(100).trusted)
        key_rotation.rotate_persisted_key(self.key_dir, SERVER_ID)
        result = request(200)
        self.assertTrue(result.key_rotation_accepted)
        self.assertEqual(TrustStore(store).get_epoch(SERVER_ID), 1)
        self.assertTrue(verify_log(log).all_clean)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_OQS, "requires liboqs-python (the CLI always uses the real OqsDilithiumSigner)")
class CliRotationTests(unittest.TestCase):
    """The deployable path: serve, stop, rotate with the CLI, serve again."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    def _with_server(self, fn):
        port = _free_port()
        server = subprocess.Popen(
            [sys.executable, "-m", "pqc_auth.transport", "serve", "--port", str(port),
             "--key-path", str(self.dir / "keys"), "--server-id", SERVER_ID],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT, env=self.env,
        )
        try:
            for line in server.stdout:
                if line.startswith("READY"):
                    break
            return fn(port)
        finally:
            server.terminate()
            server.wait(timeout=10)

    def _request(self, port, now):
        out = subprocess.run(
            [sys.executable, "-m", "pqc_auth.transport", "request", "--port", str(port), "--slice-type", "URLLC",
             "--trust-store", str(self.dir / "trust.json"), "--server-id", SERVER_ID,
             "--audit-log", str(self.dir / "audit.jsonl"), "--now", str(now), "--detector-alert"],
            capture_output=True, text=True, cwd=REPO_ROOT, env=self.env, timeout=60,
        )
        return json.loads(next(l for l in out.stdout.splitlines() if l.startswith("{")))

    def test_rotate_cli_then_restart_is_accepted_by_a_tofu_client(self):
        first = self._with_server(lambda port: self._request(port, 100))
        self.assertTrue(first["result"]["trusted"])
        rotated = subprocess.run(
            [sys.executable, "-m", "pqc_auth.key_rotation", "rotate", "--key-path", str(self.dir / "keys"),
             "--server-id", SERVER_ID], capture_output=True, text=True, cwd=REPO_ROOT, env=self.env, timeout=60,
        )
        self.assertEqual(rotated.returncode, 0, rotated.stderr)
        self.assertEqual(json.loads(rotated.stdout.splitlines()[-1])["epoch"], 1)
        second = self._with_server(lambda port: self._request(port, 200))
        self.assertTrue(second["result"]["trusted"], second)
        self.assertEqual(TrustStore(self.dir / "trust.json").get_epoch(SERVER_ID), 1)
        self.assertTrue(verify_log(self.dir / "audit.jsonl").all_clean)


if __name__ == "__main__":
    unittest.main()
