"""Tests for pqc_auth/trust_store.py's TOFU pinning, both directly and wired
through ReauthClient. The ReauthClient tests use FakeSigner and REAL local
TCP sockets (not mocked) -- these must run without liboqs, matching the
rest of this project's "the non-oqs path is never allowed to skip"
convention.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer
from pqc_auth.trust_store import TrustStore, TrustStoreError

from tests.fake_signer import FakeSigner


class TrustStoreUnitTests(unittest.TestCase):
    """Direct tests of TrustStore itself, no transport involved."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "trust_store.json"
        self.addCleanup(self._tmpdir.cleanup)

    def test_unknown_server_id_returns_none(self):
        store = TrustStore(self.path)
        self.assertIsNone(store.get_trusted_key("never-seen"))

    def test_first_contact_then_lookup_round_trips(self):
        store = TrustStore(self.path)
        key = b"some-fake-public-key-bytes"
        store.trust_first_contact("server-a", key)
        self.assertEqual(store.get_trusted_key("server-a"), key)

    def test_first_contact_twice_for_same_server_id_raises(self):
        store = TrustStore(self.path)
        store.trust_first_contact("server-a", b"key-one")
        with self.assertRaises(TrustStoreError):
            store.trust_first_contact("server-a", b"key-two")
        # The original key must be untouched -- no silent overwrite.
        self.assertEqual(store.get_trusted_key("server-a"), b"key-one")

    def test_force_retrust_overwrites_a_known_key(self):
        store = TrustStore(self.path)
        store.trust_first_contact("server-a", b"key-one")
        store.force_retrust("server-a", b"key-two")
        self.assertEqual(store.get_trusted_key("server-a"), b"key-two")

    def test_force_retrust_also_works_for_a_never_seen_server_id(self):
        store = TrustStore(self.path)
        store.force_retrust("server-b", b"key-x")
        self.assertEqual(store.get_trusted_key("server-b"), b"key-x")

    def test_persists_across_instances(self):
        TrustStore(self.path).trust_first_contact("server-a", b"key-one")
        reloaded = TrustStore(self.path)
        self.assertEqual(reloaded.get_trusted_key("server-a"), b"key-one")


class ReauthClientRequiresBothTofuParamsTests(unittest.TestCase):
    def test_trust_store_path_without_server_id_raises(self):
        with self.assertRaises(ValueError):
            ReauthClient("127.0.0.1", 1, verify_fn=FakeSigner.verify_with_public_key, trust_store_path="/tmp/x")

    def test_server_id_without_trust_store_path_raises(self):
        with self.assertRaises(ValueError):
            ReauthClient("127.0.0.1", 1, verify_fn=FakeSigner.verify_with_public_key, server_id="s")


class ReauthClientTofuTests(unittest.TestCase):
    """(a)-(d) from the task: first contact, persistence across a NEW client
    instance, rejection of a genuinely different server identity, and
    acceptance after an explicit re-trust -- all over real sockets."""

    SERVER_ID = "pqc-auth-test-server"

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.trust_store_path = Path(self._tmpdir.name) / "trust_store.json"
        self.addCleanup(self._tmpdir.cleanup)

        self.signer = FakeSigner(key=b"the-original-server-identity-key")
        self.controller = DualTriggerReauthController(signer=self.signer)
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.addCleanup(self.server.stop)

    def _client_against(self, server: ReauthServer) -> ReauthClient:
        return ReauthClient(
            server.host,
            server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            trust_store_path=str(self.trust_store_path),
            server_id=self.SERVER_ID,
        )

    def test_a_first_contact_is_accepted_and_persisted(self):
        client = self._client_against(self.server)
        result = client.request_reauth("URLLC", 0)

        self.assertTrue(result.due)
        self.assertTrue(result.trusted)
        self.assertFalse(result.trust_store_key_changed)

        store = TrustStore(self.trust_store_path)
        self.assertEqual(store.get_trusted_key(self.SERVER_ID), self.signer.public_key)

    def test_b_new_client_instance_same_server_id_and_server_is_accepted_unchanged(self):
        self._client_against(self.server).request_reauth("URLLC", 0)  # learns the key

        second_client = self._client_against(self.server)  # brand-new instance
        result = second_client.request_reauth("URLLC", 5, detector_alert=True)

        self.assertTrue(result.due)
        self.assertTrue(result.trusted)
        self.assertFalse(result.trust_store_key_changed)

    def test_c_a_server_with_a_different_identity_under_the_same_server_id_is_rejected(self):
        self._client_against(self.server).request_reauth("URLLC", 0)  # learns self.signer's key

        different_signer = FakeSigner(key=b"a-completely-different-server-identity")
        different_controller = DualTriggerReauthController(signer=different_signer)
        different_server = ReauthServer(different_controller)
        different_server.start()
        self.addCleanup(different_server.stop)

        client_against_new_server = self._client_against(different_server)
        result = client_against_new_server.request_reauth("URLLC", 0)

        self.assertTrue(result.due)
        self.assertFalse(result.trusted)
        # Specifically the "key changed" path, not "first contact" -- first
        # contact would have produced trusted=True instead.
        self.assertTrue(result.trust_store_key_changed)

        # The originally-trusted key must be untouched by a rejected change.
        store = TrustStore(self.trust_store_path)
        self.assertEqual(store.get_trusted_key(self.SERVER_ID), self.signer.public_key)

    def test_d_after_explicit_retrust_the_new_key_is_accepted_going_forward(self):
        self._client_against(self.server).request_reauth("URLLC", 0)  # learns self.signer's key

        different_signer = FakeSigner(key=b"a-completely-different-server-identity")
        different_controller = DualTriggerReauthController(signer=different_signer)
        different_server = ReauthServer(different_controller)
        different_server.start()
        self.addCleanup(different_server.stop)

        # Operator consciously accepts the rotation -- this must not
        # happen automatically anywhere else in this test.
        TrustStore(self.trust_store_path).force_retrust(self.SERVER_ID, different_signer.public_key)

        client_after_retrust = self._client_against(different_server)
        result = client_after_retrust.request_reauth("URLLC", 0)

        self.assertTrue(result.due)
        self.assertTrue(result.trusted)
        self.assertFalse(result.trust_store_key_changed)

    def test_e_first_contact_with_an_invalid_signature_is_not_trusted_and_not_persisted(self):
        """Regression test: process_response() used to call
        trust_first_contact() unconditionally on the first response for a
        server_id, before verify_fn ever ran -- so a forged or corrupted
        first packet (no valid signature required) would permanently
        poison the trust store, locking out the real server's later
        genuine responses as spurious "key changed" rejections. A
        tampered first-contact response must be rejected like any other
        signature failure and must leave the trust store empty."""
        client = self._client_against(self.server)
        captured = client._send_request("URLLC", 0, detector_alert=False)
        tampered = dict(captured)
        tampered_sig = bytearray(bytes.fromhex(tampered["signature"]))
        tampered_sig[0] ^= 0xFF
        tampered["signature"] = tampered_sig.hex()

        result = client.process_response(tampered, now=0)

        self.assertTrue(result.due)
        self.assertFalse(result.trusted)
        # This must be an ordinary signature failure, not a spurious
        # "key changed" rejection -- there was nothing stored yet to
        # change from.
        self.assertFalse(result.trust_store_key_changed)

        store = TrustStore(self.trust_store_path)
        self.assertIsNone(store.get_trusted_key(self.SERVER_ID))

    def test_f_a_later_genuine_first_contact_still_succeeds_after_a_rejected_forged_one(self):
        """Proves the fix doesn't just reject the forged packet -- it also
        doesn't leave the trust store in a state that locks out the real
        server's later, genuine first response."""
        forging_client = self._client_against(self.server)
        captured = forging_client._send_request("URLLC", 0, detector_alert=False)
        tampered = dict(captured)
        tampered_sig = bytearray(bytes.fromhex(tampered["signature"]))
        tampered_sig[0] ^= 0xFF
        tampered["signature"] = tampered_sig.hex()
        forging_client.process_response(tampered, now=0)  # rejected, not persisted

        # A later, genuine response for the same server_id (now=30, past
        # the periodic interval the forged request already consumed on
        # the real server-side controller) must be accepted as ordinary
        # first contact, not rejected as a spurious key change.
        real_client = self._client_against(self.server)
        result = real_client.request_reauth("URLLC", 30)

        self.assertTrue(result.due)
        self.assertTrue(result.trusted)
        self.assertFalse(result.trust_store_key_changed)

        store = TrustStore(self.trust_store_path)
        self.assertEqual(store.get_trusted_key(self.SERVER_ID), self.signer.public_key)


if __name__ == "__main__":
    unittest.main()
