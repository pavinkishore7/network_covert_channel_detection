"""Tests for pqc_auth/transport.py using FakeSigner -- these open REAL local
TCP sockets (not mocked) and must pass without liboqs, since they're the
ones proving the two-party transport works everywhere. A second test class
below repeats the round trip with a real OqsDilithiumSigner, gated with
pytest.importorskip("oqs") exactly like tests/test_dilithium.py -- it is
allowed to be skipped; the FakeSigner tests are not.
"""

from __future__ import annotations

import unittest

from pqc_auth.reauth import DualTriggerReauthController
import json

from pqc_auth.transport import ReauthClient, ReauthServer, signed_message

from tests.fake_signer import FakeSigner


class FakeSignerTransportTests(unittest.TestCase):
    def setUp(self):
        self.controller = DualTriggerReauthController(signer=FakeSigner())
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(
            self.server.host, self.server.port, verify_fn=FakeSigner.verify_with_public_key
        )

    def tearDown(self):
        self.server.stop()

    def test_periodic_reauth_round_trip_over_real_socket(self):
        result = self.client.request_reauth("URLLC", 0)
        self.assertTrue(result.due)
        self.assertEqual(result.reason.value, "periodic")
        self.assertTrue(result.trusted)
        self.assertFalse(result.rejected_as_replay)
        self.assertEqual(result.backend, "FakeSigner")

    def test_not_due_yet_over_real_socket(self):
        self.client.request_reauth("URLLC", 0)  # first periodic reauth
        result = self.client.request_reauth("URLLC", 5)  # well inside the 30s interval
        self.assertFalse(result.due)
        self.assertIsNone(result.trusted)

    def test_detector_alert_round_trip_over_real_socket(self):
        self.client.request_reauth("URLLC", 0)
        result = self.client.request_reauth("URLLC", 5, detector_alert=True)
        self.assertTrue(result.due)
        self.assertEqual(result.reason.value, "detector_alert")
        self.assertTrue(result.trusted)

    def test_replayed_response_is_rejected_even_though_signature_is_valid(self):
        # A real round trip first, to prove the transport itself works and
        # to capture a genuinely valid signed response.
        captured, challenge = self.client._send_request("URLLC", 0, detector_alert=False)
        first = self.client.process_response(captured, now=0, expected_challenge=challenge)
        self.assertTrue(first.due)
        self.assertTrue(first.trusted)
        self.assertFalse(first.rejected_as_replay)

        # An attacker answers the client's NEXT request (which carries a new
        # challenge) with the captured response. The signature is still
        # cryptographically valid -- verify_fn alone would accept it -- but
        # it is bound to the old challenge.
        replayed = self.client.process_response(captured, now=1, expected_challenge=b"\x01" * 32)
        self.assertTrue(replayed.due)
        self.assertFalse(replayed.trusted)
        self.assertTrue(replayed.challenge_mismatch)
        self.assertFalse(replayed.rejected_as_replay)

    def test_seen_nonce_set_still_rejects_a_repeat_as_defence_in_depth(self):
        # Only reachable if the same challenge were reused, which
        # request_reauth() never does; checks the secondary layer alone.
        captured, challenge = self.client._send_request("URLLC", 0, detector_alert=False)
        self.assertTrue(self.client.process_response(captured, now=0, expected_challenge=challenge).trusted)
        again = self.client.process_response(captured, now=1, expected_challenge=challenge)
        self.assertFalse(again.trusted)
        self.assertTrue(again.rejected_as_replay)

    def test_tampered_signature_in_response_is_not_trusted(self):
        captured, challenge = self.client._send_request("URLLC", 0, detector_alert=False)
        tampered = dict(captured)
        tampered_sig = bytearray(bytes.fromhex(tampered["signature"]))
        tampered_sig[0] ^= 0xFF
        tampered["signature"] = tampered_sig.hex()
        result = self.client.process_response(tampered, now=0, expected_challenge=challenge)
        self.assertTrue(result.due)
        self.assertFalse(result.trusted)
        self.assertFalse(result.rejected_as_replay)  # rejected on the crypto check, not as a replay

    def test_replay_window_forgets_old_nonces(self):
        # A short window client so we can exercise expiry without waiting
        # on the module default (300s).
        short_window_client = ReauthClient(
            self.server.host, self.server.port, verify_fn=FakeSigner.verify_with_public_key,
            replay_window_seconds=10,
        )
        captured, challenge = short_window_client._send_request("URLLC", 0, detector_alert=False)
        first = short_window_client.process_response(captured, now=0, expected_challenge=challenge)
        self.assertTrue(first.trusted)
        # Well past the 10s window -- the nonce should have been pruned,
        # so the seen-nonce layer alone no longer flags it (same challenge
        # passed on purpose, to isolate that layer; a real replay would
        # carry a stale challenge and fail challenge_mismatch first). This is a deliberate memory-bound
        # trade-off (see DEFAULT_REPLAY_WINDOW_SECONDS's docstring), not a
        # security gap for THIS test's timeline, which never resends a
        # message within the window.
        replay_after_expiry = short_window_client.process_response(captured, now=100, expected_challenge=challenge)
        self.assertTrue(replay_after_expiry.trusted)
        self.assertFalse(replay_after_expiry.rejected_as_replay)


class PublicKeyPinningTests(unittest.TestCase):
    """The server is backed by a DIFFERENT signer than the one the client
    is pinned to, so every response it sends genuinely, correctly verifies
    -- just under the wrong key. Proves pinning rejects on the key alone,
    not by piggybacking on a signature failure."""

    def setUp(self):
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
        )

    def tearDown(self):
        self.server.stop()

    def test_pinning_rejects_a_genuinely_valid_signature_from_a_different_keypair(self):
        captured, challenge = self.client._send_request("URLLC", 0, detector_alert=False)
        message = signed_message(captured["payload"])
        signature = bytes.fromhex(captured["signature"])
        public_key = bytes.fromhex(captured["public_key"])

        # This is the OTHER signer's response, not the pinned one.
        self.assertEqual(public_key, self.other_signer.public_key)
        self.assertNotEqual(public_key, self.pinned_signer.public_key)
        # And it is a perfectly genuine signature -- verify_fn alone, with
        # no pinning, would accept it. Proves the rejection below isn't
        # just piggybacking on a signature that would have failed anyway.
        self.assertTrue(FakeSigner.verify_with_public_key(message, signature, public_key))

        result = self.client.process_response(captured, now=0, expected_challenge=challenge)
        self.assertTrue(result.due)
        self.assertFalse(result.trusted)
        self.assertFalse(result.rejected_as_replay)
        self.assertTrue(result.pinned_key_mismatch)

    def test_pinning_rejection_does_not_consume_the_nonce(self):
        captured, challenge = self.client._send_request("URLLC", 0, detector_alert=False)
        result = self.client.process_response(captured, now=0, expected_challenge=challenge)
        self.assertTrue(result.pinned_key_mismatch)
        self.assertNotIn(bytes.fromhex(json.loads(captured["payload"])["server_nonce"]), self.client._seen_nonces)

    def test_response_from_the_correctly_pinned_key_is_still_trusted(self):
        # Same client, but pointed at a server backed by the PINNED signer
        # this time -- confirms pinning doesn't just reject everything.
        matching_controller = DualTriggerReauthController(signer=self.pinned_signer)
        matching_server = ReauthServer(matching_controller)
        matching_server.start()
        self.addCleanup(matching_server.stop)
        matching_client = ReauthClient(
            matching_server.host,
            matching_server.port,
            verify_fn=FakeSigner.verify_with_public_key,
            expected_public_key=self.pinned_signer.public_key,
        )
        result = matching_client.request_reauth("URLLC", 0)
        self.assertTrue(result.trusted)
        self.assertFalse(result.pinned_key_mismatch)


class DilithiumTransportTests(unittest.TestCase):
    """Same round trip, real OqsDilithiumSigner. Skipped, not failed, when
    liboqs isn't available -- see tests/test_dilithium.py."""

    @classmethod
    def setUpClass(cls):
        import pytest

        pytest.importorskip("oqs")

    def setUp(self):
        from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

        self.controller = DualTriggerReauthController(signer=OqsDilithiumSigner())
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.client = ReauthClient(self.server.host, self.server.port, verify_fn=verify_with_public_key)

    def tearDown(self):
        self.server.stop()

    def test_periodic_reauth_round_trip_over_real_socket(self):
        result = self.client.request_reauth("URLLC", 0)
        self.assertTrue(result.due)
        self.assertTrue(result.trusted)
        self.assertEqual(result.backend, "OqsDilithiumSigner")


if __name__ == "__main__":
    unittest.main()
