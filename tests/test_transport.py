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
from pqc_auth.transport import ReauthClient, ReauthServer

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
        # to capture a genuinely valid (nonce, signature, public_key) tuple.
        captured = self.client._send_request("URLLC", 0, detector_alert=False)
        first = self.client.process_response(captured, now=0)
        self.assertTrue(first.due)
        self.assertTrue(first.trusted)
        self.assertFalse(first.rejected_as_replay)

        # An attacker resends the exact same captured response later. The
        # signature is still cryptographically valid -- verify_fn alone
        # would accept it -- but replay tracking must reject it anyway.
        replayed = self.client.process_response(captured, now=1)
        self.assertTrue(replayed.due)
        self.assertFalse(replayed.trusted)
        self.assertTrue(replayed.rejected_as_replay)

    def test_tampered_signature_in_response_is_not_trusted(self):
        captured = self.client._send_request("URLLC", 0, detector_alert=False)
        tampered = dict(captured)
        tampered_sig = bytearray(bytes.fromhex(tampered["signature"]))
        tampered_sig[0] ^= 0xFF
        tampered["signature"] = tampered_sig.hex()
        result = self.client.process_response(tampered, now=0)
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
        captured = short_window_client._send_request("URLLC", 0, detector_alert=False)
        first = short_window_client.process_response(captured, now=0)
        self.assertTrue(first.trusted)
        # Well past the 10s window -- the nonce should have been pruned,
        # so this is treated as a fresh (still cryptographically valid)
        # message rather than a replay. This is a deliberate memory-bound
        # trade-off (see DEFAULT_REPLAY_WINDOW_SECONDS's docstring), not a
        # security gap for THIS test's timeline, which never resends a
        # message within the window.
        replay_after_expiry = short_window_client.process_response(captured, now=100)
        self.assertTrue(replay_after_expiry.trusted)
        self.assertFalse(replay_after_expiry.rejected_as_replay)


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
