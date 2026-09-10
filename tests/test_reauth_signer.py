import unittest

from pqc_auth.dilithium import OqsDilithiumSigner
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason, Signer

from tests.fake_signer import FakeSigner


class SignerProtocolTests(unittest.TestCase):
    def test_oqs_dilithium_signer_satisfies_signer_protocol(self):
        # Structural check only — does not instantiate OqsDilithiumSigner
        # (that requires liboqs), just confirms its sign()/verify() methods
        # give it the right shape. See pqc_auth/reauth.py's Signer docstring.
        self.assertTrue(issubclass(OqsDilithiumSigner, Signer))

    def test_fake_signer_satisfies_signer_protocol(self):
        self.assertTrue(issubclass(FakeSigner, Signer))


class ReauthWithoutSignerTests(unittest.TestCase):
    def test_reauth_matches_due_semantics_with_no_signer(self):
        controller = DualTriggerReauthController()
        outcome = controller.reauth("URLLC", 0)
        self.assertEqual(outcome.reason, ReauthReason.PERIODIC)
        self.assertIsNone(outcome.verified)
        self.assertIsNone(outcome.nonce)
        self.assertIsNone(outcome.signature)

    def test_reauth_returns_none_when_not_due(self):
        controller = DualTriggerReauthController()
        controller.reauth("URLLC", 0)
        self.assertIsNone(controller.reauth("URLLC", 5))


class ReauthWithSignerTests(unittest.TestCase):
    def test_reauth_with_signer_performs_verified_round_trip(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        outcome = controller.reauth("URLLC", 0)
        self.assertEqual(outcome.reason, ReauthReason.PERIODIC)
        self.assertIsNotNone(outcome.nonce)
        self.assertIsNotNone(outcome.signature)
        self.assertTrue(outcome.verified)
        # Not just trusting the controller's own report — verify independently.
        self.assertTrue(signer.verify(outcome.nonce, outcome.signature))

    def test_reauth_with_detector_alert_uses_alert_cooldown(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        controller.reauth("URLLC", 0)
        alert_outcome = controller.reauth("URLLC", 5, detector_alert=True)
        self.assertEqual(alert_outcome.reason, ReauthReason.DETECTOR_ALERT)
        self.assertTrue(alert_outcome.verified)
        # Inside the alert cooldown: no re-auth, signer not consulted again.
        self.assertIsNone(controller.reauth("URLLC", 8, detector_alert=True))

    def test_tampered_signature_fails_verification(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        outcome = controller.reauth("URLLC", 0)
        tampered = bytes([outcome.signature[0] ^ 0xFF]) + outcome.signature[1:]
        self.assertNotEqual(tampered, outcome.signature)
        self.assertFalse(signer.verify(outcome.nonce, tampered))

    def test_tampered_message_fails_verification(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        outcome = controller.reauth("URLLC", 0)
        tampered_nonce = outcome.nonce + b"extra-byte"
        self.assertFalse(signer.verify(tampered_nonce, outcome.signature))

    def test_wrong_signer_instance_fails_verification(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        outcome = controller.reauth("URLLC", 0)
        other_signer = FakeSigner(key=b"a-completely-different-test-key")
        self.assertFalse(other_signer.verify(outcome.nonce, outcome.signature))


if __name__ == "__main__":
    unittest.main()
