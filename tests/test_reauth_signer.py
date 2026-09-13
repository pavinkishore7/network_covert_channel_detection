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


class FakeSignerPublicKeyVerifyTests(unittest.TestCase):
    """FakeSigner.verify_with_public_key mirrors
    pqc_auth.dilithium.verify_with_public_key's call shape so a verifying
    party can be tested against either backend without special-casing."""

    def test_verifies_without_any_signer_instance(self):
        signer = FakeSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature = signer.sign(message)
        # No signer instance passed below -- only the "public key" bytes.
        self.assertTrue(FakeSigner.verify_with_public_key(message, signature, signer.public_key))

    def test_matches_instance_verify(self):
        signer = FakeSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature = signer.sign(message)
        self.assertEqual(
            signer.verify(message, signature),
            FakeSigner.verify_with_public_key(message, signature, signer.public_key),
        )

    def test_wrong_public_key_fails_verification(self):
        signer = FakeSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature = signer.sign(message)
        wrong_key = b"a-completely-different-test-key"
        self.assertFalse(FakeSigner.verify_with_public_key(message, signature, wrong_key))


class DryRunTests(unittest.TestCase):
    """due()/reauth()'s dry_run=True is the mechanism pqc_auth/live_loop.py
    relies on to use a SINGLE DualTriggerReauthController instance instead
    of two kept in lockstep (see reauth.py's module docstring). These
    tests prove it's genuinely non-mutating, not just documented as such:
    repeating a dry run must give the identical answer every time, which
    would NOT hold if it were silently updating scheduling state."""

    def test_due_dry_run_reports_the_real_answer_without_mutating_state(self):
        controller = DualTriggerReauthController()
        self.assertEqual(controller.due("URLLC", 0, dry_run=True), ReauthReason.PERIODIC)
        # If that call had mutated _last_reauth, this second call at the
        # same (slice_type, now) would still see PERIODIC (now - last == 0
        # doesn't clear >= interval_seconds either way) -- so repeat at a
        # time inside the interval, where a real (mutating) PERIODIC call
        # would make a later call return None, to actually distinguish the
        # two.
        self.assertEqual(controller.due("URLLC", 0, dry_run=True), ReauthReason.PERIODIC)
        self.assertEqual(controller.due("URLLC", 5, dry_run=True), ReauthReason.PERIODIC)  # still "first ever" from the dry run's point of view

    def test_due_dry_run_does_not_consume_the_real_periodic_window(self):
        controller = DualTriggerReauthController()
        # Repeated dry runs first...
        for _ in range(3):
            self.assertEqual(controller.due("URLLC", 0, dry_run=True), ReauthReason.PERIODIC)
        # ...then the REAL (mutating) call must still see "never fired yet".
        self.assertEqual(controller.due("URLLC", 0), ReauthReason.PERIODIC)
        # And NOW a later call inside the interval is correctly suppressed.
        self.assertIsNone(controller.due("URLLC", 5))

    def test_due_dry_run_after_a_real_fire_reflects_the_real_state(self):
        controller = DualTriggerReauthController()
        controller.due("URLLC", 0)  # real fire -- URLLC now due again only at t>=30
        self.assertIsNone(controller.due("URLLC", 5, dry_run=True))
        self.assertEqual(controller.due("URLLC", 30, dry_run=True), ReauthReason.PERIODIC)
        # The dry run at t=30 must not have consumed that window either.
        self.assertEqual(controller.due("URLLC", 30), ReauthReason.PERIODIC)

    def test_reauth_dry_run_never_calls_the_signer(self):
        class ExplodingSigner:
            def sign(self, message: bytes) -> bytes:
                raise AssertionError("dry_run=True must never call sign()")

            def verify(self, message: bytes, signature: bytes) -> bool:
                raise AssertionError("dry_run=True must never call verify()")

        controller = DualTriggerReauthController(signer=ExplodingSigner())
        # If dry_run touched the signer at all, ExplodingSigner would raise
        # and fail this test -- reaching the assertions below is itself
        # proof it didn't.
        outcome = controller.reauth("URLLC", 0, dry_run=True)
        self.assertEqual(outcome.reason, ReauthReason.PERIODIC)
        self.assertIsNone(outcome.verified)
        self.assertIsNone(outcome.nonce)
        self.assertIsNone(outcome.signature)

    def test_reauth_dry_run_does_not_consume_the_real_periodic_window(self):
        signer = FakeSigner()
        controller = DualTriggerReauthController(signer=signer)
        for _ in range(3):
            dry_outcome = controller.reauth("URLLC", 0, dry_run=True)
            self.assertEqual(dry_outcome.reason, ReauthReason.PERIODIC)
        # The real call afterwards must still see the un-consumed window
        # and must actually sign, unlike every dry run before it.
        real_outcome = controller.reauth("URLLC", 0)
        self.assertEqual(real_outcome.reason, ReauthReason.PERIODIC)
        self.assertTrue(real_outcome.verified)
        self.assertIsNotNone(real_outcome.nonce)
        self.assertIsNotNone(real_outcome.signature)


if __name__ == "__main__":
    unittest.main()
