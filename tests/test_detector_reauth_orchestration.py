import unittest

from pqc_auth.orchestration import drive_reauth_from_detector_flags
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason

from tests.fake_signer import FakeSigner


class DetectorReauthOrchestrationTests(unittest.TestCase):
    def test_slice_with_no_alerts_only_reauths_on_periodic_schedule(self):
        controller = DualTriggerReauthController()  # URLLC interval = 30s
        timeline = [0, 10, 20, 30, 45]
        reasons = []
        for now in timeline:
            decisions = drive_reauth_from_detector_flags({"URLLC": False}, controller, now)
            reasons.append(decisions[0].outcome.reason if decisions else None)
        self.assertEqual(reasons, [ReauthReason.PERIODIC, None, None, ReauthReason.PERIODIC, None])

    def test_slice_with_mid_cycle_alert_reauths_early_respecting_cooldown(self):
        controller = DualTriggerReauthController()  # URLLC: interval=30s, alert_cooldown=10s
        first = drive_reauth_from_detector_flags({"URLLC": False}, controller, 0)
        self.assertEqual(first[0].outcome.reason, ReauthReason.PERIODIC)

        # Detector alert fires well before the 30s periodic interval elapses.
        alerted = drive_reauth_from_detector_flags({"URLLC": True}, controller, 5)
        self.assertEqual(alerted[0].outcome.reason, ReauthReason.DETECTOR_ALERT)

        # Still inside the 10s alert cooldown -> not due again yet.
        cooldown = drive_reauth_from_detector_flags({"URLLC": True}, controller, 8)
        self.assertEqual(cooldown, [])

        # Cooldown has elapsed (16 - 5 = 11 >= 10) -> alerts again.
        after_cooldown = drive_reauth_from_detector_flags({"URLLC": True}, controller, 16)
        self.assertEqual(after_cooldown[0].outcome.reason, ReauthReason.DETECTOR_ALERT)

    def test_multiple_slices_handled_independently(self):
        controller = DualTriggerReauthController()
        decisions = drive_reauth_from_detector_flags(
            {"URLLC": False, "eMBB": False, "mMTC": False}, controller, 0
        )
        slice_types = {d.slice_type for d in decisions}
        self.assertEqual(slice_types, {"URLLC", "eMBB", "mMTC"})
        self.assertTrue(all(d.outcome.reason == ReauthReason.PERIODIC for d in decisions))

    def test_with_signer_configured_reports_true_verified_outcome(self):
        controller = DualTriggerReauthController(signer=FakeSigner())
        decisions = drive_reauth_from_detector_flags({"URLLC": False}, controller, 0)
        self.assertTrue(decisions[0].outcome.verified)


if __name__ == "__main__":
    unittest.main()
