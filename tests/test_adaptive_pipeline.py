import unittest

import numpy as np

from detector.legacy.adaptive_residual import AdaptiveResidualDetector
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason


class AdaptivePipelineTests(unittest.TestCase):
    def test_clean_calibration_rejects_stronger_slice_shift(self):
        rng = np.random.default_rng(7)
        train = rng.normal(0, 1, size=(40, 12, 8))
        valid = rng.normal(0, 1, size=(20, 12, 8))
        masks = np.ones_like(train, dtype=bool)
        detector = AdaptiveResidualDetector(percentile=95).fit(train, masks)
        detector.calibrate(valid, np.ones_like(valid, dtype=bool))
        attacked = valid.copy()
        attacked[:, :4, :4] += 4.0
        self.assertGreater(detector.predict(attacked, np.ones_like(attacked, dtype=bool)).mean(), 0.9)

    def test_dual_trigger_keeps_periodic_backstop(self):
        controller = DualTriggerReauthController()
        self.assertEqual(controller.due("URLLC", 0), ReauthReason.PERIODIC)
        self.assertEqual(controller.due("URLLC", 5, detector_alert=True), ReauthReason.DETECTOR_ALERT)
        self.assertIsNone(controller.due("URLLC", 8, detector_alert=True))
        self.assertEqual(controller.due("URLLC", 35), ReauthReason.PERIODIC)


if __name__ == "__main__":
    unittest.main()
