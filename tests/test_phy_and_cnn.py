import unittest

import numpy as np

from detector.legacy.convolutional_autoencoder import ConvolutionalPatchAutoencoder
from slicing_sim.mixed_numerology import MixedNumerologyRAN, Numerology


class PhyAndCnnTests(unittest.TestCase):
    def test_guard_band_and_cancellation_improve_sinr(self):
        no_guard = {name: Numerology(item.subcarrier_spacing_khz, item.allocated_subcarriers, 0)
                    for name, item in MixedNumerologyRAN(seed=3).numerologies.items()}
        guarded = {name: Numerology(item.subcarrier_spacing_khz, item.allocated_subcarriers, 2)
                   for name, item in MixedNumerologyRAN(seed=3).numerologies.items()}
        no_guard_sinr = MixedNumerologyRAN(seed=3, numerologies=no_guard).evaluate()[0]["sinr_db"]
        guarded_sinr = MixedNumerologyRAN(seed=3, numerologies=guarded).evaluate()[0]["sinr_db"]
        cancelled_sinr = MixedNumerologyRAN(seed=3, numerologies=no_guard).evaluate(cancellation_efficiency=0.75)[0]["sinr_db"]
        self.assertGreater(guarded_sinr, no_guard_sinr)
        self.assertGreater(cancelled_sinr, no_guard_sinr)

    def test_convolutional_autoencoder_flags_local_anomaly(self):
        rng = np.random.default_rng(5)
        train = rng.normal(0, 0.1, size=(20, 16, 16))
        valid = rng.normal(0, 0.1, size=(10, 16, 16))
        detector = ConvolutionalPatchAutoencoder(percentile=95).fit(train)
        detector.calibrate(valid)
        attacked = valid.copy()
        attacked[:, 5:8, 5:8] += 4.0
        self.assertGreater(detector.predict(attacked).mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
