from __future__ import annotations

import unittest

import numpy as np

from network_covert_channel.covert_injector import (
    AdaptiveCovertInjector,
    CovertInjectorConfig,
    NonAdaptiveCovertInjector,
    inject_bits_into_gaps,
)


class InjectBitsIntoGapsTests(unittest.TestCase):
    def test_bit_zero_passes_through_unperturbed(self):
        clean = np.array([0.01, 0.01, 0.01])
        bits = np.array([0, 0, 0])
        perturbed = inject_bits_into_gaps(clean, bits, offset_s=0.05)
        np.testing.assert_array_equal(perturbed, clean)

    def test_bit_one_adds_the_scalar_offset(self):
        clean = np.array([0.01, 0.01, 0.01])
        bits = np.array([1, 0, 1])
        perturbed = inject_bits_into_gaps(clean, bits, offset_s=0.05)
        np.testing.assert_allclose(perturbed, [0.06, 0.01, 0.06])

    def test_per_position_offset_array_is_respected(self):
        clean = np.array([0.01, 0.01, 0.01])
        bits = np.array([1, 1, 0])
        offsets = np.array([0.02, 0.05, 0.10])
        perturbed = inject_bits_into_gaps(clean, bits, offset_s=offsets)
        np.testing.assert_allclose(perturbed, [0.03, 0.06, 0.01])

    def test_shorter_bit_sequence_leaves_remaining_gaps_unperturbed(self):
        clean = np.array([0.01, 0.01, 0.01, 0.01])
        bits = np.array([1, 1])
        perturbed = inject_bits_into_gaps(clean, bits, offset_s=0.05)
        np.testing.assert_allclose(perturbed, [0.06, 0.06, 0.01, 0.01])

    def test_does_not_mutate_the_input_array(self):
        clean = np.array([0.01, 0.01, 0.01])
        original = clean.copy()
        inject_bits_into_gaps(clean, np.array([1, 1, 1]), offset_s=0.05)
        np.testing.assert_array_equal(clean, original)

    def test_empty_bit_sequence_returns_gaps_unperturbed(self):
        clean = np.array([0.01, 0.01])
        perturbed = inject_bits_into_gaps(clean, np.array([]), offset_s=0.05)
        np.testing.assert_array_equal(perturbed, clean)

    def test_mismatched_offset_array_length_raises(self):
        clean = np.array([0.01, 0.01, 0.01])
        bits = np.array([1, 1, 1])
        with self.assertRaises(ValueError):
            inject_bits_into_gaps(clean, bits, offset_s=np.array([0.01, 0.02]))


class NonAdaptiveCovertInjectorTests(unittest.TestCase):
    def test_generate_covert_bits_is_deterministic_given_seed(self):
        injector = NonAdaptiveCovertInjector(CovertInjectorConfig(n_covert_bits=16, seed=7))
        bits_a = injector.generate_covert_bits()

        injector_again = NonAdaptiveCovertInjector(CovertInjectorConfig(n_covert_bits=16, seed=7))
        bits_b = injector_again.generate_covert_bits()

        np.testing.assert_array_equal(bits_a, bits_b)

    def test_inject_applies_fixed_offset_regardless_of_local_gap_scale(self):
        clean = np.array([0.001, 0.001, 0.06, 0.06])  # one tight pair, one wide pair
        injector = NonAdaptiveCovertInjector(CovertInjectorConfig(offset_s=0.01, seed=1))
        bits = np.array([1, 0, 1, 0])
        perturbed = injector.inject(clean, bits)
        # the SAME absolute offset is added at every bit=1 position, whether
        # the local gap scale there is tight (index 0) or wide (index 2)
        np.testing.assert_allclose(perturbed[0] - clean[0], perturbed[2] - clean[2])

    def test_inject_without_explicit_bits_generates_its_own(self):
        clean = np.full(16, 0.01)
        injector = NonAdaptiveCovertInjector(CovertInjectorConfig(n_covert_bits=16, offset_s=0.01, seed=3))
        perturbed = injector.inject(clean)
        self.assertEqual(len(perturbed), len(clean))
        self.assertTrue(np.any(perturbed != clean))  # seed=3 should produce at least one bit=1


class AdaptiveCovertInjectorTests(unittest.TestCase):
    def test_offset_never_exceeds_the_non_adaptive_ceiling(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(0.01, 0.005, size=200)
        clean = np.clip(clean, 1e-4, None)
        bits = np.ones(200, dtype=int)

        cfg = CovertInjectorConfig(offset_s=0.01, seed=0)
        adaptive = AdaptiveCovertInjector(cfg, window=10)
        perturbed = adaptive.inject(clean, bits)

        applied_offset = perturbed - clean
        self.assertTrue(np.all(applied_offset <= 0.01 + 1e-12))
        self.assertTrue(np.all(applied_offset >= 0.0))

    def test_offset_shrinks_in_a_high_local_jitter_region_vs_a_low_jitter_region(self):
        # First half: tight, steady gaps (low local jitter).
        # Second half: wide, highly variable gaps (high local jitter).
        low_jitter = np.full(60, 0.01)
        rng = np.random.default_rng(1)
        high_jitter = rng.exponential(0.03, size=60)
        clean = np.concatenate([low_jitter, high_jitter])
        bits = np.ones(120, dtype=int)

        cfg = CovertInjectorConfig(offset_s=0.01, seed=1)
        adaptive = AdaptiveCovertInjector(cfg, window=20, reference_jitter_s=np.std(low_jitter) or 1e-6)
        perturbed = adaptive.inject(clean, bits)
        applied_offset = perturbed - clean

        # offsets late in the low-jitter half should sit at (or very near)
        # the ceiling; offsets late in the high-jitter half should be
        # meaningfully smaller on average.
        low_region_mean = applied_offset[40:60].mean()
        high_region_mean = applied_offset[100:120].mean()
        self.assertGreater(low_region_mean, high_region_mean)

    def test_reference_jitter_estimated_from_input_when_not_given(self):
        clean = np.full(50, 0.01)
        bits = np.ones(50, dtype=int)
        cfg = CovertInjectorConfig(offset_s=0.01, seed=2)
        adaptive = AdaptiveCovertInjector(cfg, window=10)
        perturbed = adaptive.inject(clean, bits)
        # a perfectly uniform clean sequence has ~zero jitter everywhere,
        # so offsets should sit at (or very near) the ceiling throughout
        applied_offset = perturbed - clean
        self.assertTrue(np.all(applied_offset > 0.009))


if __name__ == "__main__":
    unittest.main()
