from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from network_covert_channel.covert_demo import (
    SliceAccuracySweepResult,
    SliceDemoResult,
    build_covert_traffic_plan,
    netns_privileges_available,
    run_live_demo,
    run_synthetic_accuracy_sweep,
    run_synthetic_demo,
    run_synthetic_slice_demo,
)
from network_covert_channel.covert_injector import CovertInjectorConfig, NonAdaptiveCovertInjector
from network_covert_channel.traffic import build_traffic_plan
from slicing_sim.ofdm_grid import SLICE_TYPES


class BuildCovertTrafficPlanTests(unittest.TestCase):
    def test_only_gap_before_s_is_perturbed_other_fields_pass_through(self):
        clean_plan = build_traffic_plan(
            "URLLC", 20, src_ip="10.200.0.11", dst_ip="10.200.0.1", rng=np.random.default_rng(1)
        )
        injector = NonAdaptiveCovertInjector(CovertInjectorConfig(n_covert_bits=20, offset_s=0.05, seed=1))
        covert_plan, bits = build_covert_traffic_plan(clean_plan, injector)

        self.assertEqual(len(covert_plan), len(clean_plan))
        for clean_entry, covert_entry, bit in zip(clean_plan, covert_plan, bits):
            self.assertEqual(covert_entry.slice_type, clean_entry.slice_type)
            self.assertEqual(covert_entry.src_ip, clean_entry.src_ip)
            self.assertEqual(covert_entry.dst_ip, clean_entry.dst_ip)
            self.assertEqual(covert_entry.src_port, clean_entry.src_port)
            self.assertEqual(covert_entry.dst_port, clean_entry.dst_port)
            self.assertEqual(covert_entry.payload_bytes, clean_entry.payload_bytes)
            if bit:
                self.assertAlmostEqual(covert_entry.gap_before_s, clean_entry.gap_before_s + 0.05, places=9)
            else:
                self.assertAlmostEqual(covert_entry.gap_before_s, clean_entry.gap_before_s, places=9)

    def test_does_not_mutate_the_clean_plan(self):
        clean_plan = build_traffic_plan(
            "eMBB", 10, src_ip="10.200.0.12", dst_ip="10.200.0.1", rng=np.random.default_rng(2)
        )
        original_gaps = [e.gap_before_s for e in clean_plan]
        injector = NonAdaptiveCovertInjector(CovertInjectorConfig(n_covert_bits=10, offset_s=0.05, seed=2))
        build_covert_traffic_plan(clean_plan, injector)
        self.assertEqual([e.gap_before_s for e in clean_plan], original_gaps)


class SyntheticDemoTests(unittest.TestCase):
    def test_run_synthetic_slice_demo_returns_a_populated_result(self):
        result = run_synthetic_slice_demo("URLLC", n_packets=100, offset_s=0.02, seed=1)
        self.assertIsInstance(result, SliceDemoResult)
        self.assertEqual(result.slice_type, "URLLC")
        self.assertGreater(result.threshold, 0.0)
        self.assertIsInstance(result.clean_flag, bool)
        self.assertIsInstance(result.covert_flag, bool)

    def test_run_synthetic_demo_covers_every_slice_type(self):
        results = run_synthetic_demo(offset_s=0.02, n_packets=100, seed=1)
        self.assertEqual(set(results.keys()), set(SLICE_TYPES))

    def test_large_offset_end_to_end_run_flags_covert_traffic_on_every_slice(self):
        # 20ms is well within the range test_network_timing_detector.py's
        # DetectionAccuracyTests measures as >=90% reliable on every slice.
        results = run_synthetic_demo(offset_s=0.02, n_packets=300, seed=2026)
        for slice_type, result in results.items():
            with self.subTest(slice_type=slice_type):
                self.assertTrue(result.covert_flag, f"{slice_type} covert traffic was not flagged at a large offset")


class SyntheticAccuracySweepTests(unittest.TestCase):
    def test_sweep_returns_a_result_per_slice_and_offset_label_with_rates_in_bounds(self):
        offsets = {"large": 0.02, "marginal": 0.0006}
        sweep = run_synthetic_accuracy_sweep(offsets, n_trials=10, n_packets=100, seed_base=1)
        self.assertEqual(set(sweep.keys()), {(s, label) for s in SLICE_TYPES for label in offsets})
        for result in sweep.values():
            self.assertIsInstance(result, SliceAccuracySweepResult)
            self.assertGreaterEqual(result.detection_rate, 0.0)
            self.assertLessEqual(result.detection_rate, 1.0)
            self.assertGreaterEqual(result.false_alarm_rate, 0.0)
            self.assertLessEqual(result.false_alarm_rate, 1.0)


class PrivilegeProbeTests(unittest.TestCase):
    def test_netns_privileges_available_returns_a_bool_and_a_message(self):
        available, message = netns_privileges_available()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)


class RunLiveDemoGuardTests(unittest.TestCase):
    """run_live_demo must refuse to touch NetnsTopology at all when
    privileges or a capture tool are unavailable -- these tests patch
    both checks directly rather than depending on this machine's actual
    privilege state, so they're deterministic in CI either way."""

    def test_raises_without_touching_topology_when_privileges_unavailable(self):
        with mock.patch(
            "network_covert_channel.covert_demo.netns_privileges_available",
            return_value=(False, "no CAP_NET_ADMIN"),
        ), mock.patch("network_covert_channel.covert_demo.NetnsTopology") as mock_topo:
            with self.assertRaisesRegex(RuntimeError, "no CAP_NET_ADMIN"):
                run_live_demo()
            mock_topo.assert_not_called()

    def test_raises_without_touching_topology_when_no_capture_tool(self):
        with mock.patch(
            "network_covert_channel.covert_demo.netns_privileges_available", return_value=(True, "ok")
        ), mock.patch(
            "network_covert_channel.covert_demo.preferred_capture_tool", return_value=None
        ), mock.patch("network_covert_channel.covert_demo.NetnsTopology") as mock_topo:
            with self.assertRaisesRegex(RuntimeError, "tshark or tcpdump"):
                run_live_demo()
            mock_topo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
