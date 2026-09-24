"""Tests for network_covert_channel/traffic.py.

No real socket send, no live interface, no root needed: send_traffic_plan
is tested with a fake sender and a fake sleeper (no real wall-clock
delay), and everything else is pure computation.
"""

from __future__ import annotations

import unittest

import numpy as np

from network_covert_channel.traffic import (
    DEFAULT_PAYLOAD_BYTES,
    TrafficPacketPlan,
    build_packet,
    build_traffic_plan,
    generate_inter_packet_gaps,
)
from slicing_sim.ofdm_grid import SLICE_TYPES


class InterPacketGapTests(unittest.TestCase):
    def test_rejects_unknown_slice_type(self):
        with self.assertRaises(ValueError):
            generate_inter_packet_gaps("not-a-slice", 10, np.random.default_rng(0))

    def test_rejects_non_positive_packet_count(self):
        with self.assertRaises(ValueError):
            generate_inter_packet_gaps("URLLC", 0, np.random.default_rng(0))

    def test_gaps_are_positive(self):
        gaps = generate_inter_packet_gaps("mMTC", 200, np.random.default_rng(1))
        self.assertTrue(np.all(gaps > 0))

    def test_deterministic_given_seeded_rng(self):
        gaps_a = generate_inter_packet_gaps("eMBB", 50, np.random.default_rng(42))
        gaps_b = generate_inter_packet_gaps("eMBB", 50, np.random.default_rng(42))
        np.testing.assert_array_equal(gaps_a, gaps_b)

    def test_burstiness_visibly_increases_timing_variance(self):
        # Same seed sequence reused per slice type (each call gets a fresh
        # generator so results are directly comparable). URLLC has the
        # lowest burstiness in SLICE_PROFILES and mMTC the highest, so the
        # coefficient of variation of inter-packet gaps should be ordered
        # the same way.
        n = 2000
        cv_by_slice = {}
        for slice_type in SLICE_TYPES:
            gaps = generate_inter_packet_gaps(slice_type, n, np.random.default_rng(7))
            cv_by_slice[slice_type] = np.std(gaps) / np.mean(gaps)
        self.assertLess(cv_by_slice["URLLC"], cv_by_slice["eMBB"])
        self.assertLess(cv_by_slice["eMBB"], cv_by_slice["mMTC"])


class TrafficPlanTests(unittest.TestCase):
    def test_build_traffic_plan_produces_requested_packet_count(self):
        plan = build_traffic_plan(
            "URLLC", 25, "10.200.0.11", "10.200.0.1", rng=np.random.default_rng(3)
        )
        self.assertEqual(len(plan), 25)
        for entry in plan:
            self.assertIsInstance(entry, TrafficPacketPlan)
            self.assertEqual(entry.src_ip, "10.200.0.11")
            self.assertEqual(entry.dst_ip, "10.200.0.1")
            self.assertEqual(entry.payload_bytes, DEFAULT_PAYLOAD_BYTES)
            self.assertGreater(entry.gap_before_s, 0)

    def test_build_packet_sets_ip_and_udp_fields(self):
        entry = TrafficPacketPlan(
            slice_type="eMBB",
            src_ip="10.200.0.12",
            dst_ip="10.200.0.1",
            src_port=51000,
            dst_port=52000,
            payload_bytes=32,
            gap_before_s=0.001,
        )
        pkt = build_packet(entry)
        self.assertEqual(pkt["IP"].src, "10.200.0.12")
        self.assertEqual(pkt["IP"].dst, "10.200.0.1")
        self.assertEqual(pkt["UDP"].sport, 51000)
        self.assertEqual(pkt["UDP"].dport, 52000)
        self.assertEqual(len(pkt["Raw"].load), 32)


class SendTrafficPlanTests(unittest.TestCase):
    def test_calls_sender_once_per_entry_and_sleeps_the_configured_gap(self):
        from network_covert_channel.traffic import send_traffic_plan

        plan = [
            TrafficPacketPlan("URLLC", "10.0.0.1", "10.0.0.2", 1, 2, 8, 0.01),
            TrafficPacketPlan("URLLC", "10.0.0.1", "10.0.0.2", 1, 2, 8, 0.02),
        ]
        sent = []
        slept = []
        send_traffic_plan(plan, sender=sent.append, sleeper=slept.append)
        self.assertEqual(len(sent), 2)
        self.assertEqual(slept, [0.01, 0.02])


if __name__ == "__main__":
    unittest.main()


# --- steady_jitter_frac (added 2026-09-24) ---------------------------------
import hashlib as _hashlib

import numpy as _np

from network_covert_channel.traffic import generate_inter_packet_gaps as _gen

# SHA-256 of generate_inter_packet_gaps(slice, 1000, default_rng(2026)) recorded
# BEFORE the steady_jitter_frac parameter was added: the default must not change output.
_PRE_EDIT_HASHES = {
    "URLLC": "e37855b61dc910af395ff0e2af03fe860e318ec43d84b7abce693a492e0ec340",
    "eMBB": "d61e45305cc5a07328b92b205326466cc8c11ebd61227d5b5fe93e981e60821f",
    "mMTC": "7fb25b8698dc67fb29b930af26bcd5159b9b551618793272f3c30540528c9732",
}


def test_default_steady_jitter_reproduces_pre_edit_output_bit_for_bit():
    for s, h in _PRE_EDIT_HASHES.items():
        assert _hashlib.sha256(_gen(s, 1000, _np.random.default_rng(2026)).tobytes()).hexdigest() == h
        assert _hashlib.sha256(_gen(s, 1000, _np.random.default_rng(2026), steady_jitter_frac=0.05).tobytes()).hexdigest() == h


def test_larger_steady_jitter_widens_urllc_spread():
    narrow = _gen("URLLC", 20000, _np.random.default_rng(1), steady_jitter_frac=0.05)
    wide = _gen("URLLC", 20000, _np.random.default_rng(1), steady_jitter_frac=0.5)
    assert wide.std() > narrow.std()


def test_negative_steady_jitter_rejected():
    import pytest
    with pytest.raises(ValueError):
        _gen("URLLC", 10, _np.random.default_rng(0), steady_jitter_frac=-0.1)
