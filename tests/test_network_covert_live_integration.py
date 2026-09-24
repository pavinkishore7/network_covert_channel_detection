"""Live integration test for the Phase 2 timing covert channel + detector.

NOT run by default -- same reason and same exclusion mechanism as
tests/test_network_live_integration.py (pytest.ini's ``addopts = -m "not
integration"``): this needs root/CAP_NET_ADMIN to create real network
namespaces plus tshark/tcpdump to capture real traffic. Run it explicitly:

    sudo venv/bin/python -m pytest tests/test_network_covert_live_integration.py -m integration -v

This is the live-topology counterpart to
tests/test_network_covert_demo.py's synthetic (in-memory) tests: it sends
a clean-only traffic plan and a covert-carrying plan (via
covert_demo.build_covert_traffic_plan) over the SAME real veth per slice,
captures each with capture.py, and asserts that
timing_detector.TimingKSDetector flags the covert capture as anomalous.

Privilege check performed when this test was written (2026-09-22, WSL2
dev environment, freshly re-verified in this session -- see
network_covert_channel/covert_demo.py's module docstring for the exact
commands run and their output): no root, no passwordless sudo, and
neither tcpdump nor tshark on PATH. This test was NOT run for real in
that session; it self-skips with a clear message if run without the
required privileges/tools, rather than failing confusingly mid-setup,
exactly like tests/test_network_live_integration.py already does for
Phase 1.
"""

from __future__ import annotations

import unittest

import pytest

from network_covert_channel.capture import preferred_capture_tool
from network_covert_channel.covert_demo import netns_privileges_available, run_live_demo
from slicing_sim.ofdm_grid import SLICE_TYPES

CAPTURE_DURATION_S = 15
PACKETS_PER_SLICE = 200
COVERT_OFFSET_S = 0.02  # large/easy offset -- this test checks the pipeline works live, not detection limits


@pytest.mark.integration
class LiveCovertChannelDemoTest(unittest.TestCase):
    """Requires root/CAP_NET_ADMIN and tshark/tcpdump. Skips itself with a
    clear message if run without them, reusing the same
    netns_privileges_available() probe run_live_demo() itself checks, so
    this test's skip condition can never drift from what would actually
    make run_live_demo() raise."""

    def setUp(self):
        available, reason = netns_privileges_available()
        if not available:
            self.skipTest(reason)
        if preferred_capture_tool() is None:
            self.skipTest("requires tshark or tcpdump on PATH to capture traffic; neither was found")

    def test_covert_traffic_is_flagged_anomalous_on_every_slice_over_real_topology(self):
        results = run_live_demo(
            n_packets=PACKETS_PER_SLICE,
            offset_s=COVERT_OFFSET_S,
            capture_duration_s=CAPTURE_DURATION_S,
        )
        self.assertEqual(set(results.keys()), set(SLICE_TYPES))
        for slice_type, result in results.items():
            with self.subTest(slice_type=slice_type):
                self.assertTrue(result.covert_flag, f"{slice_type}: covert traffic was not flagged live")


if __name__ == "__main__":
    unittest.main()
