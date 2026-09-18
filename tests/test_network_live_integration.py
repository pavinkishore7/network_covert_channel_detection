"""Live integration test for the network-layer covert-channel testbed.

NOT run by default. Excluded by pytest.ini's ``addopts = -m "not
integration"`` because it needs root / CAP_NET_ADMIN to create real Linux
network namespaces, veth pairs, and tc qdiscs -- privileges this project's
main test suite is required to run without (see
network_covert_channel/README.md). Run it explicitly, as root, on a real
Linux machine:

    sudo venv/bin/python -m pytest tests/test_network_live_integration.py -m integration -v

This is the actual Phase 1 review-checkpoint evidence: it stands up the
three-namespace topology, generates ~15s of traffic on all three slices
concurrently, captures each slice's veth, and asserts the resulting pcaps
show three distinguishable inter-arrival-time distributions (URLLC
tightest/most uniform, mMTC loosest/most bursty) -- the property the whole
downstream covert-channel-in-timing threat model depends on existing in
the first place.

This test was NOT run in the session that wrote it: that environment (WSL2)
had neither root nor passwordless sudo (`ip netns add` failed with
"Permission denied", `sudo -n true` demanded a password) -- see step 0's
findings in network_covert_channel/README.md. It is designed and ready to
run, not verified against a real kernel.
"""

from __future__ import annotations

import subprocess
import time
import unittest

import numpy as np
import pytest

from network_covert_channel.capture import parse_pcap_to_dataframe, pcap_filename
from network_covert_channel.topology import NetnsTopology
from network_covert_channel.traffic import build_traffic_plan, send_traffic_plan
from slicing_sim.ofdm_grid import SLICE_TYPES

CAPTURE_DURATION_S = 15
PACKETS_PER_SLICE = 200


@pytest.mark.integration
class LiveTopologyDemoTest(unittest.TestCase):
    """Requires root/CAP_NET_ADMIN. Skips itself with a clear message if
    run without them, rather than failing confusingly mid-setup."""

    def setUp(self):
        probe = subprocess.run(
            ["ip", "netns", "add", "__ncc_priv_probe__"], capture_output=True, text=True
        )
        if probe.returncode != 0:
            self.skipTest(
                "requires root/CAP_NET_ADMIN to create network namespaces "
                f"(probe failed: {probe.stderr.strip()})"
            )
        subprocess.run(["ip", "netns", "delete", "__ncc_priv_probe__"], capture_output=True)

        self.topo = NetnsTopology()
        self.addCleanup(self.topo.teardown)

    def test_three_slices_produce_distinguishable_timing_distributions(self):
        self.topo.setup()

        pcap_paths = {}
        rng = np.random.default_rng(2026)
        for slice_type in SLICE_TYPES:
            host_if, _ = self.topo.veth_names(slice_type)
            pcap_paths[slice_type] = f"/tmp/{pcap_filename(slice_type)}"

            plan = build_traffic_plan(
                slice_type,
                PACKETS_PER_SLICE,
                src_ip=self.topo.ip_for(slice_type),
                dst_ip=self.topo.bridge_ip(),
                rng=rng,
            )

            cmd = ["tshark", "-i", host_if, "-w", pcap_paths[slice_type], "-a", f"duration:{CAPTURE_DURATION_S}"]
            capture_handle = subprocess.Popen(cmd)
            time.sleep(1)  # let tshark attach before traffic starts

            send_traffic_plan(plan, iface=host_if)
            capture_handle.wait(timeout=CAPTURE_DURATION_S + 10)

        distributions = {}
        for slice_type, path in pcap_paths.items():
            df = parse_pcap_to_dataframe(path)
            gaps = df["inter_arrival_s"].dropna()
            distributions[slice_type] = float(np.std(gaps) / np.mean(gaps))  # coefficient of variation

        self.assertLess(distributions["URLLC"], distributions["eMBB"])
        self.assertLess(distributions["eMBB"], distributions["mMTC"])


if __name__ == "__main__":
    unittest.main()
