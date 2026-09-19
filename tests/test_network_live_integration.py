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

from network_covert_channel.capture import build_capture_cmd, parse_pcap_to_dataframe, pcap_filename, preferred_capture_tool
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
        try:
            probe = subprocess.run(
                ["ip", "netns", "add", "__ncc_priv_probe__"], capture_output=True, text=True
            )
        except OSError as exc:
            # `ip` itself isn't on PATH at all -- a different, earlier
            # failure than "found ip but lack CAP_NET_ADMIN" (below), and
            # worth telling apart in the skip message: one means "install
            # iproute2", the other means "run as root/with the capability".
            self.skipTest(f"requires the 'ip' binary (iproute2), not found on PATH ({exc})")
        if probe.returncode != 0:
            self.skipTest(
                "requires root/CAP_NET_ADMIN to create network namespaces "
                f"(probe failed: {probe.stderr.strip()})"
            )
        subprocess.run(["ip", "netns", "delete", "__ncc_priv_probe__"], capture_output=True)

        self.topo = NetnsTopology()
        self.addCleanup(self.topo.teardown)

    def test_three_slices_produce_distinguishable_timing_distributions(self):
        capture_tool = preferred_capture_tool()
        if capture_tool is None:
            self.skipTest("requires tshark or tcpdump on PATH to capture traffic; neither was found")

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

            cmd = build_capture_cmd(capture_tool, host_if, pcap_paths[slice_type], duration_s=CAPTURE_DURATION_S)
            capture_handle = subprocess.Popen(cmd)
            time.sleep(1)  # let the capture tool attach before traffic starts

            send_traffic_plan(plan, iface=host_if)
            try:
                capture_handle.wait(timeout=CAPTURE_DURATION_S + 10)
            except subprocess.TimeoutExpired:
                # build_capture_cmd only gives tshark a self-stop flag
                # (`-a duration:N`); tcpdump has no equivalent (see
                # capture.py's run_capture docstring), so if
                # preferred_capture_tool() picked tcpdump here it will
                # still be running at this point -- make sure it doesn't
                # leak as a background process now that this path is
                # reachable (it never was while the tool was hardcoded to
                # tshark, which always self-terminates).
                capture_handle.kill()
                capture_handle.wait()

        distributions = {}
        for slice_type, path in pcap_paths.items():
            df = parse_pcap_to_dataframe(path)
            gaps = df["inter_arrival_s"].dropna()
            distributions[slice_type] = float(np.std(gaps) / np.mean(gaps))  # coefficient of variation

        self.assertLess(distributions["URLLC"], distributions["eMBB"])
        self.assertLess(distributions["eMBB"], distributions["mMTC"])


if __name__ == "__main__":
    unittest.main()
