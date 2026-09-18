"""Tests for network_covert_channel/capture.py.

`parse_pcap_to_dataframe` is tested against a synthetic pcap this test
builds and writes with scapy itself -- no live capture, no tshark/tcpdump
binary, no root required. `run_capture`/`build_capture_cmd` are tested via
pure command construction and an injected fake runner.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scapy.all import IP, UDP, Raw, wrpcap

from network_covert_channel.capture import (
    build_capture_cmd,
    parse_pcap_to_dataframe,
    pcap_filename,
    run_capture,
)


class PcapFilenameTests(unittest.TestCase):
    def test_filename_embeds_slice_and_timestamp(self):
        name = pcap_filename("URLLC", timestamp=1700000000.0)
        self.assertTrue(name.startswith("urllc_"))
        self.assertTrue(name.endswith(".pcap"))
        self.assertRegex(name, r"^urllc_\d{8}T\d{6}Z\.pcap$")


class BuildCaptureCmdTests(unittest.TestCase):
    def test_tshark_command_includes_duration_and_filter(self):
        cmd = build_capture_cmd("tshark", "veth-urllc-h", "/tmp/out.pcap", duration_s=15, capture_filter="udp")
        self.assertEqual(cmd[0], "tshark")
        self.assertIn("-i", cmd)
        self.assertIn("veth-urllc-h", cmd)
        self.assertIn("-w", cmd)
        self.assertIn("/tmp/out.pcap", cmd)
        self.assertIn("duration:15", cmd)
        self.assertIn("udp", cmd)

    def test_tcpdump_command_includes_filter(self):
        cmd = build_capture_cmd("tcpdump", "veth-embb-h", "/tmp/out2.pcap", capture_filter="udp")
        self.assertEqual(cmd[0], "tcpdump")
        self.assertIn("veth-embb-h", cmd)
        self.assertIn("/tmp/out2.pcap", cmd)
        self.assertIn("udp", cmd)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            build_capture_cmd("wireshark-gui", "eth0", "/tmp/x.pcap")


class RunCaptureTests(unittest.TestCase):
    def test_run_capture_invokes_runner_with_built_command_and_timeout(self):
        seen = {}

        def fake_runner(cmd, timeout=None, capture_output=None, text=None):
            seen["cmd"] = cmd
            seen["timeout"] = timeout
            return subprocess.CompletedProcess(cmd, 0, "", "")

        result = run_capture("tshark", "veth-mmtc-h", "/tmp/o.pcap", duration_s=10, runner=fake_runner)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(seen["cmd"][0], "tshark")
        self.assertEqual(seen["timeout"], 15)

    def test_tcpdump_timeout_expiry_is_treated_as_normal_stop(self):
        def timing_out_runner(cmd, timeout=None, capture_output=None, text=None):
            raise subprocess.TimeoutExpired(cmd, timeout)

        result = run_capture("tcpdump", "veth-urllc-h", "/tmp/o2.pcap", duration_s=5, runner=timing_out_runner)
        self.assertEqual(result.returncode, 0)


class ParsePcapTests(unittest.TestCase):
    def test_parses_synthetic_pcap_into_expected_dataframe(self):
        pkt1 = IP(src="10.200.0.11", dst="10.200.0.1") / UDP(sport=1000, dport=2000) / Raw(load=b"a" * 10)
        pkt2 = IP(src="10.200.0.11", dst="10.200.0.1") / UDP(sport=1000, dport=2000) / Raw(load=b"b" * 20)
        pkt1.time = 1000.0
        pkt2.time = 1000.25

        with tempfile.TemporaryDirectory() as tmpdir:
            pcap_path = Path(tmpdir) / "synthetic.pcap"
            wrpcap(str(pcap_path), [pkt1, pkt2])

            df = parse_pcap_to_dataframe(pcap_path)

        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns), ["timestamp", "size", "src", "dst", "inter_arrival_s"]
        )
        self.assertEqual(df.iloc[0]["src"], "10.200.0.11")
        self.assertEqual(df.iloc[0]["dst"], "10.200.0.1")
        self.assertTrue(df.iloc[0]["inter_arrival_s"] != df.iloc[0]["inter_arrival_s"])  # NaN
        self.assertAlmostEqual(df.iloc[1]["inter_arrival_s"], 0.25, places=3)
        self.assertGreater(df.iloc[1]["size"], df.iloc[0]["size"])  # pkt2 has a bigger payload


if __name__ == "__main__":
    unittest.main()
