"""Tests for network_covert_channel/topology.py.

None of these require root / CAP_NET_ADMIN or a real network namespace --
every test injects a fake command runner and asserts on the constructed
argv, matching this project's requirement that the unit test suite never
needs elevated privileges to pass.
"""

from __future__ import annotations

import subprocess
import unittest

from network_covert_channel.topology import (
    TOTAL_LINK_CAPACITY_MBIT,
    NetnsTopology,
    TopologyCommandError,
    derive_qos_profile,
)
from slicing_sim.ofdm_grid import SLICE_PROFILES, SLICE_TYPES


class FakeRunner:
    """Records every argv it's called with. ``fail_on`` maps a command's
    first two argv elements (e.g. ("ip", "netns")) to a returncode/stderr
    to simulate a specific command failing, so tests can exercise
    setup()'s rollback path deterministically."""

    def __init__(self, fail_on: dict[tuple[str, str], tuple[int, str]] | None = None):
        self.calls: list[list[str]] = []
        self._fail_on = fail_on or {}

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        key = tuple(argv[:2])
        if key in self._fail_on:
            code, stderr = self._fail_on[key]
            return subprocess.CompletedProcess(argv, returncode=code, stdout="", stderr=stderr)
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")


class QoSProfileTests(unittest.TestCase):
    def test_rejects_unknown_slice_type(self):
        with self.assertRaises(ValueError):
            derive_qos_profile("not-a-slice")

    def test_rate_mbit_matches_subcarrier_frac_proportion(self):
        profiles = {s: derive_qos_profile(s) for s in SLICE_TYPES}
        for slice_type in SLICE_TYPES:
            expected = SLICE_PROFILES[slice_type]["subcarrier_frac"] * TOTAL_LINK_CAPACITY_MBIT
            self.assertAlmostEqual(profiles[slice_type].rate_mbit, expected, places=3)
        # subcarrier_frac sums to 1.0 across slices, so rates should sum to
        # the total assumed link capacity.
        total = sum(p.rate_mbit for p in profiles.values())
        self.assertAlmostEqual(total, TOTAL_LINK_CAPACITY_MBIT, places=3)

    def test_urllc_has_tightest_latency_and_highest_priority(self):
        urllc = derive_qos_profile("URLLC")
        embb = derive_qos_profile("eMBB")
        mmtc = derive_qos_profile("mMTC")
        # URLLC has the lowest burstiness in SLICE_PROFILES -> should map
        # to the smallest delay/jitter/loss and the highest-priority band.
        self.assertLess(urllc.delay_ms, embb.delay_ms)
        self.assertLess(embb.delay_ms, mmtc.delay_ms)
        self.assertLess(urllc.jitter_ms, mmtc.jitter_ms)
        self.assertLess(urllc.loss_pct, mmtc.loss_pct)
        self.assertEqual(urllc.priority_band, 0)
        self.assertEqual(mmtc.priority_band, 2)


class NamingAndAddressingTests(unittest.TestCase):
    def setUp(self):
        self.topo = NetnsTopology(runner=FakeRunner())

    def test_netns_names_are_distinct_and_prefixed(self):
        names = {self.topo.netns_name(s) for s in SLICE_TYPES}
        self.assertEqual(len(names), 3)
        for name in names:
            self.assertTrue(name.startswith("ns-"))

    def test_veth_names_within_ifnamsiz_limit(self):
        for slice_type in SLICE_TYPES:
            host_if, ns_if = self.topo.veth_names(slice_type)
            self.assertLessEqual(len(host_if), 15)
            self.assertLessEqual(len(ns_if), 15)
            self.assertNotEqual(host_if, ns_if)

    def test_ip_addresses_are_distinct(self):
        ips = {self.topo.ip_for(s) for s in SLICE_TYPES}
        self.assertEqual(len(ips), 3)
        for ip in ips:
            self.assertTrue(ip.startswith(self.topo.subnet_base))


class CommandConstructionTests(unittest.TestCase):
    def setUp(self):
        self.topo = NetnsTopology(runner=FakeRunner())

    def test_build_create_bridge_cmds(self):
        cmds = self.topo.build_create_bridge_cmds()
        self.assertEqual(cmds[0], ["ip", "link", "add", "br-ncc0", "type", "bridge"])
        self.assertIn(["ip", "link", "set", "br-ncc0", "up"], cmds)

    def test_build_tc_shaping_cmds_uses_profile_values(self):
        profile = self.topo.qos_profile("URLLC")
        cmds = self.topo.build_tc_shaping_cmds("URLLC")
        netem_cmd = cmds[0]
        self.assertIn("netem", netem_cmd)
        self.assertIn(f"{profile.delay_ms}ms", netem_cmd)
        self.assertIn(f"{profile.loss_pct}%", netem_cmd)
        tbf_cmd = cmds[1]
        self.assertIn("tbf", tbf_cmd)
        self.assertIn(f"{profile.rate_mbit}mbit", tbf_cmd)

    def test_build_setup_cmds_orders_bridge_before_slices(self):
        cmds = self.topo.build_setup_cmds()
        bridge_idx = cmds.index(["ip", "link", "add", "br-ncc0", "type", "bridge"])
        first_netns_idx = next(
            i for i, c in enumerate(cmds) if c[:3] == ["ip", "netns", "add"]
        )
        self.assertLess(bridge_idx, first_netns_idx)

    def test_build_teardown_cmds_deletes_every_slice_and_the_bridge(self):
        cmds = self.topo.build_teardown_cmds()
        for slice_type in SLICE_TYPES:
            self.assertIn(["ip", "netns", "delete", self.topo.netns_name(slice_type)], cmds)
        self.assertIn(["ip", "link", "delete", "br-ncc0"], cmds)


class SetupTeardownExecutionTests(unittest.TestCase):
    def test_setup_runs_every_constructed_command_via_the_injected_runner(self):
        runner = FakeRunner()
        topo = NetnsTopology(runner=runner)
        topo.setup()
        self.assertEqual(runner.calls, topo.build_setup_cmds())

    def test_teardown_is_safe_when_setup_was_never_called(self):
        # Simulates every delete failing (nothing exists yet) -- teardown
        # must not raise.
        runner = FakeRunner(fail_on={
            ("ip", "netns"): (1, "Cannot remove namespace file"),
            ("ip", "link"): (1, "Cannot find device"),
        })
        topo = NetnsTopology(runner=runner)
        topo.teardown()  # must not raise
        self.assertTrue(len(runner.calls) > 0)

    def test_setup_failure_triggers_rollback_teardown(self):
        # Fail the veth-creation step; setup() must raise but also must
        # have attempted a full teardown before re-raising.
        class SelectiveFailRunner(FakeRunner):
            def __call__(self, argv):
                self.calls.append(argv)
                if argv[:3] == ["ip", "link", "add"] and "veth" in argv:
                    return subprocess.CompletedProcess(argv, 1, "", "simulated veth failure")
                return subprocess.CompletedProcess(argv, 0, "", "")

        runner = SelectiveFailRunner()
        topo = NetnsTopology(runner=runner)
        with self.assertRaises(TopologyCommandError):
            topo.setup()
        # The bridge delete (part of teardown) must have been attempted.
        self.assertIn(["ip", "link", "delete", topo.bridge_name], runner.calls)

    def test_teardown_never_raises_even_if_runner_itself_raises(self):
        def exploding_runner(argv):
            raise OSError("no such command")

        topo = NetnsTopology(runner=exploding_runner)
        topo.teardown()  # must not raise


if __name__ == "__main__":
    unittest.main()
