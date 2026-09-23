"""Live run of integration/auth_over_topology.py on the real namespace topology.

NOT run by default (pytest.ini excludes ``integration``). Needs root or
CAP_NET_ADMIN, plus liboqs. Either of:

    sudo venv/bin/python -m pytest tests/test_auth_over_topology_integration.py -m integration -v -rs

    # no root: an unprivileged user+net namespace, with a private /run for `ip netns`
    unshare --user --map-root-user --mount --net sh -c \\
        'mount -t tmpfs tmpfs /run && venv/bin/python -m pytest tests/test_auth_over_topology_integration.py -m integration -v -rs'

Skips itself, with the reason, when namespaces can't be created -- the same
probe tests/test_network_live_integration.py uses
(network_covert_channel.topology.netns_privilege_skip_reason).

What is asserted is what the current code is supposed to guarantee:
challenge-bound replays rejected (including by a fresh client process),
slices served while idle attacker connections are held, the server
surviving malformed requests. The
link-failure runs are only checked for "the client recovered once the link
came back and nothing hung"; what the client does DURING the fault is
recorded in the report, not asserted, because changing it is later work.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from integration.auth_over_topology import AuthOverTopologyRunner, AuthTopologyConfig, build_plan, format_summary
from network_covert_channel.topology import NetnsTopology, netns_privilege_skip_reason
from slicing_sim.ofdm_grid import SLICE_TYPES


@pytest.mark.integration
class AuthOverTopologyLiveTest(unittest.TestCase):
    def setUp(self):
        skip_reason = netns_privilege_skip_reason()
        if skip_reason is not None:
            self.skipTest(skip_reason)
        try:
            import oqs  # noqa: F401  # type: ignore[import-not-found]
        except (ImportError, RuntimeError, SystemExit) as exc:
            self.skipTest(f"requires liboqs-python for the real OqsDilithiumSigner ({exc})")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_full_plan_on_real_topology(self):
        topo = NetnsTopology()
        cfg = AuthTopologyConfig(workdir=Path(self.tmp.name), rtt_samples=3)
        runner = AuthOverTopologyRunner(cfg, topo)
        try:
            report = runner.execute(build_plan(cfg, topo))
        finally:
            print(format_summary(runner.report))

        legit = [c for c in report["clients"] if c["group"] == "legit"]
        self.assertEqual(len(legit), 2 * len(SLICE_TYPES))
        for client in legit:
            self.assertEqual([r["result"]["trusted"] for r in client["requests"]], [True] * cfg.rtt_samples, client["name"])
            if client["trust_mode"] == "tofu":
                self.assertTrue(client["tofu_store_matches_oob_key"])

        replay = next(c for c in report["clients"] if c["group"] == "replay_same_process")["requests"]
        self.assertTrue(replay[0]["result"]["trusted"])
        self.assertTrue(replay[1]["result"]["challenge_mismatch"])
        self.assertFalse(replay[1]["result"]["trusted"])
        # the case that came back TRUSTED before wire version 2
        (fresh,) = next(c for c in report["clients"] if c["group"] == "replay_fresh_process")["requests"]
        self.assertFalse(fresh["result"]["trusted"])
        self.assertTrue(fresh["result"]["challenge_mismatch"])

        ready = report["ready_at"]["idle"]
        idle_clients = [c for c in report["clients"] if c["group"] == "idle_attack"]
        self.assertEqual(len(idle_clients), len(SLICE_TYPES))
        for client in idle_clients:
            self.assertTrue(client["requests"][0]["result"]["trusted"], client)
            # served while the attacker's connections were still held open
            self.assertLess(client["wall_end"] - ready, cfg.server_connection_timeout)
        idle_done = next(p for p in report["probes"] if "idle" in p["name"])["records"][-1]
        self.assertEqual(idle_done["closed_by_server"], cfg.idle_attack_connections)
        self.assertTrue(all(t >= cfg.server_connection_timeout for t in idle_done["closed_after_s"]))

        malformed = next(p for p in report["probes"] if p["name"].startswith("malformed"))
        self.assertTrue(malformed["server_alive_after"])
        self.assertEqual(len(malformed["records"]), 9)
        for client in (c for c in report["clients"] if c["group"] == "after_malformed"):
            self.assertTrue(client["requests"][0]["result"]["trusted"], client)
        self.assertNotIn("error:internal_error", report["server"]["events"])
        self.assertGreater(report["server"]["sign_ms"]["n"], 0)

        rogue = {c["trust_mode"]: c["requests"][0]["result"] for c in report["clients"] if c["group"] == "rogue"}
        self.assertFalse(rogue["pinned"]["trusted"])
        self.assertTrue(rogue["pinned"]["pinned_key_mismatch"])
        self.assertFalse(rogue["tofu"]["trusted"])
        self.assertTrue(rogue["tofu"]["trust_store_key_changed"])

        for client in report["clients"]:
            if not client["group"].startswith("link_"):
                continue
            self.assertIsNone(client.get("hung_after_s"), client["name"])
            phases = [r["phase"] for r in client["requests"]]
            self.assertIn("after_restore", phases, client["name"])
            for r in client["requests"]:
                if r["phase"] != "during_fault":
                    self.assertTrue(r["result"]["trusted"], (client["name"], r))

        self.assertEqual(report["audit"]["exit_code"], 0, report["audit"])
        self.assertEqual(report["teardown"]["leaked_netns"], [])
        self.assertEqual(report["teardown"]["still_running"], [])


if __name__ == "__main__":
    unittest.main()
