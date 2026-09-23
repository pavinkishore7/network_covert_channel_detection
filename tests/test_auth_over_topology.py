"""Tests for integration/auth_over_topology.py.

None of these need root / CAP_NET_ADMIN: the plan builders are pure, and
the runner is exercised with an injected fake ``run`` and ``popen`` -- same
convention as tests/test_network_topology.py. The real-topology run is
tests/test_auth_over_topology_integration.py (marked ``integration``).
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from integration.auth_over_topology import (
    CORE,
    ENDPOINTS,
    ROGUE,
    AuthOverTopologyRunner,
    AuthTopologyConfig,
    PlanStepError,
    Step,
    build_client_cmd,
    build_endpoint_setup_cmds,
    build_link_fault_cmds,
    build_link_restore_cmds,
    build_plan,
    build_teardown_cmds,
    summarize_rtts,
    summarize_served_log,
)
from network_covert_channel.topology import NetnsTopology
from slicing_sim.ofdm_grid import SLICE_TYPES


def _arg(argv, flag):
    return argv[list(argv).index(flag) + 1]


class FakeRun:
    """Records argv; ``fail_when(argv)`` -> CompletedProcess/exception to inject."""

    def __init__(self, fail_when=None, stdout_for=None):
        self.calls: list[list[str]] = []
        self._fail_when = fail_when or (lambda argv: None)
        self._stdout_for = stdout_for or (lambda argv: "")

    def __call__(self, argv, timeout=None):
        self.calls.append(list(argv))
        injected = self._fail_when(argv)
        if isinstance(injected, BaseException):
            raise injected
        if injected is not None:
            return injected
        return subprocess.CompletedProcess(argv, 0, stdout=self._stdout_for(argv), stderr="")


class FakePopen:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _client_line(index, trusted=True, error=None, **flags):
    if error is not None:
        return json.dumps({"index": index, "slice_type": "URLLC", "target": "x:1", "error": error, "elapsed_ms": 1.0})
    result = {"due": True, "trusted": trusted, "rejected_as_replay": False, "pinned_key_mismatch": False,
              "trust_store_key_changed": False, "reason": "periodic", "backend": "OqsDilithiumSigner", **flags}
    return json.dumps({"index": index, "slice_type": "URLLC", "target": "x:1", "error": None, "result": result, "rtt_ms": 1.0 + index})


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.topo = NetnsTopology()
        self.cfg = AuthTopologyConfig(workdir=Path("/wd"), python="PY")

    def test_endpoints_do_not_collide_with_slice_addresses_or_names(self):
        slice_ips = {self.topo.ip_for(s) for s in SLICE_TYPES}
        for endpoint in ENDPOINTS:
            self.assertNotIn(endpoint.ip(self.topo), slice_ips | {self.topo.bridge_ip()})
            for ifname in endpoint.veth_names:
                self.assertLessEqual(len(ifname), 15)  # IFNAMSIZ
        self.assertEqual(CORE.ip(self.topo), "10.200.0.2")

    def test_endpoint_setup_attaches_to_the_phase1_bridge_without_shaping(self):
        cmds = build_endpoint_setup_cmds(CORE, self.topo)
        self.assertIn(["ip", "link", "set", "veth-core-h", "master", self.topo.bridge_name], cmds)
        self.assertIn(["ip", "netns", "exec", "ns-core", "ip", "addr", "add", "10.200.0.2/24", "dev", "veth-core-c"], cmds)
        self.assertFalse(any(c[0] == "tc" for c in cmds))

    def test_teardown_removes_endpoints_before_the_bridge(self):
        cmds = build_teardown_cmds(self.topo)
        self.assertEqual(cmds[-1], ["ip", "link", "delete", self.topo.bridge_name])
        for endpoint in ENDPOINTS:
            self.assertIn(["ip", "netns", "delete", endpoint.netns], cmds)
            self.assertIn(["ip", "link", "delete", endpoint.veth_names[0]], cmds)
        for slice_type in SLICE_TYPES:
            self.assertIn(["ip", "netns", "delete", self.topo.netns_name(slice_type)], cmds)

    def test_blackhole_fault_drops_both_directions_and_restore_reuses_topology_netem(self):
        fault = build_link_fault_cmds(self.topo, "URLLC", "blackhole")
        self.assertEqual(fault[0][:5], ["tc", "qdisc", "change", "dev", "veth-urllc-h"])
        self.assertEqual(fault[1][:4], ["ip", "netns", "exec", "ns-urllc"])
        self.assertTrue(all(c[-2:] == ["loss", "100%"] for c in fault))
        restore = build_link_restore_cmds(self.topo, "URLLC", "blackhole")
        original = self.topo.build_tc_shaping_cmds("URLLC")[0]
        self.assertEqual(restore[0], original[:2] + ["change"] + original[3:])
        self.assertEqual(restore[1][-5:], ["qdisc", "del", "dev", "veth-urllc-c", "root"])

    def test_link_down_fault_and_restore(self):
        self.assertEqual(build_link_fault_cmds(self.topo, "eMBB", "link_down")[0][-1], "down")
        self.assertEqual(build_link_restore_cmds(self.topo, "eMBB", "link_down")[0][-1], "up")
        with self.assertRaises(ValueError):
            build_link_fault_cmds(self.topo, "eMBB", "unplug")

    def test_client_cmd_runs_in_its_slice_namespace_with_exactly_one_trust_mode(self):
        pinned = build_client_cmd(self.cfg, self.topo, "mMTC", "pinned", server="10.200.0.2", port=7000, now=5)
        self.assertEqual(pinned[:4], ["ip", "netns", "exec", "ns-mmtc"])
        self.assertEqual(_arg(pinned, "--expected-pubkey-file"), str(self.cfg.oob_pubkey_file))
        self.assertNotIn("--trust-store", pinned)
        tofu = build_client_cmd(self.cfg, self.topo, "mMTC", "tofu", server="10.200.0.2", port=7000, now=5)
        self.assertEqual(_arg(tofu, "--trust-store"), str(self.cfg.trust_store("mMTC")))
        self.assertEqual(_arg(tofu, "--server-id"), self.cfg.server_id)
        self.assertNotIn("--expected-pubkey-file", tofu)
        with self.assertRaises(ValueError):
            build_client_cmd(self.cfg, self.topo, "mMTC", "none", server="x", port=1, now=0)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.topo = NetnsTopology()
        self.cfg = AuthTopologyConfig(workdir=Path("/wd"), python="PY")
        self.plan = build_plan(self.cfg, self.topo)

    def _steps(self, group):
        return [s for s in self.plan if s.meta.get("group") == group]

    def test_order_setup_server_pubkey_clients_adversaries_audit(self):
        kinds = [s.kind for s in self.plan]
        self.assertEqual(kinds[:4], ["commands", "commands", "start", "export_pubkey"])
        self.assertEqual(kinds[-1], "audit")
        self.assertIn("pqc_auth.audit_verify", self.plan[-1].argv)
        self.assertEqual(self.plan[2].argv[:4], ("ip", "netns", "exec", "ns-core"))
        self.assertEqual(_arg(self.plan[2].argv, "--key-path"), str(self.cfg.key_path(CORE)))

    def test_every_slice_runs_both_pinned_and_tofu_against_core(self):
        legit = self._steps("legit")
        self.assertEqual({(s.meta["slice_type"], s.meta["trust_mode"]) for s in legit},
                         {(sl, m) for sl in SLICE_TYPES for m in ("pinned", "tofu")})
        for s in legit:
            self.assertEqual(_arg(s.argv, "--server"), CORE.ip(self.topo))
            self.assertEqual(_arg(s.argv, "--count"), str(self.cfg.rtt_samples))

    def test_pubkey_is_exported_before_any_client_runs(self):
        export = next(i for i, s in enumerate(self.plan) if s.kind == "export_pubkey")
        first_client = next(i for i, s in enumerate(self.plan) if s.kind in ("client", "link_fault"))
        self.assertLess(export, first_client)
        self.assertEqual(self.plan[export].meta["dst"], str(self.cfg.oob_pubkey_file))

    def test_logical_clock_never_goes_backwards(self):
        nows = [float(_arg(s.argv, "--now")) for s in self.plan if s.kind in ("client", "link_fault")]
        self.assertEqual(nows, sorted(nows))
        self.assertEqual(len(nows), len(set(nows)))

    def test_replay_same_process_captures_then_follows_up_at_the_replayer(self):
        (same,) = self._steps("replay_same_process")
        self.assertEqual(_arg(same.argv, "--server"), CORE.ip(self.topo))
        self.assertEqual(_arg(same.argv, "--then"), f"{ROGUE.ip(self.topo)}:{self.cfg.replay_port}")
        self.assertEqual(_arg(same.argv, "--capture-response"), str(self.cfg.captured_response))
        # the follow-up must land inside the client's replay window
        self.assertLess(float(_arg(same.argv, "--now-step")), 300)
        (fresh,) = self._steps("replay_fresh_process")
        self.assertEqual(_arg(fresh.argv, "--port"), str(self.cfg.replay_port))

    def test_rogue_server_uses_its_own_key_and_both_trust_modes_point_at_it(self):
        start = next(s for s in self.plan if s.kind == "start" and s.meta["proc"] == "rogue")
        self.assertEqual(_arg(start.argv, "--key-path"), str(self.cfg.key_path(ROGUE)))
        self.assertNotEqual(self.cfg.key_path(ROGUE), self.cfg.key_path(CORE))
        rogue = self._steps("rogue")
        self.assertEqual({s.meta["trust_mode"] for s in rogue}, {"pinned", "tofu"})
        for s in rogue:
            self.assertEqual(_arg(s.argv, "--server"), ROGUE.ip(self.topo))
        # the TOFU client reuses the store the legit batch already populated
        tofu_rogue = next(s for s in rogue if s.meta["trust_mode"] == "tofu")
        legit_tofu = next(s for s in self._steps("legit")
                          if s.meta["trust_mode"] == "tofu" and s.meta["slice_type"] == tofu_rogue.meta["slice_type"])
        self.assertEqual(_arg(tofu_rogue.argv, "--trust-store"), _arg(legit_tofu.argv, "--trust-store"))

    def test_idle_attack_runs_from_rogue_and_every_slice_is_served_while_it_holds(self):
        kinds_names = [(s.kind, s.name) for s in self.plan]
        start = next(i for i, s in enumerate(self.plan) if s.kind == "start" and s.meta.get("proc") == "idle")
        finish = next(i for i, s in enumerate(self.plan) if s.kind == "finish" and s.meta.get("proc") == "idle")
        between = self.plan[start + 1:finish]
        self.assertEqual({s.meta["slice_type"] for s in between}, set(SLICE_TYPES), kinds_names)
        self.assertTrue(all(s.meta["group"] == "idle_attack" for s in between))
        attacker = self.plan[start].argv
        self.assertEqual(attacker[:4], ("ip", "netns", "exec", ROGUE.netns))
        self.assertEqual(_arg(attacker, "--server"), CORE.ip(self.topo))
        self.assertLess(self.cfg.idle_attack_connections, 32)  # below ReauthServer's default cap
        self.assertGreater(self.cfg.idle_attack_hold_s, self.cfg.server_connection_timeout)
        core = next(s for s in self.plan if s.kind == "start" and s.meta["proc"] == "core")
        self.assertEqual(float(_arg(core.argv, "--connection-timeout")), self.cfg.server_connection_timeout)

    def test_malformed_probe_is_followed_by_a_served_request_per_slice(self):
        i = next(i for i, s in enumerate(self.plan) if s.kind == "probe")
        self.assertEqual(self.plan[i].argv[:4], ("ip", "netns", "exec", ROGUE.netns))
        self.assertEqual(self.plan[i].meta["alive_proc"], "core")
        after = [s for s in self.plan[i + 1:] if s.meta.get("group") == "after_malformed"]
        self.assertEqual({s.meta["slice_type"] for s in after}, set(SLICE_TYPES))

    def test_server_stats_read_the_core_served_log_before_the_audit(self):
        kinds = [s.kind for s in self.plan]
        self.assertEqual(kinds[-2:], ["server_stats", "audit"])
        self.assertEqual(self.plan[-2].meta["served_log"], str(self.cfg.served_log(CORE)))

    def test_link_fault_steps_cover_both_modes(self):
        modes = {s.meta["mode"] for s in self.plan if s.kind == "link_fault"}
        self.assertEqual(modes, {"blackhole", "link_down"})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.topo = NetnsTopology()
        self.cfg = AuthTopologyConfig(workdir=Path(self.tmp.name), python="PY", ready_timeout_s=2, hang_timeout_s=2)
        self.teardown_cmds = build_teardown_cmds(self.topo)

    def _runner(self, run, popen=None):
        popen = popen or (lambda argv: FakePopen(["READY {}"]))
        return AuthOverTopologyRunner(self.cfg, self.topo, run=run, popen=popen, log=lambda _msg: None)

    def assertTornDown(self, run):
        for argv in self.teardown_cmds:
            self.assertIn(argv, run.calls)

    def test_setup_failure_still_tears_everything_down(self):
        run = FakeRun(fail_when=lambda argv: subprocess.CompletedProcess(argv, 2, "", "boom")
                      if argv[:3] == ["ip", "netns", "add"] and argv[3] == "ns-core" else None)
        runner = self._runner(run)
        with self.assertRaises(PlanStepError):
            runner.execute(build_plan(self.cfg, self.topo))
        self.assertTornDown(run)
        self.assertIn("PlanStepError", runner.report["error"])

    def test_missing_binary_during_setup_is_converted_and_torn_down(self):
        run = FakeRun(fail_when=lambda argv: FileNotFoundError("ip") if argv[:2] == ["ip", "link"] else None)
        with self.assertRaises(PlanStepError):
            self._runner(run).execute(build_plan(self.cfg, self.topo))
        self.assertTornDown(run)

    def test_server_exiting_before_ready_tears_down_and_reports(self):
        run = FakeRun()
        popen = lambda argv: FakePopen(["Traceback: oqs missing"])
        with self.assertRaises(PlanStepError) as ctx:
            self._runner(run, popen).execute(build_plan(self.cfg, self.topo))
        self.assertIn("exited before READY", str(ctx.exception))
        self.assertTornDown(run)

    def test_interrupt_mid_plan_stops_started_processes_and_tears_down(self):
        started = []

        def popen(argv):
            started.append(FakePopen(["READY {}"]))
            return started[-1]

        run = FakeRun(fail_when=lambda argv: KeyboardInterrupt() if "request" in argv else None)
        plan = [s for s in build_plan(self.cfg, self.topo) if s.kind != "export_pubkey"]
        with self.assertRaises(KeyboardInterrupt):
            self._runner(run, popen).execute(plan)
        self.assertEqual(len(started), 1)
        self.assertTrue(started[0].terminated)
        self.assertTornDown(run)

    def test_teardown_ignores_failures_and_reports_leaks(self):
        run = FakeRun(fail_when=lambda argv: OSError("nope") if argv[:3] == ["ip", "link", "delete"] else None,
                      stdout_for=lambda argv: "ns-core (id: 3)\nother\n" if argv == ["ip", "netns", "list"] else "")
        runner = self._runner(run)
        runner.teardown()
        runner.teardown()  # idempotent
        self.assertEqual(runner.report["teardown"]["leaked_netns"], ["ns-core"])

    def test_client_hang_is_recorded_not_raised(self):
        step = Step("client", "c", argv=("PY", "request"), meta={"group": "legit", "slice_type": "URLLC", "trust_mode": "pinned"})
        run = FakeRun(fail_when=lambda argv: subprocess.TimeoutExpired(argv, 2, output=_client_line(0).encode())
                      if "request" in argv else None)
        runner = self._runner(run)
        runner.execute([step])
        (client,) = runner.report["clients"]
        self.assertIsNotNone(client["hung_after_s"])
        self.assertEqual(len(client["requests"]), 1)

    def test_link_fault_applies_after_n_results_and_restores_after_span(self):
        meta = {"group": "link_blackhole", "slice_type": "URLLC", "trust_mode": "pinned", "mode": "blackhole",
                "fault_cmds": [["FAULT"]], "restore_cmds": [["RESTORE"]], "fault_after": 2, "fault_span": 2}
        step = Step("link_fault", "lf", argv=("PY",), meta=meta)
        lines = ["liboqs-python faulthandler is disabled", _client_line(0), _client_line(1),
                 _client_line(2, error={"type": "TimeoutError", "message": "timed out"}),
                 _client_line(3, error={"type": "TimeoutError", "message": "timed out"}),
                 _client_line(4), _client_line(5)]
        run = FakeRun()
        runner = self._runner(run, popen=lambda argv: FakePopen(lines))
        runner.execute([step])
        (client,) = runner.report["clients"]
        self.assertEqual([r["phase"] for r in client["requests"]],
                         ["before_fault"] * 2 + ["during_fault"] * 2 + ["after_restore"] * 2)
        self.assertEqual([e["event"] for e in client["events"]], ["fault_applied", "fault_removed"])
        self.assertLess(run.calls.index(["FAULT"]), run.calls.index(["RESTORE"]))

    def test_link_fault_restores_even_if_client_dies_mid_fault(self):
        meta = {"group": "link_blackhole", "slice_type": "URLLC", "trust_mode": "pinned", "mode": "blackhole",
                "fault_cmds": [["FAULT"]], "restore_cmds": [["RESTORE"]], "fault_after": 1, "fault_span": 5}
        run = FakeRun()
        runner = self._runner(run, popen=lambda argv: FakePopen([_client_line(0)]))
        runner.execute([Step("link_fault", "lf", argv=("PY",), meta=meta)])
        self.assertIn(["RESTORE"], run.calls)
        self.assertEqual(runner.report["clients"][0]["events"][-1]["event"], "fault_removed_after_client_ended")

    def test_probe_records_output_and_whether_the_server_survived(self):
        core = FakePopen(["READY {}"])
        run = FakeRun(stdout_for=lambda argv: '{"case": "x", "server_reply": "invalid_json"}\n' if "probe" in argv else "")
        plan = [Step("start", "core", argv=("PY",), meta={"proc": "core"}),
                Step("probe", "p", argv=("PY", "probe"), meta={"alive_proc": "core"})]
        runner = self._runner(run, popen=lambda argv: core)
        runner.execute(plan)
        (probe,) = runner.report["probes"]
        self.assertTrue(probe["server_alive_after"])
        self.assertEqual(probe["records"], [{"case": "x", "server_reply": "invalid_json"}])

    def test_finish_waits_for_the_process_and_keeps_its_json_output(self):
        idle = FakePopen(["READY {}", '{"event": "idle_attacker_done", "closed_by_server": 8}'])
        plan = [Step("start", "idle", argv=("PY",), meta={"proc": "idle"}),
                Step("finish", "wait", meta={"proc": "idle", "timeout_s": 2})]
        runner = self._runner(FakeRun(), popen=lambda argv: idle)
        runner.execute(plan)
        self.assertEqual(runner.report["probes"][0]["records"][0]["closed_by_server"], 8)
        self.assertFalse(runner.report["probes"][0]["timed_out"])

    def test_served_log_summary_measures_sign_and_verify_and_tallies_errors(self):
        records = [
            {"event": "served", "due": True, "sign_ms": 0.5, "verify_ms": 0.2},
            {"event": "served", "due": True, "sign_ms": 1.5, "verify_ms": 0.1},
            {"event": "served", "due": False, "sign_ms": None, "verify_ms": None},
            {"event": "error", "code": "invalid_json"},
            {"event": "rejected_at_capacity"},
        ]
        summary = summarize_served_log(records)
        self.assertEqual(summary["sign_ms"]["n"], 2)
        self.assertEqual(summary["sign_ms"]["median_ms"], 1.0)
        self.assertEqual(summary["verify_ms"]["max_ms"], 0.2)
        self.assertEqual(summary["events"], {"error:invalid_json": 1, "rejected_at_capacity": 1, "served": 3})

    def test_rtt_summary_only_counts_legit_successful_requests(self):
        report = {"clients": [
            {"group": "legit", "slice_type": "URLLC", "requests": [json.loads(_client_line(i)) for i in range(3)]},
            {"group": "rogue", "slice_type": "URLLC", "requests": [json.loads(_client_line(9))]},
            {"group": "legit", "slice_type": "eMBB",
             "requests": [json.loads(_client_line(0, error={"type": "TimeoutError", "message": ""}))]},
        ]}
        summary = summarize_rtts(report)
        self.assertEqual(set(summary), {"URLLC"})
        self.assertEqual(summary["URLLC"]["n"], 3)
        self.assertEqual(summary["URLLC"]["max_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
