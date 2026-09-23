"""Run pqc_auth's re-auth exchange across network_covert_channel's real topology.

    python -m integration.auth_over_topology plan              # print the plan; no root needed
    python -m integration.auth_over_topology run [--workdir D] # execute it; needs root/CAP_NET_ADMIN

Until now ReauthServer and ReauthClient only ever talked over loopback,
inside one process. Here the server and every client are SEPARATE PROCESSES
in SEPARATE NETWORK NAMESPACES, connected through the Phase 1 bridge and the
per-slice tc shaping (network_covert_channel/topology.py). They share
nothing in memory: the only way a client learns the server's key is through
a file (explicit pin) or over the wire on first contact (TOFU).

Topology used (NetnsTopology's three slice namespaces, plus two namespaces
this module adds on the same bridge -- topology.py itself has no "core"
role, since in Phase 1 the root namespace's bridge address was the only
peer):

    ns-core   10.200.0.2  real ReauthServer (persisted key_path)
    ns-rogue  10.200.0.3  adversary: rogue ReauthServer (different key) +
                          replay responder
    ns-urllc  10.200.0.11 ┐
    ns-embb   10.200.0.12 ├ one client process per request batch, per slice
    ns-mmtc   10.200.0.13 ┘

ns-core and ns-rogue get no tc shaping: they sit on the backhaul side of the
per-slice shaping, which stays where Phase 1 put it (the slice veths).

Same split as topology.py: every ``build_*`` function below is pure (config
in, argv lists / plan steps out -- no subprocess, no I/O), and
``AuthOverTopologyRunner`` is the thin, injectable layer that executes a plan.
Teardown always runs (try/finally), is best-effort, and is blind to how far
setup got -- the same reasoning as ``NetnsTopology.teardown()``.

The latency numbers this produces are veth-in-a-VM numbers: signature +
transport overhead on a software topology. They are NOT radio or 5G latency
and say nothing about 3GPP TS 22.261 URLLC targets. See the README.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from network_covert_channel.topology import NetnsTopology, netns_privilege_skip_reason

REPO_ROOT = Path(__file__).resolve().parent.parent

LINK_FAULT_MODES = ("blackhole", "link_down")


# -- extra endpoint namespaces ---------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """A namespace this module adds to the Phase 1 bridge (not a slice)."""

    name: str
    host_octet: int

    @property
    def netns(self) -> str:
        return f"ns-{self.name}"

    @property
    def veth_names(self) -> tuple[str, str]:
        return f"veth-{self.name}-h", f"veth-{self.name}-c"

    def ip(self, topo: NetnsTopology) -> str:
        return f"{topo.subnet_base}.{self.host_octet}"


CORE = Endpoint("core", 2)
ROGUE = Endpoint("rogue", 3)
ENDPOINTS = (CORE, ROGUE)


def build_endpoint_setup_cmds(endpoint: Endpoint, topo: NetnsTopology) -> list[list[str]]:
    """Same command shape as NetnsTopology.build_slice_setup_cmds, minus the
    tc shaping (see the module docstring)."""
    host_if, ns_if = endpoint.veth_names
    ns = endpoint.netns
    return [
        ["ip", "netns", "add", ns],
        ["ip", "link", "add", host_if, "type", "veth", "peer", "name", ns_if],
        ["ip", "link", "set", host_if, "master", topo.bridge_name],
        ["ip", "link", "set", host_if, "up"],
        ["ip", "link", "set", ns_if, "netns", ns],
        ["ip", "netns", "exec", ns, "ip", "link", "set", "lo", "up"],
        ["ip", "netns", "exec", ns, "ip", "addr", "add", f"{endpoint.ip(topo)}/24", "dev", ns_if],
        ["ip", "netns", "exec", ns, "ip", "link", "set", ns_if, "up"],
    ]


def build_endpoint_teardown_cmds(endpoint: Endpoint) -> list[list[str]]:
    host_if, _ = endpoint.veth_names
    return [["ip", "netns", "delete", endpoint.netns], ["ip", "link", "delete", host_if]]


def build_teardown_cmds(topo: NetnsTopology, endpoints: Sequence[Endpoint] = ENDPOINTS) -> list[list[str]]:
    """Endpoints first (their host-side veths are bridge ports), then the
    Phase 1 topology, whose own teardown deletes the bridge last."""
    cmds: list[list[str]] = []
    for endpoint in endpoints:
        cmds += build_endpoint_teardown_cmds(endpoint)
    return cmds + topo.build_teardown_cmds()


# -- link faults (the "kill the path mid-run" case) ------------------------


def build_link_fault_cmds(topo: NetnsTopology, slice_type: str, mode: str) -> list[list[str]]:
    """``blackhole``: netem loss 100% in BOTH directions (host-side veth
    egress = toward the slice; ns-side veth egress = toward the bridge), so
    packets silently vanish and the link still looks up. ``link_down``: the
    slice's own interface goes administratively down, so its kernel knows."""
    host_if, ns_if = topo.veth_names(slice_type)
    ns = topo.netns_name(slice_type)
    if mode == "blackhole":
        return [
            ["tc", "qdisc", "change", "dev", host_if, "root", "handle", "1:", "netem", "loss", "100%"],
            ["ip", "netns", "exec", ns, "tc", "qdisc", "add", "dev", ns_if, "root", "netem", "loss", "100%"],
        ]
    if mode == "link_down":
        return [["ip", "netns", "exec", ns, "ip", "link", "set", ns_if, "down"]]
    raise ValueError(f"unknown link fault mode {mode!r}, expected one of {LINK_FAULT_MODES}")


def build_link_restore_cmds(topo: NetnsTopology, slice_type: str, mode: str) -> list[list[str]]:
    host_if, ns_if = topo.veth_names(slice_type)
    ns = topo.netns_name(slice_type)
    if mode == "blackhole":
        # Restore the slice's original netem parameters: topology.py's own
        # netem command, with `add` -> `change`.
        netem = list(topo.build_tc_shaping_cmds(slice_type)[0])
        netem[2] = "change"
        return [netem, ["ip", "netns", "exec", ns, "tc", "qdisc", "del", "dev", ns_if, "root"]]
    if mode == "link_down":
        return [["ip", "netns", "exec", ns, "ip", "link", "set", ns_if, "up"]]
    raise ValueError(f"unknown link fault mode {mode!r}, expected one of {LINK_FAULT_MODES}")


# -- process command lines -------------------------------------------------


@dataclass(frozen=True)
class AuthTopologyConfig:
    workdir: Path
    python: str = sys.executable
    port: int = 7000
    replay_port: int = 7001
    server_id: str = "core-reauth"
    rtt_samples: int = 10  # requests per (slice, trust mode) batch
    client_timeout: float = 5.0  # passed to ReauthClient; its own default, unchanged
    hang_timeout_s: float = 90.0  # orchestrator-side kill: beyond this a client counts as HUNG
    ready_timeout_s: float = 30.0
    link_fault_slice: str = "URLLC"
    link_fault_requests: int = 6
    link_fault_after: int = 2  # fault is applied after this many results
    link_fault_span: int = 2  # ... and removed after this many more
    link_fault_interval: float = 0.5
    t0: float = 1_000_000.0  # logical clock for request `now` values
    now_step: float = 1000.0  # > every DEFAULT_POLICIES interval, so each request is due

    def key_path(self, endpoint: Endpoint) -> Path:
        return self.workdir / "keys" / endpoint.name

    @property
    def oob_pubkey_file(self) -> Path:
        # The explicit-pin "out-of-band channel". A file on the same disk as
        # the server's key directory is a STAND-IN for a real trusted channel
        # (provisioning, a signed config, a human comparing fingerprints) --
        # it demonstrates the client side of pinning, it is not a key
        # distribution mechanism.
        return self.workdir / "oob" / "core_public_key.bin"

    def trust_store(self, slice_type: str) -> Path:
        return self.workdir / "trust" / f"{slice_type}.json"

    @property
    def client_audit_log(self) -> Path:
        return self.workdir / "logs" / "client_audit.jsonl"

    def served_log(self, endpoint: Endpoint) -> Path:
        return self.workdir / "logs" / f"served_{endpoint.name}.jsonl"

    @property
    def captured_response(self) -> Path:
        return self.workdir / "captured_response.json"


def _in_netns(ns: str, argv: list[str]) -> list[str]:
    return ["ip", "netns", "exec", ns, *argv]


def build_server_cmd(cfg: AuthTopologyConfig, endpoint: Endpoint, topo: NetnsTopology) -> list[str]:
    return _in_netns(endpoint.netns, [
        cfg.python, "-m", "pqc_auth.transport", "serve",
        "--bind", endpoint.ip(topo), "--port", str(cfg.port),
        "--key-path", str(cfg.key_path(endpoint)),
        "--audit-log", str(cfg.served_log(endpoint)),
    ])


def build_replay_responder_cmd(cfg: AuthTopologyConfig, topo: NetnsTopology) -> list[str]:
    return _in_netns(ROGUE.netns, [
        cfg.python, "-m", "integration.auth_over_topology", "replay-responder",
        "--bind", ROGUE.ip(topo), "--port", str(cfg.replay_port),
        "--response-file", str(cfg.captured_response),
    ])


def build_client_cmd(
    cfg: AuthTopologyConfig,
    topo: NetnsTopology,
    slice_type: str,
    trust_mode: str,
    *,
    server: str,
    port: int,
    now: float,
    count: int = 1,
    now_step: float = 0.0,
    interval: float = 0.0,
    then: Sequence[str] = (),
    capture_response: Path | None = None,
) -> list[str]:
    if trust_mode == "pinned":
        trust = ["--expected-pubkey-file", str(cfg.oob_pubkey_file)]
    elif trust_mode == "tofu":
        trust = ["--trust-store", str(cfg.trust_store(slice_type)), "--server-id", cfg.server_id]
    else:
        raise ValueError(f"unknown trust_mode {trust_mode!r}")
    argv = [
        cfg.python, "-m", "pqc_auth.transport", "request",
        "--server", server, "--port", str(port), "--slice-type", slice_type, *trust,
        "--audit-log", str(cfg.client_audit_log),
        "--timeout", str(cfg.client_timeout),
        "--now", repr(float(now)), "--now-step", repr(float(now_step)),
        "--count", str(count), "--interval", repr(float(interval)),
    ]
    for target in then:
        argv += ["--then", target]
    if capture_response is not None:
        argv += ["--capture-response", str(capture_response)]
    return _in_netns(topo.netns_name(slice_type), argv)


# -- the plan ----------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One plan step. ``kind`` selects how the runner executes it:

    commands      run ``cmds`` in order, fail fast
    start         launch ``argv`` in the background, wait for its READY line
    export_pubkey copy ``src`` -> ``dst`` (the out-of-band pin)
    client        run ``argv`` to completion, collect one JSON line per request
    link_fault    run client ``argv``; after ``fault_after`` lines run
                  ``fault_cmds``, after ``fault_span`` more run ``restore_cmds``
    audit         run ``argv`` (pqc_auth.audit_verify) and record its verdict
    """

    kind: str
    name: str
    argv: tuple[str, ...] = ()
    cmds: tuple[tuple[str, ...], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def _cmds(cmds: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(c) for c in cmds)


def build_plan(cfg: AuthTopologyConfig, topo: NetnsTopology) -> list[Step]:
    core_ip, rogue_ip = CORE.ip(topo), ROGUE.ip(topo)
    now = cfg.t0
    steps: list[Step] = []

    def advance(requests: int) -> float:
        nonlocal now
        start = now
        now += (requests + 1) * cfg.now_step
        return start

    steps.append(Step("commands", "topology: bridge + slice namespaces + tc shaping", cmds=_cmds(topo.build_setup_cmds())))
    endpoint_cmds: list[list[str]] = []
    for endpoint in ENDPOINTS:
        endpoint_cmds += build_endpoint_setup_cmds(endpoint, topo)
    steps.append(Step("commands", "topology: core + rogue namespaces", cmds=_cmds(endpoint_cmds)))

    steps.append(Step("start", "core ReauthServer", argv=tuple(build_server_cmd(cfg, CORE, topo)), meta={"proc": "core"}))
    steps.append(Step(
        "export_pubkey", "export core public key (out-of-band stand-in)",
        meta={"src": str(cfg.key_path(CORE) / "public_key.bin"), "dst": str(cfg.oob_pubkey_file)},
    ))

    # Every slice runs one batch with explicit pinning and one with TOFU, so
    # the split is exactly half/half and each slice exercises both.
    for slice_type in topo.slices:
        for trust_mode in ("pinned", "tofu"):
            argv = build_client_cmd(
                cfg, topo, slice_type, trust_mode, server=core_ip, port=cfg.port,
                now=advance(cfg.rtt_samples), count=cfg.rtt_samples, now_step=cfg.now_step,
            )
            steps.append(Step(
                "client", f"{slice_type} {trust_mode} -> core", argv=tuple(argv),
                meta={"group": "legit", "slice_type": slice_type, "trust_mode": trust_mode},
            ))

    # a. Replay. The attacker endpoint serves whatever response the client
    # captured from the real server. Case 1: the SAME client process that
    # accepted it is sent it again (--then) -- must be rejected_as_replay.
    # Case 2: a FRESH client process gets it -- recorded as-is, because the
    # client's seen-nonce memory is per process (see the README).
    replay_slice = topo.slices[0]
    steps.append(Step("start", "replay responder (adversary)", argv=tuple(build_replay_responder_cmd(cfg, topo)), meta={"proc": "replay"}))
    argv = build_client_cmd(
        cfg, topo, replay_slice, "pinned", server=core_ip, port=cfg.port, now=advance(2), now_step=1.0,
        then=[f"{rogue_ip}:{cfg.replay_port}"], capture_response=cfg.captured_response,
    )
    steps.append(Step("client", f"replay: same client process ({replay_slice})", argv=tuple(argv),
                      meta={"group": "replay_same_process", "slice_type": replay_slice, "trust_mode": "pinned"}))
    argv = build_client_cmd(cfg, topo, replay_slice, "pinned", server=rogue_ip, port=cfg.replay_port, now=advance(1))
    steps.append(Step("client", f"replay: fresh client process ({replay_slice})", argv=tuple(argv),
                      meta={"group": "replay_fresh_process", "slice_type": replay_slice, "trust_mode": "pinned"}))

    # b. Rogue server: a real ReauthServer with its own, different keypair.
    # Both clients below already trust the real core server -- one via the
    # out-of-band pin, one via the TOFU store the legit batch populated.
    rogue_slice = topo.slices[1 % len(topo.slices)]
    steps.append(Step("start", "rogue ReauthServer (adversary)", argv=tuple(build_server_cmd(cfg, ROGUE, topo)), meta={"proc": "rogue"}))
    for trust_mode in ("pinned", "tofu"):
        argv = build_client_cmd(cfg, topo, rogue_slice, trust_mode, server=rogue_ip, port=cfg.port, now=advance(1))
        steps.append(Step("client", f"rogue server: {rogue_slice} {trust_mode} -> rogue", argv=tuple(argv),
                          meta={"group": "rogue", "slice_type": rogue_slice, "trust_mode": trust_mode}))

    # c. Link failure mid-run: one client doing periodic re-auth; the path is
    # cut after `link_fault_after` results and restored `link_fault_span`
    # results later. Nothing here changes what the client does about it.
    for mode in LINK_FAULT_MODES:
        argv = build_client_cmd(
            cfg, topo, cfg.link_fault_slice, "pinned", server=core_ip, port=cfg.port,
            now=advance(cfg.link_fault_requests), count=cfg.link_fault_requests,
            now_step=cfg.now_step, interval=cfg.link_fault_interval,
        )
        steps.append(Step(
            "link_fault", f"link failure ({mode}) on {cfg.link_fault_slice}", argv=tuple(argv),
            meta={
                "group": f"link_{mode}", "slice_type": cfg.link_fault_slice, "trust_mode": "pinned", "mode": mode,
                "fault_cmds": build_link_fault_cmds(topo, cfg.link_fault_slice, mode),
                "restore_cmds": build_link_restore_cmds(topo, cfg.link_fault_slice, mode),
                "fault_after": cfg.link_fault_after, "fault_span": cfg.link_fault_span,
            },
        ))

    steps.append(Step("audit", "pqc_auth.audit_verify on the client log",
                      argv=(cfg.python, "-m", "pqc_auth.audit_verify", str(cfg.client_audit_log))))
    return steps


# -- execution ---------------------------------------------------------------


class PlanStepError(RuntimeError):
    """A plan step failed in a way that makes continuing meaningless."""


def _default_run(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout,
                          cwd=REPO_ROOT, env=_child_env())


def _default_popen(argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=REPO_ROOT, env=_child_env())


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    return env


class _Proc:
    """A Popen-like with a reader thread, so stdout lines can be waited on
    with a deadline instead of a blocking readline that could hang the
    orchestrator itself."""

    def __init__(self, handle):
        self.handle = handle
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        self.transcript: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            for line in self.handle.stdout:
                self.lines.put(line.rstrip("\n"))
        finally:
            self.lines.put(None)

    def next_line(self, timeout: float) -> str | None:
        """Next stdout line; None at EOF; raises queue.Empty on timeout."""
        line = self.lines.get(timeout=timeout)
        if line is not None:
            self.transcript.append(line)
        return line

    def stop(self, grace_s: float = 5.0) -> int | None:
        if self.handle.poll() is None:
            self.handle.terminate()
            try:
                self.handle.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self.handle.kill()
                self.handle.wait()
        return self.handle.poll()


def _is_noise(line: str) -> bool:
    # liboqs-python prints this on import; it's not ours to parse.
    return "faulthandler" in line


def _parse_client_lines(lines: Sequence[str]) -> list[dict]:
    records = []
    for line in lines:
        line = line.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


class AuthOverTopologyRunner:
    """Executes a plan from ``build_plan``. ``run(argv, timeout=None)`` and
    ``popen(argv)`` are injectable, like NetnsTopology's ``runner``."""

    def __init__(
        self,
        cfg: AuthTopologyConfig,
        topo: NetnsTopology,
        run: Callable[..., subprocess.CompletedProcess] | None = None,
        popen: Callable[[list[str]], Any] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.cfg = cfg
        self.topo = topo
        self._run = run or _default_run
        self._popen = popen or _default_popen
        self._log = log
        self.procs: dict[str, _Proc] = {}
        self.report: dict[str, Any] = {"steps": [], "clients": [], "audit": None, "teardown": None, "error": None}

    # public ---------------------------------------------------------------

    def execute(self, plan: Sequence[Step]) -> dict[str, Any]:
        for sub in ("keys", "oob", "trust", "logs"):
            (self.cfg.workdir / sub).mkdir(parents=True, exist_ok=True)
        try:
            for step in plan:
                self._log(f"==> {step.name}")
                getattr(self, f"_step_{step.kind}")(step)
                self.report["steps"].append({"name": step.name, "kind": step.kind, "ok": True})
        except BaseException as exc:
            self.report["error"] = f"{type(exc).__name__}: {exc}"
            self._log(f"!! plan aborted at this step: {self.report['error']}")
            raise
        finally:
            self.teardown()
        return self.report

    def teardown(self) -> None:
        """Stop every process we started, then delete every namespace/link
        this config describes, ignoring failures. Safe to call repeatedly."""
        stopped = {}
        for name, proc in self.procs.items():
            try:
                stopped[name] = {"returncode": proc.stop(), "transcript_tail": proc.transcript[-5:]}
            except Exception as exc:  # noqa: BLE001 - keep tearing down
                stopped[name] = {"error": f"{type(exc).__name__}: {exc}"}
        for argv in build_teardown_cmds(self.topo):
            try:
                self._run(argv)
            except Exception:  # noqa: BLE001 - best-effort by design
                pass
        remaining = None
        try:
            listing = self._run(["ip", "netns", "list"])
            ours = {self.topo.netns_name(s) for s in self.topo.slices} | {e.netns for e in ENDPOINTS}
            remaining = sorted(n for n in ours if any(line.split()[:1] == [n] for line in (listing.stdout or "").splitlines()))
        except Exception as exc:  # noqa: BLE001
            remaining = f"could not list: {exc}"
        self.report["teardown"] = {
            "processes": stopped,
            "still_running": sorted(n for n, p in self.procs.items() if p.handle.poll() is None),
            "leaked_netns": remaining,
        }

    # steps ----------------------------------------------------------------

    def _check(self, argv: Sequence[str]) -> None:
        try:
            result = self._run(list(argv))
        except OSError as exc:
            raise PlanStepError(f"could not execute {argv[0]!r}: {' '.join(argv)}: {exc}") from exc
        if result.returncode != 0:
            raise PlanStepError(f"command failed (exit {result.returncode}): {' '.join(argv)}\nstderr: {result.stderr}")

    def _step_commands(self, step: Step) -> None:
        for argv in step.cmds:
            self._check(argv)

    def _step_start(self, step: Step) -> None:
        proc = _Proc(self._popen(list(step.argv)))
        self.procs[step.meta["proc"]] = proc
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlanStepError(f"{step.name}: no READY line within {self.cfg.ready_timeout_s}s")
            try:
                line = proc.next_line(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise PlanStepError(f"{step.name}: exited before READY; output: {proc.transcript}")
            if line.startswith("READY"):
                self._log(f"    {line}")
                return

    def _step_export_pubkey(self, step: Step) -> None:
        Path(step.meta["dst"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(step.meta["src"], step.meta["dst"])

    def _step_client(self, step: Step) -> None:
        started = time.monotonic()
        try:
            result = self._run(list(step.argv), timeout=self.cfg.hang_timeout_s)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            self._record(step, _parse_client_lines(output.splitlines()),
                         hung_after_s=time.monotonic() - started, exit_code=None)
            return
        self._record(step, _parse_client_lines((result.stdout or "").splitlines()), exit_code=result.returncode,
                     stderr=[l for l in (result.stderr or "").splitlines() if not _is_noise(l)][-5:])

    def _step_link_fault(self, step: Step) -> None:
        proc = _Proc(self._popen(list(step.argv)))
        self.procs[f"client:{step.name}"] = proc
        records: list[dict] = []
        events: list[dict] = []
        started = time.monotonic()
        deadline = started + self.cfg.hang_timeout_s
        phase = "before_fault"
        hung = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                hung = True
                break
            try:
                line = proc.next_line(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                break
            parsed = _parse_client_lines([line])
            if not parsed:
                continue
            record = {**parsed[0], "phase": phase}
            records.append(record)
            if phase == "before_fault" and len(records) == step.meta["fault_after"]:
                for argv in step.meta["fault_cmds"]:
                    self._check(argv)
                events.append({"event": "fault_applied", "t_s": round(time.monotonic() - started, 3), "cmds": step.meta["fault_cmds"]})
                phase = "during_fault"
            elif phase == "during_fault" and len(records) == step.meta["fault_after"] + step.meta["fault_span"]:
                for argv in step.meta["restore_cmds"]:
                    self._check(argv)
                events.append({"event": "fault_removed", "t_s": round(time.monotonic() - started, 3), "cmds": step.meta["restore_cmds"]})
                phase = "after_restore"
        if phase == "during_fault":
            # Restore even if the client died/hung mid-fault, so later steps
            # (and a human inspecting the topology) see a working link.
            for argv in step.meta["restore_cmds"]:
                self._run(list(argv))
            events.append({"event": "fault_removed_after_client_ended", "t_s": round(time.monotonic() - started, 3)})
        exit_code = proc.stop()
        self._record(step, records, exit_code=exit_code, events=events,
                     hung_after_s=(time.monotonic() - started) if hung else None)

    def _step_audit(self, step: Step) -> None:
        result = self._run(list(step.argv), timeout=self.cfg.hang_timeout_s)
        lines = [l for l in (result.stdout or "").splitlines() if not _is_noise(l)]
        self.report["audit"] = {
            "exit_code": result.returncode,
            "summary": [l for l in lines if l.startswith("OVERALL") or "record(s) checked" in l],
            "failures": [l for l in lines if "FAIL" in l and not l.startswith("OVERALL")],
        }

    def _record(self, step: Step, records: list[dict], **extra) -> None:
        entry = {"name": step.name, **{k: v for k, v in step.meta.items() if k in ("group", "slice_type", "trust_mode", "mode")},
                 "requests": records, **extra}
        if step.meta.get("trust_mode") == "tofu":
            entry["tofu_store_matches_oob_key"] = self._tofu_store_matches(step.meta["slice_type"])
        self.report["clients"].append(entry)
        for r in records:
            self._log(f"    {_describe_request(r)}")
        if extra.get("hung_after_s") is not None:
            self._log(f"    HUNG: killed by orchestrator after {extra['hung_after_s']:.1f}s")

    def _tofu_store_matches(self, slice_type: str) -> bool | None:
        store, oob = self.cfg.trust_store(slice_type), self.cfg.oob_pubkey_file
        if not store.exists() or not oob.exists():
            return None
        learned = json.loads(store.read_text()).get(self.cfg.server_id)
        return learned == oob.read_bytes().hex()


# -- report formatting (pure) ----------------------------------------------


def _describe_request(r: dict) -> str:
    phase = f"[{r['phase']}] " if "phase" in r else ""
    head = f"{phase}#{r.get('index')} {r.get('slice_type')} -> {r.get('target')}"
    if r.get("error"):
        return f"{head}: EXCEPTION {r['error']['type']}: {r['error']['message']} (after {r['elapsed_ms']:.0f} ms)"
    res = r["result"]
    flags = [k for k in ("rejected_as_replay", "pinned_key_mismatch", "trust_store_key_changed") if res.get(k)]
    verdict = "TRUSTED" if res.get("trusted") else ("not due" if not res.get("due") else "REJECTED")
    return f"{head}: {verdict}{' (' + ', '.join(flags) + ')' if flags else ''} rtt={r['rtt_ms']:.2f} ms"


def summarize_rtts(report: dict) -> dict[str, dict[str, float]]:
    """Per-slice RTT stats over the legitimate batches only (both trust modes)."""
    by_slice: dict[str, list[float]] = {}
    for client in report["clients"]:
        if client.get("group") != "legit":
            continue
        for r in client["requests"]:
            if r.get("error") is None and "rtt_ms" in r:
                by_slice.setdefault(client["slice_type"], []).append(r["rtt_ms"])
    out = {}
    for slice_type, values in by_slice.items():
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        out[slice_type] = {"n": len(values), "min_ms": ordered[0], "median_ms": statistics.median(ordered),
                           "p95_ms": p95, "max_ms": ordered[-1]}
    return out


def format_summary(report: dict) -> str:
    lines = ["", "=" * 72, "AUTH OVER TOPOLOGY -- SUMMARY", "=" * 72]
    lines.append("Per-slice verification (legit batches):")
    for c in report["clients"]:
        if c.get("group") != "legit":
            continue
        reqs = c["requests"]
        trusted = sum(1 for r in reqs if r.get("result", {}).get("trusted"))
        tofu = f", TOFU store == OOB key: {c['tofu_store_matches_oob_key']}" if "tofu_store_matches_oob_key" in c else ""
        lines.append(f"  {c['slice_type']:<5} {c['trust_mode']:<6} {trusted}/{len(reqs)} trusted (exit {c.get('exit_code')}){tofu}")
    lines.append("Adversarial runs:")
    for c in report["clients"]:
        if c.get("group") == "legit":
            continue
        lines.append(f"  {c['name']}:")
        for e in c.get("events", []):
            lines.append(f"      -- {e['event']} at t={e['t_s']}s")
        for r in c["requests"]:
            lines.append(f"      {_describe_request(r)}")
        if c.get("hung_after_s") is not None:
            lines.append(f"      HUNG: killed by orchestrator after {c['hung_after_s']:.1f}s")
    lines.append("RTT per slice (veth-in-a-VM: signature + transport overhead on a software topology;")
    lines.append("NOT radio/5G latency, NOT comparable to TS 22.261 URLLC targets):")
    for slice_type, s in summarize_rtts(report).items():
        lines.append(f"  {slice_type:<5} n={s['n']:<3} min={s['min_ms']:.2f}  median={s['median_ms']:.2f}  "
                     f"p95={s['p95_ms']:.2f}  max={s['max_ms']:.2f} ms")
    audit = report.get("audit") or {}
    lines.append(f"audit_verify: exit {audit.get('exit_code')} {' | '.join(audit.get('summary', []))}")
    for f in audit.get("failures", []):
        lines.append(f"  {f}")
    td = report.get("teardown") or {}
    lines.append(f"teardown: leaked netns={td.get('leaked_netns')} still-running processes={td.get('still_running')}")
    if report.get("error"):
        lines.append(f"PLAN ABORTED: {report['error']}")
    return "\n".join(lines)


# -- the attacker's replay responder ---------------------------------------


def serve_replay(bind: str, port: int, response_file: str) -> int:
    """Answer every request with the captured response bytes, read fresh on
    each connection (the file is written by the victim client moments before
    it connects here). A deliberately dumb adversary: it never signs."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    sock.listen(4)
    sock.settimeout(0.5)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print("READY " + json.dumps({"host": bind, "port": port, "role": "replay-responder"}), flush=True)
    with sock:
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(5.0)
                try:
                    conn.recv(4096)
                    conn.sendall(Path(response_file).read_text().strip().encode() + b"\n")
                except OSError:
                    pass
    return 0


# -- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m integration.auth_over_topology")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default=None, help="where keys/logs/trust stores go (default: a new temp dir)")
        p.add_argument("--rtt-samples", type=int, default=AuthTopologyConfig.rtt_samples)
        p.add_argument("--report", default=None, help="also write the full JSON report here")
    rr = sub.add_parser("replay-responder", help="adversary endpoint; used by the plan itself")
    rr.add_argument("--bind", required=True)
    rr.add_argument("--port", type=int, required=True)
    rr.add_argument("--response-file", required=True)
    args = parser.parse_args(argv)

    if args.command == "replay-responder":
        return serve_replay(args.bind, args.port, args.response_file)

    import tempfile

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="auth_over_topology_"))
    cfg = AuthTopologyConfig(workdir=workdir.resolve(), rtt_samples=args.rtt_samples)
    topo = NetnsTopology()
    plan = build_plan(cfg, topo)

    if args.command == "plan":
        for step in plan:
            print(f"[{step.kind}] {step.name}")
            for c in step.cmds:
                print("    " + " ".join(c))
            if step.argv:
                print("    " + " ".join(step.argv))
        return 0

    skip_reason = netns_privilege_skip_reason()
    if skip_reason is not None:
        print(f"cannot run: {skip_reason}", file=sys.stderr)
        return 2
    print(f"workdir: {workdir}")
    runner = AuthOverTopologyRunner(cfg, topo)
    try:
        report = runner.execute(plan)
    finally:
        report = runner.report
        print(format_summary(report))
        report["rtt_summary"] = summarize_rtts(report)
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    ok = report.get("error") is None and (report.get("audit") or {}).get("exit_code") == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
