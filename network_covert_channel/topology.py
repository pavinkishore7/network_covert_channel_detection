"""
Network-layer topology for the inter-packet-timing covert channel.

This is the infrastructure for a SECOND, independent threat vector, distinct
from covert_channel/attacker.py's PHY-layer OFDM-grid channel. That one
perturbs a simulated interference grid; this one hides information in the
timing of real packets moving through a real Linux network. The two are
kept in separate top-level packages on purpose -- they are conceptually and
structurally unrelated apart from both being "covert channels" and both
being organized around the same three network slices.

``NetnsTopology`` builds three Linux network namespaces (one per slice:
URLLC, eMBB, mMTC), each given a veth pair whose host-side end plugs into a
shared bridge in the root namespace, and whose namespace-side end carries
the slice's traffic. Per-slice QoS differentiation is applied with `tc`
netem (delay/jitter/loss) and tbf (rate limiting) on the host-side veth --
representing the shaping a real RAN/gNB would enforce per slice before
traffic reaches the shared backhaul.

Command construction is deliberately split from execution: every
``build_*_cmds`` method is a pure function of ``self`` (slice names, IP
scheme, QoS profile) that returns a list of argv lists -- no subprocess
call, no I/O. A thin ``self._run`` callable (defaulting to a
``subprocess.run`` wrapper, but injectable) is the only thing that ever
executes anything. Tests pass a fake runner and assert on the constructed
argv directly; none of them need a real namespace or root, matching this
project's requirement that no unit test needs CAP_NET_ADMIN to pass.

Privilege check performed when this module was written (2026-09-18, WSL2
dev environment): ``ip netns add`` failed with "Permission denied" and
``sudo -n true`` required a password (no passwordless sudo) -- see
network_covert_channel/README.md for the full record. This module was
therefore built and unit-tested entirely against a mocked runner; the code
paths that actually call ``ip``/``tc`` have not been exercised against a
real kernel in this session. Running it for real requires root or
CAP_NET_ADMIN on a real Linux host (WSL2's own netns support is separately
uncertain even with root, since it's not this kernel's ordinary configuration
-- untested either way here).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from slicing_sim.ofdm_grid import SLICE_PROFILES, SLICE_TYPES

# Arbitrary shared "backhaul" capacity assumption used only to turn
# subcarrier_frac (which already sums to 1.0 across the three slices) into
# absolute per-slice bandwidth numbers. This is a modeling choice for this
# testbed, not a measured or standards-derived value -- same honesty
# standard slicing_sim/ofdm_grid.py holds itself to for SLICE_PROFILES.
TOTAL_LINK_CAPACITY_MBIT = 100.0

DEFAULT_BRIDGE_NAME = "br-ncc0"
DEFAULT_SUBNET_BASE = "10.200.0"  # a /24; bridge = .1, slices get .11/.12/.13 in SLICE_TYPES order


class TopologyCommandError(RuntimeError):
    """Raised when a command run during ``setup()`` exits non-zero. Not
    raised during ``teardown()`` -- teardown is best-effort by design (see
    its docstring)."""


@dataclass(frozen=True)
class NetworkQoSProfile:
    """Per-slice network-layer QoS parameters, derived from
    ``slicing_sim.ofdm_grid.SLICE_PROFILES``. See ``derive_qos_profile``
    for exactly how each field is computed and why -- these are design
    assumptions translating a PHY-layer profile into plausible network-layer
    numbers, NOT measured values.
    """

    slice_type: str
    rate_mbit: float
    delay_ms: float
    jitter_ms: float
    loss_pct: float
    priority_band: int  # 0 = highest priority; lower band = served first


def derive_qos_profile(slice_type: str) -> NetworkQoSProfile:
    """Translate ``SLICE_PROFILES[slice_type]`` (subcarrier_frac,
    burstiness) into network-layer QoS parameters.

    Reasoning for each mapping (a modeling choice, not a measurement):
      - rate_mbit: proportional to subcarrier_frac, since that PHY-layer
        field already represents "share of the resource grid this slice
        gets" -- the direct network-layer analogue is "share of link
        capacity". Because the three subcarrier_frac values already sum to
        1.0, the three rate_mbit values sum to TOTAL_LINK_CAPACITY_MBIT.
      - delay_ms / jitter_ms: scaled from burstiness. URLLC has the lowest
        burstiness in SLICE_PROFILES (0.2) and is explicitly described
        there as "latency-critical", so lower burstiness is mapped to a
        tighter delay/jitter budget. This reproduces the intended ordering
        (URLLC tightest, then eMBB, then mMTC loosest) without inventing a
        second independent knob.
      - loss_pct: also scaled from burstiness -- a slice whose PHY traffic
        is already sporadic/bursty (mMTC, burstiness 0.8) is modeled as
        tolerating more loss than one that needs steady, reliable delivery
        (URLLC, burstiness 0.2).
      - priority_band: slices ranked by delay_ms ascending get bands
        0, 1, 2 in that order (0 = served first by a priority-aware
        scheduler). Not currently enforced by a tc prio qdisc in this
        Phase 1 build (see build_tc_shaping_cmds's docstring) -- carried
        as metadata for a later phase that might want it.
    """
    if slice_type not in SLICE_PROFILES:
        raise ValueError(f"unknown slice_type {slice_type!r}, expected one of {SLICE_TYPES}")

    profile = SLICE_PROFILES[slice_type]
    burstiness = profile["burstiness"]

    rate_mbit = profile["subcarrier_frac"] * TOTAL_LINK_CAPACITY_MBIT
    delay_ms = 1.0 + burstiness * 15.0
    jitter_ms = delay_ms * 0.25
    loss_pct = burstiness * 1.5

    ranked = sorted(SLICE_TYPES, key=lambda s: SLICE_PROFILES[s]["burstiness"])
    priority_band = ranked.index(slice_type)

    return NetworkQoSProfile(
        slice_type=slice_type,
        rate_mbit=round(rate_mbit, 3),
        delay_ms=round(delay_ms, 3),
        jitter_ms=round(jitter_ms, 3),
        loss_pct=round(loss_pct, 3),
        priority_band=priority_band,
    )


def _default_run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


class NetnsTopology:
    """Builds/tears down the three-namespace, bridge-connected topology.

    ``runner`` is a callable ``(argv: list[str]) -> subprocess.CompletedProcess``
    used for every command this class executes. Defaults to a plain
    ``subprocess.run`` wrapper; tests inject a fake to avoid touching the
    real kernel.
    """

    def __init__(
        self,
        slices: Sequence[str] = SLICE_TYPES,
        bridge_name: str = DEFAULT_BRIDGE_NAME,
        subnet_base: str = DEFAULT_SUBNET_BASE,
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ):
        self.slices = tuple(slices)
        self.bridge_name = bridge_name
        self.subnet_base = subnet_base
        self._run = runner or _default_run

    # -- naming / addressing -------------------------------------------------

    def netns_name(self, slice_type: str) -> str:
        return f"ns-{slice_type.lower()}"

    def veth_names(self, slice_type: str) -> tuple[str, str]:
        """(host_side_ifname, namespace_side_ifname). Both kept under the
        15-char Linux IFNAMSIZ limit for every current SLICE_TYPES value."""
        short = slice_type.lower()
        return f"veth-{short}-h", f"veth-{short}-c"

    def bridge_ip(self) -> str:
        return f"{self.subnet_base}.1"

    def ip_for(self, slice_type: str) -> str:
        # .11, .12, .13, ... in SLICE_TYPES order -- stable regardless of
        # which subset of slices this instance was built with.
        host_octet = 11 + SLICE_TYPES.index(slice_type)
        return f"{self.subnet_base}.{host_octet}"

    def qos_profile(self, slice_type: str) -> NetworkQoSProfile:
        return derive_qos_profile(slice_type)

    @staticmethod
    def _tbf_burst_kbit(profile: NetworkQoSProfile) -> int:
        # A practical rule of thumb (burst >= rate / HZ) rounded up to a
        # sane minimum; not derived from the slice profile itself.
        return max(32, round(profile.rate_mbit * 32))

    # -- pure command construction -------------------------------------------

    def build_create_bridge_cmds(self) -> list[list[str]]:
        return [
            ["ip", "link", "add", self.bridge_name, "type", "bridge"],
            ["ip", "addr", "add", f"{self.bridge_ip()}/24", "dev", self.bridge_name],
            ["ip", "link", "set", self.bridge_name, "up"],
        ]

    def build_create_netns_cmds(self, slice_type: str) -> list[list[str]]:
        return [["ip", "netns", "add", self.netns_name(slice_type)]]

    def build_create_veth_cmds(self, slice_type: str) -> list[list[str]]:
        host_if, ns_if = self.veth_names(slice_type)
        return [["ip", "link", "add", host_if, "type", "veth", "peer", "name", ns_if]]

    def build_attach_to_bridge_cmds(self, slice_type: str) -> list[list[str]]:
        host_if, _ = self.veth_names(slice_type)
        return [
            ["ip", "link", "set", host_if, "master", self.bridge_name],
            ["ip", "link", "set", host_if, "up"],
        ]

    def build_move_into_netns_cmds(self, slice_type: str) -> list[list[str]]:
        _, ns_if = self.veth_names(slice_type)
        return [["ip", "link", "set", ns_if, "netns", self.netns_name(slice_type)]]

    def build_configure_netns_cmds(self, slice_type: str) -> list[list[str]]:
        ns = self.netns_name(slice_type)
        _, ns_if = self.veth_names(slice_type)
        addr = f"{self.ip_for(slice_type)}/24"
        return [
            ["ip", "netns", "exec", ns, "ip", "link", "set", "lo", "up"],
            ["ip", "netns", "exec", ns, "ip", "addr", "add", addr, "dev", ns_if],
            ["ip", "netns", "exec", ns, "ip", "link", "set", ns_if, "up"],
        ]

    def build_tc_shaping_cmds(self, slice_type: str) -> list[list[str]]:
        """netem (delay/jitter/loss) as the root qdisc, tbf (rate limit) as
        its child, applied on the HOST-side veth end -- i.e. shaping is
        enforced at the point representing the gNB/RAN, before traffic
        reaches the shared bridge. ``priority_band`` is not enforced here
        (would need a tc prio qdisc across all three host-side veths
        together, e.g. via a shared parent on the bridge); left for a later
        phase if slice-vs-slice scheduling contention turns out to matter
        for the covert-channel signal.
        """
        host_if, _ = self.veth_names(slice_type)
        profile = self.qos_profile(slice_type)
        return [
            [
                "tc", "qdisc", "add", "dev", host_if, "root", "handle", "1:", "netem",
                "delay", f"{profile.delay_ms}ms", f"{profile.jitter_ms}ms",
                "loss", f"{profile.loss_pct}%",
            ],
            [
                "tc", "qdisc", "add", "dev", host_if, "parent", "1:", "handle", "10:", "tbf",
                "rate", f"{profile.rate_mbit}mbit",
                "burst", f"{self._tbf_burst_kbit(profile)}kbit",
                "latency", "50ms",
            ],
        ]

    def build_slice_setup_cmds(self, slice_type: str) -> list[list[str]]:
        """All commands for one slice, in order, after the bridge exists."""
        return (
            self.build_create_netns_cmds(slice_type)
            + self.build_create_veth_cmds(slice_type)
            + self.build_attach_to_bridge_cmds(slice_type)
            + self.build_move_into_netns_cmds(slice_type)
            + self.build_configure_netns_cmds(slice_type)
            + self.build_tc_shaping_cmds(slice_type)
        )

    def build_setup_cmds(self) -> list[list[str]]:
        cmds = list(self.build_create_bridge_cmds())
        for slice_type in self.slices:
            cmds += self.build_slice_setup_cmds(slice_type)
        return cmds

    def build_teardown_cmds(self) -> list[list[str]]:
        """Deletes every resource this instance could have created.
        Deliberately blind to whether setup ever actually ran, or how far
        it got -- see ``teardown()``'s docstring for why."""
        cmds: list[list[str]] = []
        for slice_type in self.slices:
            # Deleting the netns removes the ns-side veth end with it; the
            # host-side end lives in the root namespace and needs its own
            # delete. tc qdiscs on it are removed automatically when the
            # interface itself is deleted, so no separate `tc qdisc del`.
            cmds.append(["ip", "netns", "delete", self.netns_name(slice_type)])
            host_if, _ = self.veth_names(slice_type)
            cmds.append(["ip", "link", "delete", host_if])
        cmds.append(["ip", "link", "delete", self.bridge_name])
        return cmds

    # -- execution -------------------------------------------------------

    def _execute(self, cmds: list[list[str]]) -> None:
        for argv in cmds:
            result = self._run(argv)
            if result.returncode != 0:
                raise TopologyCommandError(
                    f"command failed (exit {result.returncode}): {' '.join(argv)}\n"
                    f"stderr: {result.stderr}"
                )

    def setup(self) -> None:
        """Builds the full topology. On any command failure, tears down
        whatever may have been partially created before re-raising --
        never leaves orphaned namespaces/interfaces behind on a crash."""
        try:
            self._execute(self.build_setup_cmds())
        except TopologyCommandError:
            self.teardown()
            raise

    def teardown(self) -> None:
        """Best-effort cleanup: runs every delete command for every
        resource this instance's config describes, ignoring failures
        (including a command not existing, or the resource never having
        existed at all). Deliberately not "smart" about tracking exactly
        what setup() actually created -- that would make teardown() itself
        a source of bugs on the crash path it exists to make safe. Safe to
        call multiple times, and safe to call when setup() was never
        called or only got partway through.
        """
        for argv in self.build_teardown_cmds():
            try:
                self._run(argv)
            except Exception:
                pass
