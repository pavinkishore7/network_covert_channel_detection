"""
Per-slice traffic generation for the network-layer covert-channel testbed.

Environment check performed when this module was written (2026-09-18, WSL2
dev environment): ``which iperf3`` found nothing, and ``python3 -c "import
scapy"`` failed (not installed). iperf3 needs an external binary that isn't
installable here without root (apt needs sudo, and there's no passwordless
sudo -- see network_covert_channel/README.md and step 0 of the Phase 1
prompt this package was built from). scapy is a pure-Python package that
``pip install scapy`` pulled in cleanly (now in requirements.txt) and gives
direct, per-packet control over send timing -- exactly what's needed to
make ``SLICE_PROFILES[...]["burstiness"]`` visibly reshape the
inter-packet-gap distribution. That is why traffic here is built with scapy
rather than shelled out to iperf3. ``iperf3_available()`` below is provided
so a future phase running on a host that DOES have iperf3 can detect that
and add an iperf3 path without touching this one; no such path exists yet.

Timing model: ``generate_inter_packet_gaps`` blends two component
processes using ``burstiness`` as the mix weight:
  - a "steady" component: gaps drawn tightly around a fixed baseline
    (low variance) -- what a burstiness-0 slice would look like.
  - a "bursty" component: a two-state ON/OFF renewal process, alternating
    short intra-burst gaps with long inter-burst idle periods -- what a
    burstiness-1 slice would look like.
Intermediate burstiness values are a weighted blend of the two. This
reuses the same ``SLICE_PROFILES`` burstiness scalar that
``slicing_sim/ofdm_grid.py`` uses for per-symbol active-set resampling, but
the mechanics here are necessarily different (that one operates per OFDM
symbol on a subcarrier mask; this one operates per packet on a timing
series) -- the tie is conceptual (same knob, same three slices), not a
literal reuse of the same formula. This is a MODELING CHOICE made to give
each slice a visibly distinguishable inter-packet-timing signature for
later covert-channel/detector work, not a validated model of real
URLLC/eMBB/mMTC traffic statistics -- state that plainly if asked.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scapy.all import IP, UDP, Raw, send

from slicing_sim.ofdm_grid import SLICE_PROFILES, SLICE_TYPES

DEFAULT_BASE_GAP_S = 0.01  # 10ms baseline spacing shared across slices before burstiness reshapes it
DEFAULT_PAYLOAD_BYTES = 64
DEFAULT_PORT = 50000


def iperf3_available() -> bool:
    return shutil.which("iperf3") is not None


def scapy_available() -> bool:
    try:
        import scapy  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class TrafficPacketPlan:
    """One packet to send, plus the delay (seconds) to wait immediately
    before sending it (including before the very first packet)."""

    slice_type: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    payload_bytes: int
    gap_before_s: float


def generate_inter_packet_gaps(
    slice_type: str,
    n_packets: int,
    rng: np.random.Generator,
    base_gap_s: float = DEFAULT_BASE_GAP_S,
) -> np.ndarray:
    """Inter-packet gaps (seconds), shaped by ``slice_type``'s burstiness.
    See module docstring for the blend model. Deterministic given ``rng``'s
    state, so tests can seed it and compare distributions across slices.
    """
    if slice_type not in SLICE_PROFILES:
        raise ValueError(f"unknown slice_type {slice_type!r}, expected one of {SLICE_TYPES}")
    if n_packets <= 0:
        raise ValueError("n_packets must be positive")

    burstiness = SLICE_PROFILES[slice_type]["burstiness"]

    steady = rng.normal(base_gap_s, base_gap_s * 0.05, size=n_packets)

    on = True
    bursty = np.empty(n_packets)
    for i in range(n_packets):
        if rng.random() < 0.3:  # state-transition check, independent of burstiness itself
            on = not on
        bursty[i] = (
            rng.exponential(base_gap_s * 0.2) if on else rng.exponential(base_gap_s * 6.0)
        )

    gaps = (1.0 - burstiness) * steady + burstiness * bursty
    return np.clip(gaps, 1e-4, None)


def build_traffic_plan(
    slice_type: str,
    n_packets: int,
    src_ip: str,
    dst_ip: str,
    src_port: int = DEFAULT_PORT,
    dst_port: int = DEFAULT_PORT,
    payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
    rng: np.random.Generator | None = None,
) -> list[TrafficPacketPlan]:
    """Pure plan construction -- no sending, no live interface needed.
    Fully testable and inspectable before anything touches a socket."""
    rng = rng if rng is not None else np.random.default_rng()
    gaps = generate_inter_packet_gaps(slice_type, n_packets, rng)
    return [
        TrafficPacketPlan(
            slice_type=slice_type,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            payload_bytes=payload_bytes,
            gap_before_s=float(gap),
        )
        for gap in gaps
    ]


def build_packet(entry: TrafficPacketPlan):
    """Pure scapy packet construction from a plan entry -- no sending."""
    payload = b"\x00" * entry.payload_bytes
    return (
        IP(src=entry.src_ip, dst=entry.dst_ip)
        / UDP(sport=entry.src_port, dport=entry.dst_port)
        / Raw(load=payload)
    )


def send_traffic_plan(
    plan: list[TrafficPacketPlan],
    iface: str | None = None,
    sender: Callable[[object], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Executes a plan built by ``build_traffic_plan``: sleeps
    ``gap_before_s``, then sends the packet, for each entry in order.

    ``sender`` defaults to scapy's ``send()`` (a raw socket send, which
    needs CAP_NET_RAW/root at runtime -- not exercised by the unit test
    suite). Tests inject a fake sender/sleeper so this is fully testable
    without a live interface, root, or real wall-clock delays.
    """
    _send = sender or (lambda pkt: send(pkt, iface=iface, verbose=False))
    for entry in plan:
        if entry.gap_before_s > 0:
            sleeper(entry.gap_before_s)
        _send(build_packet(entry))
