"""
Packet capture + parsing for the network-layer covert-channel testbed.

Environment check performed when this module was written (2026-09-18, WSL2
dev environment): ``which tshark`` and ``which tcpdump`` both found
nothing, and neither ``pyshark`` nor ``scapy`` was pre-installed (scapy was
subsequently ``pip install``ed for network_covert_channel/traffic.py, see
requirements.txt). This module wraps whichever of tshark/tcpdump is
present on the host for the part that actually runs a LIVE capture
(``run_capture``) -- that needs a real interface plus
CAP_NET_RAW/CAP_NET_ADMIN, so it is exercised only by the live-integration
step described in network_covert_channel/README.md, not by the unit test
suite. The PARSING utility (``parse_pcap_to_dataframe``) is fully testable
offline: it uses ``scapy.rdpcap`` against a synthetic pcap that a test
writes with ``scapy.utils.wrpcap`` itself, no live capture required.
pyshark was not used because it wasn't installed in the checked
environment, and scapy already covers both packet crafting (traffic.py)
and pcap parsing (here), so it was the simpler choice.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from scapy.all import IP, rdpcap


def tshark_available() -> bool:
    return shutil.which("tshark") is not None


def tcpdump_available() -> bool:
    return shutil.which("tcpdump") is not None


def preferred_capture_tool() -> str | None:
    """tshark if present (richer built-in duration/filter support), else
    tcpdump, else None if neither is on PATH."""
    if tshark_available():
        return "tshark"
    if tcpdump_available():
        return "tcpdump"
    return None


def pcap_filename(slice_type: str, timestamp: float | None = None) -> str:
    """e.g. 'urllc_20260918T120000Z.pcap' -- slice type and capture start
    time both embedded so files from a multi-slice run sort and identify
    themselves without needing a separate manifest."""
    ts = timestamp if timestamp is not None else time.time()
    ts_str = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts))
    return f"{slice_type.lower()}_{ts_str}.pcap"


def build_capture_cmd(
    tool: str,
    interface: str,
    output_path: str,
    duration_s: float | None = None,
    capture_filter: str | None = None,
) -> list[str]:
    """Pure command construction -- no execution -- so tests can assert on
    the argv without tshark/tcpdump installed or a live interface."""
    if tool == "tshark":
        cmd = ["tshark", "-i", interface, "-w", output_path]
        if duration_s is not None:
            cmd += ["-a", f"duration:{int(duration_s)}"]
        if capture_filter:
            cmd += ["-f", capture_filter]
        return cmd
    if tool == "tcpdump":
        cmd = ["tcpdump", "-i", interface, "-w", output_path]
        if capture_filter:
            cmd.append(capture_filter)
        return cmd
    raise ValueError(f"unknown capture tool {tool!r}, expected 'tshark' or 'tcpdump'")


def run_capture(
    tool: str,
    interface: str,
    output_path: str,
    duration_s: float,
    capture_filter: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> subprocess.CompletedProcess:
    """Runs a capture for approximately ``duration_s`` seconds.

    tshark has a built-in autostop (``-a duration:N``, included in the
    built command); tcpdump has no equivalent flag, so for tcpdump the
    ``subprocess`` timeout below IS the stop mechanism, not just a safety
    margin. Requires CAP_NET_RAW/CAP_NET_ADMIN on ``interface`` at runtime
    -- not exercised by the unit test suite (tests inject ``runner``).
    """
    _run = runner or subprocess.run
    cmd = build_capture_cmd(tool, interface, output_path, duration_s, capture_filter)
    try:
        return _run(cmd, timeout=duration_s + 5, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=exc.stdout or "", stderr=exc.stderr or ""
        )


def parse_pcap_to_dataframe(path: str | Path) -> pd.DataFrame:
    """Reads a pcap with scapy and returns one row per packet:
    ``timestamp`` (float, seconds since epoch), ``size`` (bytes on the
    wire), ``src``/``dst`` (None if the packet has no IP layer), and
    ``inter_arrival_s`` (gap since the previous packet in file order; NaN
    for the first packet).
    """
    packets = rdpcap(str(path))
    rows = []
    prev_ts: float | None = None
    for pkt in packets:
        ts = float(pkt.time)
        has_ip = pkt.haslayer(IP)
        rows.append(
            {
                "timestamp": ts,
                "size": len(pkt),
                "src": pkt[IP].src if has_ip else None,
                "dst": pkt[IP].dst if has_ip else None,
                "inter_arrival_s": (ts - prev_ts) if prev_ts is not None else float("nan"),
            }
        )
        prev_ts = ts
    return pd.DataFrame(rows, columns=["timestamp", "size", "src", "dst", "inter_arrival_s"])
