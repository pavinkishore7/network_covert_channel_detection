"""
Phase 2, step 3: end-to-end demo tying covert_injector.py and
timing_detector.py together against the rest of network_covert_channel/.

Two paths, exactly mirroring how topology.py/capture.py already split
pure/testable construction from privileged execution:

  - ``run_synthetic_slice_demo`` / ``run_synthetic_demo``: entirely
    in-memory. Builds a clean traffic plan via traffic.py's
    ``build_traffic_plan``, injects a covert channel into its gaps via
    covert_injector.py, and scores both the clean and covert gap sequences
    with a freshly calibrated ``TimingKSDetector``. No socket, subprocess,
    or privilege of any kind -- this is what the unit test suite and any
    caller without root gets.
  - ``run_live_demo``: actually stands up ``NetnsTopology``, sends both a
    clean-only and a covert-carrying traffic plan per slice over real
    veths, captures each with capture.py, and runs the SAME detector
    against the real captured gaps. Requires root/CAP_NET_ADMIN (checked
    via ``netns_privileges_available``, a thin wrapper around
    topology.netns_privilege_skip_reason() -- the one privilege check that
    tests/test_network_live_integration.py also uses) plus tshark/tcpdump. Self-skips with a
    clear reason if either is missing rather than failing confusingly.

``build_covert_traffic_plan`` is the "thin wrapper" the covert_injector.py
module docstring promises: it takes a plan already built by traffic.py's
unmodified ``build_traffic_plan`` and returns a new plan with
``gap_before_s`` perturbed by an injector, ready to hand to traffic.py's
unmodified ``send_traffic_plan``. Neither traffic.py, topology.py, nor
capture.py is modified by this module -- everything here is new
composition on top of Phase 1's existing, unmodified building blocks.

Privilege check performed when this module was written (2026-09-22, same
WSL2 dev environment Phase 1 was built in, freshly re-verified in this
session rather than assumed from Phase 1's record):
    $ id -u
    1000
    $ ip netns add __ncc_phase2_probe__
    mkdir /run/netns failed: Permission denied
    $ sudo -n true
    sudo: a password is required
    $ which tcpdump tshark ip
    /usr/sbin/ip
Neither root/CAP_NET_ADMIN nor a capture tool (tshark/tcpdump) is
available. ``run_live_demo`` was NOT executed for real in this session --
only the synthetic path was run, and its actual results are recorded in
network_covert_channel/README.md and this Phase's PR description, not
fabricated.
"""

from __future__ import annotations

import dataclasses
import functools
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from network_covert_channel.capture import (
    build_capture_cmd,
    parse_pcap_to_dataframe,
    pcap_filename,
    preferred_capture_tool,
)
from network_covert_channel.covert_injector import (
    AdaptiveCovertInjector,
    CovertInjectorConfig,
    NonAdaptiveCovertInjector,
)
from network_covert_channel.timing_detector import TimingKSDetector
from network_covert_channel.topology import NetnsTopology, netns_privilege_skip_reason
from network_covert_channel.traffic import (
    TrafficPacketPlan,
    build_traffic_plan,
    generate_inter_packet_gaps,
    send_traffic_plan,
)
from slicing_sim.ofdm_grid import SLICE_TYPES

DEFAULT_N_PACKETS = 300
DEFAULT_CALIBRATION_TRIALS = 100
DEFAULT_CALIBRATION_PERCENTILE = 95.0


@dataclass(frozen=True)
class SliceDemoResult:
    """One slice's clean-vs-covert detection outcome."""

    slice_type: str
    offset_s: float
    adaptive: bool
    threshold: float
    clean_statistic: float
    clean_flag: bool
    covert_statistic: float
    covert_flag: bool


def netns_privileges_available() -> tuple[bool, str]:
    """(available, message) form of topology.netns_privilege_skip_reason(),
    the single privilege check for everything that needs real namespaces
    -- delegated to rather than reimplemented, so the live demo, its test
    and Phase 1's live test can never disagree. message explains why when
    available is False."""
    reason = netns_privilege_skip_reason()
    if reason is not None:
        return False, reason
    return True, "root/CAP_NET_ADMIN available"


def _make_injector(offset_s: float, adaptive: bool, n_covert_bits: int, seed: int):
    cfg = CovertInjectorConfig(n_covert_bits=n_covert_bits, offset_s=offset_s, seed=seed)
    return AdaptiveCovertInjector(cfg) if adaptive else NonAdaptiveCovertInjector(cfg)


def build_covert_traffic_plan(
    clean_plan: list[TrafficPacketPlan],
    injector: NonAdaptiveCovertInjector | AdaptiveCovertInjector,
    bits: np.ndarray | None = None,
) -> tuple[list[TrafficPacketPlan], np.ndarray]:
    """Perturbs a traffic.py plan's gap_before_s field via an injector.
    Pure w.r.t. traffic.py itself -- builds a NEW list of
    TrafficPacketPlan entries (dataclasses.replace, since the dataclass is
    frozen) rather than mutating clean_plan, and every other field
    (slice_type, src_ip, dst_ip, ports, payload_bytes) is carried over
    unchanged. The result is a plan send_traffic_plan (unmodified) can
    send exactly as-is.
    """
    clean_gaps = np.array([entry.gap_before_s for entry in clean_plan])
    bits = injector.generate_covert_bits() if bits is None else np.asarray(bits)
    covert_gaps = injector.inject(clean_gaps, bits)
    covert_plan = [
        dataclasses.replace(entry, gap_before_s=float(gap)) for entry, gap in zip(clean_plan, covert_gaps)
    ]
    return covert_plan, bits


def run_synthetic_slice_demo(
    slice_type: str,
    n_packets: int = DEFAULT_N_PACKETS,
    offset_s: float = 0.02,
    adaptive: bool = False,
    seed: int = 2026,
) -> SliceDemoResult:
    """Entirely in-memory: no socket, subprocess, or privilege needed.

    Builds a clean plan via build_traffic_plan, perturbs a copy of it via
    build_covert_traffic_plan, calibrates a TimingKSDetector against fresh
    independent clean draws of the same slice, and scores both the clean
    and covert gap sequences against an independent clean baseline
    sample.
    """
    plan_rng = np.random.default_rng(seed)
    clean_plan = build_traffic_plan(
        slice_type, n_packets, src_ip="10.200.0.11", dst_ip="10.200.0.1", rng=plan_rng
    )

    injector = _make_injector(offset_s, adaptive, n_packets, seed)
    covert_plan, _bits = build_covert_traffic_plan(clean_plan, injector)

    clean_gaps = np.array([e.gap_before_s for e in clean_plan])
    covert_gaps = np.array([e.gap_before_s for e in covert_plan])

    detector = TimingKSDetector(seed=seed)
    calib_rng = np.random.default_rng(seed + 1)
    sampler = functools.partial(generate_inter_packet_gaps, slice_type, n_packets, calib_rng)
    threshold = detector.calibrate(
        sampler, n_trials=DEFAULT_CALIBRATION_TRIALS, percentile=DEFAULT_CALIBRATION_PERCENTILE
    )

    baseline = generate_inter_packet_gaps(slice_type, n_packets, np.random.default_rng(seed + 2))
    clean_result = detector.score(clean_gaps, baseline)
    covert_result = detector.score(covert_gaps, baseline)

    return SliceDemoResult(
        slice_type=slice_type,
        offset_s=offset_s,
        adaptive=adaptive,
        threshold=threshold,
        clean_statistic=clean_result.statistic,
        clean_flag=clean_result.anomaly,
        covert_statistic=covert_result.statistic,
        covert_flag=covert_result.anomaly,
    )


def run_synthetic_demo(
    offset_s: float = 0.02,
    adaptive: bool = False,
    n_packets: int = DEFAULT_N_PACKETS,
    seed: int = 2026,
) -> dict[str, SliceDemoResult]:
    """The synthetic path across all three slices at once."""
    return {
        slice_type: run_synthetic_slice_demo(
            slice_type, n_packets=n_packets, offset_s=offset_s, adaptive=adaptive, seed=seed
        )
        for slice_type in SLICE_TYPES
    }


@dataclass(frozen=True)
class SliceAccuracySweepResult:
    """Aggregate detection accuracy over many independent trials, for one
    slice at one offset -- the "reports detection accuracy" step 3.1
    requires, computed here as a checked-in, reproducible function rather
    than left as one-off exploration. false_alarm_rate is measured on
    CLEAN-vs-baseline trials (offset_s is irrelevant to those);
    detection_rate is measured on COVERT-vs-baseline trials at offset_s."""

    slice_type: str
    offset_s: float
    n_trials: int
    false_alarm_rate: float
    detection_rate: float


def run_synthetic_accuracy_sweep(
    offsets_s: dict[str, float],
    n_trials: int = 30,
    n_packets: int = DEFAULT_N_PACKETS,
    adaptive: bool = False,
    seed_base: int = 2026,
) -> dict[tuple[str, str], SliceAccuracySweepResult]:
    """For every (offset label -> offset_s) pair, and every slice, draws
    n_trials independent clean samples and n_trials independent
    injected-covert samples (fresh RNG draws each trial -- not the same
    sample reused), scores each against a fixed clean baseline with a
    freshly calibrated TimingKSDetector, and returns the aggregate
    false-alarm/detection rates. Keyed by (slice_type, offset_label) so
    results print in a stable, labeled order.
    """
    results: dict[tuple[str, str], SliceAccuracySweepResult] = {}
    for slice_type in SLICE_TYPES:
        detector = TimingKSDetector(seed=seed_base)
        calib_rng = np.random.default_rng(seed_base + 1)
        sampler = functools.partial(generate_inter_packet_gaps, slice_type, n_packets, calib_rng)
        detector.calibrate(sampler, n_trials=DEFAULT_CALIBRATION_TRIALS, percentile=DEFAULT_CALIBRATION_PERCENTILE)
        baseline = generate_inter_packet_gaps(slice_type, n_packets, np.random.default_rng(seed_base + 2))

        false_alarms = 0
        for t in range(n_trials):
            rng = np.random.default_rng(seed_base + 3000 + t)
            clean = generate_inter_packet_gaps(slice_type, n_packets, rng)
            if detector.is_anomalous(clean, baseline):
                false_alarms += 1
        false_alarm_rate = false_alarms / n_trials

        for label, offset_s in offsets_s.items():
            hits = 0
            for t in range(n_trials):
                rng = np.random.default_rng(seed_base + 4000 + t)
                clean = generate_inter_packet_gaps(slice_type, n_packets, rng)
                injector = _make_injector(offset_s, adaptive, n_packets, seed_base + 4000 + t)
                covert = injector.inject(clean, injector.generate_covert_bits())
                if detector.is_anomalous(covert, baseline):
                    hits += 1
            results[(slice_type, label)] = SliceAccuracySweepResult(
                slice_type=slice_type,
                offset_s=offset_s,
                n_trials=n_trials,
                false_alarm_rate=false_alarm_rate,
                detection_rate=hits / n_trials,
            )
    return results


def run_live_demo(
    n_packets: int = DEFAULT_N_PACKETS,
    offset_s: float = 0.02,
    adaptive: bool = False,
    capture_duration_s: float = 15.0,
    tmp_dir: str = "/tmp",
    seed: int = 2026,
) -> dict[str, SliceDemoResult]:
    """Runs the real thing over NetnsTopology: for each slice, sends a
    clean-only plan and (separately) a covert-carrying plan over the
    slice's real veth, captures each with tshark/tcpdump, parses the
    resulting pcaps' inter_arrival_s via capture.py, and scores the real
    captured gap sequences with the same TimingKSDetector used by the
    synthetic path.

    Raises RuntimeError immediately, before touching the topology, if
    netns_privileges_available() or preferred_capture_tool() say this
    can't actually run -- callers (the live pytest) should check those
    themselves first and skip rather than relying on this exception.
    """
    available, reason = netns_privileges_available()
    if not available:
        raise RuntimeError(f"cannot run live demo: {reason}")
    capture_tool = preferred_capture_tool()
    if capture_tool is None:
        raise RuntimeError("cannot run live demo: requires tshark or tcpdump on PATH; neither was found")

    topo = NetnsTopology()
    topo.setup()
    try:
        results: dict[str, SliceDemoResult] = {}
        detector = TimingKSDetector(seed=seed)
        calib_rng = np.random.default_rng(seed + 1)

        for slice_type in SLICE_TYPES:
            host_if, _ = topo.veth_names(slice_type)

            # --- clean run ---
            clean_plan_rng = np.random.default_rng(seed)
            clean_plan = build_traffic_plan(
                slice_type, n_packets, src_ip=topo.ip_for(slice_type), dst_ip=topo.bridge_ip(), rng=clean_plan_rng
            )
            clean_path = f"{tmp_dir}/{pcap_filename(f'{slice_type}_clean')}"
            clean_gaps = _capture_one_plan(capture_tool, host_if, clean_path, clean_plan, capture_duration_s)

            # --- covert run ---
            injector = _make_injector(offset_s, adaptive, n_packets, seed)
            covert_plan, _bits = build_covert_traffic_plan(clean_plan, injector)
            covert_path = f"{tmp_dir}/{pcap_filename(f'{slice_type}_covert')}"
            covert_gaps = _capture_one_plan(capture_tool, host_if, covert_path, covert_plan, capture_duration_s)

            sampler = functools.partial(generate_inter_packet_gaps, slice_type, n_packets, calib_rng)
            threshold = detector.calibrate(
                sampler, n_trials=DEFAULT_CALIBRATION_TRIALS, percentile=DEFAULT_CALIBRATION_PERCENTILE
            )
            baseline = generate_inter_packet_gaps(slice_type, n_packets, np.random.default_rng(seed + 2))
            clean_result = detector.score(clean_gaps, baseline)
            covert_result = detector.score(covert_gaps, baseline)

            results[slice_type] = SliceDemoResult(
                slice_type=slice_type,
                offset_s=offset_s,
                adaptive=adaptive,
                threshold=threshold,
                clean_statistic=clean_result.statistic,
                clean_flag=clean_result.anomaly,
                covert_statistic=covert_result.statistic,
                covert_flag=covert_result.anomaly,
            )
        return results
    finally:
        topo.teardown()


def _capture_one_plan(
    capture_tool: str,
    host_if: str,
    output_path: str,
    plan: list[TrafficPacketPlan],
    capture_duration_s: float,
) -> np.ndarray:
    cmd = build_capture_cmd(capture_tool, host_if, output_path, duration_s=capture_duration_s)
    capture_handle = subprocess.Popen(cmd)
    time.sleep(1)  # let the capture tool attach before traffic starts
    send_traffic_plan(plan, iface=host_if)
    try:
        capture_handle.wait(timeout=capture_duration_s + 10)
    except subprocess.TimeoutExpired:
        capture_handle.kill()
        capture_handle.wait()
    df = parse_pcap_to_dataframe(output_path)
    return df["inter_arrival_s"].dropna().to_numpy()


def plot_clean_vs_covert(
    results: dict[str, SliceDemoResult],
    clean_gaps_by_slice: dict[str, np.ndarray],
    covert_gaps_by_slice: dict[str, np.ndarray],
    output_path: str,
) -> str:
    """Saves a per-slice clean-vs-covert gap-distribution comparison PNG.
    Only ever called with real, already-computed gap arrays -- never
    fabricated. Returns output_path for convenience."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SLICE_TYPES), figsize=(5 * len(SLICE_TYPES), 4), sharey=False)
    for ax, slice_type in zip(axes, SLICE_TYPES):
        ax.hist(clean_gaps_by_slice[slice_type], bins=40, alpha=0.6, label="clean", density=True)
        ax.hist(covert_gaps_by_slice[slice_type], bins=40, alpha=0.6, label="covert", density=True)
        result = results[slice_type]
        ax.set_title(
            f"{slice_type}\noffset={result.offset_s * 1000:.2f}ms  "
            f"D_clean={result.clean_statistic:.3f}  D_covert={result.covert_statistic:.3f}"
        )
        ax.set_xlabel("inter-packet gap (s)")
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print("Synthetic (in-memory) Phase 2 demo -- no root required.\n")

    print("--- single end-to-end run (one draw per slice; illustrative, not an accuracy estimate) ---")
    results = run_synthetic_demo(offset_s=0.02, adaptive=False)
    for slice_type, r in results.items():
        print(
            f"  {slice_type:6s}: threshold={r.threshold:.4f}  "
            f"clean D={r.clean_statistic:.4f} flag={r.clean_flag}  "
            f"covert D={r.covert_statistic:.4f} flag={r.covert_flag}"
        )

    print("\n--- aggregate detection-accuracy sweep (30 independent trials per cell) ---")
    offsets = {"large (20ms)": 0.02, "moderate (4ms)": 0.004, "marginal (0.6ms)": 0.0006}
    sweep = run_synthetic_accuracy_sweep(offsets, n_trials=30)
    for slice_type in SLICE_TYPES:
        fa = sweep[(slice_type, "large (20ms)")].false_alarm_rate  # same for every label at this slice
        print(f"  {slice_type:6s}: false-alarm rate (clean-vs-clean) = {fa:.2%}")
        for label in offsets:
            r = sweep[(slice_type, label)]
            print(f"    offset={label:18s}: detection rate = {r.detection_rate:.2%}")

    print()
    available, reason = netns_privileges_available()
    print(f"Live-demo privileges available: {available} ({reason})")
