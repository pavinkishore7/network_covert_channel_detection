"""Phase 2 evidence: per-slice detection rate and false-alarm rate of the
KS timing detector, with 95% Clopper-Pearson intervals.

    python -m network_covert_channel.sweep [--trials N] [--seed S]

Writes ``results/phase2_sweep.csv``, one row per (slice, injector, offset).

Design (every choice is also printed when the script runs):
  - Window: the demo's DEFAULT_N_PACKETS (300 packets), scored against one
    fixed clean baseline window per slice, as covert_demo.py does.
  - Offsets are multiples of each slice's own clean gap standard deviation
    (sigma_slice, measured from SIGMA_SAMPLE_GAPS clean gaps), so the same
    row means the same relative perturbation on every slice.
  - Detector: TimingKSDetector calibrated PER SLICE on that slice's clean
    gaps (95th percentile of CALIBRATION_PAIRS clean-vs-clean pairs).
  - detection_rate: fraction of ``trials`` covert windows flagged.
    far: fraction of ``trials`` fresh clean windows flagged (per slice,
    shared by every row of that slice).
  - mean_abs_perturbation_ms: mean |covert gap - clean gap| over ALL packets
    in the window (bit-0 packets contribute 0) -- what the injector
    actually applied, not its configured ceiling.
  - covert_bits_per_s: the injector encodes one bit per packet, so this is
    the covert window's packet rate, bits / sum(covert gaps), averaged over
    trials (the added delays lower the packet rate slightly).
  - CIs: scipy.stats.binomtest(k, n).proportion_ci(method="exact").
"""

from __future__ import annotations

import argparse
import csv
import functools
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from network_covert_channel.covert_demo import DEFAULT_N_PACKETS
from network_covert_channel.covert_injector import (
    AdaptiveCovertInjector,
    CovertInjectorConfig,
    NonAdaptiveCovertInjector,
)
from network_covert_channel.timing_detector import TimingKSDetector
from network_covert_channel.traffic import generate_inter_packet_gaps
from slicing_sim.ofdm_grid import SLICE_TYPES

DEFAULT_SEED = 2026
DEFAULT_TRIALS = 500
OFFSETS_OVER_SIGMA = (0.1, 0.25, 0.5, 1.0, 2.0)
INJECTORS = {"non_adaptive": NonAdaptiveCovertInjector, "adaptive": AdaptiveCovertInjector}
SIGMA_SAMPLE_GAPS = 200_000
CALIBRATION_PAIRS = 1000
CALIBRATION_PERCENTILE = 95.0
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "phase2_sweep.csv"

COLUMNS = [
    "slice", "injector", "offset_over_sigma", "offset_ms", "mean_abs_perturbation_ms",
    "detection_rate", "det_ci_low", "det_ci_high", "far", "far_ci_low", "far_ci_high", "covert_bits_per_s",
]


def _rate_with_ci(k: int, n: int) -> tuple[float, float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(ci.low), float(ci.high)


def _rng(seed: int, *path: int) -> np.random.Generator:
    """Independent, reproducible stream per (slice, purpose, trial)."""
    return np.random.default_rng(np.random.SeedSequence([seed, *path]))


def slice_profile(slice_type: str, seed: int) -> dict:
    si = SLICE_TYPES.index(slice_type)
    gaps = generate_inter_packet_gaps(slice_type, SIGMA_SAMPLE_GAPS, _rng(seed, si, 0))
    return {"sigma_s": float(gaps.std()), "mean_gap_s": float(gaps.mean())}


def run_sweep(trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED, n_packets: int = DEFAULT_N_PACKETS):
    rows, profiles = [], {}
    detector = TimingKSDetector(seed=seed)
    for si, slice_type in enumerate(SLICE_TYPES):
        profile = profiles[slice_type] = slice_profile(slice_type, seed)
        sampler = functools.partial(generate_inter_packet_gaps, slice_type, n_packets, _rng(seed, si, 1))
        detector.calibrate(sampler, n_trials=CALIBRATION_PAIRS, percentile=CALIBRATION_PERCENTILE,
                           slice_type=slice_type)
        baseline = generate_inter_packet_gaps(slice_type, n_packets, _rng(seed, si, 2))

        clean_hits = sum(
            detector.is_anomalous(generate_inter_packet_gaps(slice_type, n_packets, _rng(seed, si, 3, t)),
                                  baseline, slice_type=slice_type)
            for t in range(trials)
        )
        far, far_lo, far_hi = _rate_with_ci(clean_hits, trials)
        profile.update(far=far, far_ci=(far_lo, far_hi), threshold=detector.thresholds_[slice_type])

        for ii, (injector_name, injector_cls) in enumerate(INJECTORS.items()):
            for oi, k in enumerate(OFFSETS_OVER_SIGMA):
                offset_s = k * profile["sigma_s"]
                hits, abs_pert, bit_rates = 0, [], []
                for t in range(trials):
                    clean = generate_inter_packet_gaps(slice_type, n_packets, _rng(seed, si, 4, ii, oi, t))
                    injector = injector_cls(CovertInjectorConfig(
                        n_covert_bits=n_packets, offset_s=offset_s,
                        seed=int(np.random.SeedSequence([seed, si, 5, ii, oi, t]).generate_state(1)[0]),
                    ))
                    covert = injector.inject(clean)
                    abs_pert.append(float(np.mean(np.abs(covert - clean))))
                    bit_rates.append(n_packets / float(covert.sum()))
                    hits += detector.is_anomalous(covert, baseline, slice_type=slice_type)
                det, det_lo, det_hi = _rate_with_ci(int(hits), trials)
                rows.append({
                    "slice": slice_type, "injector": injector_name, "offset_over_sigma": k,
                    "offset_ms": offset_s * 1e3, "mean_abs_perturbation_ms": float(np.mean(abs_pert)) * 1e3,
                    "detection_rate": det, "det_ci_low": det_lo, "det_ci_high": det_hi,
                    "far": far, "far_ci_low": far_lo, "far_ci_high": far_hi,
                    "covert_bits_per_s": float(np.mean(bit_rates)),
                })
    return rows, profiles


def write_csv(rows: list[dict], path: Path = OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: (f"{row[c]:.6g}" if isinstance(row[c], float) else row[c]) for c in COLUMNS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m network_covert_channel.sweep")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    print(f"seed={args.seed} trials/cell={args.trials} window={DEFAULT_N_PACKETS} packets "
          f"calibration={CALIBRATION_PAIRS} clean pairs @ p{CALIBRATION_PERCENTILE:g} (per slice) "
          f"sigma from {SIGMA_SAMPLE_GAPS} clean gaps")
    started = time.perf_counter()
    rows, profiles = run_sweep(trials=args.trials, seed=args.seed)
    write_csv(rows)
    for slice_type, p in profiles.items():
        window_s = DEFAULT_N_PACKETS * p["mean_gap_s"]
        print(f"{slice_type:6s} sigma={p['sigma_s'] * 1e3:.2f} ms mean_gap={p['mean_gap_s'] * 1e3:.2f} ms "
              f"window={window_s:.2f} s ({1 / window_s:.4f} windows/s) threshold D={p['threshold']:.4f} "
              f"FAR={p['far']:.2%} [{p['far_ci'][0]:.2%}, {p['far_ci'][1]:.2%}]")
    print(f"wrote {OUTPUT_CSV} ({len(rows)} rows) in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
