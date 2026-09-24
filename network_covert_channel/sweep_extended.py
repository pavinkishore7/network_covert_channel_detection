"""Extended timing-channel sweep: where detection breaks down, and how that depends on
traffic regularity.

Why (see network_covert_channel/README.md, "Where detection breaks down"): the original
sweep (results/phase2_sweep.csv) saturates at its smallest offset, 0.1 sigma. sigma there is
the overall std of a slice's gaps, which the bursty exponential component inflates, while
the steady component has only 5% jitter by default. A shift of 0.1 sigma is therefore several
steady-core standard deviations, and KS catches it easily.

This script reuses the original sweep's detector, injectors, window size, calibration and
seeding helpers, and measures:
  (a) offsets from 0.002 to 0.1 sigma at the default jitter, both injectors, all slices;
  (b) steady_jitter_frac in {0.05, 0.1, 0.2, 0.5} at fixed absolute offsets (0.01, 0.05 and
      0.1 x the DEFAULT-jitter sigma, so the delay in ms is the same across jitter values),
      non-adaptive injector, detector recalibrated on clean traffic at each jitter.

Output: results/phase2_sweep_extended.csv (floats rounded to 6 decimals).
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from network_covert_channel.covert_demo import DEFAULT_N_PACKETS
from network_covert_channel.covert_injector import CovertInjectorConfig
from network_covert_channel.sweep import (CALIBRATION_PAIRS, CALIBRATION_PERCENTILE, DEFAULT_SEED, DEFAULT_TRIALS,
                                          INJECTORS, SIGMA_SAMPLE_GAPS, _rate_with_ci, _rng)
from network_covert_channel.timing_detector import TimingKSDetector
from network_covert_channel.traffic import DEFAULT_BASE_GAP_S, DEFAULT_STEADY_JITTER_FRAC, generate_inter_packet_gaps
from slicing_sim.ofdm_grid import SLICE_PROFILES, SLICE_TYPES

OFFSETS_A = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
JITTERS_B = (0.05, 0.1, 0.2, 0.5)
OFFSETS_B = (0.01, 0.05, 0.1)
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "results" / "phase2_sweep_extended.csv"
# stream tags distinct from sweep.py's (0..5) so no random stream is shared with the original sweep
TAG = 100


def _steady_sd_ms(slice_type: str, jitter: float) -> float:
    b = SLICE_PROFILES[slice_type]["burstiness"]
    return jitter * DEFAULT_BASE_GAP_S * (1.0 - b) * 1e3


def _cell(slice_type, si, jitter, jj, injector_name, ii, k, oi, sigma_default_s, sigma_s,
          detector, baseline, trials, seed, far_tuple, part):
    offset_s = k * sigma_default_s
    gen = functools.partial(generate_inter_packet_gaps, steady_jitter_frac=jitter)
    hits, pert, rates = 0, [], []
    for t in range(trials):
        clean = gen(slice_type, DEFAULT_N_PACKETS, _rng(seed, TAG, si, jj, 4, ii, oi, t))
        inj = INJECTORS[injector_name](CovertInjectorConfig(
            n_covert_bits=DEFAULT_N_PACKETS, offset_s=offset_s,
            seed=int(np.random.SeedSequence([seed, TAG, si, jj, 5, ii, oi, t]).generate_state(1)[0])))
        covert = inj.inject(clean)
        pert.append(float(np.mean(np.abs(covert - clean))))
        rates.append(DEFAULT_N_PACKETS / float(covert.sum()))
        hits += detector.is_anomalous(covert, baseline, slice_type=slice_type)
    det, lo, hi = _rate_with_ci(int(hits), trials)
    far, flo, fhi = far_tuple
    return {"part": part, "slice": slice_type, "injector": injector_name, "steady_jitter_frac": jitter,
            "offset_over_sigma": k, "offset_ms": offset_s * 1e3,
            "offset_over_steady_sd": offset_s * 1e3 / _steady_sd_ms(slice_type, jitter),
            "sigma_ms_at_this_jitter": sigma_s * 1e3, "mean_abs_perturbation_ms": float(np.mean(pert)) * 1e3,
            "detection_rate": det, "det_ci_low": lo, "det_ci_high": hi, "far": far, "far_ci_low": flo,
            "far_ci_high": fhi, "covert_bits_per_s": float(np.mean(rates)), "trials": trials}


def run(trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rows = []
    for si, slice_type in enumerate(SLICE_TYPES):
        sigma_default_s = float(generate_inter_packet_gaps(slice_type, SIGMA_SAMPLE_GAPS, _rng(seed, si, 0)).std())
        for jj, jitter in enumerate(JITTERS_B):
            gen = functools.partial(generate_inter_packet_gaps, steady_jitter_frac=jitter)
            sigma_s = float(gen(slice_type, SIGMA_SAMPLE_GAPS, _rng(seed, TAG, si, jj, 0)).std())
            detector = TimingKSDetector(seed=seed)
            detector.calibrate(functools.partial(gen, slice_type, DEFAULT_N_PACKETS, _rng(seed, TAG, si, jj, 1)),
                               n_trials=CALIBRATION_PAIRS, percentile=CALIBRATION_PERCENTILE, slice_type=slice_type)
            baseline = gen(slice_type, DEFAULT_N_PACKETS, _rng(seed, TAG, si, jj, 2))
            clean_hits = sum(detector.is_anomalous(gen(slice_type, DEFAULT_N_PACKETS, _rng(seed, TAG, si, jj, 3, t)),
                                                   baseline, slice_type=slice_type) for t in range(trials))
            far_tuple = _rate_with_ci(int(clean_hits), trials)
            if jitter == DEFAULT_STEADY_JITTER_FRAC:
                for ii, name in enumerate(INJECTORS):
                    for oi, k in enumerate(OFFSETS_A):
                        rows.append(_cell(slice_type, si, jitter, jj, name, ii, k, oi, sigma_default_s, sigma_s,
                                          detector, baseline, trials, seed, far_tuple, "a_offsets"))
            for oi, k in enumerate(OFFSETS_B):
                rows.append(_cell(slice_type, si, jitter, jj, "non_adaptive", 0, k, 100 + oi, sigma_default_s,
                                  sigma_s, detector, baseline, trials, seed, far_tuple, "b_jitter"))
            print(f"{slice_type} jitter={jitter} done", flush=True)
    return pd.DataFrame(rows)


def breakdown_table(df: pd.DataFrame) -> pd.DataFrame:
    """Smallest tested offset/sigma with detection >= 90%, per slice, injector and jitter."""
    out = []
    for (part, s, inj, j), g in df.groupby(["part", "slice", "injector", "steady_jitter_frac"]):
        ok = g[g.detection_rate >= 0.9].sort_values("offset_over_sigma")
        out.append({"part": part, "slice": s, "injector": inj, "steady_jitter_frac": j,
                    "smallest_offset_over_sigma_det90": ok.offset_over_sigma.iloc[0] if len(ok) else None,
                    "offset_ms": ok.offset_ms.iloc[0] if len(ok) else None,
                    "offset_over_steady_sd": ok.offset_over_steady_sd.iloc[0] if len(ok) else None})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m network_covert_channel.sweep_extended")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    df = run(args.trials, args.seed).round(6)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(breakdown_table(df).to_string(index=False))
    print(f"wrote {OUTPUT_CSV} ({len(df)} rows) in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
