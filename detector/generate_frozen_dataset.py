"""Generate the frozen dataset consumed by detector/evaluate_structured_dae.py.

For each of 7 SNR levels (0..30 dB, step 5) and 200 independent scenarios,
build one clean OFDM grid and derive exactly 3 rows from it, in order:
  [0] clean grid (label 0)
  [1] clean grid + NonAdaptiveAttacker injection (label 1)
  [2] clean grid + AdaptiveAttacker injection (label 2)
so every triplet of 3 consecutive rows shares one base grid, matching the
matched_split protocol in detector/evaluate_structured_dae.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig, NonAdaptiveAttacker
from detector.evaluate_structured_dae import matched_split
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

SNR_LEVELS = np.arange(0, 35, 5)
SCENARIOS_PER_SNR = 200


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_rows, y_rows, snr_rows = [], [], []
    scenario_seed = 0
    for snr_db in SNR_LEVELS:
        for _ in range(SCENARIOS_PER_SNR):
            sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=float(snr_db), seed=scenario_seed))
            allocations = sim.allocate_slices()
            clean_grid = sim.combined_interference_grid(allocations)
            target_mask = allocations["eMBB"].subcarrier_mask

            naive = NonAdaptiveAttacker(AttackerConfig(seed=scenario_seed))
            adaptive = AdaptiveAttacker(AttackerConfig(seed=scenario_seed))

            naive_grid = naive.inject(clean_grid, target_mask)
            adaptive_grid = adaptive.inject(clean_grid, target_mask)

            X_rows.extend([clean_grid, naive_grid, adaptive_grid])
            y_rows.extend([0, 1, 2])
            snr_rows.extend([snr_db, snr_db, snr_db])

            scenario_seed += 1

    X = np.stack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    snr = np.asarray(snr_rows, dtype=np.int64)
    return X, y, snr


def main() -> None:
    X, y, snr = build_dataset()

    expected_y = np.tile([0, 1, 2], SCENARIOS_PER_SNR * len(SNR_LEVELS))
    expected_snr = np.repeat(SNR_LEVELS, 600)
    assert X.shape == (4200, 200, 64), f"unexpected X shape: {X.shape}"
    assert np.array_equal(y, expected_y), "y layout does not match expected tiling"
    assert np.array_equal(snr, expected_snr), "snr layout does not match expected repeat"

    # Confirm matched_split accepts this layout before writing anything.
    matched_split(y, snr)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    np.save(results_dir / "dataset_X.npy", X)
    np.save(results_dir / "dataset_y.npy", y)
    np.save(results_dir / "dataset_snr.npy", snr)
    print(f"Saved dataset_X{X.shape}, dataset_y{y.shape}, dataset_snr{snr.shape} to {results_dir}/")


if __name__ == "__main__":
    main()
