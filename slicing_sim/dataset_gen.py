"""
Generate labeled OFDM-grid datasets for covert-channel detection.

Classes:
    0 = clean
    1 = non-adaptive attack
    2 = adaptive attack

Full dataset:
    7 SNR values x 200 samples x 3 classes = 4200 samples

Each sample has shape:
    (n_symbols, n_subcarriers) = (200, 64)

Smoke test:
    1 SNR x 10 samples x 3 classes = 30 samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from slicing_sim.ofdm_grid import OFDMGridConfig, NetworkSlicingSimulator
from covert_channel.attacker import (
    AttackerConfig,
    NonAdaptiveAttacker,
    AdaptiveAttacker,
)


RESULTS_DIR = Path("results")

DEFAULT_SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]
DEFAULT_SAMPLES_PER_SNR = 200

N_SUBCARRIERS = 64
N_SYMBOLS = 200
TARGET_SLICE = "eMBB"

CLASS_CLEAN = 0
CLASS_NON_ADAPTIVE = 1
CLASS_ADAPTIVE = 2


def generate_sample(
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one clean, non-adaptive, and adaptive sample.

    All three samples originate from the same clean interference grid
    and target-slice allocation.
    """

    # One simulator controls allocation randomness and AWGN randomness.
    sim_config = OFDMGridConfig(
        n_subcarriers=N_SUBCARRIERS,
        n_symbols=N_SYMBOLS,
        snr_db=snr_db,
        seed=seed,
    )

    simulator = NetworkSlicingSimulator(sim_config)

    allocations = simulator.allocate_slices()
    clean_grid = simulator.combined_interference_grid(allocations)

    target_mask = allocations[TARGET_SLICE].subcarrier_mask

    # Use deterministic but different attacker seeds.
    nonadaptive_config = AttackerConfig(
        n_covert_bits=32,
        target_slice=TARGET_SLICE,
        seed=seed + 1,
    )

    adaptive_config = AttackerConfig(
        n_covert_bits=32,
        target_slice=TARGET_SLICE,
        seed=seed + 2,
    )

    nonadaptive = NonAdaptiveAttacker(nonadaptive_config)
    adaptive = AdaptiveAttacker(adaptive_config)

    nonadaptive_grid = nonadaptive.inject(
        clean_grid,
        target_mask,
    )

    adaptive_grid = adaptive.inject(
        clean_grid,
        target_mask,
    )

    expected_shape = (N_SYMBOLS, N_SUBCARRIERS)

    assert clean_grid.shape == expected_shape, (
        f"Clean grid shape {clean_grid.shape} != {expected_shape}"
    )

    assert nonadaptive_grid.shape == expected_shape, (
        f"Non-adaptive grid shape {nonadaptive_grid.shape} != {expected_shape}"
    )

    assert adaptive_grid.shape == expected_shape, (
        f"Adaptive grid shape {adaptive_grid.shape} != {expected_shape}"
    )

    return clean_grid, nonadaptive_grid, adaptive_grid


def generate_dataset(
    snr_values: list[float],
    samples_per_snr: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the complete dataset.

    Returns:
        X       : grid samples, shape (N, 200, 64)
        y       : class labels, shape (N,)
        snr_arr : SNR associated with every sample, shape (N,)
    """

    samples: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[float] = []

    for snr_db in snr_values:
        print(
            f"Generating SNR={snr_db} dB "
            f"({samples_per_snr} samples/class)..."
        )

        for sample_idx in range(samples_per_snr):
            seed = base_seed + (
                int(snr_db) * samples_per_snr
                + sample_idx
            ) * 10

            clean, nonadaptive, adaptive = generate_sample(
                snr_db=snr_db,
                seed=seed,
            )

            samples.extend(
                [
                    clean,
                    nonadaptive,
                    adaptive,
                ]
            )

            labels.extend(
                [
                    CLASS_CLEAN,
                    CLASS_NON_ADAPTIVE,
                    CLASS_ADAPTIVE,
                ]
            )

            snrs.extend(
                [
                    snr_db,
                    snr_db,
                    snr_db,
                ]
            )

    X = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    snr_arr = np.asarray(snrs, dtype=np.float32)

    expected_samples = len(snr_values) * samples_per_snr * 3

    assert X.shape == (
        expected_samples,
        N_SYMBOLS,
        N_SUBCARRIERS,
    ), f"Unexpected X shape: {X.shape}"

    assert y.shape == (expected_samples,), (
        f"Unexpected y shape: {y.shape}"
    )

    assert snr_arr.shape == (expected_samples,), (
        f"Unexpected SNR shape: {snr_arr.shape}"
    )

    assert np.isfinite(X).all(), "Dataset contains NaN or Inf values."

    return X, y, snr_arr


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    snr_arr: np.ndarray,
    prefix: str,
) -> None:
    """Save dataset arrays under results/."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(RESULTS_DIR / f"{prefix}_X.npy", X)
    np.save(RESULTS_DIR / f"{prefix}_y.npy", y)
    np.save(RESULTS_DIR / f"{prefix}_snr.npy", snr_arr)

    print("\nSaved:")
    print(f"  {RESULTS_DIR / f'{prefix}_X.npy'}")
    print(f"  {RESULTS_DIR / f'{prefix}_y.npy'}")
    print(f"  {RESULTS_DIR / f'{prefix}_snr.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate covert-channel detection dataset."
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small 1-SNR/10-sample-per-class smoke dataset.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )

    args = parser.parse_args()

    if args.smoke:
        snr_values = [10]
        samples_per_snr = 10
        prefix = "dataset_smoke"
    else:
        snr_values = DEFAULT_SNR_VALUES
        samples_per_snr = DEFAULT_SAMPLES_PER_SNR
        prefix = "dataset"

    print("=== Covert-channel dataset generation ===")
    print(f"SNR values       : {snr_values}")
    print(f"Samples/SNR/class: {samples_per_snr}")
    print(f"Classes           : 3")
    print(f"Base seed         : {args.seed}")

    X, y, snr_arr = generate_dataset(
        snr_values=snr_values,
        samples_per_snr=samples_per_snr,
        base_seed=args.seed,
    )

    print("\nDataset summary:")
    print(f"  X shape   : {X.shape}")
    print(f"  y shape   : {y.shape}")
    print(f"  SNR shape : {snr_arr.shape}")

    unique, counts = np.unique(y, return_counts=True)

    print("\nClass counts:")
    for label, count in zip(unique, counts):
        print(f"  class {label}: {count}")

    save_dataset(
        X=X,
        y=y,
        snr_arr=snr_arr,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
