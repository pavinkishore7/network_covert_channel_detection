"""
Evaluate the CNN autoencoder covert-channel detector.

Experimental design
--------------------
The dataset contains:

    7 SNR values
    x 200 simulation samples per SNR
    x 3 classes

Class:
    0 = clean
    1 = non-adaptive attack
    2 = adaptive attack

Each simulation sample produces a matched triplet:

    clean
    non-adaptive
    adaptive

Therefore splitting is performed by simulation sample index so that
matched attack variants never cross train/validation/test boundaries.

Split per SNR:
    0-139   -> training
    140-169 -> validation
    170-199 -> test

Training:
    CLEAN ONLY

Validation:
    CLEAN ONLY

Testing:
    CLEAN
    NON-ADAPTIVE
    ADAPTIVE

Primary reconstruction score:
    topk = mean of the highest-error 1% of cells

Baseline:
    mean = full-grid mean squared reconstruction error

The detector is evaluated against the attacker models simulated by this
project. Results are NOT a claim about arbitrary real-world covert
channels.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from detector.cnn_autoencoder import CNNAutoencoderDetector


# ---------------------------------------------------------------------
# Paths and experimental constants
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")

X_PATH = RESULTS_DIR / "dataset_X.npy"
Y_PATH = RESULTS_DIR / "dataset_y.npy"
SNR_PATH = RESULTS_DIR / "dataset_snr.npy"

OUTPUT_CSV = RESULTS_DIR / "detector_results.csv"
OUTPUT_NPZ = RESULTS_DIR / "detector_errors.npz"

SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]

SAMPLES_PER_SNR = 200

TRAIN_COUNT = 140
VAL_COUNT = 30
TEST_COUNT = 30

TRAIN_END = TRAIN_COUNT
VAL_END = TRAIN_COUNT + VAL_COUNT

CLASS_CLEAN = 0
CLASS_NON_ADAPTIVE = 1
CLASS_ADAPTIVE = 2

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8

CALIBRATION_PERCENTILE = 95.0

PRIMARY_MODE = "topk"
BASELINE_MODE = "mean"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and TensorFlow random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the frozen full dataset."""

    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {X_PATH}")

    if not Y_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {Y_PATH}")

    if not SNR_PATH.exists():
        raise FileNotFoundError(f"Missing SNR metadata: {SNR_PATH}")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    snr = np.load(SNR_PATH)

    expected_shape = (
        len(SNR_VALUES) * SAMPLES_PER_SNR * 3,
        200,
        64,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected X shape: {X.shape}; "
            f"expected {expected_shape}"
        )

    if y.shape != (expected_shape[0],):
        raise ValueError(f"Unexpected y shape: {y.shape}")

    if snr.shape != (expected_shape[0],):
        raise ValueError(f"Unexpected SNR shape: {snr.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or Inf values.")

    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError(
            f"Unexpected classes: {np.unique(y)}"
        )

    return X, y, snr


# ---------------------------------------------------------------------
# Seed-aware split
# ---------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Split by original simulation sample.

    The dataset ordering is:

        SNR 0:
            sample 0: clean, nonadaptive, adaptive
            sample 1: clean, nonadaptive, adaptive
            ...
        SNR 5:
            ...

    This function reconstructs that grouping and ensures the three
    outputs generated from one simulation never cross a split boundary.
    """

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    samples_per_triplet = 3

    for snr_index, snr_value in enumerate(SNR_VALUES):

        snr_start = (
            snr_index
            * SAMPLES_PER_SNR
            * samples_per_triplet
        )

        for sample_idx in range(SAMPLES_PER_SNR):

            base = (
                snr_start
                + sample_idx * samples_per_triplet
            )

            triplet = [
                base,
                base + 1,
                base + 2,
            ]

            # Verify the dataset ordering before using it.
            expected_classes = [0, 1, 2]

            actual_classes = [
                int(y[i])
                for i in triplet
            ]

            if actual_classes != expected_classes:
                raise ValueError(
                    f"Unexpected class ordering at SNR={snr_value}, "
                    f"sample={sample_idx}: {actual_classes}"
                )

            actual_snrs = [
                float(snr[i])
                for i in triplet
            ]

            if not all(
                value == float(snr_value)
                for value in actual_snrs
            ):
                raise ValueError(
                    f"Unexpected SNR metadata at SNR={snr_value}, "
                    f"sample={sample_idx}: {actual_snrs}"
                )

            if sample_idx < TRAIN_END:
                train_indices.extend(triplet)

            elif sample_idx < VAL_END:
                val_indices.extend(triplet)

            else:
                test_indices.extend(triplet)

    # --------------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------------

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    val_indices = np.asarray(
        val_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    # --------------------------------------------------------------
    # Extract only the required classes for each stage
    # --------------------------------------------------------------

    # Training: clean ONLY.
    train_clean = train_indices[
        y[train_indices] == CLASS_CLEAN
    ]

    # Validation: clean ONLY.
    val_clean = val_indices[
        y[val_indices] == CLASS_CLEAN
    ]

    # Test: all three classes.
    test_clean = test_indices[
        y[test_indices] == CLASS_CLEAN
    ]

    test_nonadaptive = test_indices[
        y[test_indices] == CLASS_NON_ADAPTIVE
    ]

    test_adaptive = test_indices[
        y[test_indices] == CLASS_ADAPTIVE
    ]

    return {
        "train_clean": train_clean,
        "val_clean": val_clean,
        "test_clean": test_clean,
        "test_nonadaptive": test_nonadaptive,
        "test_adaptive": test_adaptive,
        "train_all": train_indices,
        "val_all": val_indices,
        "test_all": test_indices,
    }


# ---------------------------------------------------------------------
# Basic split validation
# ---------------------------------------------------------------------

def validate_split(
    split: dict[str, np.ndarray],
    y: np.ndarray,
) -> None:
    """Validate expected split sizes."""

    expected_train_clean = len(SNR_VALUES) * TRAIN_COUNT
    expected_val_clean = len(SNR_VALUES) * VAL_COUNT
    expected_test_per_class = len(SNR_VALUES) * TEST_COUNT

    assert len(split["train_clean"]) == expected_train_clean
    assert len(split["val_clean"]) == expected_val_clean

    assert len(split["test_clean"]) == expected_test_per_class
    assert (
        len(split["test_nonadaptive"])
        == expected_test_per_class
    )
    assert (
        len(split["test_adaptive"])
        == expected_test_per_class
    )

    assert np.all(
        y[split["train_clean"]] == CLASS_CLEAN
    )

    assert np.all(
        y[split["val_clean"]] == CLASS_CLEAN
    )

    assert np.all(
        y[split["test_clean"]] == CLASS_CLEAN
    )

    assert np.all(
        y[split["test_nonadaptive"]]
        == CLASS_NON_ADAPTIVE
    )

    assert np.all(
        y[split["test_adaptive"]]
        == CLASS_ADAPTIVE
    )

    # No train/test overlap.
    train_set = set(split["train_all"])
    val_set = set(split["val_all"])
    test_set = set(split["test_all"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_binary_metrics(
    clean_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Calculate threshold-based detection metrics.

    Clean:
        label 0

    Attack:
        label 1
    """

    y_true = np.concatenate(
        [
            np.zeros(len(clean_errors), dtype=np.int64),
            np.ones(len(attack_errors), dtype=np.int64),
        ]
    )

    scores = np.concatenate(
        [
            clean_errors,
            attack_errors,
        ]
    )

    predictions = (
        scores > threshold
    ).astype(np.int64)

    # Attack detection rate / TPR.
    attack_predictions = predictions[
        len(clean_errors):
    ]

    detection_rate = float(
        attack_predictions.mean()
    )

    # Clean false-positive rate.
    clean_predictions = predictions[
        :len(clean_errors)
    ]

    false_positive_rate = float(
        clean_predictions.mean()
    )

    precision = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_true,
            scores,
        )
    )

    pr_auc = float(
        average_precision_score(
            y_true,
            scores,
        )
    )

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "clean_mean_error": float(
            np.mean(clean_errors)
        ),
        "attack_mean_error": float(
            np.mean(attack_errors)
        ),
        "clean_median_error": float(
            np.median(clean_errors)
        ),
        "attack_median_error": float(
            np.median(attack_errors)
        ),
    }


# ---------------------------------------------------------------------
# Per-SNR evaluation
# ---------------------------------------------------------------------

def evaluate_by_snr(
    detector: CNNAutoencoderDetector,
    X: np.ndarray,
    snr: np.ndarray,
    clean_indices: np.ndarray,
    attack_indices: np.ndarray,
    threshold: float,
    mode: str,
    attack_name: str,
) -> list[dict[str, float | str]]:
    """Calculate metrics separately for every SNR."""

    rows: list[dict[str, float | str]] = []

    for snr_value in SNR_VALUES:

        clean_snr_indices = clean_indices[
            snr[clean_indices] == snr_value
        ]

        attack_snr_indices = attack_indices[
            snr[attack_indices] == snr_value
        ]

        clean_errors = detector.reconstruction_error(
            X[clean_snr_indices],
            mode=mode,
        )

        attack_errors = detector.reconstruction_error(
            X[attack_snr_indices],
            mode=mode,
        )

        metrics = calculate_binary_metrics(
            clean_errors=clean_errors,
            attack_errors=attack_errors,
            threshold=threshold,
        )

        row: dict[str, float | str] = {
            "mode": mode,
            "attack": attack_name,
            "snr_db": float(snr_value),
            **metrics,
        }

        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_experiment(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 2026,
) -> None:

    print("=" * 70)
    print("CNN AUTOENCODER COVERT-CHANNEL DETECTION")
    print("=" * 70)

    print("\nLoading frozen dataset...")

    X, y, snr = load_dataset()

    print(f"X shape:   {X.shape}")
    print(f"y shape:   {y.shape}")
    print(f"SNR shape: {snr.shape}")

    # --------------------------------------------------------------
    # Split
    # --------------------------------------------------------------

    print("\nCreating seed-aware train/validation/test split...")

    split = split_dataset(
        X=X,
        y=y,
        snr=snr,
    )

    validate_split(
        split=split,
        y=y,
    )

    print(
        f"Training clean grids   : "
        f"{len(split['train_clean'])}"
    )

    print(
        f"Validation clean grids : "
        f"{len(split['val_clean'])}"
    )

    print(
        f"Test clean grids       : "
        f"{len(split['test_clean'])}"
    )

    print(
        f"Test non-adaptive      : "
        f"{len(split['test_nonadaptive'])}"
    )

    print(
        f"Test adaptive          : "
        f"{len(split['test_adaptive'])}"
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    set_global_seed(seed)

    detector = CNNAutoencoderDetector(
        input_shape=X.shape[1:],
        latent_dim=16,
    )

    print("\nModel parameters:")
    print(
        f"  {detector.model.count_params():,}"
    )

    print("\nTraining on CLEAN grids ONLY...")
    print(f"Epochs     : {epochs}")
    print(f"Batch size : {batch_size}")

    history = detector.fit(
        X[split["train_clean"]],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    final_loss = float(
        history.history["loss"][-1]
    )

    print(
        f"\nFinal training loss: "
        f"{final_loss:.8f}"
    )

    # --------------------------------------------------------------
    # Threshold calibration
    # --------------------------------------------------------------

    print("\nCalibrating threshold using CLEAN validation data...")

    primary_threshold = detector.calibrate(
        X[split["val_clean"]],
        percentile=CALIBRATION_PERCENTILE,
        mode=PRIMARY_MODE,
    )

    print(
        f"Primary mode : {PRIMARY_MODE}"
    )

    print(
        f"Percentile   : "
        f"{CALIBRATION_PERCENTILE}"
    )

    print(
        f"Threshold    : "
        f"{primary_threshold:.8f}"
    )

    # --------------------------------------------------------------
    # Clean validation FPR
    # --------------------------------------------------------------

    val_errors = detector.reconstruction_error(
        X[split["val_clean"]],
        mode=PRIMARY_MODE,
    )

    val_fpr = float(
        np.mean(
            val_errors > primary_threshold
        )
    )

    print(
        f"Validation false-positive rate: "
        f"{val_fpr:.2%}"
    )

    # --------------------------------------------------------------
    # Overall test reconstruction errors
    # --------------------------------------------------------------

    print("\nCalculating held-out test reconstruction errors...")

    test_clean_errors = detector.reconstruction_error(
        X[split["test_clean"]],
        mode=PRIMARY_MODE,
    )

    test_nonadaptive_errors = detector.reconstruction_error(
        X[split["test_nonadaptive"]],
        mode=PRIMARY_MODE,
    )

    test_adaptive_errors = detector.reconstruction_error(
        X[split["test_adaptive"]],
        mode=PRIMARY_MODE,
    )

    # --------------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------------

    nonadaptive_metrics = calculate_binary_metrics(
        clean_errors=test_clean_errors,
        attack_errors=test_nonadaptive_errors,
        threshold=primary_threshold,
    )

    adaptive_metrics = calculate_binary_metrics(
        clean_errors=test_clean_errors,
        attack_errors=test_adaptive_errors,
        threshold=primary_threshold,
    )

    print("\n" + "=" * 70)
    print("PRIMARY RESULTS — TOPK")
    print("=" * 70)

    print("\nNon-adaptive attacker:")

    print(
        f"  Detection rate : "
        f"{nonadaptive_metrics['detection_rate']:.2%}"
    )

    print(
        f"  False-positive  : "
        f"{nonadaptive_metrics['false_positive_rate']:.2%}"
    )

    print(
        f"  Precision       : "
        f"{nonadaptive_metrics['precision']:.2%}"
    )

    print(
        f"  Recall          : "
        f"{nonadaptive_metrics['recall']:.2%}"
    )

    print(
        f"  ROC-AUC         : "
        f"{nonadaptive_metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC          : "
        f"{nonadaptive_metrics['pr_auc']:.4f}"
    )

    print("\nAdaptive attacker:")

    print(
        f"  Detection rate : "
        f"{adaptive_metrics['detection_rate']:.2%}"
    )

    print(
        f"  False-positive  : "
        f"{adaptive_metrics['false_positive_rate']:.2%}"
    )

    print(
        f"  Precision       : "
        f"{adaptive_metrics['precision']:.2%}"
    )

    print(
        f"  Recall          : "
        f"{adaptive_metrics['recall']:.2%}"
    )

    print(
        f"  ROC-AUC         : "
        f"{adaptive_metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC          : "
        f"{adaptive_metrics['pr_auc']:.4f}"
    )

    # --------------------------------------------------------------
    # Per-SNR primary results
    # --------------------------------------------------------------

    primary_rows = []

    primary_rows.extend(
        evaluate_by_snr(
            detector=detector,
            X=X,
            snr=snr,
            clean_indices=split["test_clean"],
            attack_indices=split["test_nonadaptive"],
            threshold=primary_threshold,
            mode=PRIMARY_MODE,
            attack_name="non-adaptive",
        )
    )

    primary_rows.extend(
        evaluate_by_snr(
            detector=detector,
            X=X,
            snr=snr,
            clean_indices=split["test_clean"],
            attack_indices=split["test_adaptive"],
            threshold=primary_threshold,
            mode=PRIMARY_MODE,
            attack_name="adaptive",
        )
    )

    print("\n" + "=" * 70)
    print("PER-SNR PRIMARY RESULTS")
    print("=" * 70)

    print(
        f"{'Attack':<16}"
        f"{'SNR':>6}"
        f"{'TPR':>10}"
        f"{'FPR':>10}"
        f"{'ROC-AUC':>12}"
        f"{'PR-AUC':>12}"
    )

    for row in primary_rows:
        print(
            f"{str(row['attack']):<16}"
            f"{float(row['snr_db']):>6.0f}"
            f"{float(row['detection_rate']):>10.2%}"
            f"{float(row['false_positive_rate']):>10.2%}"
            f"{float(row['roc_auc']):>12.4f}"
            f"{float(row['pr_auc']):>12.4f}"
        )

    # --------------------------------------------------------------
    # Mean-score baseline
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE RESULTS — MEAN RECONSTRUCTION ERROR")
    print("=" * 70)

    mean_threshold = detector.calibrate(
        X[split["val_clean"]],
        percentile=CALIBRATION_PERCENTILE,
        mode=BASELINE_MODE,
    )

    mean_clean_errors = detector.reconstruction_error(
        X[split["test_clean"]],
        mode=BASELINE_MODE,
    )

    mean_nonadaptive_errors = detector.reconstruction_error(
        X[split["test_nonadaptive"]],
        mode=BASELINE_MODE,
    )

    mean_adaptive_errors = detector.reconstruction_error(
        X[split["test_adaptive"]],
        mode=BASELINE_MODE,
    )

    mean_nonadaptive_metrics = calculate_binary_metrics(
        clean_errors=mean_clean_errors,
        attack_errors=mean_nonadaptive_errors,
        threshold=mean_threshold,
    )

    mean_adaptive_metrics = calculate_binary_metrics(
        clean_errors=mean_clean_errors,
        attack_errors=mean_adaptive_errors,
        threshold=mean_threshold,
    )

    print("\nNon-adaptive:")
    print(
        f"  Detection rate : "
        f"{mean_nonadaptive_metrics['detection_rate']:.2%}"
    )
    print(
        f"  ROC-AUC        : "
        f"{mean_nonadaptive_metrics['roc_auc']:.4f}"
    )

    print("\nAdaptive:")
    print(
        f"  Detection rate : "
        f"{mean_adaptive_metrics['detection_rate']:.2%}"
    )
    print(
        f"  ROC-AUC        : "
        f"{mean_adaptive_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------------------
    # Save reconstruction errors
    # --------------------------------------------------------------

    np.savez(
        OUTPUT_NPZ,
        topk_threshold=primary_threshold,
        mean_threshold=mean_threshold,
        topk_test_clean_errors=test_clean_errors,
        topk_test_nonadaptive_errors=test_nonadaptive_errors,
        topk_test_adaptive_errors=test_adaptive_errors,
        mean_test_clean_errors=mean_clean_errors,
        mean_test_nonadaptive_errors=mean_nonadaptive_errors,
        mean_test_adaptive_errors=mean_adaptive_errors,
        validation_topk_errors=val_errors,
    )

    # --------------------------------------------------------------
    # Save per-SNR results
    # --------------------------------------------------------------

    fieldnames = [
        "mode",
        "attack",
        "snr_db",
        "detection_rate",
        "false_positive_rate",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "threshold",
        "clean_mean_error",
        "attack_mean_error",
        "clean_median_error",
        "attack_median_error",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in primary_rows:
            writer.writerow(row)

    print("\nSaved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_NPZ}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT LIMITATION:"
    )

    print(
        "These results measure detection of the non-adaptive and "
        "adaptive attacker models simulated by this project. "
        "They do not establish detection of arbitrary real-world "
        "covert channels."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate CNN autoencoder covert-channel detector."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Global random seed.",
    )

    args = parser.parse_args()

    run_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
