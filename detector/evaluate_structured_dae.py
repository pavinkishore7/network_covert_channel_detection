"""Run one predeclared structured-DAE experiment on the frozen dataset.

This script intentionally has no test-set model selection.  Its train/valid/
test indices are based on matched simulation triplets, never class-wise rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig


def target_masks_for_rows(row_indices: np.ndarray) -> np.ndarray:
    """Reconstruct the per-scenario eMBB target_mask used by
    generate_frozen_dataset.py for each given row of the frozen dataset.

    The mask is NOT fixed across the dataset — allocate_slices() is
    re-randomized per scenario, so it varies both in time (bursty
    allocation) and by seed. It's deterministic given only the seed
    (allocate_slices does not depend on snr_db), and
    generate_frozen_dataset.py assigns scenario_seed = 0, 1, 2, ...
    sequentially, one per scenario, writing 3 rows (clean, non-adaptive,
    adaptive) per scenario in order — so row index r maps to
    scenario_seed = r // 3. Nothing about this uses attack labels; it's
    the known network configuration (subcarrier allocation), reconstructed
    the same deterministic way it was originally generated.
    """
    masks = np.empty((len(row_indices), 200, 64), dtype=bool)
    for i, row in enumerate(row_indices):
        scenario_seed = int(row) // 3
        sim = NetworkSlicingSimulator(OFDMGridConfig(seed=scenario_seed))
        masks[i] = sim.allocate_slices()["eMBB"].subcarrier_mask
    return masks


def matched_split(y: np.ndarray, snr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices for 140/30/30 matched triplets at every SNR."""
    train, valid, test = [], [], []
    for value in np.unique(snr):
        rows = np.flatnonzero(snr == value)
        if len(rows) != 600 or not np.array_equal(y[rows].reshape(-1, 3), np.tile([0, 1, 2], (200, 1))):
            raise ValueError(f"SNR {value}: expected 200 ordered [clean, non-adaptive, adaptive] triplets")
        triplets = rows.reshape(200, 3)
        train.extend(triplets[:140].ravel())
        valid.extend(triplets[140:170].ravel())
        test.extend(triplets[170:].ravel())
    return np.asarray(train), np.asarray(valid), np.asarray(test)


def _metrics(clean_scores: np.ndarray, attack_scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    scores = np.r_[clean_scores, attack_scores]
    return {"fpr": float((clean_scores > threshold).mean()), "detection_rate": float((attack_scores > threshold).mean()),
            "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def detection_rate_at_fpr(clean_scores: np.ndarray, attack_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """TPR at a fixed FPR on this band's ROC curve, via linear interpolation
    between the two bracketing (fpr, tpr) points from sklearn.roc_curve.

    This decouples "does the model separate clean from attack" from "did
    this band's small test sample happen to land a threshold-crossing
    sample" — the latter is what makes the raw threshold-based detection_rate
    noisy at n=30 test samples per band.
    """
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    scores = np.r_[clean_scores, attack_scores]
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    # Keep the split utility importable for protocol tests even on systems
    # that have not installed the TensorFlow experiment dependency yet.
    from detector.autoencoder_detector import AutoencoderDetector

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--mask-probability", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    np.random.seed(args.seed)
    paths = {name: args.results_dir / f"dataset_{name}.npy" for name in ("X", "y", "snr")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Frozen dataset missing: " + ", ".join(missing))
    X, y, snr = (np.load(paths[name]) for name in ("X", "y", "snr"))
    if X.shape != (4200, 200, 64) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Unexpected frozen dataset shape or labels")
    train, valid, test = matched_split(y, snr)
    model = AutoencoderDetector.structured_dae_preset(X.shape[1:], seed=args.seed)
    model.fit(X[train][y[train] == 0], epochs=args.epochs, batch_size=args.batch_size,
              mask_probability=args.mask_probability, verbose=2)

    # Calibrate each SNR band's threshold from ALL non-test clean samples at
    # that SNR (train + validation, ~170 per band) rather than just the
    # 30-sample validation slice — with n=30 the 95th percentile is close to
    # the 2nd-highest value, so a single outlier swings the threshold and the
    # downstream detection_rate substantially. This does not touch test data.
    calib = np.concatenate([train, valid])
    calib_y, calib_snr = y[calib], snr[calib]
    calib_masks = target_masks_for_rows(calib)
    calib_scores = model.reconstruction_error(X[calib], mode="mean", region_mask=calib_masks)
    thresholds = {}
    for snr_value in np.unique(calib_snr):
        clean_calib = calib_scores[(calib_snr == snr_value) & (calib_y == 0)]
        thresholds[snr_value] = float(np.percentile(clean_calib, 95))

    test_masks = target_masks_for_rows(test)
    scores = model.reconstruction_error(X[test], mode="mean", region_mask=test_masks)
    test_y, test_snr = y[test], snr[test]
    rows = []
    for snr_value in np.unique(test_snr):
        select = test_snr == snr_value
        clean = scores[select & (test_y == 0)]
        threshold = thresholds[snr_value]
        for label, name in ((1, "non_adaptive"), (2, "adaptive")):
            attack_scores = scores[select & (test_y == label)]
            row = {"snr": snr_value, "attack": name, "threshold": threshold,
                   "n_flagged": int((attack_scores > threshold).sum()), "n_total": int(len(attack_scores)),
                   "n_fp": int((clean > threshold).sum()), "n_clean": int(len(clean)),
                   **_metrics(clean, attack_scores, threshold),
                   "detection_rate_at_5pct_fpr": detection_rate_at_fpr(clean, attack_scores, target_fpr=0.05)}
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(args.results_dir / "structured_dae_results.csv", index=False)
    np.savez(args.results_dir / "structured_dae_errors.npz", scores=scores, y=test_y, snr=test_snr,
             thresholds=np.array(sorted(thresholds.items())))
    (args.results_dir / "structured_dae_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(output.to_string(index=False))
    print("\nPer-SNR thresholds (clean validation 95th percentile):")
    for snr_value in sorted(thresholds):
        print(f"  SNR {snr_value:>3}: {thresholds[snr_value]:.6f}")


if __name__ == "__main__":
    main()
