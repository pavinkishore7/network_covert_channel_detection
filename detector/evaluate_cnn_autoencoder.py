"""Run the settled CNNAutoencoderDetector architecture on the frozen dataset.

Mirrors evaluate_structured_dae.py's protocol exactly (same matched_split,
same train+valid combined per-SNR calibration, same output columns) so the
two detectors are directly comparable. StructuredDAE and CNNAutoencoderDetector
are different architectures — this script evaluates the one documented as
"settled" in cnn_autoencoder.py; evaluate_structured_dae.py evaluates the
other. Neither result stands in for the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from detector.evaluate_structured_dae import matched_split, target_masks_for_rows


def _metrics(clean_scores: np.ndarray, attack_scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    scores = np.r_[clean_scores, attack_scores]
    return {"fpr": float((clean_scores > threshold).mean()), "detection_rate": float((attack_scores > threshold).mean()),
            "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def detection_rate_at_fpr(clean_scores: np.ndarray, attack_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    scores = np.r_[clean_scores, attack_scores]
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def main() -> None:
    from detector.cnn_autoencoder import CNNAutoencoderDetector

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
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
    model = CNNAutoencoderDetector(X.shape[1:], seed=args.seed)
    model.fit(X[train][y[train] == 0], epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # Same protocol as evaluate_structured_dae.py: calibrate each SNR band's
    # threshold from ALL non-test clean samples at that SNR (train + valid,
    # ~170 per band), not just the 30-sample validation slice.
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
    output.to_csv(args.results_dir / "cnn_autoencoder_results.csv", index=False)
    np.savez(args.results_dir / "cnn_autoencoder_errors.npz", scores=scores, y=test_y, snr=test_snr,
             thresholds=np.array(sorted(thresholds.items())))
    (args.results_dir / "cnn_autoencoder_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(output.to_string(index=False))
    print("\nPer-SNR thresholds (clean train+valid 95th percentile):")
    for snr_value in sorted(thresholds):
        print(f"  SNR {snr_value:>3}: {thresholds[snr_value]:.6f}")


if __name__ == "__main__":
    main()
