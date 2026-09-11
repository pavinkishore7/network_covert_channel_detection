"""Generate a held-out result for the CPU-compatible convolutional autoencoder."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from detector.legacy.convolutional_autoencoder import ConvolutionalPatchAutoencoder
from detector.run_adaptive_benchmark import _auc, _collect, _pr_auc


def main() -> None:
    train, _, _ = _collect(range(0, 180), "clean")
    valid, _, _ = _collect(range(180, 270), "clean")
    clean, _, _ = _collect(range(270, 360), "clean")
    adaptive, _, _ = _collect(range(270, 360), "adaptive")
    detector = ConvolutionalPatchAutoencoder(latent_dim=4, percentile=95).fit(train)
    threshold = detector.calibrate(valid)
    clean_scores, attack_scores = detector.reconstruction_error(clean), detector.reconstruction_error(adaptive)
    labels, scores = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))], np.r_[clean_scores, attack_scores]
    row = {
        "model": "linear_convolutional_patch_autoencoder",
        "attack": "adaptive",
        "threshold": round(threshold, 8),
        "false_positive_rate": round(float((clean_scores > threshold).mean()), 6),
        "detection_recall": round(float((attack_scores > threshold).mean()), 6),
        "roc_auc": round(_auc(labels, scores), 6),
        "pr_auc": round(_pr_auc(labels, scores), 6),
        "test_clean_samples": len(clean_scores),
        "test_attack_samples": len(attack_scores),
    }
    output = Path("results/convolutional_autoencoder_v1.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(output); print(row)


if __name__ == "__main__":
    main()
