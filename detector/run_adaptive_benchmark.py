"""Generate a reproducible, held-out adaptive-attacker benchmark.

The benchmark is intentionally simulation-scoped.  It records FPR, recall,
ROC-AUC and PR-AUC separately for non-adaptive and adaptive attacks and never
uses the test partition to fit a threshold.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig, NonAdaptiveAttacker
from detector.adaptive_residual import AdaptiveResidualDetector
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig
from slicing_sim.channel import ChannelImpairmentConfig


def _sample(seed: int, kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    # SNR and impairment-profile variation are held in disjoint partitions.
    snr = (8, 12, 16, 20, 24)[seed % 5]
    profile = ("awgn", "urban_micro", "high_mobility")[seed % 3]
    sim = NetworkSlicingSimulator(OFDMGridConfig(
        seed=seed, snr_db=snr, channel=ChannelImpairmentConfig(profile=profile)
    ))
    allocations = sim.allocate_slices()
    clean = sim.combined_interference_grid(allocations)
    mask = allocations["eMBB"].subcarrier_mask
    if kind == "clean":
        return clean, mask, f"{profile}:{snr}"
    config = AttackerConfig(n_covert_bits=64, seed=seed, target_slice="eMBB")
    attacker = NonAdaptiveAttacker(config) if kind == "non_adaptive" else AdaptiveAttacker(config)
    return attacker.inject(clean, mask), mask, f"{profile}:{snr}"


def _collect(seeds: range, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [_sample(seed, kind) for seed in seeds]
    return (np.stack([sample[0] for sample in samples]), np.stack([sample[1] for sample in samples]),
            np.asarray([sample[2] for sample in samples]))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return float(((positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean()))


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = max(1, int(labels.sum()))
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def _row(name: str, clean_scores: np.ndarray, attack_scores: np.ndarray, thresholds: np.ndarray) -> dict[str, float | str]:
    labels = np.r_[np.zeros(len(clean_scores), dtype=int), np.ones(len(attack_scores), dtype=int)]
    scores = np.r_[clean_scores, attack_scores]
    return {
        "attack": name,
        "threshold": round(float(thresholds.mean()), 6),
        "false_positive_rate": round(float((clean_scores > thresholds).mean()), 6),
        "detection_recall": round(float((attack_scores > thresholds).mean()), 6),
        "roc_auc": round(_auc(labels, scores), 6),
        "pr_auc": round(_pr_auc(labels, scores), 6),
        "test_clean_samples": len(clean_scores),
        "test_attack_samples": len(attack_scores),
    }


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    # Fixed seed ranges make the split reproducible and disjoint.
    train, train_masks, train_profiles = _collect(range(0, 600), "clean")
    valid, valid_masks, valid_profiles = _collect(range(600, 900), "clean")
    clean, clean_masks, clean_profiles = _collect(range(900, 1200), "clean")
    naive, naive_masks, _ = _collect(range(900, 1200), "non_adaptive")
    adaptive, adaptive_masks, _ = _collect(range(900, 1200), "adaptive")
    detectors, thresholds = {}, {}
    for profile in np.unique(train_profiles):
        detector = AdaptiveResidualDetector(top_fraction=0.01, percentile=99.0).fit(
            train[train_profiles == profile], train_masks[train_profiles == profile]
        )
        thresholds[profile] = detector.calibrate(valid[valid_profiles == profile], valid_masks[valid_profiles == profile])
        detectors[profile] = detector

    def score_by_profile(grids, masks):
        return np.asarray([detectors[p].score(grids[i:i + 1], masks[i:i + 1])[0] for i, p in enumerate(clean_profiles)])
    clean_scores = score_by_profile(clean, clean_masks)
    naive_scores = score_by_profile(naive, naive_masks)
    adaptive_scores = score_by_profile(adaptive, adaptive_masks)
    sample_thresholds = np.asarray([thresholds[p] for p in clean_profiles])
    rows = [
        _row("non_adaptive", clean_scores, naive_scores, sample_thresholds),
        _row("adaptive", clean_scores, adaptive_scores, sample_thresholds),
    ]
    output = results_dir / "adaptive_benchmark_v1.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "detector": "AdaptiveResidualDetector",
        "scope": "OFDM impairment simulation; not a field-performance claim",
        "split": {"train_clean": 600, "validation_clean": 300, "test_per_class": 300},
        "snr_db": [8, 12, 16, 20, 24],
        "channel_profiles": ["awgn", "urban_micro", "high_mobility"],
        "threshold_percentile": 99.0,
        "top_fraction": 0.01,
    }
    (results_dir / "adaptive_benchmark_v1.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
