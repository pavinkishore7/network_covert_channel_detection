"""Numpy-only residual guard for evaluating adaptive covert-channel attacks.

This is a reproducible benchmark detector, not a replacement for the CNN
autoencoder.  It provides a dependency-light guardrail while TensorFlow models
are trained separately: clean training grids define robust per-cell baselines;
scores combine the strongest normalized residuals with target-slice energy.
"""

from __future__ import annotations

import numpy as np


class AdaptiveResidualDetector:
    """Clean-trained, robust residual detector with calibrated thresholding."""

    def __init__(self, top_fraction: float = 0.01, percentile: float = 99.0):
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        self.top_fraction = top_fraction
        self.percentile = percentile
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_center_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.threshold_: float | None = None

    @staticmethod
    def _features(grids: np.ndarray, target_masks: np.ndarray) -> np.ndarray:
        if target_masks.shape != grids.shape:
            raise ValueError("target_masks must match grids")
        rows = []
        for grid, mask in zip(grids, target_masks):
            values = grid[mask]
            if not len(values):
                raise ValueError("each target mask must select at least one resource element")
            centered = values - np.median(values)
            rows.append((
                float(np.std(values)),
                float(np.mean(np.abs(centered))),
                float(np.percentile(np.abs(centered), 95)),
                float(np.mean(centered ** 4) / (np.var(values) ** 2 + 1e-8)),
            ))
        return np.asarray(rows)

    def fit(self, clean_grids: np.ndarray, target_masks: np.ndarray | None = None) -> "AdaptiveResidualDetector":
        if clean_grids.ndim != 3 or len(clean_grids) < 2:
            raise ValueError("clean_grids must have shape (samples, symbols, subcarriers)")
        self.center_ = np.median(clean_grids, axis=0)
        mad = np.median(np.abs(clean_grids - self.center_), axis=0)
        self.scale_ = np.maximum(1.4826 * mad, 1e-4)
        if target_masks is not None:
            features = self._features(clean_grids, target_masks)
            self.feature_center_ = np.median(features, axis=0)
            self.feature_scale_ = np.maximum(1.4826 * np.median(np.abs(features - self.feature_center_), axis=0), 1e-5)
        return self

    def score(self, grids: np.ndarray, target_masks: np.ndarray | None = None) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before score().")
        if grids.ndim != 3 or grids.shape[1:] != self.center_.shape:
            raise ValueError("grid shape differs from clean training data")
        residual = np.abs((grids - self.center_) / self.scale_)
        flat = residual.reshape(len(grids), -1)
        k = max(1, int(flat.shape[1] * self.top_fraction))
        topk = np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)
        if target_masks is None:
            return topk
        if target_masks.shape != grids.shape:
            raise ValueError("target_masks must match grids")
        # Mean target-slice residual adds telemetry available to a slice monitor;
        # top-k prevents a sparse adaptive perturbation from being averaged away.
        masked_mean = np.array([
            residual[i][target_masks[i]].mean() if target_masks[i].any() else 0.0
            for i in range(len(grids))
        ])
        if self.feature_center_ is None or self.feature_scale_ is None:
            return 0.8 * topk + 0.2 * masked_mean
        feature_z = np.abs((self._features(grids, target_masks) - self.feature_center_) / self.feature_scale_)
        # Distributional features make the guard sensitive to an attacker that
        # shapes individual chips to look normal but changes slice-level tails.
        feature_score = np.partition(feature_z, -2, axis=1)[:, -2:].mean(axis=1)
        return 0.35 * topk + 0.10 * masked_mean + 0.55 * feature_score

    def calibrate(self, clean_validation: np.ndarray, target_masks: np.ndarray | None = None) -> float:
        self.threshold_ = float(np.percentile(self.score(clean_validation, target_masks), self.percentile))
        return self.threshold_

    def predict(self, grids: np.ndarray, target_masks: np.ndarray | None = None) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict().")
        return self.score(grids, target_masks) > self.threshold_
