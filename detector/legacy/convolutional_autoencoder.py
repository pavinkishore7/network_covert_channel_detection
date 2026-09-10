"""Dependency-free linear convolutional autoencoder for OFDM grids.

It learns a shared low-rank representation of 3x3 local patches from clean
grids. This is a linear convolutional autoencoder baseline: the same learned
encoder/decoder is applied at every grid location, so it can run where the
TensorFlow CNN implementation is unavailable.
"""

from __future__ import annotations

import numpy as np


class ConvolutionalPatchAutoencoder:
    def __init__(self, latent_dim: int = 4, percentile: float = 99.0, top_fraction: float = 0.01):
        if not 1 <= latent_dim <= 9:
            raise ValueError("latent_dim must be in [1, 9]")
        self.latent_dim = latent_dim
        self.percentile = percentile
        self.top_fraction = top_fraction
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.threshold_: float | None = None

    @staticmethod
    def _patches(grids: np.ndarray) -> np.ndarray:
        padded = np.pad(grids, ((0, 0), (1, 1), (1, 1)), mode="edge")
        windows = [padded[:, row:row + grids.shape[1], col:col + grids.shape[2]] for row in range(3) for col in range(3)]
        return np.stack(windows, axis=-1).reshape(-1, 9)

    def fit(self, clean_grids: np.ndarray) -> "ConvolutionalPatchAutoencoder":
        patches = self._patches(clean_grids.astype("float64"))
        self.mean_ = patches.mean(axis=0)
        _, _, vectors = np.linalg.svd(patches - self.mean_, full_matrices=False)
        self.components_ = vectors[:self.latent_dim]
        return self

    def reconstruction_error(self, grids: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit() before scoring grids.")
        patches = self._patches(grids.astype("float64"))
        centered = patches - self.mean_
        reconstruction = (centered @ self.components_.T) @ self.components_ + self.mean_
        patch_error = np.mean((patches - reconstruction) ** 2, axis=1)
        per_grid = patch_error.reshape(len(grids), -1)
        k = max(1, int(per_grid.shape[1] * self.top_fraction))
        return np.partition(per_grid, -k, axis=1)[:, -k:].mean(axis=1)

    def calibrate(self, clean_validation: np.ndarray) -> float:
        self.threshold_ = float(np.percentile(self.reconstruction_error(clean_validation), self.percentile))
        return self.threshold_

    def predict(self, grids: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict().")
        return self.reconstruction_error(grids) > self.threshold_
