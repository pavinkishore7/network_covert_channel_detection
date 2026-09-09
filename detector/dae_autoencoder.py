"""Clean-only structured denoising autoencoder for OFDM grids.

Only :meth:`fit` corrupts inputs.  Validation and test grids are always fed
to the network unchanged, and normalization statistics are fitted once from
the clean training split.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _reduce(flat: np.ndarray, mode: str, top_fraction: float) -> np.ndarray:
    if mode == "mean":
        return flat.mean(axis=1)
    if mode == "max":
        return flat.max(axis=1)
    if mode == "topk":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        k = max(1, int(top_fraction * flat.shape[1]))
        return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)
    raise ValueError("mode must be one of: mean, max, topk")


def _reduce_1d(vals: np.ndarray, mode: str, top_fraction: float) -> float:
    if mode == "mean":
        return float(vals.mean())
    if mode == "max":
        return float(vals.max())
    if mode == "topk":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        k = max(1, int(top_fraction * vals.size))
        return float(np.partition(vals, -k)[-k:].mean())
    raise ValueError("mode must be one of: mean, max, topk")


def _reduce_masked(sq: np.ndarray, region_mask: np.ndarray, mode: str, top_fraction: float) -> np.ndarray:
    """sq: (N, H, W) squared error. region_mask: (H, W) or (N, H, W) bool."""
    region_mask = np.asarray(region_mask, dtype=bool)
    if region_mask.ndim == 2:
        region_mask = np.broadcast_to(region_mask, sq.shape)
    if region_mask.shape != sq.shape:
        raise ValueError(f"region_mask shape {region_mask.shape} does not match error grid shape {sq.shape}")
    out = np.empty(len(sq), dtype=np.float64)
    for i in range(len(sq)):
        vals = sq[i][region_mask[i]]
        if vals.size == 0:
            raise ValueError(f"region_mask for sample {i} selects no cells")
        out[i] = _reduce_1d(vals, mode, top_fraction)
    return out


class StructuredDAE:
    """A CNN DAE with local cell, time, and frequency masking corruption."""

    def __init__(self, input_shape: tuple[int, int], latent_dim: int = 32, seed: int = 2026):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        tf.keras.utils.set_random_seed(seed)
        self.mu_: float | None = None
        self.sigma_: float | None = None
        self.threshold_: float | None = None
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        h, w = self.input_shape
        inp = keras.Input((h, w, 1))
        x = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inp)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        encoded_shape = x.shape[1:]
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent", kernel_initializer="he_normal")(x)
        x = layers.Dense(int(np.prod(encoded_shape)), activation="relu", kernel_initializer="he_normal")(latent)
        x = layers.Reshape(encoded_shape)(x)
        x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.Resizing(h, w)(x)
        out = layers.Conv2D(1, 3, activation="linear", padding="same")(x)
        model = keras.Model(inp, out, name="structured_dae")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _normalize(self, grids: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Call fit() before scoring data.")
        return ((grids.astype("float32") - self.mu_) / self.sigma_)[..., None]

    def corrupt(self, clean_normalized: np.ndarray, mask_probability: float = 0.08) -> np.ndarray:
        """Mask cells and short contiguous time/frequency regions to zero.

        This routine accepts normalized data and never mutates its input.
        The masking distribution is independent of class labels and is used
        exclusively while fitting clean grids.
        """
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be between 0 and 1")
        x = clean_normalized.copy()
        n, h, w, _ = x.shape
        x[self.rng.random((n, h, w, 1)) < mask_probability] = 0.0
        # One short time and frequency dropout per example.  Their size is
        # deliberately bounded so this is local-structure denoising, not a
        # hidden attack simulation.
        for i in range(n):
            t_len = int(self.rng.integers(1, min(9, h) + 1))
            f_len = int(self.rng.integers(1, min(9, w) + 1))
            t0 = int(self.rng.integers(0, h - t_len + 1))
            f0 = int(self.rng.integers(0, w - f_len + 1))
            x[i, t0:t0 + t_len, :, :] = 0.0
            x[i, :, f0:f0 + f_len, :] = 0.0
        return x

    def fit(self, clean_train: np.ndarray, *, epochs: int = 30, batch_size: int = 16,
            mask_probability: float = 0.08, verbose: int = 0):
        """Fit statistics and train from corrupted clean inputs to clean targets."""
        self.mu_ = float(clean_train.mean())
        self.sigma_ = float(clean_train.std() + 1e-8)
        target = self._normalize(clean_train)
        source = self.corrupt(target, mask_probability)
        return self.model.fit(source, target, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def reconstruction_error(self, grids: np.ndarray, mode: str = "topk", top_fraction: float = 0.01,
                              region_mask: np.ndarray | None = None) -> np.ndarray:
        """region_mask, if given, restricts scoring to True cells only —
        either a single (H, W) mask for every sample, or a per-sample
        (N, H, W) stack (the eMBB target mask varies per scenario in the
        frozen dataset, so scoring it needs the latter)."""
        x = self._normalize(grids)
        recon = self.model.predict(x, verbose=0)
        sq = np.square(x - recon)[..., 0]
        if region_mask is None:
            return _reduce(sq.reshape(len(sq), -1), mode, top_fraction)
        return _reduce_masked(sq, region_mask, mode, top_fraction)

    def calibrate(self, clean_validation: np.ndarray, percentile: float = 95.0, **score_kwargs) -> float:
        self.threshold_ = float(np.percentile(self.reconstruction_error(clean_validation, **score_kwargs), percentile))
        return self.threshold_
