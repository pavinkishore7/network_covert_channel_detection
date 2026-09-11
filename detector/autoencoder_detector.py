"""Unified TensorFlow autoencoder detector for covert-channel detection.

Settled architecture per team decision: CNN autoencoder ONLY.
(Earlier deck drafts inconsistently said "CNN+LSTM" in some slides and
"CNN Autoencoder" in others — this is now the single source of truth.
Update the PPT to match this, not the other way around.)

Approach: train on CLEAN (non-attacked) OFDM interference grids only, so
the model learns the normal statistical structure of legitimate multi-slice
interference. At inference, reconstruction error on a grid containing a
covert perturbation should be higher than on clean grids — anomaly = high
reconstruction error.

Honest limitation, stated once here, must also appear on your slide:
this detector is trained on and evaluated against the NON-ADAPTIVE and
ADAPTIVE attacker outputs THIS TEAM SIMULATES. Its ROC/AUC numbers say how
well it distinguishes YOUR attacker model from clean traffic — not a
general claim about detecting arbitrary covert channels in the wild.

This module replaces what were previously two separate near-duplicate
files (``cnn_autoencoder.py``'s ``CNNAutoencoderDetector`` and
``dae_autoencoder.py``'s ``StructuredDAE``) with a single class,
``AutoencoderDetector``, parameterized by the hyperparameters that used to
distinguish the two: filter widths, latent dimension, and whether training
corrupts its inputs (denoising) before reconstructing the clean target. Use
the ``cnn_preset`` / ``structured_dae_preset`` constructors to reproduce
the two original architectures exactly.
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


class AutoencoderDetector:
    """Conv2D encoder/decoder autoencoder anomaly detector for OFDM grids.

    Parameters mirror what previously distinguished ``CNNAutoencoderDetector``
    from ``StructuredDAE``: ``filters`` and ``latent_dim`` set the layer
    widths, ``denoise`` switches on corrupt-input/clean-target training (with
    ``mask_probability`` controlling the corruption rate), and
    ``default_mode`` sets the reconstruction-error reduction used when a
    caller doesn't specify one explicitly. Use :meth:`cnn_preset` or
    :meth:`structured_dae_preset` instead of calling this constructor
    directly, unless you need a genuinely new configuration.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        filters: tuple[int, int],
        latent_dim: int,
        *,
        denoise: bool = False,
        mask_probability: float = 0.08,
        default_epochs: int = 20,
        default_batch_size: int = 8,
        default_mode: str = "mean",
        seed: int = 2026,
    ):
        """
        input_shape: (n_symbols, n_subcarriers) of a single OFDM grid window.
        """
        self.input_shape = input_shape
        self.filters = filters
        self.latent_dim = latent_dim
        self.denoise = denoise
        self.mask_probability = mask_probability
        self.default_epochs = default_epochs
        self.default_batch_size = default_batch_size
        self.default_mode = default_mode
        self.rng = np.random.default_rng(seed)
        tf.keras.utils.set_random_seed(seed)
        self.model = self._build_model()
        self.threshold_: float | None = None  # set by calibrate()
        self.mu_: float | None = None
        self.sigma_: float | None = None

    @classmethod
    def cnn_preset(cls, input_shape: tuple[int, int], seed: int = 2026) -> "AutoencoderDetector":
        """Equivalent to the old ``CNNAutoencoderDetector(input_shape, latent_dim=16, seed=seed)``."""
        return cls(
            input_shape,
            filters=(16, 8),
            latent_dim=16,
            denoise=False,
            default_epochs=20,
            default_batch_size=8,
            default_mode="mean",
            seed=seed,
        )

    @classmethod
    def structured_dae_preset(cls, input_shape: tuple[int, int], seed: int = 2026) -> "AutoencoderDetector":
        """Equivalent to the old ``StructuredDAE(input_shape, latent_dim=32, seed=seed)``."""
        return cls(
            input_shape,
            filters=(32, 16),
            latent_dim=32,
            denoise=True,
            mask_probability=0.08,
            default_epochs=30,
            default_batch_size=16,
            default_mode="topk",
            seed=seed,
        )

    def _build_model(self) -> keras.Model:
        h, w = self.input_shape
        f1, f2 = self.filters
        inp = keras.Input(shape=(h, w, 1))

        # Encoder
        x = layers.Conv2D(f1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inp)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(f2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        encoded_shape = x.shape[1:]  # remember for decoder upsampling
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent", kernel_initializer="he_normal")(x)

        # Decoder
        flat_units = int(np.prod(encoded_shape))
        x = layers.Dense(flat_units, activation="relu", kernel_initializer="he_normal")(latent)
        x = layers.Reshape(encoded_shape)(x)
        x = layers.Conv2DTranspose(f2, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.Conv2DTranspose(f1, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        # Crop/pad back to exact input size (pooling can round dimensions)
        x = layers.Resizing(h, w)(x)
        out = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

        model = keras.Model(inp, out, name="autoencoder_detector")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _normalize(self, grids: np.ndarray) -> np.ndarray:
        """Normalize grids with the clean-training statistics locked by ``fit``."""
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Call fit() before preparing data for detection.")
        x = (grids.astype("float32") - self.mu_) / self.sigma_
        return x[..., np.newaxis]

    def corrupt(self, clean_normalized: np.ndarray, mask_probability: float = 0.08) -> np.ndarray:
        """Mask cells and short contiguous time/frequency regions to zero.

        This routine accepts normalized data and never mutates its input.
        The masking distribution is independent of class labels and is used
        exclusively while fitting clean grids (denoising presets only).
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

    def fit(self, clean_grids: np.ndarray, epochs: int | None = None, batch_size: int | None = None,
            mask_probability: float | None = None, verbose: int = 0):
        """Train ONLY on clean (non-attacked) grids and lock their scale.

        When ``denoise`` is set (structured-DAE preset), the network is fed
        corrupted inputs and trained to reconstruct the clean target; when
        it isn't (cnn preset), it trains directly on clean grids as its own
        target, matching the original ``CNNAutoencoderDetector`` behavior.
        """
        epochs = self.default_epochs if epochs is None else epochs
        batch_size = self.default_batch_size if batch_size is None else batch_size
        mask_probability = self.mask_probability if mask_probability is None else mask_probability
        clean_grids = clean_grids.astype("float32")
        self.mu_ = float(clean_grids.mean())
        self.sigma_ = float(clean_grids.std() + 1e-8)
        target = self._normalize(clean_grids)
        source = self.corrupt(target, mask_probability) if self.denoise else target
        history = self.model.fit(source, target, epochs=epochs, batch_size=batch_size,
                                  validation_split=0.1 if not self.denoise else 0.0, verbose=verbose)
        return history

    def reconstruction_error(self, grids: np.ndarray, mode: str | None = None, top_fraction: float = 0.01,
                              region_mask: np.ndarray | None = None) -> np.ndarray:
        """Per-sample MSE reconstruction error. Higher = more anomalous.

        mode=None uses this instance's ``default_mode`` ("mean" for the cnn
        preset, "topk" for the structured-DAE preset — matching the original
        two classes' defaults). "max" takes the single worst cell. "topk"
        averages the top_fraction worst cells.

        region_mask, if given, restricts scoring to True cells only — either
        a single (H, W) boolean mask applied to every sample, or a per-sample
        (N, H, W) stack (the eMBB target mask varies per scenario in this
        dataset, so callers scoring the frozen dataset should pass the latter).
        """
        mode = self.default_mode if mode is None else mode
        x = self._normalize(grids)
        recon = self.model.predict(x, verbose=0)
        sq = np.square(x - recon)[..., 0]
        if region_mask is None:
            return _reduce(sq.reshape(len(sq), -1), mode, top_fraction)
        return _reduce_masked(sq, region_mask, mode, top_fraction)

    def calibrate(self, clean_val_grids: np.ndarray, percentile: float = 95.0, **score_kwargs) -> float:
        """Set detection threshold from the tail of the CLEAN validation
        error distribution. percentile=95 means ~5% false-alarm rate on
        clean data BY CONSTRUCTION — report this alongside any detection
        accuracy number, don't quote accuracy without the FPR it was
        calibrated at."""
        errors = self.reconstruction_error(clean_val_grids, **score_kwargs)
        self.threshold_ = float(np.percentile(errors, percentile))
        return self.threshold_

    def predict_anomaly(self, grids: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict_anomaly().")
        return self.reconstruction_error(grids) > self.threshold_


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig
    from covert_channel.attacker import AdaptiveAttacker, AttackerConfig

    # Generate a small set of clean grids for a smoke test (NOT a real
    # training run — real training needs far more samples and epochs).
    n_samples = 40
    clean = []
    attacked = []
    for i in range(n_samples):
        cfg = OFDMGridConfig(seed=i)
        sim = NetworkSlicingSimulator(cfg)
        alloc = sim.allocate_slices()
        grid = sim.combined_interference_grid(alloc)
        clean.append(grid)

        atk = AdaptiveAttacker(AttackerConfig(seed=i))
        attacked.append(atk.inject(grid, alloc["eMBB"].subcarrier_mask))

    clean = np.array(clean)
    attacked = np.array(attacked)

    split = int(0.75 * n_samples)
    detector = AutoencoderDetector.cnn_preset(input_shape=clean.shape[1:])
    detector.fit(clean[:split], epochs=5, verbose=0)  # smoke test only
    detector.calibrate(clean[split:])

    clean_flags = detector.predict_anomaly(clean[split:])
    attack_flags = detector.predict_anomaly(attacked[split:])

    print(f"False alarm rate on held-out clean data: {clean_flags.mean():.2%}")
    print(f"Detection rate on adaptive-attacker data: {attack_flags.mean():.2%}")
    print("NOTE: 5 epochs on 30 samples is a smoke test, not a real result. "
          "Do not report these numbers anywhere.")
