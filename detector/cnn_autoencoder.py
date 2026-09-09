"""
CNN Autoencoder for covert-channel detection.

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
"""

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


class CNNAutoencoderDetector:
    def __init__(self, input_shape: tuple[int, int], latent_dim: int = 16, seed: int = 2026):
        """
        input_shape: (n_symbols, n_subcarriers) of a single OFDM grid window.
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        tf.keras.utils.set_random_seed(seed)
        self.model = self._build_model()
        self.threshold_: float | None = None  # set by calibrate()
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def _build_model(self) -> keras.Model:
        h, w = self.input_shape
        inp = keras.Input(shape=(h, w, 1))

        # Encoder
        x = layers.Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inp)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(8, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        encoded_shape = x.shape[1:]  # remember for decoder upsampling
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent", kernel_initializer="he_normal")(x)

        # Decoder
        flat_units = int(np.prod(encoded_shape))
        x = layers.Dense(flat_units, activation="relu", kernel_initializer="he_normal")(latent)
        x = layers.Reshape(encoded_shape)(x)
        x = layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        # Crop/pad back to exact input size (pooling can round dimensions)
        x = layers.Resizing(h, w)(x)
        out = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

        model = keras.Model(inp, out, name="cnn_autoencoder_detector")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _prep(self, grids: np.ndarray) -> np.ndarray:
        """Normalize grids with the clean-training statistics locked by ``fit``."""
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Call fit() before preparing data for detection.")
        x = (grids.astype("float32") - self.mu_) / self.sigma_
        return x[..., np.newaxis]

    def fit(self, clean_grids: np.ndarray, epochs: int = 20, batch_size: int = 8, verbose: int = 0):
        """Train ONLY on clean (non-attacked) grids and lock their scale."""
        clean_grids = clean_grids.astype("float32")
        self.mu_ = float(clean_grids.mean())
        self.sigma_ = float(clean_grids.std() + 1e-8)
        x = self._prep(clean_grids)
        history = self.model.fit(x, x, epochs=epochs, batch_size=batch_size,
                                  validation_split=0.1, verbose=verbose)
        return history

    def reconstruction_error(self, grids: np.ndarray, mode: str = "mean", top_fraction: float = 0.01,
                              region_mask: np.ndarray | None = None) -> np.ndarray:
        """Per-sample MSE reconstruction error. Higher = more anomalous.

        mode="mean" (default, original behavior) averages over every cell.
        mode="max" takes the single worst cell. mode="topk" averages the
        top_fraction worst cells — matches StructuredDAE.reconstruction_error
        so the two detectors can be compared under the same scoring scheme.

        region_mask, if given, restricts scoring to True cells only — either
        a single (H, W) boolean mask applied to every sample, or a per-sample
        (N, H, W) stack (the eMBB target mask varies per scenario in this
        dataset, so callers scoring the frozen dataset should pass the latter).
        """
        x = self._prep(grids)
        recon = self.model.predict(x, verbose=0)
        sq = np.square(x - recon)[..., 0]
        if region_mask is None:
            return _reduce(sq.reshape(len(sq), -1), mode, top_fraction)
        return _reduce_masked(sq, region_mask, mode, top_fraction)

    def calibrate(self, clean_val_grids: np.ndarray, percentile: float = 95.0, **score_kwargs):
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
    detector = CNNAutoencoderDetector(input_shape=clean.shape[1:])
    detector.fit(clean[:split], epochs=5, verbose=0)  # smoke test only
    detector.calibrate(clean[split:])

    clean_flags = detector.predict_anomaly(clean[split:])
    attack_flags = detector.predict_anomaly(attacked[split:])

    print(f"False alarm rate on held-out clean data: {clean_flags.mean():.2%}")
    print(f"Detection rate on adaptive-attacker data: {attack_flags.mean():.2%}")
    print("NOTE: 5 epochs on 30 samples is a smoke test, not a real result. "
          "Do not report these numbers anywhere.")
