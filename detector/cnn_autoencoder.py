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


class CNNAutoencoderDetector:
    def __init__(self, input_shape: tuple[int, int], latent_dim: int = 16):
        """
        input_shape: (n_symbols, n_subcarriers) of a single OFDM grid window.
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = self._build_model()
        self.threshold_: float | None = None  # set by calibrate()
        self.mode_: str = "mean"  # set by calibrate(), used by predict_anomaly()
        # Fixed normalization from clean training data — never recomputed
        # per batch (doing so would absorb variance-shift attacks into the
        # normalizer and hide them from reconstruction error).
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def _build_model(self) -> keras.Model:
        h, w = self.input_shape
        inp = keras.Input(shape=(h, w, 1))

        # Encoder
        x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
        x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        encoded_shape = x.shape[1:]  # remember for decoder upsampling
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        # Decoder
        flat_units = int(np.prod(encoded_shape))
        x = layers.Dense(flat_units, activation="relu")(latent)
        x = layers.Reshape(encoded_shape)(x)
        x = layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)
        # Crop/pad back to exact input size (pooling can round dimensions)
        x = layers.Resizing(h, w)(x)
        out = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

        model = keras.Model(inp, out, name="cnn_autoencoder_detector")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _prep(self, grids: np.ndarray) -> np.ndarray:
        """grids: (N, n_symbols, n_subcarriers) -> normalized (N, h, w, 1).

        Always uses mu_/sigma_ locked in during fit() on clean training data.
        """
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Call fit() before using the detector — "
                               "normalization stats come from clean training data.")
        x = grids.astype("float32")
        x = (x - self.mu_) / self.sigma_
        return x[..., np.newaxis]

    def fit(self, clean_grids: np.ndarray, epochs: int = 20, batch_size: int = 8, verbose: int = 0):
        """Train ONLY on clean (non-attacked) grids."""
        x_raw = clean_grids.astype("float32")
        self.mu_ = float(x_raw.mean())
        self.sigma_ = float(x_raw.std() + 1e-8)
        x = self._prep(clean_grids)
        history = self.model.fit(x, x, epochs=epochs, batch_size=batch_size,
                                  validation_split=0.1, verbose=verbose)
        return history

    def reconstruction_error(self, grids: np.ndarray, mode: str = "mean") -> np.ndarray:
        """mode='mean': original full-grid MSE (diluted by sparse attacks).
        mode='max': per-cell squared error, max over grid — sensitive to
        sparse, localized anomalies even if only ~30 cells are touched.
        mode='topk': mean of top 1% highest-error cells — more robust than
        max, still sensitive to sparse localized perturbation."""
        x = self._prep(grids)
        recon = self.model.predict(x, verbose=0)
        sq_err = (x - recon) ** 2
        if mode == "mean":
            return np.mean(sq_err, axis=(1, 2, 3))
        elif mode == "max":
            return np.max(sq_err, axis=(1, 2, 3))
        elif mode == "topk":
            flat = sq_err.reshape(sq_err.shape[0], -1)
            k = max(1, int(0.01 * flat.shape[1]))
            topk = np.partition(flat, -k, axis=1)[:, -k:]
            return topk.mean(axis=1)
        else:
            raise ValueError(f"unknown mode: {mode}")

    def calibrate(self, clean_val_grids: np.ndarray, percentile: float = 95.0,
                  mode: str = "mean"):
        """Set detection threshold from the tail of the CLEAN validation
        error distribution, using the same mode that will be used at
        inference. percentile=95 means ~5% false-alarm rate on clean data
        BY CONSTRUCTION — report this alongside any detection accuracy
        number, don't quote accuracy without the FPR it was calibrated
        at."""
        errors = self.reconstruction_error(clean_val_grids, mode=mode)
        self.threshold_ = float(np.percentile(errors, percentile))
        self.mode_ = mode
        return self.threshold_

    def predict_anomaly(self, grids: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict_anomaly().")
        return self.reconstruction_error(grids, mode=self.mode_) > self.threshold_


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
