"""
CNN Autoencoder for covert-channel detection — normalization fix.

BUG FIXED: the original _prep() recomputed mu/sigma from whatever array
was passed to it, on every call. This meant test/attack data was being
normalized using statistics derived from itself, silently erasing any
anomaly that expressed itself as a shift in the grid's global mean or
variance -- exactly the kind of anomaly an attacker optimizing to match
the legitimate interference distribution would produce. Fix: compute
mu/sigma ONCE from clean training data in fit(), store them, and reuse
those locked values in every subsequent call to _prep().
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNNAutoencoderDetector:
    def __init__(self, input_shape: tuple[int, int], latent_dim: int = 16):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = self._build_model()
        self.threshold_: float | None = None
        self.mu_: float | None = None       # locked at fit() time
        self.sigma_: float | None = None    # locked at fit() time

    def _build_model(self) -> keras.Model:
        h, w = self.input_shape
        inp = keras.Input(shape=(h, w, 1))

        x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        encoded_shape = x.shape[1:]
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        flat_units = int(np.prod(encoded_shape))
        x = layers.Dense(flat_units, activation="relu")(latent)
        x = layers.Reshape(encoded_shape)(x)
        x = layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Resizing(h, w)(x)
        out = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

        model = keras.Model(inp, out, name="cnn_autoencoder_detector")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _prep(self, grids: np.ndarray) -> np.ndarray:
        """grids: (N, n_symbols, n_subcarriers) -> normalized (N, h, w, 1).

        Uses self.mu_/self.sigma_ locked during fit(). Raises if called
        before fit() -- there is no such thing as a sensible default that
        silently falls back to per-call stats, because that IS the bug.
        """
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError(
                "Normalization stats not locked yet. Call fit() on clean "
                "training data before _prep()/reconstruction_error()."
            )
        x = grids.astype("float32")
        x = (x - self.mu_) / self.sigma_
        return x[..., np.newaxis]

    def fit(self, clean_grids: np.ndarray, epochs: int = 20, batch_size: int = 8, verbose: int = 0):
        """Train ONLY on clean (non-attacked) grids. Locks mu_/sigma_ here,
        from this data, once, before any training or evaluation happens."""
        clean_grids = clean_grids.astype("float32")
        self.mu_ = float(clean_grids.mean())
        self.sigma_ = float(clean_grids.std() + 1e-8)

        x = self._prep(clean_grids)
        history = self.model.fit(x, x, epochs=epochs, batch_size=batch_size,
                                  validation_split=0.1, verbose=verbose)
        return history

    def reconstruction_error(self, grids: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error, using LOCKED normalization
        stats from training -- not stats derived from `grids` itself."""
        x = self._prep(grids)
        recon = self.model.predict(x, verbose=0)
        return np.mean((x - recon) ** 2, axis=(1, 2, 3))

    def calibrate(self, clean_val_grids: np.ndarray, percentile: float = 95.0):
        errors = self.reconstruction_error(clean_val_grids)
        self.threshold_ = float(np.percentile(errors, percentile))
        return self.threshold_

    def predict_anomaly(self, grids: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict_anomaly().")
        return self.reconstruction_error(grids) > self.threshold_