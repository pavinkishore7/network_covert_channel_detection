from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DAEAutoencoderDetector:
    """
    CNN Denoising Autoencoder for covert-channel detection.

    Training:
        corrupted clean grids -> clean grids

    Inference:
        clean / attacker grids -> reconstruction error

    Corruption is used ONLY during training.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int = 16,
        noise_std: float = 0.10,
    ):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.noise_std = noise_std

        self.model = self._build_model()

        self.threshold_: float | None = None
        self.mode_: str = "topk"

        self.mu_: float | None = None
        self.sigma_: float | None = None

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    def _build_model(self) -> keras.Model:

        h, w = self.input_shape

        inp = keras.Input(
            shape=(h, w, 1),
            name="ofdm_grid_input",
        )

        x = layers.Conv2D(
            16, 3, activation="relu", padding="same"
        )(inp)

        x = layers.Conv2D(
            8, 3, activation="relu", padding="same"
        )(x)

        x = layers.MaxPooling2D(
            2, padding="same"
        )(x)

        encoded_shape = x.shape[1:]

        x = layers.Flatten()(x)

        latent = layers.Dense(
            self.latent_dim,
            activation="relu",
            name="latent",
        )(x)

        flat_units = int(np.prod(encoded_shape))

        x = layers.Dense(
            flat_units,
            activation="relu",
        )(latent)

        x = layers.Reshape(encoded_shape)(x)

        x = layers.Conv2DTranspose(
            8,
            3,
            strides=2,
            activation="relu",
            padding="same",
        )(x)

        x = layers.Conv2DTranspose(
            16,
            3,
            strides=2,
            activation="relu",
            padding="same",
        )(x)

        x = layers.Resizing(
            h,
            w,
        )(x)

        out = layers.Conv2D(
            1,
            3,
            activation="linear",
            padding="same",
        )(x)

        model = keras.Model(
            inp,
            out,
            name="cnn_denoising_autoencoder",
        )

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        return model

    # --------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------

    def _prep(self, grids: np.ndarray) -> np.ndarray:

        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError(
                "Call fit() before inference."
            )

        x = grids.astype("float32")

        x = (x - self.mu_) / self.sigma_

        return x[..., np.newaxis]

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    def fit(
        self,
        clean_grids: np.ndarray,
        epochs: int = 20,
        batch_size: int = 8,
        verbose: int = 1,
    ):

        if clean_grids.ndim != 3:
            raise ValueError(
                "clean_grids must have shape "
                "(N, n_symbols, n_subcarriers)."
            )

        if tuple(clean_grids.shape[1:]) != tuple(
            self.input_shape
        ):
            raise ValueError(
                f"Expected (*, {self.input_shape[0]}, "
                f"{self.input_shape[1]}), "
                f"got {clean_grids.shape}."
            )

        clean = clean_grids.astype("float32")

        # Learn normalization ONLY from clean training data.
        self.mu_ = float(clean.mean())
        self.sigma_ = float(clean.std() + 1e-8)

        target = self._prep(clean)

        # Controlled corruption.
        rng = np.random.default_rng(42)

        noisy = clean + rng.normal(
            0.0,
            self.noise_std * self.sigma_,
            size=clean.shape,
        ).astype("float32")

        noisy = self._prep(noisy)

        history = self.model.fit(
            noisy,
            target,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=verbose,
        )

        return history

    # --------------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------------

    def reconstruction_error(
        self,
        grids: np.ndarray,
        mode: str = "topk",
    ) -> np.ndarray:

        if grids.ndim != 3:
            raise ValueError(
                "grids must have shape "
                "(N, n_symbols, n_subcarriers)."
            )

        x = self._prep(grids)

        recon = self.model.predict(
            x,
            verbose=0,
        )

        sq_err = (x - recon) ** 2

        if mode == "mean":

            return np.mean(
                sq_err,
                axis=(1, 2, 3),
            )

        if mode == "max":

            return np.max(
                sq_err,
                axis=(1, 2, 3),
            )

        if mode == "topk":

            flat = sq_err.reshape(
                sq_err.shape[0],
                -1,
            )

            k = max(
                1,
                int(0.01 * flat.shape[1]),
            )

            topk = np.partition(
                flat,
                -k,
                axis=1,
            )[:, -k:]

            return topk.mean(axis=1)

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    def calibrate(
        self,
        clean_val_grids: np.ndarray,
        percentile: float = 95.0,
        mode: str = "topk",
    ) -> float:

        errors = self.reconstruction_error(
            clean_val_grids,
            mode=mode,
        )

        self.threshold_ = float(
            np.percentile(
                errors,
                percentile,
            )
        )

        self.mode_ = mode

        return self.threshold_

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    def predict_anomaly(
        self,
        grids: np.ndarray,
    ) -> np.ndarray:

        if self.threshold_ is None:
            raise RuntimeError(
                "Call calibrate() first."
            )

        errors = self.reconstruction_error(
            grids,
            mode=self.mode_,
        )

        return errors > self.threshold_
