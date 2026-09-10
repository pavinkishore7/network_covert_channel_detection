# detector/legacy

NumPy-only, dependency-light detectors: `AdaptiveResidualDetector`
(`adaptive_residual.py`, a robust median/MAD residual guard) and
`ConvolutionalPatchAutoencoder` (`convolutional_autoencoder.py`, a
PCA-based patch autoencoder). Both run without TensorFlow installed.

Per `docs/DECISIONS.md`'s 2026-09 entry, these exist as a reproducible
fallback/benchmark for environments that can't run the TensorFlow
experiments — **not** as a replacement for, or an equivalent to, the
project's canonical detector. The results reported for this project come
from `detector/autoencoder_detector.py` (`AutoencoderDetector`, via its
`cnn_preset` and `structured_dae_preset`), evaluated in
`evaluate_cnn_autoencoder.py` / `evaluate_structured_dae.py`.
