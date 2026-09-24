"""Fully-convolutional CNN detector (TensorFlow) for PHY covert channels.

Why this design (see docs/DECISIONS.md, 2026-09-24): the earlier CNN autoencoder squeezed
every grid through a 16-number dense bottleneck and scored the mean reconstruction error.
It collapsed to a near-constant reconstruction, and the mean diluted a 32-cell burst ~400x.
This model avoids both failure modes by construction:

* no dense bottleneck: every layer is convolutional, so the network outputs a per-cell
  evidence map the same size as the input and cannot "average the attack away";
* a smooth global maximum (log-sum-exp) of that map gives one score per grid, so the score
  depends on the most suspicious region wherever it is (translation-invariant, like the
  handcrafted scan);
* inputs are the raw grid plus the blind level residual (distance of each cell from the
  legal power levels), so the network starts from physically meaningful evidence; an
  optional third channel carries the allocation-aware residual for an observer that knows
  the schedule.

Training is supervised on SHORT CROPS (32 symbols x 64 subcarriers): positive crops contain
the whole covert burst; negative crops come from clean grids AND from attacked grids away
from the burst (hard negatives, which teach the network to localise rather than to react
to grid-level statistics). Because the network is fully convolutional, it is then applied
to full 200 x 64 grids unchanged.

Scope: simulation only (AWGN), the attacker classes in covert_channel/attacker.py.
"""

from __future__ import annotations

import numpy as np

from detector.scan_detector import allocation_residual, level_residual

CROP_T = 32


def build_inputs(grids: np.ndarray, scheduled: np.ndarray | None = None) -> np.ndarray:
    """Stack model input channels: raw grid, |level deviation| (and |observed - scheduled|)."""
    g = np.asarray(grids, dtype=np.float32)
    if g.ndim == 2:
        g = g[None]
    chans = [g, np.sqrt(level_residual(g)).astype(np.float32)]
    if scheduled is not None:
        s = np.asarray(scheduled, dtype=np.float32)
        if s.ndim == 2:
            s = s[None]
        chans.append(np.sqrt(allocation_residual(g, s)).astype(np.float32))
    return np.stack(chans, axis=-1)


_LSE = None


def _lse_pool_layer():
    global _LSE
    if _LSE is not None:
        return _LSE
    import tensorflow as tf
    from tensorflow import keras

    @tf.keras.utils.register_keras_serializable(package="covert")
    class LogSumExpPool(keras.layers.Layer):
        """Smooth maximum over all cells: logsumexp(evidence) - log(H*W).

        Behaves like max pooling (dominated by the most suspicious cells, independent of
        where they are) but passes gradient to every cell, which plain max pooling does
        not; with plain max pooling the network failed to train (val AUC ~0.55 after 6
        epochs vs ~0.86 with this layer, same data)."""

        def call(self, e):
            hw = tf.cast(tf.shape(e)[1] * tf.shape(e)[2], e.dtype)
            return tf.reduce_logsumexp(e, axis=[1, 2, 3])[:, None] - tf.math.log(hw)

    _LSE = LogSumExpPool
    return _LSE


def build_model(n_channels: int, seed: int = 2026):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(None, None, n_channels))
    x = inp
    # (filters, kernel, dilation): local 3x3 context, a dilated 3x3, then 1x9 kernels that
    # look along a symbol (the covert burst runs along the frequency axis of 1-2 symbols)
    for f, k, d in [(16, 3, 1), (16, 3, 2), (32, (1, 9), 1), (32, 3, 1), (32, (1, 9), 1)]:
        x = layers.Conv2D(f, k, padding="same", dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    evidence = layers.Conv2D(1, 1, padding="same", name="evidence_map")(x)  # per-cell logit
    score = _lse_pool_layer()(name="pool")(evidence)
    out = layers.Activation("sigmoid")(score)
    model = keras.Model(inp, out, name="fcn_scan_detector")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def logit_scores(model, inputs: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Pre-sigmoid max-pooled evidence (monotone in the probability, better resolved near 1)."""
    import tensorflow as tf
    sub = tf.keras.Model(model.input, model.get_layer("pool").output)
    return sub.predict(inputs, batch_size=batch_size, verbose=0)[:, 0]


def crop_examples(clean: np.ndarray, attacked: np.ndarray, rng: np.random.Generator,
                  sched: np.ndarray | None = None):
    """From one (clean, attacked) grid pair, return 3 crops: positive, clean negative, hard negative."""
    T = clean.shape[0]
    changed_t = np.flatnonzero((attacked != clean).any(axis=1))
    lo, hi = changed_t.min(), changed_t.max()
    start = int(rng.integers(max(0, hi - CROP_T + 1), min(lo, T - CROP_T) + 1))
    pos = slice(start, start + CROP_T)
    neg = slice(*(lambda s: (s, s + CROP_T))(int(rng.integers(0, T - CROP_T + 1))))
    # hard negative: a crop of the attacked grid that does not touch the burst
    candidates = [s for s in range(0, T - CROP_T + 1, 4) if s + CROP_T <= lo or s > hi]
    hs = int(rng.choice(candidates))
    hard = slice(hs, hs + CROP_T)
    s_ = (lambda sl: None if sched is None else sched[sl])
    return ([build_inputs(attacked[pos], s_(pos))[0], build_inputs(clean[neg], s_(neg))[0],
             build_inputs(attacked[hard], s_(hard))[0]], [1, 0, 0])
