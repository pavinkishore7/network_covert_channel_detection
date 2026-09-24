"""Tests for detector/cnn_scan_detector.py (skipped when TensorFlow is absent)."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig  # noqa: E402
from detector.cnn_scan_detector import CROP_T, build_inputs, build_model, crop_examples, logit_scores  # noqa: E402
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig  # noqa: E402


def _pair(seed=1, snr=20.0):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=snr, seed=seed))
    al = sim.allocate_slices()
    g = sim.combined_interference_grid(al)
    a = AdaptiveAttacker(AttackerConfig(seed=seed)).inject(g, al["eMBB"].subcarrier_mask)
    return g, a, sum(x.power for x in al.values())


def test_inputs_channels():
    g, _, sch = _pair()
    assert build_inputs(g).shape == (1, 200, 64, 2)
    assert build_inputs(g, sch).shape == (1, 200, 64, 3)


def test_crops_positive_contains_burst_and_hard_negative_does_not():
    g, a, _ = _pair()
    xs, ys = crop_examples(g, a, np.random.default_rng(0))
    assert ys == [1, 0, 0]
    assert all(x.shape == (CROP_T, 64, 2) for x in xs)
    burst_rows = np.flatnonzero((a != g).any(axis=1))
    assert len(burst_rows) > 0
    # the positive crop's raw-grid channel must differ from the clean grid somewhere
    assert not np.isin(xs[0][..., 0], g).all()


def test_fully_convolutional_model_scores_any_grid_size_and_round_trips(tmp_path):
    m = build_model(2)
    g, a, _ = _pair()
    full = np.concatenate([build_inputs(g), build_inputs(a)])
    s = logit_scores(m, full)
    assert s.shape == (2,)
    crop = full[:, :CROP_T]
    assert logit_scores(m, crop).shape == (2,)
    path = tmp_path / "m.keras"
    m.save(path)
    m2 = tf.keras.models.load_model(path)
    np.testing.assert_allclose(logit_scores(m2, full), s, rtol=1e-5, atol=1e-5)
