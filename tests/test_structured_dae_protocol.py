import numpy as np
import pytest

from detector.evaluate_structured_dae import matched_split


def _frozen_layout():
    y = np.tile([0, 1, 2], 200 * 7)
    snr = np.repeat(np.arange(0, 35, 5), 600)
    return y, snr


def test_matched_split_preserves_triplets_and_counts():
    y, snr = _frozen_layout()
    train, valid, test = matched_split(y, snr)
    assert len(train) == 2940 and len(valid) == len(test) == 630
    assert not set(train) & set(valid) and not set(train) & set(test) and not set(valid) & set(test)
    for part in (train, valid, test):
        assert np.array_equal(y[part].reshape(-1, 3), np.tile([0, 1, 2], (len(part) // 3, 1)))


def test_matched_split_rejects_broken_triplet_order():
    y, snr = _frozen_layout()
    y[1] = 2
    with pytest.raises(ValueError, match="ordered"):
        matched_split(y, snr)
