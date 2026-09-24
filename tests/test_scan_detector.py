"""Tests for detector/scan_detector.py and the attacker placement option."""

from __future__ import annotations

import numpy as np
import pytest

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig, NonAdaptiveAttacker
from detector.scan_detector import (
    ALLOCATED_POWER_BAND,
    ScanDetector,
    allocation_residual,
    level_residual,
    multiscale_features,
)
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig


def _clean(seed: int, snr: float = 20.0):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=snr, seed=seed))
    alloc = sim.allocate_slices()
    grid = sim.combined_interference_grid(alloc)
    return grid, alloc


def test_level_residual_is_zero_on_clean_levels():
    lo, hi = ALLOCATED_POWER_BAND
    g = np.array([[0.0, lo, (lo + hi) / 2, hi]])
    assert np.all(level_residual(g) == 0.0)
    assert level_residual(np.array([[hi + 0.3]]))[0, 0] == pytest.approx(0.09)
    assert level_residual(np.array([[-0.2]]))[0, 0] == pytest.approx(0.04)


def test_allocation_residual_is_exact_difference():
    assert allocation_residual(np.array([[1.5]]), np.array([[1.0]]))[0, 0] == pytest.approx(0.25)


def test_multiscale_feature_shape():
    f = multiscale_features(np.zeros((3, 200, 64)))
    assert f.shape == (3, 17)


def _block(pos: tuple[int, int], rng) -> np.ndarray:
    m = rng.normal(0, 0.01, size=(200, 64)) ** 2
    t, f = pos
    m[t:t + 2, f:f + 32] += 0.25
    return m


def test_scan_flags_block_anywhere_and_passes_clean():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.01, size=(200, 200, 64)) ** 2
    det = ScanDetector()
    det.calibrate(clean)
    assert det.flag(rng.normal(0, 0.01, size=(50, 200, 64)) ** 2).mean() < 0.2
    scores = [det.score(_block(p, np.random.default_rng(1))[None])[0] for p in [(0, 0), (97, 20), (198, 32)]]
    assert all(s > det.threshold_ for s in scores)
    # position invariance: same block, same noise, different place -> near-identical score
    assert max(scores) - min(scores) < 0.05 * max(scores)


def test_calibrate_rejects_tiny_calibration_sets():
    with pytest.raises(ValueError):
        ScanDetector().calibrate(np.zeros((10, 200, 64)))


def test_placement_first_reproduces_legacy_positions():
    grid, alloc = _clean(3)
    mask = alloc["eMBB"].subcarrier_mask
    first_cells = np.argwhere(mask)[:32]
    out = NonAdaptiveAttacker(AttackerConfig(seed=3, placement="first")).inject(grid, mask)
    changed = np.argwhere(out != grid)
    assert np.array_equal(changed, first_cells)


def test_random_burst_moves_but_keeps_size_and_magnitude():
    starts, mags = set(), []
    for seed in range(20):
        grid, alloc = _clean(seed)
        mask = alloc["eMBB"].subcarrier_mask
        out = AdaptiveAttacker(AttackerConfig(seed=seed)).inject(grid, mask)
        diff = np.argwhere(out != grid)
        assert len(diff) == 32
        assert mask[diff[:, 0], diff[:, 1]].all()  # only the target slice's cells
        starts.add(int(diff[0, 0]))
        mags.append(np.abs(out - grid).max())
    assert len(starts) > 5  # not always the same symbol
    assert max(mags) <= 0.5 * 0.6 + 1e-9  # adaptive ceiling unchanged


def test_unknown_placement_rejected():
    grid, alloc = _clean(1)
    with pytest.raises(ValueError):
        NonAdaptiveAttacker(AttackerConfig(seed=1, placement="nope")).inject(grid, alloc["eMBB"].subcarrier_mask)


def test_detector_separates_non_adaptive_at_high_snr():
    det = ScanDetector()
    clean_maps, atk_maps = [], []
    for seed in range(200):
        g, a = _clean(10_000 + seed, snr=25.0)
        clean_maps.append(level_residual(g))
    det.calibrate(np.stack(clean_maps[:150]))
    for seed in range(50):
        g, a = _clean(20_000 + seed, snr=25.0)
        atk_maps.append(level_residual(NonAdaptiveAttacker(AttackerConfig(seed=seed)).inject(g, a["eMBB"].subcarrier_mask)))
    assert det.flag(np.stack(atk_maps)).mean() > 0.9
    assert det.flag(np.stack(clean_maps[150:])).mean() < 0.2
