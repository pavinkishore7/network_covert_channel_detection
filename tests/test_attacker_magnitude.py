import numpy as np
import pytest

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig, NonAdaptiveAttacker
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig


def _mean_abs_delta(attacker, clean_grid, target_mask):
    perturbed = attacker.inject(clean_grid, target_mask)
    return np.mean(np.abs(perturbed - clean_grid))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_adaptive_perturbation_smaller_than_non_adaptive(seed):
    sim = NetworkSlicingSimulator(OFDMGridConfig(seed=seed))
    allocations = sim.allocate_slices()
    clean_grid = sim.combined_interference_grid(allocations)
    target_mask = allocations["eMBB"].subcarrier_mask

    naive = NonAdaptiveAttacker(AttackerConfig(seed=seed))
    adaptive = AdaptiveAttacker(AttackerConfig(seed=seed))

    naive_delta = _mean_abs_delta(naive, clean_grid, target_mask)
    adaptive_delta = _mean_abs_delta(adaptive, clean_grid, target_mask)

    assert adaptive_delta < naive_delta, (
        f"seed={seed}: adaptive delta {adaptive_delta:.6f} should be smaller "
        f"than non-adaptive delta {naive_delta:.6f}"
    )


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_adaptive_ceiling_keeps_perturbation_below_non_adaptive(seed):
    sim = NetworkSlicingSimulator(OFDMGridConfig(seed=seed))
    allocations = sim.allocate_slices()
    clean_grid = sim.combined_interference_grid(allocations)
    target_mask = allocations["eMBB"].subcarrier_mask

    naive = NonAdaptiveAttacker(AttackerConfig(seed=seed))
    adaptive = AdaptiveAttacker(AttackerConfig(seed=seed))

    naive_delta = _mean_abs_delta(naive, clean_grid, target_mask)
    adaptive_delta = _mean_abs_delta(adaptive, clean_grid, target_mask)

    assert adaptive_delta < naive_delta, (
        f"seed={seed}: adaptive delta {adaptive_delta:.6f} should be smaller "
        f"than non-adaptive delta {naive_delta:.6f}"
    )


def test_adaptive_ceiling_binds_when_shaped_magnitude_would_exceed_it(monkeypatch):
    # Force sqrt_law_magnitude to return an artificially huge value, and use
    # a synthetic high-variance grid so local_std is also large — together
    # magnitude * local_std is far beyond fixed_magnitude * 0.6, so the only
    # thing keeping the per-cell delta bounded is the np.clip in inject().
    monkeypatch.setattr(AdaptiveAttacker, "sqrt_law_magnitude", lambda self, n: 50.0)

    # local_std is computed from positive grid values, so the grid needs real
    # variance among them (a constant grid has std=0, which would zero out
    # the shaped magnitude regardless of clipping and hide the bug).
    grid = np.array([[10.0, 90.0, 20.0, 80.0]] * 4)
    target_mask = np.ones((4, 4), dtype=bool)

    fixed_magnitude = 0.5
    ceiling = fixed_magnitude * 0.6

    adaptive = AdaptiveAttacker(AttackerConfig(n_covert_bits=16, seed=1))
    perturbed = adaptive.inject(grid, target_mask, fixed_magnitude=fixed_magnitude)

    delta = perturbed - grid
    touched = delta != 0
    assert touched.any(), "expected the attacker to perturb at least one cell"
    assert np.allclose(np.abs(delta[touched]), ceiling), (
        f"expected every perturbed cell to be clipped to exactly {ceiling}, "
        f"got magnitudes {np.unique(np.abs(delta[touched]))}"
    )


def test_adaptive_magnitude_shrinks_with_more_channel_uses():
    attacker = AdaptiveAttacker(AttackerConfig(seed=1))
    small = attacker.sqrt_law_magnitude(4)
    large = attacker.sqrt_law_magnitude(400)
    assert large < small
    # sqrt-law: magnitude(n) = base / sqrt(n), so ratio should match sqrt(400/4) = 10
    assert small / large == pytest.approx(np.sqrt(400 / 4), rel=1e-6)
