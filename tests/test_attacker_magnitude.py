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


def test_adaptive_magnitude_shrinks_with_more_channel_uses():
    attacker = AdaptiveAttacker(AttackerConfig(seed=1))
    small = attacker.sqrt_law_magnitude(4)
    large = attacker.sqrt_law_magnitude(400)
    assert large < small
    # sqrt-law: magnitude(n) = base / sqrt(n), so ratio should match sqrt(400/4) = 10
    assert small / large == pytest.approx(np.sqrt(400 / 4), rel=1e-6)
