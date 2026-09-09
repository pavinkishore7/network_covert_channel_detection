"""
Adaptive covert-channel attacker.

Injects a covert signal into the shared OFDM interference grid by
perturbing subcarrier power levels in a slice the attacker doesn't own,
bounded by the square-root law (Bash-Goeckel-Towsley): under AWGN, a
covert transmitter can reliably send O(sqrt(n)) bits over n channel uses
while keeping detection probability bounded — NOT O(n) bits, which is
what a naive/non-adaptive attacker would try and get caught doing.

Two attacker modes are implemented:
  - NonAdaptiveAttacker: fixed perturbation magnitude, easy to catch.
    Exists as a baseline so the adaptive attacker's improvement is
    measurable, not just asserted.
  - AdaptiveAttacker: perturbation magnitude scales down as the covert
    message length grows, respecting the sqrt-law bound, and adds
    interference *shaping* (perturbation follows the shape of natural
    interference variance) to reduce statistical distinguishability.

Caveat that belongs on your slide, not just in this docstring:
the sqrt-law bound applied here assumes AWGN. It is a THEORETICAL BOUND,
not a guarantee the CNN autoencoder can't detect shaped perturbations
below that bound — detectability and the sqrt-law capacity bound are
related but not the same claim. Don't conflate "bounded by sqrt-law" with
"undetectable."
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AttackerConfig:
    n_covert_bits: int = 32
    target_slice: str = "eMBB"     # slice whose subcarriers get perturbed
    seed: int | None = None


class NonAdaptiveAttacker:
    """Fixed-magnitude perturbation. Baseline for comparison — this is what
    'transmitting at a fixed, easily-caught rate' (your novelty slide) means
    concretely."""

    def __init__(self, config: AttackerConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate_covert_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.cfg.n_covert_bits)

    def inject(self, interference_grid: np.ndarray, target_mask: np.ndarray,
               fixed_magnitude: float = 0.5) -> np.ndarray:
        """Perturb subcarriers within target_mask by a FIXED magnitude,
        regardless of how much covert data there is or how noisy the
        surrounding grid is. This is intentionally the naive strategy."""
        grid = interference_grid.copy()
        bits = self.generate_covert_bits()
        target_idx = np.argwhere(target_mask)

        if len(target_idx) == 0:
            return grid

        n_slots = min(len(bits), len(target_idx))
        for i in range(n_slots):
            t, f = target_idx[i]
            grid[t, f] += fixed_magnitude if bits[i] else -fixed_magnitude

        return grid


class AdaptiveAttacker:
    """Perturbation magnitude bounded by sqrt-law scaling, shaped to match
    local interference variance. This is the attacker your detector actually
    needs to be evaluated against for the adaptive-attacker claim to be
    honest."""

    def __init__(self, config: AttackerConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate_covert_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.cfg.n_covert_bits)

    def sqrt_law_magnitude(self, n_channel_uses: int, base_magnitude: float = 4.0) -> float:
        """Per-symbol perturbation magnitude scaled so cumulative detectability
        stays bounded as channel uses grow — magnitude ~ 1/sqrt(n) per use,
        giving O(sqrt(n)) total covert information, per the sqrt-law."""
        return base_magnitude / np.sqrt(max(n_channel_uses, 1))

    def inject(self, interference_grid: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        """Shape perturbation to local interference standard deviation so the
        covert signal doesn't stick out as a flat, unnaturally uniform
        perturbation (which is what makes the non-adaptive attacker easy to
        catch — its perturbation has a different statistical signature than
        natural interference)."""
        grid = interference_grid.copy()
        bits = self.generate_covert_bits()
        target_idx = np.argwhere(target_mask)

        if len(target_idx) == 0:
            return grid

        # ``n`` in the square-root law is the number of channel uses that
        # actually carry covert symbols, not every allocated resource element.
        # Using the full mask previously understated the adaptive signal by
        # treating untouched cells as transmissions.
        n_channel_uses = min(len(bits), len(target_idx))
        magnitude = self.sqrt_law_magnitude(n_channel_uses)

        # local noise scale: perturbation should be comparable to, not
        # wildly different from, the natural variance around each subcarrier
        local_std = np.std(interference_grid[interference_grid > 0]) if np.any(interference_grid > 0) else 1.0

        n_slots = min(len(bits), len(target_idx))
        for i in range(n_slots):
            t, f = target_idx[i]
            shaped = magnitude * local_std * self.rng.normal(1.0, 0.15)
            grid[t, f] += shaped if bits[i] else -shaped

        return grid


class ReactiveJammer:
    """Defensive simulation of an energy-aware reactive partial-band jammer.

    It transmits only when observed slice power crosses a threshold, modelling
    a common availability threat without interacting with any real radio.
    """

    def __init__(self, seed: int | None = None, duty_cycle: float = 0.15):
        self.rng = np.random.default_rng(seed)
        self.duty_cycle = duty_cycle

    def inject(self, grid: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        result = grid.copy()
        active = target_mask & (grid > np.quantile(grid[target_mask], 0.7))
        candidates = np.argwhere(active)
        count = int(len(candidates) * self.duty_cycle)
        if count:
            selected = candidates[self.rng.choice(len(candidates), count, replace=False)]
            result[selected[:, 0], selected[:, 1]] += self.rng.normal(0.8, 0.12, size=count)
        return result


class PilotSpoofer:
    """Defensive model of pilot-resource manipulation / synchronization spoofing."""

    def __init__(self, seed: int | None = None, pilot_period: int = 14):
        self.rng = np.random.default_rng(seed)
        self.pilot_period = pilot_period

    def inject(self, grid: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        result = grid.copy()
        pilot_rows = np.arange(0, grid.shape[0], self.pilot_period)
        pilot_mask = target_mask.copy()
        pilot_mask[np.setdiff1d(np.arange(grid.shape[0]), pilot_rows)] = False
        result[pilot_mask] += self.rng.normal(0.35, 0.05, size=int(pilot_mask.sum()))
        return result


if __name__ == "__main__":
    from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

    sim_cfg = OFDMGridConfig(seed=1)
    sim = NetworkSlicingSimulator(sim_cfg)
    allocations = sim.allocate_slices()
    clean_grid = sim.combined_interference_grid(allocations)

    target_mask = allocations["eMBB"].subcarrier_mask

    naive = NonAdaptiveAttacker(AttackerConfig(seed=1))
    adaptive = AdaptiveAttacker(AttackerConfig(seed=1))

    naive_grid = naive.inject(clean_grid, target_mask)
    adaptive_grid = adaptive.inject(clean_grid, target_mask)

    print("Perturbation magnitude comparison (mean abs delta vs clean grid):")
    print(f"  Non-adaptive: {np.mean(np.abs(naive_grid - clean_grid)):.4f}")
    print(f"  Adaptive:     {np.mean(np.abs(adaptive_grid - clean_grid)):.4f}")
    print("Adaptive should be noticeably smaller — that's the point.")
