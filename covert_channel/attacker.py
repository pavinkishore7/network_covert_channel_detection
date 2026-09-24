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
    # Where the covert symbols go inside the target slice's cells.
    #   "random_burst" (default): a contiguous burst of target cells (row-major,
    #       i.e. time-then-frequency order) starting at a random OFDM symbol, so
    #       the attack can appear anywhere in the grid.
    #   "first": the first n target cells in row-major order, i.e. always at the
    #       start of the grid. This was the only behaviour before 2026-09-24; it
    #       is kept only so earlier results can be reproduced exactly. It is a
    #       modelling artifact (a real attacker has no reason to always transmit
    #       in the first symbol).
    placement: str = "random_burst"


def _select_slots(target_idx: np.ndarray, n_bits: int, placement: str,
                  rng: np.random.Generator) -> np.ndarray:
    """Pick the (t, f) cells that carry covert symbols.

    Only the *position* of the burst is randomised; its size and the
    per-symbol magnitude are unchanged, so detectability is not made easier or
    harder by this choice other than removing the fixed-position artifact.
    """
    n_slots = min(n_bits, len(target_idx))
    if placement == "first":
        return target_idx[:n_slots]
    if placement != "random_burst":
        raise ValueError(f"unknown placement: {placement!r}")
    start = int(rng.integers(0, len(target_idx) - n_slots + 1))
    return target_idx[start:start + n_slots]


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

        slots = _select_slots(target_idx, len(bits), self.cfg.placement, self.rng)
        for i, (t, f) in enumerate(slots):
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

    def inject(self, interference_grid: np.ndarray, target_mask: np.ndarray,
               fixed_magnitude: float = 0.5) -> np.ndarray:
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

        # Hard ceiling: regardless of n_channel_uses/local_std, the adaptive
        # perturbation must never exceed 60% of the non-adaptive baseline —
        # base_magnitude alone doesn't guarantee this, so enforce it directly.
        ceiling = fixed_magnitude * 0.6

        slots = _select_slots(target_idx, len(bits), self.cfg.placement, self.rng)
        for i, (t, f) in enumerate(slots):
            shaped = magnitude * local_std * self.rng.normal(1.0, 0.15)
            shaped = np.clip(shaped, -ceiling, ceiling)
            grid[t, f] += shaped if bits[i] else -shaped

        return grid


class BandLimitedAdaptiveAttacker(AdaptiveAttacker):
    """Stress-test attacker that knows the clean power model and hides in it.

    Same sqrt-law magnitude and shaping as AdaptiveAttacker, but every
    perturbed cell is clipped so its value stays inside the allocated-power
    band a clean cell can take (``band``; see slicing_sim/ofdm_grid.py).
    A detector that only checks "is this cell at a legal power level" cannot
    see it by construction; only a detector that knows what was actually
    scheduled can. It is deliberately STRONGER than AdaptiveAttacker, added
    to test the detector, not to make it look good.
    """

    def __init__(self, config: AttackerConfig, band: tuple[float, float] = (0.8, 1.0)):
        super().__init__(config)
        self.band = band

    def inject(self, interference_grid: np.ndarray, target_mask: np.ndarray,
               fixed_magnitude: float = 0.5) -> np.ndarray:
        perturbed = super().inject(interference_grid, target_mask, fixed_magnitude)
        changed = perturbed != interference_grid
        lo, hi = self.band
        # keep each touched cell inside the legal band (or leave it as the
        # clean value if the clean value itself was already outside it)
        clean = interference_grid[changed]
        inside = (clean >= lo) & (clean <= hi)
        clipped = np.clip(perturbed[changed], lo, hi)
        perturbed[changed] = np.where(inside, clipped, clean)
        return perturbed


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
