"""
Network slicing + OFDM resource-grid simulation.

Produces a per-slice OFDM resource allocation over time: which subcarriers
are active, their power/interference levels, per symbol, per slice.
This resource grid is the shared object that:
  - covert_channel.attacker perturbs (to hide a covert signal in it)
  - detector.autoencoder_detector learns the "normal" distribution of

Slice types modeled: URLLC, eMBB, mMTC — each with different subcarrier
allocation patterns and timing tolerances (used later for the slice-aware
re-auth timer intervals).
"""

import numpy as np
from slicing_sim.channel import ChannelImpairmentConfig, apply_impairments
from dataclasses import dataclass, field


SLICE_TYPES = ("URLLC", "eMBB", "mMTC")

# Rough per-slice-type profile: fraction of subcarriers typically allocated,
# and how "bursty" vs steady the traffic is. These are design assumptions
# for simulation purposes, NOT measured values — state this plainly if asked.
SLICE_PROFILES = {
    "URLLC": {"subcarrier_frac": 0.15, "burstiness": 0.2},   # small, steady, latency-critical
    "eMBB":  {"subcarrier_frac": 0.55, "burstiness": 0.5},   # large, bursty, throughput-driven
    "mMTC":  {"subcarrier_frac": 0.30, "burstiness": 0.8},   # many small sporadic bursts
}


@dataclass
class OFDMGridConfig:
    n_subcarriers: int = 64
    n_symbols: int = 200          # time steps (OFDM symbols) per simulation run
    snr_db: float = 20.0
    seed: int | None = None
    channel: ChannelImpairmentConfig = ChannelImpairmentConfig()


@dataclass
class SliceAllocation:
    slice_type: str
    subcarrier_mask: np.ndarray   # (n_symbols, n_subcarriers) bool — which subcarriers this slice owns per symbol
    power: np.ndarray             # (n_symbols, n_subcarriers) float — allocated power where mask is True


class NetworkSlicingSimulator:
    """Generates a shared OFDM resource grid partitioned across slices."""

    def __init__(self, config: OFDMGridConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def allocate_slices(self) -> dict[str, SliceAllocation]:
        """Partition subcarriers across the three slice types per symbol.

        Non-overlapping allocation per symbol (logical isolation assumption —
        the covert channel exploits INTERFERENCE LEAKAGE across this boundary,
        not the allocation logic itself).
        """
        n_sub = self.cfg.n_subcarriers
        n_sym = self.cfg.n_symbols
        allocations = {}

        for slice_type in SLICE_TYPES:
            profile = SLICE_PROFILES[slice_type]
            mask = np.zeros((n_sym, n_sub), dtype=bool)
            power = np.zeros((n_sym, n_sub), dtype=float)

            n_active = max(1, int(profile["subcarrier_frac"] * n_sub))
            for t in range(n_sym):
                # burstiness controls how much the active set changes symbol to symbol
                if t == 0 or self.rng.random() < profile["burstiness"]:
                    active_idx = self.rng.choice(n_sub, size=n_active, replace=False)
                else:
                    active_idx = np.where(mask[t - 1])[0]
                mask[t, active_idx] = True
                power[t, active_idx] = self.rng.uniform(0.8, 1.0, size=len(active_idx))

            allocations[slice_type] = SliceAllocation(slice_type, mask, power)

        return self._resolve_conflicts(allocations)

    def _resolve_conflicts(self, allocations: dict[str, SliceAllocation]) -> dict[str, SliceAllocation]:
        """Ensure no two slices claim the same subcarrier in the same symbol
        (logical isolation). Conflicts are resolved by priority: URLLC > eMBB > mMTC.
        """
        priority = ["URLLC", "eMBB", "mMTC"]
        claimed = np.zeros((self.cfg.n_symbols, self.cfg.n_subcarriers), dtype=bool)

        for slice_type in priority:
            alloc = allocations[slice_type]
            conflict = alloc.subcarrier_mask & claimed
            alloc.subcarrier_mask[conflict] = False
            alloc.power[conflict] = 0.0
            claimed |= alloc.subcarrier_mask

        return allocations

    def add_awgn(self, power_grid: np.ndarray) -> np.ndarray:
        """Apply AWGN at the configured SNR. This is where the AWGN-specificity
        caveat lives: real multipath channels aren't modeled here."""
        signal_power = np.mean(power_grid[power_grid > 0]) if np.any(power_grid > 0) else 1.0
        snr_linear = 10 ** (self.cfg.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = self.rng.normal(0, np.sqrt(noise_power), size=power_grid.shape)
        return power_grid + noise

    def combined_interference_grid(self, allocations: dict[str, SliceAllocation]) -> np.ndarray:
        """Sum of all slices' power allocation, i.e. what a receiver observing
        the shared spectrum actually sees — the substrate the covert channel
        rides on top of."""
        total = np.zeros((self.cfg.n_symbols, self.cfg.n_subcarriers))
        for alloc in allocations.values():
            total += alloc.power
        return apply_impairments(self.add_awgn(total), self.rng, self.cfg.channel)


if __name__ == "__main__":
    cfg = OFDMGridConfig(seed=42)
    sim = NetworkSlicingSimulator(cfg)
    allocations = sim.allocate_slices()
    grid = sim.combined_interference_grid(allocations)
    print(f"Grid shape: {grid.shape}")
    for slice_type, alloc in allocations.items():
        occ = alloc.subcarrier_mask.mean()
        print(f"{slice_type}: mean subcarrier occupancy = {occ:.2%}")
