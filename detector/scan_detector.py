"""Physics-informed multi-scale scan detector for OFDM-grid covert channels.

Why this exists (see docs/DECISIONS.md, 2026-09-24):
the covert attackers in covert_channel/attacker.py perturb a few dozen cells
of a 200 x 64 grid (~0.25%). The CNN autoencoder scores a grid by its mean
reconstruction error over every cell, which dilutes those cells ~400x, and its
small bottleneck cannot model per-scenario random slice scheduling, so its
residual is dominated by legitimate scheduling randomness. This detector
replaces both halves:

1. Residual (what counts as "unexpected" in a cell):
   - ``level_residual``: squared distance from each cell to the nearest power
     level a clean grid can take in this simulator: 0 (unallocated) or the
     allocated-power band ``ALLOCATED_POWER_BAND`` (see
     slicing_sim/ofdm_grid.py, ``allocate_slices``). Needs no side
     information. An attacker that keeps every perturbed cell inside the
     band evades it.
   - ``allocation_residual``: squared difference between the observed grid
     and the power the scheduler actually allocated. Needs the scheduled
     per-cell power, which the scheduler/gNB has but an arbitrary third-party
     receiver does not. Catches in-band perturbations too.

2. Pooling (how cell residuals become one score): a position-agnostic
   multi-scale scan. For each window size in a fixed generic grid
   (``DEFAULT_SCALES``), take the maximum window-mean over all positions;
   standardise each scale against clean calibration data; the score is the
   largest standardised value. The window grid is generic, not fitted to the
   attacker's burst shape, and the maximum over positions makes the score
   independent of where the burst sits.

Scope: simulation only, AWGN channel profile, the attacker classes in this
repository. Not a field-performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

# Allocated cells draw power from uniform(0.8, 1.0) in
# slicing_sim.ofdm_grid.NetworkSlicingSimulator.allocate_slices; unallocated
# cells are 0 before AWGN. Keep this in sync with that function.
ALLOCATED_POWER_BAND = (0.8, 1.0)

DEFAULT_SCALES: tuple[tuple[int, int], ...] = tuple(
    (t, f) for t in (1, 2, 4, 8) for f in (8, 16, 32, 64)
)


def level_residual(grids: np.ndarray, band: tuple[float, float] = ALLOCATED_POWER_BAND) -> np.ndarray:
    """Squared distance of each cell to the nearest clean power level.

    Works on a single (H, W) grid or a stack (N, H, W). Zero inside the band
    and exactly at 0.
    """
    g = np.asarray(grids, dtype=np.float64)
    lo, hi = band
    to_zero = np.abs(g)
    to_band = np.where(g < lo, lo - g, np.where(g > hi, g - hi, 0.0))
    return np.square(np.minimum(to_zero, to_band))


def allocation_residual(grids: np.ndarray, scheduled_power: np.ndarray) -> np.ndarray:
    """Squared difference between observed and scheduled per-cell power."""
    return np.square(np.asarray(grids, dtype=np.float64) - np.asarray(scheduled_power, dtype=np.float64))


def multiscale_features(residual_maps: np.ndarray,
                        scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES) -> np.ndarray:
    """Per-map scan features: max window-mean for each scale, plus the global mean.

    residual_maps: (N, H, W). Returns (N, len(scales) + 1).
    """
    maps = np.asarray(residual_maps, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[np.newaxis]
    cols = [uniform_filter(maps, size=(1, t, f), mode="constant").max(axis=(1, 2)) for t, f in scales]
    cols.append(maps.reshape(len(maps), -1).mean(axis=1))
    return np.stack(cols, axis=1)


@dataclass
class ScanDetector:
    """Multi-scale scan detector calibrated on clean residual maps only.

    Calibrate one instance per operating condition (per SNR in this
    project): the per-scale clean statistics depend on the noise level.
    """

    scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES
    target_fpr: float = 0.05
    mu_: np.ndarray | None = field(default=None, repr=False)
    sd_: np.ndarray | None = field(default=None, repr=False)
    threshold_: float | None = None

    def _scan(self, residual_maps: np.ndarray) -> np.ndarray:
        return multiscale_features(residual_maps, self.scales)[:, :-1]  # scan scales only

    def calibrate(self, clean_residual_maps: np.ndarray) -> float:
        feats = self._scan(clean_residual_maps)
        if len(feats) < 50:
            raise ValueError("calibrate on at least 50 clean maps; the max-over-scales threshold is unstable below that")
        self.mu_ = feats.mean(axis=0)
        self.sd_ = feats.std(axis=0) + 1e-12
        self.threshold_ = float(np.quantile(self._z(feats), 1.0 - self.target_fpr))
        return self.threshold_

    def _z(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.mu_) / self.sd_).max(axis=1)

    def score(self, residual_maps: np.ndarray) -> np.ndarray:
        if self.mu_ is None:
            raise RuntimeError("call calibrate() first")
        return self._z(self._scan(residual_maps))

    def flag(self, residual_maps: np.ndarray) -> np.ndarray:
        return self.score(residual_maps) > self.threshold_
