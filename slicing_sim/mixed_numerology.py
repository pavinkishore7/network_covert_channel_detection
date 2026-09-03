"""Mixed-numerology RAN-slicing measurements for the PHY simulation.

This is a compact analytical simulation: each slice has a distinct subcarrier
spacing, contiguous band and configurable guard band. Spectral leakage creates
inter-slice-band interference (ISBI); cooperative cancellation estimates and
removes a configurable fraction of that interference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc

import numpy as np


@dataclass(frozen=True)
class Numerology:
    subcarrier_spacing_khz: int
    allocated_subcarriers: int
    guard_subcarriers: int


DEFAULT_NUMEROLOGIES = {
    "URLLC": Numerology(60, 12, 2),
    "eMBB": Numerology(30, 32, 2),
    "mMTC": Numerology(15, 12, 2),
}


class MixedNumerologyRAN:
    """Generates slice bands and reports ISBI, SINR and QPSK BER estimates."""

    def __init__(self, n_symbols: int = 200, numerologies: dict[str, Numerology] | None = None, seed: int | None = None):
        self.n_symbols = n_symbols
        self.numerologies = numerologies or DEFAULT_NUMEROLOGIES
        self.rng = np.random.default_rng(seed)

    @property
    def n_subcarriers(self) -> int:
        return sum(item.allocated_subcarriers + item.guard_subcarriers for item in self.numerologies.values())

    def slice_masks(self) -> dict[str, np.ndarray]:
        masks, cursor = {}, 0
        for name, config in self.numerologies.items():
            mask = np.zeros((self.n_symbols, self.n_subcarriers), dtype=bool)
            mask[:, cursor:cursor + config.allocated_subcarriers] = True
            masks[name] = mask
            cursor += config.allocated_subcarriers + config.guard_subcarriers
        return masks

    def transmit(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        masks = self.slice_masks()
        signals = {}
        for name, mask in masks.items():
            # QPSK-like complex symbols with slice-specific time granularity.
            symbols = self.rng.choice((-1, 1), mask.shape) + 1j * self.rng.choice((-1, 1), mask.shape)
            signals[name] = symbols / np.sqrt(2) * mask
        return signals, masks

    @staticmethod
    def _leak(signal: np.ndarray) -> np.ndarray:
        # Compact out-of-band-emission approximation: adjacent subcarriers
        # leak most, then decay with distance.
        kernel = np.array([0.01, 0.02, 0.04, 0.08, 0.70, 0.08, 0.04, 0.02, 0.01])
        return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, signal)

    def evaluate(self, noise_power: float = 0.01, cancellation_efficiency: float = 0.0) -> list[dict[str, float | str]]:
        if not 0 <= cancellation_efficiency <= 1:
            raise ValueError("cancellation_efficiency must be in [0, 1]")
        signals, masks = self.transmit()
        leakage = {name: self._leak(signal) for name, signal in signals.items()}
        rows = []
        for name, mask in masks.items():
            desired = float(np.mean(np.abs(signals[name][mask]) ** 2))
            interference = sum((np.abs(leakage[other][mask]) ** 2).sum() for other in signals if other != name) / mask.sum()
            residual = interference * (1 - cancellation_efficiency)
            sinr = desired / (residual + noise_power)
            ber = 0.5 * erfc(np.sqrt(sinr))
            rows.append({
                "slice": name,
                "subcarrier_spacing_khz": self.numerologies[name].subcarrier_spacing_khz,
                "guard_subcarriers": self.numerologies[name].guard_subcarriers,
                "isbi_power": float(interference),
                "sinr_db": float(10 * np.log10(sinr)),
                "qpsk_ber_estimate": float(ber),
                "cancellation_efficiency": cancellation_efficiency,
            })
        return rows
