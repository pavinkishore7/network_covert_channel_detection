"""Defensive channel-impairment models for OFDM-grid simulation.

These are compact, reproducible approximations of effects represented by TDL/
CDL channel models, not an implementation of 3GPP TR 38.901 itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelImpairmentConfig:
    profile: str = "awgn"  # awgn, urban_micro, high_mobility
    doppler_correlation: float = 0.97
    phase_noise_std: float = 0.015
    cfo_leakage: float = 0.02
    impulsive_probability: float = 0.002


def apply_impairments(grid: np.ndarray, rng: np.random.Generator, config: ChannelImpairmentConfig) -> np.ndarray:
    if config.profile == "awgn":
        return grid
    if config.profile not in {"urban_micro", "high_mobility"}:
        raise ValueError(f"Unknown channel profile: {config.profile}")
    symbols, carriers = grid.shape
    taps = 6 if config.profile == "urban_micro" else 10
    delay = np.arange(taps)
    decay = np.exp(-delay / (2.5 if config.profile == "urban_micro" else 4.0))
    tap_gain = rng.rayleigh(scale=decay / np.sqrt(2), size=taps)
    phase = rng.uniform(0, 2 * np.pi, size=taps)
    response = np.abs(sum(tap_gain[i] * np.exp(1j * (phase[i] - 2 * np.pi * i * np.arange(carriers) / carriers)) for i in range(taps)))
    response /= np.mean(response) + 1e-8
    rho = config.doppler_correlation if config.profile == "urban_micro" else min(config.doppler_correlation, 0.88)
    fading = np.empty(symbols)
    fading[0] = 1.0
    for index in range(1, symbols):
        fading[index] = rho * fading[index - 1] + np.sqrt(1 - rho**2) * rng.normal()
    fading = np.clip(1 + 0.25 * fading, 0.2, 2.5)
    impaired = grid * fading[:, None] * response[None, :]
    # CFO/phase noise create inter-carrier leakage rather than a simple SNR loss.
    leakage = config.cfo_leakage * (np.roll(impaired, 1, axis=1) + np.roll(impaired, -1, axis=1)) / 2
    impaired = (1 - config.cfo_leakage) * impaired + leakage
    impaired *= 1 + rng.normal(0, config.phase_noise_std, size=(symbols, 1))
    impulses = rng.random(grid.shape) < config.impulsive_probability
    impaired[impulses] += rng.normal(0, 4 * np.std(grid), size=int(impulses.sum()))
    return impaired
