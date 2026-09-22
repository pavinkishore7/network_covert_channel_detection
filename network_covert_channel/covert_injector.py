"""
Phase 2, step 1: the network-layer timing covert channel itself.

network_covert_channel/traffic.py (Phase 1) generates each slice's natural
inter-packet-gap timing (generate_inter_packet_gaps) but never hides
anything in it -- Phase 1 was plumbing only (see that package's README,
"Phase 2 will need" section). This module is the injector: it perturbs a
clean gap sequence to encode a hidden bit sequence, on top of whatever
timing generate_inter_packet_gaps() would have produced anyway.

Encoding: one covert bit per packet, via an additive delay on that
packet's inter-packet gap. bit=0 -> the clean gap is sent unperturbed;
bit=1 -> the clean gap has a positive offset added to it. This is the
simplest encoding that satisfies "perturb inter-packet gaps beyond what
generate_inter_packet_gaps() would produce" from the Phase 2 prompt, and
it mirrors covert_channel/attacker.py's own per-symbol +/-magnitude
encoding (there: perturb a subcarrier's power up/down per bit; here:
perturb a gap up (only up, not down -- see below) per bit). A signed
(+/-offset) encoding was considered and rejected: a negative offset large
enough to matter risks pushing a gap below the clamp floor
generate_inter_packet_gaps() itself uses (np.clip(gaps, 1e-4, None)),
which would make bit=1 gaps partially indistinguishable from naturally
tiny bursty-component gaps for reasons unrelated to the covert signal.
A same-magnitude-but-only-additive channel avoids that confound and is
simpler to reason about, at the cost of encoding on a mean-*increase*
only (a receiver with the clean baseline could still equally invert this
choice; nothing here assumes it must be additive-only for capacity
reasons).

Channel capacity vs. detectability (the honest tradeoff this module's
tests exist to measure): DEFAULT_BASE_GAP_S is 10ms and the "steady"
component of generate_inter_packet_gaps has stdev ~5% of that (0.5ms);
the "bursty" component's scale runs from ~2ms (in-burst) up to ~60ms
(idle-period draws from Exponential(base_gap_s * 6.0)). An additive
offset that is large relative to a slice's OWN natural gap scale (e.g.
20ms on URLLC, whose burstiness=0.2 keeps it mostly in the tight steady
regime) sticks out; the same offset on mMTC (burstiness=0.8, mostly the
wide bursty regime) can vanish into that slice's own natural spread. This
is why the unit tests below deliberately include a perturbation strength
small enough to be marginal/undetectable on the least-bursty slice and
report that honestly, rather than only reporting configurations that
detect cleanly.

Two variants, mirroring covert_channel/attacker.py's
NonAdaptiveAttacker/AdaptiveAttacker split exactly (same naming, same
non-adaptive-is-the-baseline-adaptive-improves-on-it framing):

  - NonAdaptiveCovertInjector: fixed offset_s on every bit=1 packet,
    regardless of the slice's local timing behavior at that point in the
    sequence. The easy-to-catch baseline, same role as NonAdaptiveAttacker.
  - AdaptiveCovertInjector: scales the per-packet offset down when the
    slice's own local jitter (a causal rolling stdev over the preceding
    `window` gaps) is already high, so the perturbation is smaller exactly
    where it would otherwise be easiest to notice by comparison, and never
    exceeds the non-adaptive offset_s (a hard ceiling, mirroring
    AdaptiveAttacker's `ceiling = fixed_magnitude * 0.6` clamp -- the
    numbers differ, the ceiling *pattern* is the same).

Both variants build on one pure function, ``inject_bits_into_gaps``, per
the Phase 2 prompt's explicit requirement: "a function that takes a
sequence of clean inter-packet gaps plus a bit sequence to hide, and
returns the perturbed gap sequence -- no live traffic/socket dependency
required for unit tests." Applying this to a real traffic.py send path
for live use is a separate, thin wrapper in covert_demo.py
(build_covert_traffic_plan) -- this module itself never imports
scapy/subprocess and needs no privileges to test.

Scope note: only the non-adaptive variant was strictly required by the
Phase 2 prompt ("non-adaptive is the priority if you have to cut scope").
Both are implemented here -- there was no need to cut the adaptive one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CovertInjectorConfig:
    n_covert_bits: int = 64
    offset_s: float = 0.01  # additive delay applied to a bit=1 packet's gap
    seed: int | None = None


def inject_bits_into_gaps(
    clean_gaps: np.ndarray,
    bits: Sequence[int],
    offset_s: np.ndarray | float,
) -> np.ndarray:
    """Pure perturbation primitive: for each position i where bits[i] is
    truthy, adds offset_s (a scalar applied to every such position, or a
    per-position array) to clean_gaps[i]; positions where bits[i] is 0 (or
    beyond len(bits)) pass through unperturbed. Never mutates clean_gaps.

    zip length is min(len(clean_gaps), len(bits)) -- mirrors
    NonAdaptiveAttacker/AdaptiveAttacker's own
    ``n_slots = min(len(bits), len(target_idx))`` convention for what
    happens when the covert payload is shorter than the carrier.
    """
    perturbed = np.array(clean_gaps, dtype=float, copy=True)
    bits_arr = np.asarray(bits)
    n = min(len(perturbed), len(bits_arr))
    if n == 0:
        return perturbed

    if np.isscalar(offset_s):
        offsets = np.full(n, float(offset_s))
    else:
        offsets = np.asarray(offset_s, dtype=float)[:n]
        if len(offsets) != n:
            raise ValueError("offset_s array must cover every bit position up to n")

    mask = bits_arr[:n].astype(bool)
    perturbed[:n][mask] += offsets[mask]
    return perturbed


class NonAdaptiveCovertInjector:
    """Fixed-magnitude additive delay on every bit=1 packet, regardless of
    slice or local traffic state. Baseline for comparison -- this is what
    'transmitting at a fixed, easily-caught rate' means concretely here,
    same role NonAdaptiveAttacker plays for the PHY-layer channel."""

    def __init__(self, config: CovertInjectorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate_covert_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.cfg.n_covert_bits)

    def inject(self, clean_gaps: np.ndarray, bits: np.ndarray | None = None) -> np.ndarray:
        bits = self.generate_covert_bits() if bits is None else np.asarray(bits)
        return inject_bits_into_gaps(clean_gaps, bits, self.cfg.offset_s)


class AdaptiveCovertInjector:
    """Per-packet offset shrinks when the slice's own local jitter (a
    causal rolling stdev over the preceding ``window`` gaps -- causal, not
    a whole-sequence stat, because a real sender only has past gaps to
    look at) is already high, and never exceeds the non-adaptive
    offset_s ceiling. This is the adaptive attacker your detector actually
    needs to be evaluated against for an adaptive-vs-non-adaptive claim to
    be honest -- exactly the framing AdaptiveAttacker uses for the
    PHY-layer channel.

    ``reference_jitter_s``, if not given, is estimated from the full
    clean_gaps sequence passed to inject() at call time. This is a
    simplification for a pure/testable batch API, not a design requirement
    -- a real online sender would instead track this from a startup
    calibration window rather than the very sequence it's perturbing;
    stated plainly here rather than silently assumed.
    """

    def __init__(
        self,
        config: CovertInjectorConfig,
        window: int = 20,
        reference_jitter_s: float | None = None,
    ):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.window = window
        self.reference_jitter_s = reference_jitter_s

    def generate_covert_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.cfg.n_covert_bits)

    def _local_jitter(self, clean_gaps: np.ndarray, i: int, fallback: float) -> float:
        lo = max(0, i - self.window)
        window_gaps = clean_gaps[lo:i]
        if len(window_gaps) > 1:
            return float(np.std(window_gaps))
        return fallback

    def inject(self, clean_gaps: np.ndarray, bits: np.ndarray | None = None) -> np.ndarray:
        bits = self.generate_covert_bits() if bits is None else np.asarray(bits)
        n = min(len(clean_gaps), len(bits))
        if n == 0:
            return np.array(clean_gaps, dtype=float, copy=True)

        reference = self.reference_jitter_s
        if reference is None:
            reference = float(np.std(clean_gaps[:n]))
        reference = max(reference, 1e-9)

        offsets = np.empty(n)
        for i in range(n):
            local_std = max(self._local_jitter(clean_gaps, i, reference), 1e-9)
            # Ceiling: adaptive offset never exceeds the non-adaptive
            # baseline offset_s, regardless of how small local_std gets --
            # mirrors AdaptiveAttacker's fixed_magnitude*0.6 hard clamp.
            offsets[i] = self.cfg.offset_s * min(1.0, reference / local_std)

        return inject_bits_into_gaps(clean_gaps, bits, offsets)
