"""
Phase 2, step 2: a classical (non-neural) statistical detector for the
network-layer timing covert channel injected by covert_injector.py.

Deliberately NOT the CNN autoencoder in detector/autoencoder_detector.py.
Reasons, stated plainly per this project's habit of justifying every
architecture choice rather than asserting it:
  - Inter-packet-gap series are 1-D (one float per packet), not the 2-D
    (n_symbols, n_subcarriers) image-shaped grids AutoencoderDetector's
    Conv2D encoder/decoder is built for. Reshaping a 1-D timing series to
    look image-shaped just to reuse that architecture would be an
    artificial fit, not a real one.
  - This is also a legitimate point of comparison for the report: the
    PHY-layer channel needed a CNN to see a 2-D structural anomaly; does a
    1-D network-layer timing channel need one too, or does a much simpler
    classical test already catch it? Using a different method here makes
    that a real, answerable question instead of assuming the same tool
    fits both.

Method chosen: two-sample Kolmogorov-Smirnov (KS) test
(``scipy.stats.ks_2samp``) comparing an OBSERVED inter-packet-gap sample
against a CLEAN baseline sample for the same slice.

Why KS over the other candidates the Phase 2 prompt named:
  - chi-squared needs a binning choice (bin width/count) that changes the
    test's sensitivity in ways that are hard to justify a priori for a
    gap distribution that is already a burstiness-weighted blend of two
    different component shapes (see traffic.py's module docstring) --
    KS avoids binning entirely by comparing empirical CDFs directly.
  - a z-score-of-gap-variance anomaly score is cheap but blind to a shift
    that leaves variance roughly unchanged. covert_injector.py's encoding
    is an additive MEAN shift on ~half the packets (bit=1 ones), which can
    leave the perturbed sample's variance close to the clean sample's
    variance while still shifting its distribution -- a variance-only
    score would systematically miss exactly the signal this covert
    channel produces, especially at the smaller offsets. KS compares
    the full empirical CDF, so it is sensitive to a mean shift, a
    variance change, or a shape change alike, without having to guess in
    advance which one the attack will produce.
  - entropy/regularity measures were also considered; rejected as harder
    to calibrate a principled false-positive rate for than a
    distribution-comparison test that already has a standard statistic.

Threshold calibration mirrors detector.autoencoder_detector's
percentile-of-clean-tail convention (see its ``calibrate()``) rather than
a textbook fixed significance level: ``calibrate()`` here draws many
independent PAIRS of clean baseline samples for the same slice/generator
and builds a null distribution of the KS D statistic between them, then
sets ``threshold_`` at its ``percentile``-th percentile. percentile=95
means ~5% of clean-vs-clean comparisons will be flagged anomalous BY
CONSTRUCTION -- report that alongside any detection-rate number, exactly
as autoencoder_detector.py's calibrate() docstring insists for its own
threshold.

Per-slice thresholds: ``anomaly_by_slice`` judges every slice against a
threshold calibrated on THAT slice's own clean gaps (``calibrate(...,
slice_type=...)``), and raises ``SliceNotCalibratedError`` for a slice that
was never calibrated -- it never falls back to another slice's threshold.
Why this matters even though the two-sample KS null is distribution-free
(for continuous data it depends only on the two sample sizes, not on the
gap distribution): slices run at different packet rates, so a window
defined in TIME holds a different number of packets per slice, and the
null threshold moves a lot with sample size (95th percentile D ~= 0.137 at
300 packets vs ~= 0.320 at 50). A threshold calibrated at one slice's
window size is simply wrong for another's. With identical packet counts
per window the per-slice thresholds coincide, which is why the single
shared threshold went unnoticed in the fixed-300-packet demo.

Output shape: ``anomaly_by_slice`` returns ``dict[str, bool]`` --
deliberately the exact type ``pqc_auth.orchestration
.drive_reauth_from_detector_flags`` already accepts as its
``anomaly_by_slice`` parameter. This module was checked against that
signature (pqc_auth/orchestration.py, read as part of writing this
module) specifically so the shape lines up; it is NOT wired into
drive_reauth_from_detector_flags or DualTriggerReauthController anywhere
in this change -- that integration is explicitly left for a later phase
(see network_covert_channel/README.md and this PR's description).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import ks_2samp


class SliceNotCalibratedError(RuntimeError):
    """anomaly_by_slice / is_anomalous(slice_type=...) was asked about a
    slice that has no threshold of its own."""


@dataclass(frozen=True)
class KSDetectionResult:
    statistic: float
    anomaly: bool


class TimingKSDetector:
    """Two-sample KS-test anomaly detector over inter-packet-gap samples.
    See module docstring for the full method justification and the
    calibration convention this mirrors from AutoencoderDetector."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        # Single-slice use: calibrate() without slice_type, then
        # is_anomalous()/score() without slice_type.
        self.threshold_: float | None = None
        # Per-slice thresholds: calibrate(..., slice_type=s). The only
        # thresholds anomaly_by_slice() ever reads.
        self.thresholds_: dict[str, float] = {}

    def statistic(self, observed_gaps: np.ndarray, baseline_gaps: np.ndarray) -> float:
        """The KS D statistic (max absolute gap between the two empirical
        CDFs) between an observed sample and a clean baseline sample.
        Higher = more distributionally different = more anomalous."""
        return float(ks_2samp(observed_gaps, baseline_gaps).statistic)

    def calibrate(
        self,
        clean_gap_sampler: Callable[[], np.ndarray],
        n_trials: int = 200,
        percentile: float = 95.0,
        slice_type: str | None = None,
    ) -> float:
        """Sets a threshold from the tail of the CLEAN-vs-CLEAN null
        distribution of the KS D statistic: ``thresholds_[slice_type]`` when
        ``slice_type`` is given (the sampler must then draw that slice's
        clean gaps, at that slice's window size), else ``threshold_``.

        The two-sample KS null distribution depends only on the sample
        sizes, not on the gap distribution, so thresholds really differ by
        packets per window; per-slice calibration is how that difference is
        captured when windows are defined in time.

        ``clean_gap_sampler`` is a zero-arg callable returning a fresh,
        independent clean gap sample each call (e.g.
        ``functools.partial(generate_inter_packet_gaps, slice_type,
        n_packets, rng)`` with an rng not shared with whatever produces the
        gaps later scored against this threshold). ``n_trials`` independent
        PAIRS are drawn; each pair's KS statistic is one draw from the null
        distribution this calibrates against.
        """
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")
        null_stats = np.empty(n_trials)
        for i in range(n_trials):
            a = clean_gap_sampler()
            b = clean_gap_sampler()
            null_stats[i] = self.statistic(a, b)
        threshold = float(np.percentile(null_stats, percentile))
        if slice_type is None:
            self.threshold_ = threshold
        else:
            self.thresholds_[slice_type] = threshold
        return threshold

    def threshold_for(self, slice_type: str | None) -> float:
        """The threshold to judge ``slice_type`` by (``threshold_`` when
        ``slice_type`` is None). Never substitutes another slice's."""
        if slice_type is None:
            if self.threshold_ is None:
                raise RuntimeError("Call calibrate() before scoring.")
            return self.threshold_
        if slice_type not in self.thresholds_:
            raise SliceNotCalibratedError(
                f"no threshold calibrated for slice {slice_type!r} (calibrated: {sorted(self.thresholds_)}); "
                f"call calibrate(..., slice_type={slice_type!r}) on that slice's own clean gaps"
            )
        return self.thresholds_[slice_type]

    def is_anomalous(self, observed_gaps: np.ndarray, baseline_gaps: np.ndarray,
                     slice_type: str | None = None) -> bool:
        threshold = self.threshold_for(slice_type)
        return self.statistic(observed_gaps, baseline_gaps) > threshold

    def score(self, observed_gaps: np.ndarray, baseline_gaps: np.ndarray,
              slice_type: str | None = None) -> KSDetectionResult:
        threshold = self.threshold_for(slice_type)
        stat = self.statistic(observed_gaps, baseline_gaps)
        return KSDetectionResult(statistic=stat, anomaly=stat > threshold)

    def anomaly_by_slice(
        self,
        observed_gaps_by_slice: dict[str, np.ndarray],
        baseline_gaps_by_slice: dict[str, np.ndarray],
    ) -> dict[str, bool]:
        """Per-slice anomaly flags, shaped as ``dict[str, bool]`` --
        directly compatible with
        ``pqc_auth.orchestration.drive_reauth_from_detector_flags``'s
        ``anomaly_by_slice`` parameter (not wired in here, see module
        docstring). Each slice is judged by its OWN calibrated threshold;
        an uncalibrated slice raises SliceNotCalibratedError."""
        return {
            slice_type: self.is_anomalous(
                observed_gaps_by_slice[slice_type], baseline_gaps_by_slice[slice_type], slice_type=slice_type
            )
            for slice_type in observed_gaps_by_slice
        }
