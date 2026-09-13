"""Detector-to-reauth orchestration glue.

Bridges per-slice anomaly flags (e.g. from
``detector.autoencoder_detector.AutoencoderDetector.predict_anomaly()``) to
``DualTriggerReauthController``'s ``due()``/``reauth()`` scheduling
decisions. Before this module, nothing in the codebase connected a
detector's output to a re-auth decision at all (confirmed by grepping for
``DualTriggerReauthController``/``reauth`` across the repo before writing
this) — this is new glue code, not a refactor of anything existing.

What this deliberately does NOT do yet (stated plainly, matching
docs/DECISIONS.md / docs/NOVELTY.md's habit of not overclaiming):
  - No real-time integration with a running detector process. This module
    is a plain function you call with anomaly flags you've already computed
    (however you computed them); it does not poll, subscribe to, or manage
    a detector's lifecycle.
  - No persistence of re-auth history across restarts. Scheduling state
    still lives only in the ``DualTriggerReauthController`` instance's
    in-memory dicts, exactly as before this change — restart the process
    and every slice's schedule resets.
  - No handling of what happens if a real re-auth's signature verification
    fails. ``reauth()`` reports ``ReauthOutcome.verified`` truthfully, and
    this module passes that outcome straight through, but there is no
    retry, alerting, or lockout policy here for a ``False`` result. That is
    a deliberate gap for a later prompt, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

from pqc_auth.reauth import DualTriggerReauthController, ReauthOutcome


@dataclass(frozen=True)
class SliceReauthDecision:
    """One slice's reauth() outcome, paired with which slice it's for."""

    slice_type: str
    outcome: ReauthOutcome


def drive_reauth_from_detector_flags(
    anomaly_by_slice: dict[str, bool],
    controller: DualTriggerReauthController,
    now: float,
    dry_run: bool = False,
) -> list[SliceReauthDecision]:
    """Call ``controller.reauth()`` for every slice in ``anomaly_by_slice``,
    passing that slice's flag as ``detector_alert``.

    Returns only the slices where re-auth was actually due at ``now`` (i.e.
    where ``reauth()`` did not return ``None``) — this reuses
    ``DualTriggerReauthController``'s existing scheduling semantics
    unchanged rather than reimplementing any interval or cooldown logic
    here.

    ``dry_run=False`` is the original, only-ever-existing behavior: every
    call here passes straight through to ``controller.reauth()`` with its
    own default (``dry_run=False``), so nothing about existing callers
    changes. ``dry_run=True`` passes that through instead, turning every
    slice's decision into a "would this fire" query that touches neither
    ``controller``'s scheduling state nor its signer — added for
    ``pqc_auth/live_loop.py``, which needs to ask this question against
    the SAME controller instance that a real ``ReauthServer`` answers
    requests with, without that asking consuming the very re-auth window
    it's asking about.
    """
    decisions: list[SliceReauthDecision] = []
    for slice_type, anomaly in anomaly_by_slice.items():
        outcome = controller.reauth(slice_type, now, detector_alert=anomaly, dry_run=dry_run)
        if outcome is not None:
            decisions.append(SliceReauthDecision(slice_type=slice_type, outcome=outcome))
    return decisions
