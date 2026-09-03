"""Dual-trigger slice re-authentication policy.

The policy is transport-agnostic: production code supplies a Dilithium signer
and verification transport; tests use a deterministic fake signer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReauthReason(str, Enum):
    PERIODIC = "periodic"
    DETECTOR_ALERT = "detector_alert"


@dataclass(frozen=True)
class SlicePolicy:
    interval_seconds: int
    alert_cooldown_seconds: int = 10


DEFAULT_POLICIES = {
    "URLLC": SlicePolicy(30),
    "eMBB": SlicePolicy(90),
    "mMTC": SlicePolicy(300),
}


class DualTriggerReauthController:
    """Schedules periodic and detector-triggered re-auth independently."""

    def __init__(self, policies: dict[str, SlicePolicy] | None = None):
        self.policies = policies or DEFAULT_POLICIES
        self._last_reauth: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}

    def due(self, slice_type: str, now: float, detector_alert: bool = False) -> ReauthReason | None:
        if slice_type not in self.policies:
            raise ValueError(f"Unknown slice type: {slice_type}")
        policy = self.policies[slice_type]
        last = self._last_reauth.get(slice_type)
        if detector_alert and now - self._last_alert.get(slice_type, float("-inf")) >= policy.alert_cooldown_seconds:
            self._last_alert[slice_type] = now
            self._last_reauth[slice_type] = now
            return ReauthReason.DETECTOR_ALERT
        if last is None or now - last >= policy.interval_seconds:
            self._last_reauth[slice_type] = now
            return ReauthReason.PERIODIC
        return None
