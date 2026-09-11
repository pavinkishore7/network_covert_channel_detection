"""Dual-trigger slice re-authentication policy.

The policy is transport-agnostic: production code supplies a Dilithium signer
and verification transport; tests use a deterministic fake signer.

``due()`` is the original scheduling decision (periodic vs. detector-triggered
vs. not due yet) and is unchanged: it never touches a signer and always
returns a bare ``ReauthReason | None``, so existing callers and tests keep
working exactly as before. ``reauth()`` is new: it calls ``due()`` internally
and, only when a ``signer`` was configured, additionally performs a real
sign+verify round trip on a fresh per-call challenge, returning a
``ReauthOutcome`` that wraps the same ``ReauthReason``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    """Structural interface a re-auth signer must satisfy.

    ``pqc_auth.dilithium.OqsDilithiumSigner`` already satisfies this shape
    via its existing ``sign()``/``verify()`` methods — nothing there needs to
    change; this Protocol is checked structurally (duck typing), not via
    inheritance. Test code should use a fake signer from ``tests/``, never a
    production stand-in — see ``tests/fake_signer.py``.
    """

    def sign(self, message: bytes) -> bytes: ...

    def verify(self, message: bytes, signature: bytes) -> bool: ...


@runtime_checkable
class PublicKeyVerifier(Protocol):
    """Structural interface for a verifying party that holds ONLY a public
    key — never a signing capability. A real ``Signer`` (above) can sign
    AND self-verify; this is deliberately narrower: it takes the public key
    as an explicit argument on every call, so it can be satisfied by a bare
    module-level function with no signer instance behind it at all.

    Concrete examples matching this call shape, both ``(message, signature,
    public_key) -> bool``: ``pqc_auth.dilithium.verify_with_public_key``
    (a free function — nothing to instantiate) and
    ``tests.fake_signer.FakeSigner.verify_with_public_key`` (a staticmethod,
    callable without ever constructing a signer).

    ``pqc_auth.transport``'s client role is typed against this Protocol,
    not ``Signer``: a client that only ever receives a bound function
    matching this shape has no way to accidentally end up with signing
    capability, by construction rather than by convention.
    """

    def __call__(self, message: bytes, signature: bytes, public_key: bytes) -> bool: ...


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


@dataclass(frozen=True)
class ReauthOutcome:
    """Result of :meth:`DualTriggerReauthController.reauth`.

    ``verified`` is ``None`` when no signer was configured (nothing to
    verify); otherwise it is the actual boolean result of a sign+verify
    round trip on ``nonce`` — never assumed true, always the real
    ``signer.verify()`` return value.
    """

    reason: ReauthReason
    verified: bool | None
    nonce: bytes | None = None
    signature: bytes | None = None


class DualTriggerReauthController:
    """Schedules periodic and detector-triggered re-auth independently."""

    def __init__(self, policies: dict[str, SlicePolicy] | None = None, signer: Signer | None = None):
        self.policies = policies or DEFAULT_POLICIES
        self.signer = signer
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

    def _challenge(self, slice_type: str, now: float) -> bytes:
        """Fresh, unpredictable per-call challenge: timestamp + slice_type + random token."""
        return f"{now}:{slice_type}:".encode() + secrets.token_bytes(16)

    def reauth(self, slice_type: str, now: float, detector_alert: bool = False) -> ReauthOutcome | None:
        """Like :meth:`due`, but when a signer is configured and re-auth is
        due, also performs a real sign+verify round trip on a fresh
        challenge and reports whether it actually verified.

        Returns ``None`` when re-auth is not due (identical to ``due()``
        returning ``None``). Never raises on a failed verification — the
        caller decides what a failed ``verified`` means for their flow;
        this method's job is only to report the true outcome.
        """
        reason = self.due(slice_type, now, detector_alert=detector_alert)
        if reason is None:
            return None
        if self.signer is None:
            return ReauthOutcome(reason=reason, verified=None)
        nonce = self._challenge(slice_type, now)
        signature = self.signer.sign(nonce)
        verified = self.signer.verify(nonce, signature)
        return ReauthOutcome(reason=reason, verified=verified, nonce=nonce, signature=signature)
