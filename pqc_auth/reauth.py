"""Dual-trigger slice re-authentication policy.

The policy is transport-agnostic: production code supplies a Dilithium signer
and verification transport; tests use a deterministic fake signer.

``due()`` is the original scheduling decision (periodic vs. detector-triggered
vs. not due yet); its default (``dry_run=False``) behavior is unchanged: it
never touches a signer and always returns a bare ``ReauthReason | None``, so
existing callers and tests keep working exactly as before. ``reauth()`` is
new: it calls ``due()`` internally and, only when a ``signer`` was
configured, additionally performs a real sign+verify round trip on a fresh
per-call challenge, returning a ``ReauthOutcome`` that wraps the same
``ReauthReason``.

Both ``due()`` and ``reauth()`` also accept ``dry_run=True``, which answers
the identical scheduling question without writing to this controller's
internal state or (for ``reauth()``) calling the signer at all -- see
``due()``'s own docstring for why this exists: it lets a single controller
instance answer "would this fire" as many times as needed before an actual
fire happens elsewhere, without needing a second controller kept in
lockstep to ask the question safely.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable


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
    round trip on ``signed_message`` — never assumed true, always the real
    ``signer.verify()`` return value.
    """

    reason: ReauthReason
    verified: bool | None
    nonce: bytes | None = None
    signature: bytes | None = None
    # The exact bytes that were signed. Equal to ``nonce`` unless reauth()
    # was given a ``sign_message`` builder (see there).
    signed_message: bytes | None = None
    # Wall-clock cost of signer.sign() / signer.verify() for this call,
    # measured with time.perf_counter() around each one. None when nothing
    # was signed.
    sign_ms: float | None = None
    verify_ms: float | None = None


class DualTriggerReauthController:
    """Schedules periodic and detector-triggered re-auth independently."""

    def __init__(self, policies: dict[str, SlicePolicy] | None = None, signer: Signer | None = None):
        self.policies = policies or DEFAULT_POLICIES
        self.signer = signer
        self._last_reauth: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}
        # due() is a read-modify-write of _last_reauth/_last_alert. A
        # threaded caller (ReauthServer handles connections concurrently)
        # could otherwise let two simultaneous requests for the same slice
        # both read "not fired yet" and both fire. The lock lives HERE,
        # not in the server, so every caller sharing this controller
        # (e.g. live_loop's dry-run checks next to the server's real
        # fires) gets the same guarantee. Only the scheduling decision is
        # serialized; signing in reauth() happens after the lock is
        # released, since the decision it acts on is already committed.
        self._lock = threading.Lock()

    def due(self, slice_type: str, now: float, detector_alert: bool = False, dry_run: bool = False) -> ReauthReason | None:
        """Decide whether ``slice_type`` is due for re-auth right now.

        ``dry_run=False`` (the default, and the only behavior that existed
        before it) is "decide AND commit": a due decision is recorded into
        this controller's own scheduling state (``_last_reauth``/
        ``_last_alert``) in the same call that reports it, which is why two
        separate calls with identical arguments can produce different
        answers -- the first one due changes what the second one sees.

        ``dry_run=True`` answers the identical scheduling question --
        computed with the exact same logic below, not an approximation --
        WITHOUT writing to ``_last_reauth``/``_last_alert``. This is what
        lets a caller ask "would this fire right now?" as many times as it
        wants without affecting whether it actually would, and without
        needing a second controller instance kept in lockstep to answer
        the question safely (see ``pqc_auth/live_loop.py``, whose whole
        point is asking this before deciding to actually fire over the
        real transport).
        """
        if slice_type not in self.policies:
            raise ValueError(f"Unknown slice type: {slice_type}")
        with self._lock:
            return self._due_locked(slice_type, now, detector_alert, dry_run)

    def _due_locked(self, slice_type: str, now: float, detector_alert: bool, dry_run: bool) -> ReauthReason | None:
        policy = self.policies[slice_type]
        last = self._last_reauth.get(slice_type)
        alert_due = detector_alert and now - self._last_alert.get(slice_type, float("-inf")) >= policy.alert_cooldown_seconds
        if alert_due:
            if not dry_run:
                self._last_alert[slice_type] = now
                self._last_reauth[slice_type] = now
            return ReauthReason.DETECTOR_ALERT
        periodic_due = last is None or now - last >= policy.interval_seconds
        if periodic_due:
            if not dry_run:
                self._last_reauth[slice_type] = now
            return ReauthReason.PERIODIC
        return None

    def _challenge(self, slice_type: str, now: float) -> bytes:
        """Fresh, unpredictable per-call challenge: timestamp + slice_type + random token."""
        return f"{now}:{slice_type}:".encode() + secrets.token_bytes(16)

    def reauth(
        self,
        slice_type: str,
        now: float,
        detector_alert: bool = False,
        dry_run: bool = False,
        sign_message: Callable[[ReauthReason, bytes], bytes] | None = None,
    ) -> ReauthOutcome | None:
        """Like :meth:`due`, but when a signer is configured and re-auth is
        due, also performs a real sign+verify round trip on a fresh
        challenge and reports whether it actually verified.

        Returns ``None`` when re-auth is not due (identical to ``due()``
        returning ``None``). Never raises on a failed verification — the
        caller decides what a failed ``verified`` means for their flow;
        this method's job is only to report the true outcome.

        ``dry_run=True`` passes through to :meth:`due` (so no scheduling
        state is written) and additionally skips signing entirely -- the
        signer is never called, matching the "no signing on a dry run"
        requirement, since a dry run's whole point is answering "would
        this fire" without any of the side effects firing actually has.
        The returned ``ReauthOutcome`` still carries the real ``reason``
        (``PERIODIC``/``DETECTOR_ALERT``) so a caller can act on WHY it
        would fire without needing a separate code path; ``verified``,
        ``nonce``, and ``signature`` are ``None``, the same shape already
        used when no signer is configured at all.
        """
        reason = self.due(slice_type, now, detector_alert=detector_alert, dry_run=dry_run)
        if reason is None:
            return None
        if dry_run or self.signer is None:
            return ReauthOutcome(reason=reason, verified=None)
        nonce = self._challenge(slice_type, now)
        # sign_message, if given, maps (reason, this call's fresh nonce) to
        # the bytes that actually get signed -- how pqc_auth.transport binds
        # the signature to the client's challenge and the other response
        # fields. Without it, the bare nonce is signed, as before.
        message = sign_message(reason, nonce) if sign_message is not None else nonce
        started = time.perf_counter()
        signature = self.signer.sign(message)
        signed = time.perf_counter()
        verified = self.signer.verify(message, signature)
        finished = time.perf_counter()
        return ReauthOutcome(
            reason=reason, verified=verified, nonce=nonce, signature=signature, signed_message=message,
            sign_ms=(signed - started) * 1000.0, verify_ms=(finished - signed) * 1000.0,
        )
