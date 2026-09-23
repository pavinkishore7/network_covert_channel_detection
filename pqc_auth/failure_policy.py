"""What to do when a re-auth request fails.

``decide()`` is a PURE function: slice type + that slice's recent history
(outcomes of every attempt, and the policy's own earlier decisions) + the
current logical time + the slice's config  ->  one ``PolicyDecision``. No
I/O, no clock, no hidden state, so every rule below is unit-tested
directly. ``ReauthSupervisor`` is the thin stateful layer that runs a real
request, feeds the result to ``decide()``, and APPLIES the decision (audit
record, alert, quarantine flag, escalation re-auth).

Actions (one per decision; each implies the ones above it in severity):

  NONE                        nothing to do
  ALERT                       raise an operator alert (logged + callback)
  ESCALATE_TO_DETECTOR_ALERT  ALERT, and request one extra re-auth right now
                              through the EXISTING detector-alert trigger
                              (request_reauth(..., detector_alert=True)), so
                              the server's own alert cooldown still applies
  QUARANTINE_FLAG             ALERT, and mark the slice untrusted until an
                              authenticated answer (VERIFIED / NOT_DUE)
                              arrives; escalates too when the cause is an
                              authentication failure and the cooldown allows

Rules (defaults in ``SliceFailurePolicy``). These are DESIGN CHOICES, argued
here -- not empirical results:

  1. Any AUTH_FAILURE (forged, replayed, wrong-key, unsigned-status or
     malformed response) -> ESCALATE_TO_DETECTOR_ALERT. One bad signature
     is already evidence that something other than the trusted server
     answered; there is no benign "flaky crypto". Escalating asks for a
     fresh re-auth at once instead of waiting up to a full periodic
     interval (up to 300 s for mMTC) with a known-bad answer.
  2. ``auth_failures_to_quarantine`` (2) AUTH_FAILUREs within
     ``auth_failure_window_s`` (300 s, the longest periodic interval) ->
     QUARANTINE_FLAG. Two means "the escalation's fresh attempt ALSO failed
     to authenticate" (or a second incident within one mMTC interval):
     the slice's session can no longer be vouched for.
  3. ``transport_failures_to_alert`` (3) CONSECUTIVE TRANSPORT_FAILURE
     attempts -> ALERT. 3 equals the client's default retry budget, so one
     request that exhausted all of its retries alerts, while a single blip
     that a retry recovered from does not.
  4. Only if the slice is configured ``fail_closed``:
     ``transport_failures_to_quarantine`` (6, i.e. two fully failed
     requests) consecutive TRANSPORT_FAILUREs -> QUARANTINE_FLAG.
  5. REFUSED (an authenticated "I refuse this request") -> ALERT. It is a
     configuration problem; escalating would just be refused again.
  6. An authenticated answer after a quarantine -> NONE, logged as
     "quarantine_cleared".

The two failure modes this module must not create:

  a. Retry/escalation storm (failure -> forced re-auth -> failure -> ...).
     Bounded three ways: (i) the client retries ONLY transport failures,
     at most ``max_attempts`` times, with jittered backoff; (ii) transport
     failures never escalate -- during an outage an extra request only adds
     load; (iii) escalation has a per-slice cooldown
     (``escalation_cooldown_s``, 30 s = 3x the controller's 10 s
     detector-alert cooldown, so an escalation is never sent faster than
     the server would honour one). A failed escalation is itself an
     AUTH_FAILURE, but its decision finds the cooldown active and is
     downgraded (escalation_suppressed=True). So a sustained attack costs
     at most one extra re-auth per slice per 30 s.
  b. DoS amplification. An on-path attacker who can only DROP packets
     produces nothing but TIMEOUT / CONNECTION_FAILED, and by default
     (``fail_closed=False`` on every slice) transport failures can only
     ALERT, never QUARANTINE. Dropping is therefore DETECTED (alerts
     every exhausted request) but cannot be used to flip a slice into
     quarantine. ``fail_closed=True`` is available per slice as a
     deliberate choice. The tradeoff, plainly:
       - fail-closed: a drop-only attacker can force the slice into
         QUARANTINE, i.e. an attacker-induced outage of that slice (for
         URLLC, exactly the traffic that can least afford one);
       - fail-open (default): the slice keeps running on its last verified
         session while re-auth is failing -- an UNVERIFIED session for the
         length of the outage, flagged by alerts but not stopped.
     Fail-open is the default because the attacker gains nothing new
     from it (dropping is already alerted on, and the previous session
     was authenticated), whereas fail-closed hands a drop-only attacker a
     kill switch. Note that an attacker who can INJECT (not only drop) can
     manufacture AUTH_FAILUREs without any key (any garbage signature
     does) and so can reach QUARANTINE under rule 2 on any slice -- that is
     intended: an active forger on the path is precisely when the session
     should stop being trusted, and it is visible as auth failures in the
     audit log, not as a quiet outage.

The dual-trigger design is untouched: nothing here changes, delays or
replaces DualTriggerReauthController's PERIODIC schedule. Escalation only
ADDS detector-alert re-auths, and a quarantined slice is still re-authed
on schedule -- that is how it gets out of quarantine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from pqc_auth.outcomes import OutcomeCategory, RequestOutcome

MAX_HISTORY_EVENTS = 256  # per slice; the rules only ever look back one window


class PolicyAction(str, Enum):
    NONE = "none"
    ALERT = "alert"
    ESCALATE_TO_DETECTOR_ALERT = "escalate_to_detector_alert"
    QUARANTINE_FLAG = "quarantine_flag"


@dataclass(frozen=True)
class SliceFailurePolicy:
    fail_closed: bool = False
    transport_failures_to_alert: int = 3
    transport_failures_to_quarantine: int = 6  # used only when fail_closed
    auth_failures_to_quarantine: int = 2
    auth_failure_window_s: float = 300.0
    escalation_cooldown_s: float = 30.0


# Fail-open on every slice by default -- see the module docstring (b).
DEFAULT_SLICE_POLICIES: dict[str, SliceFailurePolicy] = {
    "URLLC": SliceFailurePolicy(),
    "eMBB": SliceFailurePolicy(),
    "mMTC": SliceFailurePolicy(),
}


@dataclass(frozen=True)
class OutcomeEvent:
    """One attempt's outcome, as logged (``audit_seq`` = its audit record)."""

    now: float
    outcome: RequestOutcome
    audit_seq: int | None = None


@dataclass(frozen=True)
class DecisionEvent:
    """A decision the policy already made (so cooldowns/quarantine are
    derived from history rather than kept as hidden state)."""

    now: float
    action: PolicyAction
    escalated: bool = False


HistoryEvent = OutcomeEvent | DecisionEvent


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    rule: str  # which rule fired: "ok", "quarantine_cleared", "auth_failure", ...
    reason: str
    trigger_seqs: tuple[int, ...] = ()  # audit records that justify it
    threshold: int | None = None
    escalate: bool = False  # the supervisor should send one detector-alert re-auth now
    escalation_suppressed: bool = False  # an escalation was warranted but the cooldown blocked it
    quarantined: bool = False  # slice state AFTER this decision

    @property
    def worth_logging(self) -> bool:
        return self.action is not PolicyAction.NONE or self.rule == "quarantine_cleared"


def _outcomes(history: Sequence[HistoryEvent]) -> list[OutcomeEvent]:
    return [e for e in history if isinstance(e, OutcomeEvent)]


def is_quarantined(history: Sequence[HistoryEvent]) -> bool:
    """Quarantined iff a QUARANTINE_FLAG decision is more recent than the
    most recent authenticated (OK) outcome."""
    quarantined = False
    for event in history:
        if isinstance(event, DecisionEvent) and event.action is PolicyAction.QUARANTINE_FLAG:
            quarantined = True
        elif isinstance(event, OutcomeEvent) and event.outcome.category is OutcomeCategory.OK:
            quarantined = False
    return quarantined


def _last_escalation(history: Sequence[HistoryEvent]) -> float | None:
    times = [e.now for e in history if isinstance(e, DecisionEvent) and e.escalated]
    return max(times) if times else None


def _seqs(events: Sequence[OutcomeEvent]) -> tuple[int, ...]:
    return tuple(e.audit_seq for e in events if e.audit_seq is not None)


def decide(
    slice_type: str,
    history: Sequence[HistoryEvent],
    now: float,
    policy: SliceFailurePolicy | None = None,
) -> PolicyDecision:
    """Decide on the most recent outcome in ``history`` (which must end with
    the outcome events of the request just made). Pure."""
    policy = policy or DEFAULT_SLICE_POLICIES.get(slice_type, SliceFailurePolicy())
    outcomes = _outcomes(history)
    if not outcomes:
        return PolicyDecision(PolicyAction.NONE, "no_outcome", "no outcome recorded yet")
    latest = outcomes[-1]
    category = latest.outcome.category
    latest_index = max(i for i, e in enumerate(history) if isinstance(e, OutcomeEvent))
    was_quarantined = is_quarantined(history[:latest_index])

    if category is OutcomeCategory.OK:
        if was_quarantined:
            return PolicyDecision(
                PolicyAction.NONE, "quarantine_cleared",
                f"{latest.outcome.value}: authenticated answer received, quarantine lifted",
                trigger_seqs=_seqs([latest]), quarantined=False,
            )
        return PolicyDecision(PolicyAction.NONE, "ok", latest.outcome.value)

    if category is OutcomeCategory.REFUSED:
        return PolicyDecision(
            PolicyAction.ALERT, "refused", "server returned an authenticated refusal",
            trigger_seqs=_seqs([latest]), quarantined=was_quarantined,
        )

    last_escalation = _last_escalation(history)
    cooldown_active = last_escalation is not None and now - last_escalation < policy.escalation_cooldown_s

    if category is OutcomeCategory.AUTH_FAILURE:
        in_window = [
            e for e in outcomes
            if e.outcome.category is OutcomeCategory.AUTH_FAILURE and now - e.now <= policy.auth_failure_window_s
        ]
        if len(in_window) >= policy.auth_failures_to_quarantine:
            return PolicyDecision(
                PolicyAction.QUARANTINE_FLAG, "auth_failures_to_quarantine",
                f"{len(in_window)} authentication failures within {policy.auth_failure_window_s:g}s "
                f"(threshold {policy.auth_failures_to_quarantine})",
                trigger_seqs=_seqs(in_window), threshold=policy.auth_failures_to_quarantine,
                escalate=not cooldown_active, escalation_suppressed=cooldown_active, quarantined=True,
            )
        if cooldown_active:
            return PolicyDecision(
                PolicyAction.ALERT, "auth_failure",
                f"{latest.outcome.value}; escalation suppressed (last one {now - last_escalation:g}s ago, "
                f"cooldown {policy.escalation_cooldown_s:g}s)",
                trigger_seqs=_seqs([latest]), escalation_suppressed=True, quarantined=was_quarantined,
            )
        return PolicyDecision(
            PolicyAction.ESCALATE_TO_DETECTOR_ALERT, "auth_failure",
            f"{latest.outcome.value}: forcing a fresh re-auth via the detector-alert path",
            trigger_seqs=_seqs([latest]), escalate=True, quarantined=was_quarantined,
        )

    # TRANSPORT_FAILURE: count the unbroken run of transport failures at the end.
    run: list[OutcomeEvent] = []
    for event in reversed(outcomes):
        if event.outcome.category is not OutcomeCategory.TRANSPORT_FAILURE:
            break
        run.append(event)
    run.reverse()
    if policy.fail_closed and len(run) >= policy.transport_failures_to_quarantine:
        return PolicyDecision(
            PolicyAction.QUARANTINE_FLAG, "transport_failures_to_quarantine_fail_closed",
            f"{len(run)} consecutive transport failures on a fail-closed slice "
            f"(threshold {policy.transport_failures_to_quarantine})",
            trigger_seqs=_seqs(run), threshold=policy.transport_failures_to_quarantine, quarantined=True,
        )
    if len(run) >= policy.transport_failures_to_alert:
        mode = "fail-closed" if policy.fail_closed else "fail-open: continuing on the last verified session"
        return PolicyDecision(
            PolicyAction.ALERT, "transport_failures_to_alert",
            f"{len(run)} consecutive transport failures (threshold {policy.transport_failures_to_alert}; {mode})",
            trigger_seqs=_seqs(run), threshold=policy.transport_failures_to_alert, quarantined=was_quarantined,
        )
    return PolicyDecision(
        PolicyAction.NONE, "transport_below_threshold",
        f"{len(run)} consecutive transport failure(s), below {policy.transport_failures_to_alert}",
        quarantined=was_quarantined,
    )


# -- applying decisions -------------------------------------------------------


@dataclass
class SupervisedResult:
    result: object  # pqc_auth.transport.ClientVerificationResult
    decision: PolicyDecision
    escalation: "SupervisedResult | None" = None


@dataclass
class ReauthSupervisor:
    """Runs requests through ``client`` and applies ``decide()``'s actions:
    every non-trivial decision is written to the client's audit log (with
    the audit seqs of the outcome records that justify it), ALERTs go to
    ``on_alert`` and ``alerts``, QUARANTINE_FLAG sets the slice's flag, and
    an escalation sends exactly one ``detector_alert=True`` re-auth."""

    client: object  # pqc_auth.transport.ReauthClient
    policies: dict[str, SliceFailurePolicy] = field(default_factory=lambda: dict(DEFAULT_SLICE_POLICIES))
    on_alert: Callable[[str, PolicyDecision], None] | None = None
    history: dict[str, list[HistoryEvent]] = field(default_factory=dict)
    alerts: list[tuple[str, float, PolicyDecision]] = field(default_factory=list)

    def policy_for(self, slice_type: str) -> SliceFailurePolicy:
        return self.policies.get(slice_type, SliceFailurePolicy())

    def is_quarantined(self, slice_type: str) -> bool:
        return is_quarantined(self.history.get(slice_type, []))

    def reauth(self, slice_type: str, now: float, detector_alert: bool = False) -> SupervisedResult:
        result = self.client.request_reauth(slice_type, now, detector_alert=detector_alert)
        events = self.history.setdefault(slice_type, [])
        for attempt in result.attempts:
            events.append(OutcomeEvent(now=now, outcome=attempt.outcome, audit_seq=attempt.audit_seq))
        policy = self.policy_for(slice_type)
        decision = decide(slice_type, events, now, policy)
        self._apply(slice_type, now, decision, result, policy)
        escalation = None
        if decision.escalate:
            escalation = self.reauth(slice_type, now, detector_alert=True)
        del events[:-MAX_HISTORY_EVENTS]
        return SupervisedResult(result=result, decision=decision, escalation=escalation)

    def _apply(self, slice_type, now, decision: PolicyDecision, result, policy: SliceFailurePolicy) -> None:
        if decision.action is not PolicyAction.NONE:
            self.history[slice_type].append(DecisionEvent(now=now, action=decision.action, escalated=decision.escalate))
        if decision.action is not PolicyAction.NONE:
            self.alerts.append((slice_type, now, decision))
            if self.on_alert is not None:
                self.on_alert(slice_type, decision)
        logger = getattr(self.client, "audit_logger", None)
        if logger is not None and decision.worth_logging:
            logger.log_event(
                "policy_decision",
                slice_type=slice_type,
                now=now,
                action=decision.action.value,
                rule=decision.rule,
                reason=decision.reason,
                trigger_seqs=list(decision.trigger_seqs),
                threshold=decision.threshold,
                window_s=policy.auth_failure_window_s,
                fail_closed=policy.fail_closed,
                escalated=decision.escalate,
                escalation_suppressed=decision.escalation_suppressed,
                quarantined=decision.quarantined,
                outcome=result.outcome.value if result.outcome is not None else None,
            )
