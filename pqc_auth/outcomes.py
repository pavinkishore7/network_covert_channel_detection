"""The one explicit taxonomy of how a re-auth request can end.

Every ``ReauthClient`` attempt ends in exactly one ``RequestOutcome``, and
every outcome belongs to exactly one ``OutcomeCategory``. The failure
policy (pqc_auth/failure_policy.py) and the retry logic in
pqc_auth/transport.py decide on the CATEGORY, never on ad-hoc flags:

  OK                 the server's answer was authenticated (signature valid,
                     bound to this request's challenge, from the trusted key)
  AUTH_FAILURE       a response arrived but could not be authenticated -- a
                     security event. Never retried: a forged or replayed
                     response is not a flaky network, and retrying would both
                     hide it and hand an attacker more attempts.
  TRANSPORT_FAILURE  no usable response arrived at all (timeout, refused,
                     reset, unreachable, peer closed mid-response). The only
                     category that is retried, with a new connection and a
                     fresh challenge each time.
  REFUSED            the server's AUTHENTICATED answer was "I refuse this
                     request" (e.g. unknown slice type). A configuration
                     problem, not an attack and not a flaky network:
                     retrying would get the same signed refusal.

``MALFORMED_RESPONSE`` is an AUTH_FAILURE, not a transport failure: since
wire version 2 every legitimate response is a well-formed signed payload,
so a complete line that isn't one came from something that is not a
correctly functioning trusted server (a mis-versioned peer or an on-path
injector), and it cannot be authenticated. A response cut off mid-line is
different -- that is the connection breaking, so it is CONNECTION_FAILED.

pqc_auth/audit_verify.py repeats these string values as literals (it must
not import the code it audits); tests/test_failure_policy.py checks they
stay equal.
"""

from __future__ import annotations

from enum import Enum


class OutcomeCategory(str, Enum):
    OK = "ok"
    AUTH_FAILURE = "auth_failure"
    TRANSPORT_FAILURE = "transport_failure"
    REFUSED = "refused"


class RequestOutcome(str, Enum):
    # OK: authenticated answers
    VERIFIED = "verified"  # signed "re-auth performed", accepted
    NOT_DUE = "not_due"  # signed "not due yet", accepted
    # AUTH_FAILURE: a response that could not be authenticated
    REJECTED_SIGNATURE = "rejected_signature"
    REJECTED_REPLAY = "rejected_replay"
    REJECTED_CHALLENGE_MISMATCH = "rejected_challenge_mismatch"
    REJECTED_PINNED_KEY = "rejected_pinned_key"
    REJECTED_TOFU_KEY_CHANGED = "rejected_tofu_key_changed"
    REJECTED_UNSIGNED_STATUS = "rejected_unsigned_status"  # a "not due"/error answer with no signature
    MALFORMED_RESPONSE = "malformed_response"
    # TRANSPORT_FAILURE: no usable response at all
    TIMEOUT = "timeout"
    CONNECTION_FAILED = "connection_failed"
    # REFUSED: an authenticated refusal
    SERVER_REFUSED = "server_refused"

    @property
    def category(self) -> OutcomeCategory:
        return _CATEGORY[self]


_CATEGORY = {
    RequestOutcome.VERIFIED: OutcomeCategory.OK,
    RequestOutcome.NOT_DUE: OutcomeCategory.OK,
    RequestOutcome.REJECTED_SIGNATURE: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.REJECTED_REPLAY: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.REJECTED_CHALLENGE_MISMATCH: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.REJECTED_PINNED_KEY: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.REJECTED_TOFU_KEY_CHANGED: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.REJECTED_UNSIGNED_STATUS: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.MALFORMED_RESPONSE: OutcomeCategory.AUTH_FAILURE,
    RequestOutcome.TIMEOUT: OutcomeCategory.TRANSPORT_FAILURE,
    RequestOutcome.CONNECTION_FAILED: OutcomeCategory.TRANSPORT_FAILURE,
    RequestOutcome.SERVER_REFUSED: OutcomeCategory.REFUSED,
}
