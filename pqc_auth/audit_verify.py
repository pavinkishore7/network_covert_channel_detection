"""Independent auditor for pqc_auth's audit log.

Run as::

    python -m pqc_auth.audit_verify <path-to-log.jsonl>

Deliberately does NOT import ``pqc_auth.audit_log.AuditLogger``'s own
hashing helper, nor ``pqc_auth.transport.ReauthClient``/``ReauthServer``:
it recomputes the hash chain from scratch in a few lines, so a bug in the
writer's own hashing logic can't hide from both the writer and this
checker at once. The only thing imported from the rest of ``pqc_auth`` is
``verify_with_public_key`` for oqs-backed records -- that is the actual
production verification primitive being audited, not a piece of the
logging machinery.

For the same reason, this module does not import ``tests.fake_signer``
either (nothing outside ``tests/`` is supposed to -- see that module's own
docstring): HMAC-backed (``FakeSigner``) records are verified with a
independent three-line reimplementation of the same scheme below.

Three independent checks per record:
  1. Content integrity: recomputing ``record_hash`` from the record's own
     fields (everything except ``record_hash`` itself) plus its recorded
     ``prev_hash`` must match the ``record_hash`` actually stored. Catches
     a record whose content was edited without recomputing its hash.
  2. Chain linkage: the record's ``prev_hash`` must equal the sha256 of
     the exact previous line's raw bytes (or 64 zeros for the first
     record). Catches record deletion, reordering, or insertion, and any
     edit to an earlier record even if that edit's own record_hash was
     "fixed up" to be internally consistent.
  3. Trust-outcome consistency: what this actually checks depends on
     the record's kind (and, for wire-version-2 records, the signature is
     recomputed over the exact signed payload the record carries, and the
     challenge comparison is re-done; see ``_check_signature``) --

     - Not a pinning rejection (``pinned_key_mismatch`` false/absent):
       recompute whether the recorded (nonce, signature, public_key)
       actually verifies, and confirm that result agrees with what the
       record claims (``trusted`` or ``rejected_as_replay`` -- a replayed
       record is expected to carry a genuinely valid signature that was
       rejected for being a replay, not an invalid one).
     - A pinning rejection (``pinned_key_mismatch`` true) OR a
       TOFU-detected key change (``trust_store_key_changed`` true): see
       ``_check_signature``'s docstring below for why recomputed
       cryptographic validity is deliberately NOT part of this check for
       these records, and what is checked instead. The two flags share
       one check (see the comment at that branch) because they are the
       same underlying category of record -- "this public_key is not the
       one I trust" -- and differ only in HOW the client came to trust the
       expected key, which this check has no reason to care about.
     - A challenge mismatch (``challenge_mismatch`` true, v2 only): the
       response was not an answer to the client's request. Again crypto
       validity is NOT asserted (a replayed genuine response verifies);
       instead the mismatch claim itself is recomputed from the record.
     - Records written since the failure policy (``record_type``):
       verification records must also have an ``outcome`` consistent with
       their flags; ``transport_failure`` / ``unauthenticated_response``
       records carry no signature, so no crypto check -- they must simply
       not claim trust or signed material; ``policy_decision`` records must
       be re-derivable from the earlier outcome records they cite. See
       ``_check_record`` and the functions it dispatches to.
     - A signed key rotation (``key_rotation_accepted`` true): on top of
       the ordinary checks for a trusted record, every applied rotation
       statement is decoded and re-verified as a chain from the recorded
       ``previous_public_key`` to the record's ``public_key``. A record
       claiming a rotation without such a chain FAILS. See
       ``_check_rotation_accepted``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pqc_auth.dilithium import verify_with_public_key

GENESIS_HASH = "0" * 64

# Every non-oqs Signer backend anywhere in this project implements the
# identical HMAC-SHA256(key=public_key, message=nonce) scheme --
# tests/fake_signer.py::FakeSigner, pqc_auth/demo.py::_DemoFakeSigner, and
# pqc_auth/live_loop.py::_DemoLoopSigner. They are separate, independently
# defined classes (not one imported into the others -- pqc_auth/ never
# imports tests/fake_signer.py, and demo.py/live_loop.py each carry their
# own tiny copy rather than importing from each other), but they share one
# verification scheme. A record whose "backend" field is any of these
# names is checked with the HMAC reimplementation below; anything else
# falls through to the real oqs-backed verify_with_public_key path.
_HMAC_BACKED_BACKENDS = {"FakeSigner", "_DemoFakeSigner", "_DemoLoopSigner"}

# Wire-version-2 domain tag (pqc_auth/transport.py's SIGNATURE_DOMAIN),
# repeated here rather than imported, for the same independence reason as
# everything else in this module. tests/test_transport_binding.py asserts
# the two literals stay equal.
_SIGNATURE_DOMAIN_V2 = b"pqc_auth.reauth.v2\x00"


# pqc_auth/key_rotation.py's ROTATION_DOMAIN, repeated for the same reason;
# tests/test_key_rotation.py asserts the two stay equal. The statement
# decoder below is likewise an independent reimplementation of
# key_rotation.decode_statement.
_ROTATION_DOMAIN = b"pqc_auth.key_rotation.v1\x00"


# pqc_auth/outcomes.py's RequestOutcome values and their categories,
# repeated as literals for the same independence reason as the domain tag
# above; tests/test_failure_policy.py asserts they stay equal.
_OK_OUTCOMES = {"verified", "not_due"}
_AUTH_FAILURE_OUTCOMES = {
    "rejected_signature", "rejected_replay", "rejected_challenge_mismatch", "rejected_pinned_key",
    "rejected_tofu_key_changed", "rejected_unsigned_status", "malformed_response",
}
_TRANSPORT_FAILURE_OUTCOMES = {"timeout", "connection_failed"}
_REFUSED_OUTCOMES = {"server_refused"}

# For a signature-bearing ("verification") record: what each outcome
# requires of the record's own fields. trusted=True means the answer was
# authenticated; each rejection outcome corresponds to exactly one flag
# (or, for rejected_signature, to no flag at all -- it is the plain
# "signature did not verify" case, and the crypto recheck in
# _check_signature confirms it).
_VERIFICATION_OUTCOME_RULES = {
    "verified": {"trusted": True, "flag": None, "status": "due"},
    "not_due": {"trusted": True, "flag": None, "status": "not_due"},
    "server_refused": {"trusted": True, "flag": None, "status": "error"},
    "rejected_signature": {"trusted": False, "flag": None},
    "rejected_replay": {"trusted": False, "flag": "rejected_as_replay"},
    "rejected_challenge_mismatch": {"trusted": False, "flag": "challenge_mismatch"},
    "rejected_pinned_key": {"trusted": False, "flag": "pinned_key_mismatch"},
    "rejected_tofu_key_changed": {"trusted": False, "flag": "trust_store_key_changed"},
}
_REJECTION_FLAGS = ("rejected_as_replay", "challenge_mismatch", "pinned_key_mismatch", "trust_store_key_changed")


def _canonical_json(fields: dict) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _crypto_valid(backend: str, message: bytes, signature: bytes, public_key: bytes) -> bool:
    if backend in _HMAC_BACKED_BACKENDS:
        # Independent reimplementation of the shared HMAC-SHA256
        # scheme -- deliberately not imported from any of the classes
        # that implement it (see _HMAC_BACKED_BACKENDS above).
        expected = hmac.new(public_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)
    return bool(verify_with_public_key(message, signature, public_key))


def _decode_rotation(data: bytes) -> dict:
    """Parse a rotation statement's exact signed bytes (layout documented in
    pqc_auth/key_rotation.py). Raises ValueError on anything malformed."""
    if not data.startswith(_ROTATION_DOMAIN):
        raise ValueError("wrong domain tag")
    pos = len(_ROTATION_DOMAIN)
    parts = []
    for _ in range(3):
        if len(data) < pos + 4:
            raise ValueError("truncated")
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4
        if length == 0 or len(data) < pos + length:
            raise ValueError("bad field length")
        parts.append(data[pos:pos + length])
        pos += length
    if len(data) != pos + 16:
        raise ValueError("wrong length after variable fields")
    epoch, issued_at = struct.unpack(">QQ", data[pos:])
    if len(parts[1]) != 32 or epoch == 0:
        raise ValueError("bad old_pubkey_hash length or epoch 0")
    return {"server_id": parts[0].decode("utf-8"), "old_pubkey_hash": parts[1], "new_pubkey": parts[2],
            "epoch": epoch, "issued_at": issued_at}


_ROTATION_FIELDS = ("rotation_statements", "previous_public_key", "previous_epoch", "new_epoch")


def _check_rotation_accepted(record: dict) -> tuple[bool, str]:
    """A record claiming ``key_rotation_accepted``.

    Reasoning: the claim is "the client moved its pin from
    ``previous_public_key`` to this record's ``public_key`` without a human,
    because the old key endorsed the new one". Everything that claim rests
    on is in the record, so all of it is re-checked here rather than taken
    on trust:
      1. it is an accepted, authenticated answer (trusted, an OK/refused
         outcome, no rejection flag) -- a rotation is only ever applied
         after the response itself verified under the new key; the
         ordinary signature branch then confirms that verification;
      2. there is at least one statement, epochs start above
         ``previous_epoch`` and strictly increase, and the last equals
         ``new_epoch``;
      3. link by link, the statement names the key trusted at that point
         (sha256 matches), its signature verifies UNDER THAT KEY, and it is
         for the ``server_id`` in the record's signed payload;
      4. the chain ends at the record's ``public_key``.
    A record that says a rotation was accepted but whose statements are
    missing, forged (signed by any other key), stale or broken therefore
    FAILS. What is NOT checked: that ``previous_public_key`` was really the
    client's pin at the time. The pin lives in the client's trust store,
    which the log does not mirror (``force_retrust()`` writes no record, and
    several clients may share one log), so a cross-record "previous trusted
    key" check would false-FAIL legitimate logs. Tampering with the field
    after the fact is what the hash chain (checks 1-2) catches.
    """
    if not record.get("trusted") or record.get("outcome") not in ("verified", "not_due", "server_refused", None):
        return False, "key_rotation_accepted=True on a record that was not an accepted, authenticated answer"
    if any(record.get(f) for f in _REJECTION_FLAGS):
        return False, "key_rotation_accepted=True alongside a rejection flag"
    statements = record.get("rotation_statements")
    if not isinstance(statements, list) or not statements:
        return False, "key_rotation_accepted=True but no rotation statement is recorded"
    try:
        current_key = bytes.fromhex(record["previous_public_key"])
        current_epoch = int(record["previous_epoch"])
        final_key = bytes.fromhex(record["public_key"])
        server_id = json.loads(record["signed_payload"])["server_id"]
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"key_rotation_accepted record missing previous key/epoch or signed payload: {exc}"
    backend = record.get("backend", "")
    for index, item in enumerate(statements):
        try:
            encoded = bytes.fromhex(item["statement"])
            signature = bytes.fromhex(item["signature"])
            st = _decode_rotation(encoded)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            return False, f"rotation statement #{index} is malformed: {exc}"
        if st["server_id"] != server_id:
            return False, f"rotation statement #{index} is for {st['server_id']!r}, payload server_id is {server_id!r}"
        if st["epoch"] <= current_epoch:
            return False, f"rotation statement #{index} epoch {st['epoch']} is not above {current_epoch}"
        if st["old_pubkey_hash"] != hashlib.sha256(current_key).digest():
            return False, f"rotation statement #{index} was not issued by the previously trusted key"
        try:
            valid = _crypto_valid(backend, encoded, signature, current_key)
        except Exception as exc:  # noqa: BLE001
            return False, f"rotation statement #{index} verification raised {type(exc).__name__}: {exc}"
        if not valid:
            return False, f"rotation statement #{index} signature does not verify under the previously trusted key"
        current_key, current_epoch = st["new_pubkey"], st["epoch"]
    if current_key != final_key:
        return False, "rotation chain does not end at the record's public_key"
    if current_epoch != record.get("new_epoch"):
        return False, f"record new_epoch={record.get('new_epoch')!r} but the chain ends at epoch {current_epoch}"
    return True, ""


def _check_rotation_fields(record: dict) -> tuple[bool, str]:
    if record.get("key_rotation_accepted"):
        return _check_rotation_accepted(record)
    present = [f for f in _ROTATION_FIELDS if f in record]
    if present or "key_rotation_accepted" in record:
        # Only the writer's rotation path produces these, always together
        # with key_rotation_accepted=True.
        return False, f"rotation fields {present or ['key_rotation_accepted']} without key_rotation_accepted=True"
    if "rotation_rejected_reason" in record and not record.get("trust_store_key_changed"):
        return False, "rotation_rejected_reason on a record that is not a TOFU key-change rejection"
    return True, ""


def _check_signature(record: dict) -> tuple[bool, str]:
    try:
        nonce = bytes.fromhex(record["nonce"])
        signature = bytes.fromhex(record["signature"])
        public_key = bytes.fromhex(record["public_key"])
    except (KeyError, ValueError) as exc:
        return False, f"malformed nonce/signature/public_key field: {exc}"

    trusted = bool(record.get("trusted"))
    rejected_as_replay = bool(record.get("rejected_as_replay"))
    pinned_key_mismatch = bool(record.get("pinned_key_mismatch"))
    trust_store_key_changed = bool(record.get("trust_store_key_changed"))
    challenge_mismatch = bool(record.get("challenge_mismatch"))

    # What the signature covers depends on the record's wire version.
    # v2 records carry the exact signed payload plus the challenge the
    # client sent; pre-v2 records (logs written before the challenge
    # binding existed) carry neither, and their signature covers the bare
    # nonce. Auditing an old log is not the same as accepting an old
    # response -- ReauthClient no longer accepts v1 responses at all.
    signed_payload = record.get("signed_payload")
    if signed_payload is not None:
        try:
            payload = json.loads(signed_payload)
            payload_challenge = bytes.fromhex(payload["client_challenge"])
            expected_challenge = bytes.fromhex(record["expected_challenge"])
            payload_nonce = bytes.fromhex(payload["server_nonce"])
            message = _SIGNATURE_DOMAIN_V2 + signed_payload.encode("ascii")
        except (KeyError, TypeError, ValueError, UnicodeEncodeError) as exc:
            return False, f"malformed signed_payload/expected_challenge: {exc}"
        if payload_nonce != nonce:
            return False, "record nonce does not match the signed payload's server_nonce"
        if "status" in record and record.get("status") != payload.get("status", "due"):
            return False, (
                f"record status={record.get('status')!r} but the signed payload says {payload.get('status')!r}"
            )
    else:
        if challenge_mismatch or "expected_challenge" in record:
            return False, "challenge fields present without signed_payload -- not a valid v1 or v2 record"
        payload_challenge = expected_challenge = None
        message = nonce

    if challenge_mismatch and (pinned_key_mismatch or trust_store_key_changed or rejected_as_replay):
        # process_response() returns at the FIRST failing check (key, then
        # challenge, then seen-nonce), so one record can never carry two
        # rejection reasons.
        return False, (
            "challenge_mismatch=True alongside another rejection flag -- the client stops at the first failing check"
        )

    if pinned_key_mismatch or trust_store_key_changed:
        # Reasoning (Task A.4, extended for TOFU): ReauthClient.process_response()
        # performs BOTH the explicit-pinning check and the TOFU key-change
        # check BEFORE it ever calls verify_fn (see pqc_auth/transport.py)
        # -- either one means "this public_key is not the one I trust", a
        # decision made without looking at whether a signature under that
        # (wrong) key would itself validate. Consequently the (nonce,
        # signature, public_key) triple recorded here MAY be a perfectly
        # genuine signature from a real, just-not-expected, keypair --
        # there is no bug in that; it is exactly the scenario both
        # mechanisms exist to catch (e.g. a second, legitimate-in-its-own-
        # right keypair impersonating the trusted identity, or a server's
        # identity genuinely changing under an already-known server_id).
        # Recomputing crypto_valid and comparing it to
        # trusted/rejected_as_replay -- the ordinary branch below -- is
        # therefore the WRONG check for either kind of record: it would
        # compare an answer to a question the real client deliberately
        # never asked, and a genuinely valid signature under the wrong key
        # would then make this auditor report a false FAIL on a correctly-
        # functioning rejection (a real cost: false FAILs from an auditor
        # train operators to stop trusting it, which defeats the point of
        # having one).
        #
        # pinned_key_mismatch and trust_store_key_changed are deliberately
        # checked together, not with two copies of this reasoning: they
        # are the SAME underlying category of record ("this public_key is
        # not the one I trust") and differ only in how the client arrived
        # at the key it trusted -- explicitly caller-asserted vs. learned
        # on first contact -- which is irrelevant to what this auditor can
        # independently verify. Two separate branches with the same logic
        # would be two places to get this exact reasoning right (or wrong)
        # instead of one; a future third "wrong key" mechanism should join
        # this same branch rather than growing a third copy.
        #
        # What this auditor CAN and does independently check for either
        # kind of record is the only invariant that actually follows from
        # the client's logic: a "wrong key" rejection is a rejection, full
        # stop, so it must never ALSO claim to be trusted or accepted as a
        # non-replay. That is checked here, deliberately without touching
        # crypto_valid at all.
        if trusted or rejected_as_replay:
            return False, (
                f"pinned_key_mismatch={pinned_key_mismatch!r} trust_store_key_changed={trust_store_key_changed!r} "
                f"but record also claims trusted={record.get('trusted')!r} "
                f"rejected_as_replay={record.get('rejected_as_replay')!r} "
                "-- a 'wrong key' rejection must not also be trusted or accepted as a replay"
            )
        return True, ""

    if challenge_mismatch:
        # Reasoning -- the same discipline as the "wrong key" branch above,
        # for a different question. process_response() compares the
        # payload's client_challenge with the challenge it sent BEFORE it
        # ever calls verify_fn. A challenge-mismatch record is typically a
        # REAL, OLD response replayed from an earlier exchange: genuinely
        # signed by the trusted key, so its signature verifies perfectly.
        # It may equally be a forgery whose signature doesn't verify. Both
        # are correct rejections, so "crypto valid <=> trusted" -- the
        # ordinary branch below -- is the wrong assertion here: it would
        # FAIL every correctly rejected replay of a genuine response.
        #
        # What CAN be checked independently, and is:
        #   1. it is a rejection: not trusted, not a replay acceptance;
        #   2. the claim itself is true: the challenge inside the signed
        #      payload really differs from the one the client recorded
        #      sending. (A record claiming a mismatch that isn't there
        #      means the client -- or someone editing the log -- lied.)
        if trusted:
            return False, "challenge_mismatch=True but record also claims trusted=True"
        if hmac.compare_digest(payload_challenge, expected_challenge):
            return False, (
                "challenge_mismatch=True but the signed payload's client_challenge EQUALS the expected challenge"
            )
        return True, ""

    if signed_payload is not None and not hmac.compare_digest(payload_challenge, expected_challenge):
        # Reaching verify_fn at all (trusted, plain signature failure, or
        # the seen-nonce replay check) requires the challenge to have
        # matched first; a record saying otherwise is inconsistent.
        return False, (
            "record was judged on its signature (not as challenge_mismatch) but the signed payload's "
            "client_challenge differs from the expected challenge"
        )

    try:
        crypto_valid = _crypto_valid(record.get("backend", ""), message, signature, public_key)
    except Exception as exc:  # noqa: BLE001 - report any backend failure as a check failure
        return False, f"signature verification raised {type(exc).__name__}: {exc}"

    expected_valid = trusted or rejected_as_replay
    if crypto_valid != expected_valid:
        return False, (
            f"signature check mismatch: recomputed cryptographic validity={crypto_valid} "
            f"but record claims trusted={record.get('trusted')!r} "
            f"rejected_as_replay={record.get('rejected_as_replay')!r}"
        )
    return True, ""


@dataclass
class RecordResult:
    index: int
    seq: int | None
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class LogVerificationResult:
    path: Path
    records: list[RecordResult]

    @property
    def all_clean(self) -> bool:
        return all(r.ok for r in self.records)


def _check_verification_outcome(record: dict) -> tuple[bool, str]:
    """For a verification record that names its ``outcome``: the outcome
    and the record's own fields must tell the same story.

    Why this is needed on top of _check_signature: that check proves the
    trusted/flag fields are consistent with the SIGNATURE; this one proves
    the summary the failure policy acts on (``outcome``) is consistent with
    those fields. Without it, a record could say outcome=verified while its
    flags say the key was wrong, and a policy decision citing it as
    "authenticated" would look justified. So: the outcome must be one that
    a signature-bearing record can have; ``trusted`` must match it; exactly
    the rejection flag that outcome implies must be set, and no other.
    """
    outcome = record["outcome"]
    rule = _VERIFICATION_OUTCOME_RULES.get(outcome)
    if rule is None:
        return False, f"outcome={outcome!r} cannot appear on a signature-bearing verification record"
    if bool(record.get("trusted")) != rule["trusted"]:
        return False, f"outcome={outcome!r} but record claims trusted={record.get('trusted')!r}"
    set_flags = [f for f in _REJECTION_FLAGS if record.get(f)]
    expected = [rule["flag"]] if rule["flag"] else []
    if set_flags != expected:
        return False, f"outcome={outcome!r} but rejection flags set are {set_flags} (expected {expected})"
    if "status" in rule and record.get("status") != rule["status"]:
        return False, f"outcome={outcome!r} but record status={record.get('status')!r}"
    return True, ""


def _check_signatureless_record(record: dict, allowed: set[str], kind: str) -> tuple[bool, str]:
    """``transport_failure`` and ``unauthenticated_response`` records.

    Reasoning -- same discipline as the wrong-key branch in
    _check_signature: a TIMEOUT or CONNECTION_FAILED record has no response
    and therefore no signature; an unsigned "not due" or a malformed line
    has nothing that was signed. Asking "does the signature verify" is the
    wrong question for all of them -- there is no signature to ask it of,
    and demanding one would FAIL every correctly recorded outage.

    What CAN be checked: the outcome is one this record type may carry; the
    record never claims the response was trusted; and it carries no
    signature material or rejection flag, which could only have come from a
    signed response (a record mixing the two was not written by the
    client's logic).
    """
    outcome = record.get("outcome")
    if outcome not in allowed:
        return False, f"{kind} record with outcome={outcome!r} (allowed: {sorted(allowed)})"
    if record.get("trusted"):
        return False, f"{kind} record claims trusted=True"
    for key in ("signature", "public_key", "signed_payload", *_REJECTION_FLAGS):
        if record.get(key):
            return False, f"{kind} record carries {key!r}, which only a signed response could produce"
    attempt = record.get("attempt")
    if not isinstance(attempt, int) or attempt < 1:
        return False, f"{kind} record has invalid attempt={attempt!r}"
    return True, ""


def _category_of(outcome: str | None) -> str | None:
    if outcome in _OK_OUTCOMES:
        return "ok"
    if outcome in _AUTH_FAILURE_OUTCOMES:
        return "auth_failure"
    if outcome in _TRANSPORT_FAILURE_OUTCOMES:
        return "transport_failure"
    if outcome in _REFUSED_OUTCOMES:
        return "refused"
    return None


def _check_policy_decision(record: dict, prior: dict[int, dict]) -> tuple[bool, str]:
    """A failure-policy decision must be justified by the outcome records it
    cites (``trigger_seqs``), all of which must appear EARLIER in this same
    log, for the same slice.

    Reasoning: a decision record has no signature of its own; what makes it
    trustworthy is that anyone can re-derive it from the evidence before it.
    A QUARANTINE_FLAG in particular takes a slice out of trusted service, so
    the auditor recomputes the rule that produced it from the cited records
    instead of taking the record's word for it:
      - quarantine for auth failures: >= threshold cited records, every one
        an AUTH_FAILURE outcome, all within window_s of the decision;
      - quarantine for transport failures: only legal when the slice was
        fail_closed (otherwise dropping packets could quarantine a slice --
        the DoS-amplification case the policy exists to prevent), and the
        cited records must be >= threshold TRANSPORT_FAILUREs with no
        authenticated outcome for that slice between the first cited record
        and the decision (i.e. really consecutive);
      - escalation: cited record is an AUTH_FAILURE (transport failures
        never escalate);
      - quarantine_cleared: cited record is an authenticated (OK) outcome.
    Crypto validity is not re-asked here: the cited records' own checks
    already did that, record by record.
    """
    slice_type = record.get("slice_type")
    action, rule = record.get("action"), record.get("rule")
    seqs = record.get("trigger_seqs") or []
    own_seq = record.get("seq")
    cited = []
    for seq in seqs:
        cited_record = prior.get(seq)
        if cited_record is None or not isinstance(own_seq, int) or seq >= own_seq:
            return False, f"policy decision cites seq {seq}, which is not an earlier record in this log"
        if cited_record.get("slice_type") != slice_type:
            return False, f"policy decision for {slice_type!r} cites seq {seq} for {cited_record.get('slice_type')!r}"
        cited.append(cited_record)
    categories = [_category_of(r.get("outcome")) for r in cited]
    threshold = record.get("threshold") or 0

    if action == "quarantine_flag":
        if rule == "auth_failures_to_quarantine":
            window = float(record.get("window_s") or 0)
            if len(cited) < threshold or any(c != "auth_failure" for c in categories):
                return False, f"quarantine cites {categories}, not >= {threshold} auth failures"
            if any(float(record["now"]) - float(r.get("now", float("-inf"))) > window for r in cited):
                return False, "quarantine cites an auth failure outside its window"
            return True, ""
        if rule == "transport_failures_to_quarantine_fail_closed":
            if not record.get("fail_closed"):
                return False, "quarantine for transport failures on a slice that is not fail_closed"
            if len(cited) < threshold or any(c != "transport_failure" for c in categories):
                return False, f"quarantine cites {categories}, not >= {threshold} transport failures"
            first = min(seqs)
            between = [r for s, r in prior.items() if first < s < own_seq and r.get("slice_type") == slice_type]
            if any(_category_of(r.get("outcome")) == "ok" for r in between):
                return False, "quarantine's transport failures are not consecutive (an authenticated answer intervenes)"
            return True, ""
        return False, f"quarantine_flag with unknown rule {rule!r}"
    if action == "escalate_to_detector_alert" or record.get("escalated"):
        if not cited or categories[-1] != "auth_failure":
            return False, f"escalation must be triggered by an auth failure, cites {categories}"
    if action == "none":
        if rule != "quarantine_cleared" or not cited or categories[-1] != "ok":
            return False, f"'none' decision logged without an authenticated outcome clearing quarantine ({rule!r})"
        return True, ""
    if action == "alert":
        if rule == "transport_failures_to_alert" and (len(cited) < threshold or any(c != "transport_failure" for c in categories)):
            return False, f"transport alert cites {categories}, not >= {threshold} transport failures"
        if rule == "refused" and categories != ["refused"]:
            return False, f"refusal alert cites {categories}"
        if rule == "auth_failure" and (not cited or categories[-1] != "auth_failure"):
            return False, f"auth-failure alert cites {categories}"
        return True, ""
    if action == "escalate_to_detector_alert":
        return True, ""
    return False, f"unknown policy action {action!r}"


def _check_record(record: dict, prior: dict[int, dict]) -> tuple[bool, str]:
    record_type = record.get("record_type", "verification")  # records predating record_type are all verifications
    if record_type == "transport_failure":
        return _check_signatureless_record(record, _TRANSPORT_FAILURE_OUTCOMES, "transport_failure")
    if record_type == "unauthenticated_response":
        return _check_signatureless_record(record, {"rejected_unsigned_status", "malformed_response"},
                                           "unauthenticated_response")
    if record_type == "policy_decision":
        return _check_policy_decision(record, prior)
    if record_type != "verification":
        return False, f"unknown record_type {record_type!r}"
    ok, reason = _check_signature(record)
    if ok and "outcome" in record:
        ok, reason = _check_verification_outcome(record)
    if ok:
        ok, reason = _check_rotation_fields(record)
    return ok, reason


def verify_log(path: str | Path) -> LogVerificationResult:
    path = Path(path)
    raw_lines = [line.rstrip(b"\n") for line in path.read_bytes().split(b"\n") if line.strip()]

    results: list[RecordResult] = []
    expected_prev_hash = GENESIS_HASH
    prior: dict[int, dict] = {}  # earlier records by seq, for policy decisions to cite

    for index, raw_line in enumerate(raw_lines):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            results.append(RecordResult(index=index, seq=None, reasons=[f"invalid JSON: {exc}"]))
            expected_prev_hash = hashlib.sha256(raw_line).hexdigest()
            continue

        reasons: list[str] = []

        fields_without_hash = {k: v for k, v in record.items() if k != "record_hash"}
        recomputed_record_hash = hashlib.sha256(_canonical_json(fields_without_hash).encode("utf-8")).hexdigest()
        if recomputed_record_hash != record.get("record_hash"):
            reasons.append(
                "record_hash mismatch: content was edited without recomputing record_hash "
                f"(recomputed {recomputed_record_hash}, stored {record.get('record_hash')})"
            )

        if record.get("prev_hash") != expected_prev_hash:
            reasons.append(
                "prev_hash mismatch: chain broken before this record "
                f"(expected {expected_prev_hash}, found {record.get('prev_hash')})"
            )

        record_ok, record_reason = _check_record(record, prior)
        if not record_ok:
            reasons.append(record_reason)

        results.append(RecordResult(index=index, seq=record.get("seq"), reasons=reasons))
        if isinstance(record.get("seq"), int):
            prior[record["seq"]] = record
        expected_prev_hash = hashlib.sha256(raw_line).hexdigest()

    return LogVerificationResult(path=path, records=results)


def format_report(result: LogVerificationResult) -> str:
    lines = [f"Auditing {result.path}"]
    for r in result.records:
        if r.ok:
            lines.append(f"  record #{r.index} (seq={r.seq}): PASS")
        else:
            lines.append(f"  record #{r.index} (seq={r.seq}): FAIL")
            for reason in r.reasons:
                lines.append(f"      - {reason}")
    failures = [r for r in result.records if not r.ok]
    lines.append("-" * 60)
    lines.append(f"{len(result.records)} record(s) checked, {len(failures)} failure(s)")
    lines.append(f"OVERALL: {'PASS' if result.all_clean else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m pqc_auth.audit_verify <path-to-log.jsonl>", file=sys.stderr)
        return 2
    result = verify_log(argv[0])
    print(format_report(result))
    return 0 if result.all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
