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
     whether the record is a pinning rejection --

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
"""

from __future__ import annotations

import hashlib
import hmac
import json
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


def _canonical_json(fields: dict) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


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

    backend = record.get("backend", "")
    try:
        if backend in _HMAC_BACKED_BACKENDS:
            # Independent reimplementation of the shared HMAC-SHA256
            # scheme -- deliberately not imported from any of the classes
            # that implement it (see _HMAC_BACKED_BACKENDS above).
            expected = hmac.new(public_key, nonce, hashlib.sha256).digest()
            crypto_valid = hmac.compare_digest(expected, signature)
        else:
            crypto_valid = bool(verify_with_public_key(nonce, signature, public_key))
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


def verify_log(path: str | Path) -> LogVerificationResult:
    path = Path(path)
    raw_lines = [line.rstrip(b"\n") for line in path.read_bytes().split(b"\n") if line.strip()]

    results: list[RecordResult] = []
    expected_prev_hash = GENESIS_HASH

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

        sig_ok, sig_reason = _check_signature(record)
        if not sig_ok:
            reasons.append(sig_reason)

        results.append(RecordResult(index=index, seq=record.get("seq"), reasons=reasons))
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
