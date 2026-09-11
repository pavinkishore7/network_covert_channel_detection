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
  3. Signature validity: recomputing whether the recorded
     (nonce, signature, public_key) actually verifies, and confirming
     that result agrees with what the record claims (``trusted`` or
     ``rejected_as_replay`` -- a replayed record is expected to carry a
     genuinely valid signature that was rejected for being a replay, not
     an invalid one).
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


def _canonical_json(fields: dict) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _check_signature(record: dict) -> tuple[bool, str]:
    try:
        nonce = bytes.fromhex(record["nonce"])
        signature = bytes.fromhex(record["signature"])
        public_key = bytes.fromhex(record["public_key"])
    except (KeyError, ValueError) as exc:
        return False, f"malformed nonce/signature/public_key field: {exc}"

    backend = record.get("backend", "")
    try:
        if backend == "FakeSigner":
            # Independent reimplementation of tests/fake_signer.py's
            # HMAC-SHA256 scheme -- deliberately not imported from there.
            expected = hmac.new(public_key, nonce, hashlib.sha256).digest()
            crypto_valid = hmac.compare_digest(expected, signature)
        else:
            crypto_valid = bool(verify_with_public_key(nonce, signature, public_key))
    except Exception as exc:  # noqa: BLE001 - report any backend failure as a check failure
        return False, f"signature verification raised {type(exc).__name__}: {exc}"

    expected_valid = bool(record.get("trusted")) or bool(record.get("rejected_as_replay"))
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
