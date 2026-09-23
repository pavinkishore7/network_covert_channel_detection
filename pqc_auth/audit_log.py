"""Append-only, hash-chained audit log for client-side re-auth verifications.

Each record captures one ``ReauthClient.process_response()`` outcome: what
was verified, against which public key, and whether it was trusted. Records
are chained by hash so that tampering with (or deleting) any past record is
detectable independently of this module -- see ``pqc_auth/audit_verify.py``,
which recomputes the chain from scratch without importing anything from
here.

Chain construction:
  - ``prev_hash`` on record N is the sha256 hex digest of the *exact
    serialized bytes* of record N-1's JSONL line (or 64 zeros for the first
    record in the file).
  - ``record_hash`` on record N is the sha256 hex digest of record N's own
    fields (everything except ``record_hash`` itself, including
    ``prev_hash``), serialized the same canonical way.

This means every record's hash transitively depends on every record before
it: changing anything in an old record changes that record's own bytes,
which changes the ``prev_hash`` the next record was chained against, and so
on for every record after it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS_HASH = "0" * 64


def canonical_json(fields: dict) -> str:
    """Deterministic serialization used for both writing and hashing, so a
    record's on-disk bytes are exactly what gets hashed -- no separate
    re-serialization step that could drift from what was written."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


class AuditLogger:
    """Wraps one append-only JSONL file. Safe to reopen across process
    runs: it reads the last existing line (if any) to resume the sequence
    number and hash chain rather than resetting them."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_seq, self._prev_line_hash = self._load_state()

    def _load_state(self) -> tuple[int, str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS_HASH
        last_line: bytes | None = None
        with self.path.open("rb") as f:
            for raw in f:
                raw = raw.rstrip(b"\n")
                if raw:
                    last_line = raw
        if last_line is None:
            return 0, GENESIS_HASH
        last_record = json.loads(last_line)
        return last_record["seq"] + 1, hashlib.sha256(last_line).hexdigest()

    def log(
        self,
        *,
        slice_type: str,
        reason: str,
        nonce: bytes,
        signature: bytes,
        public_key: bytes,
        backend: str,
        trusted: bool,
        rejected_as_replay: bool,
        pinned_key_mismatch: bool = False,
        trust_store_key_changed: bool = False,
        challenge_mismatch: bool = False,
        signed_payload: str | None = None,
        expected_challenge: bytes | None = None,
    ) -> dict:
        """Append one record and return it as the dict that was written.

        ``pinned_key_mismatch`` defaults to False so existing callers that
        predate public-key pinning (pqc_auth/transport.py's
        ReauthClient.expected_public_key) keep writing records of the same
        shape without any change on their part. ``trust_store_key_changed``
        follows the identical convention for TOFU pinning
        (pqc_auth/trust_store.py): existing callers/records are unaffected.

        ``signed_payload`` / ``expected_challenge`` belong to wire version 2
        (see pqc_auth/transport.py): the exact payload string the signature
        covers, and the challenge the client sent in its request. When
        given, both are written, together with ``challenge_mismatch``, so
        pqc_auth/audit_verify.py can recompute the signature over the real
        signed bytes and independently re-check the challenge comparison.
        When omitted, none of the three is written and the record keeps the
        pre-v2 shape, in which the signature covers ``nonce`` alone.
        """
        fields = {
            "seq": self._next_seq,
            "timestamp": time.time(),
            "slice_type": slice_type,
            "reason": reason,
            "nonce": nonce.hex(),
            "signature": signature.hex(),
            "public_key": public_key.hex(),
            "backend": backend,
            "trusted": bool(trusted),
            "rejected_as_replay": bool(rejected_as_replay),
            "pinned_key_mismatch": bool(pinned_key_mismatch),
            "trust_store_key_changed": bool(trust_store_key_changed),
            "prev_hash": self._prev_line_hash,
        }
        if signed_payload is not None:
            if expected_challenge is None:
                raise ValueError("signed_payload requires expected_challenge")
            fields["signed_payload"] = signed_payload
            fields["expected_challenge"] = expected_challenge.hex()
            fields["challenge_mismatch"] = bool(challenge_mismatch)
        elif challenge_mismatch:
            raise ValueError("challenge_mismatch requires signed_payload and expected_challenge")
        record_hash = hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()
        record = {**fields, "record_hash": record_hash}
        line = canonical_json(record).encode("utf-8")

        with self.path.open("ab") as f:
            f.write(line + b"\n")

        self._next_seq += 1
        self._prev_line_hash = hashlib.sha256(line).hexdigest()
        return record
