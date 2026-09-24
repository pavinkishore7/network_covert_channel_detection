"""Signed key rotation: the currently trusted key endorses its successor.

Under plain TOFU (pqc_auth/trust_store.py) a server that legitimately
rotates its key looks exactly like an impersonator: a new key under a known
``server_id``. This module adds the minimal standard fix. Before the old key
is retired, it signs a *rotation statement* naming the new key. A client
that trusts the old key can check that signature itself and move its pin to
the new key with cryptographic continuity, without a human re-trusting.

Statement fields (all covered by the old key's signature):

  server_id        the identity the statement is for (utf-8)
  old_pubkey_hash  sha256 of the key being retired -- the one the client
                   must currently have pinned
  new_pubkey       the successor key, in full
  epoch            integer, strictly increasing per server (first rotation
                   is 1; a never-rotated key is epoch 0)
  issued_at        integer unix seconds; informational only. Acceptance
                   never depends on clocks -- only on ``epoch``.

Serialization (``encode_statement``): a length-prefixed binary encoding,
NOT JSON. The signed bytes are

    ROTATION_DOMAIN
    || u32be(len(server_id))       || server_id
    || u32be(32)                   || old_pubkey_hash
    || u32be(len(new_pubkey))      || new_pubkey
    || u64be(epoch)
    || u64be(issued_at)

with fields in that fixed order and nothing else. Why this and not JSON:
  - It is injective by construction. Every variable-length field carries its
    length, so no two different statements share an encoding, and the
    decoder rejects trailing bytes, a wrong domain tag and out-of-range
    lengths. There is exactly one way to parse a statement.
  - JSON has several ways to disagree about the same bytes: duplicate keys
    (parsers differ on which one wins), key order, number representation
    (1 vs 1.0 vs 1e0; float round-trips), string escaping and Unicode
    normalization. The wire payloads in transport.py dodge most of this by
    sending the exact signed string, but a statement is stored, forwarded
    and audited long after it is issued, so the encoding itself should
    leave nothing to interpretation.
  - Integers are fixed-width, so there are no floats at all.

Domain separation: ``ROTATION_DOMAIN`` differs from transport.py's
``SIGNATURE_DOMAIN``, so a signature the server made over a re-auth payload
can never be presented as a rotation statement, or the other way round.
pqc_auth/audit_verify.py repeats this literal (it must not import this
module); tests/test_key_rotation.py checks the two stay equal.

Chains: a client that missed several rotations is sent the server's recent
statements, oldest first, and accepts them only as a contiguous, verified
chain starting from its own pin -- each link signed by the key the previous
link endorsed, epochs strictly increasing, ending at the key the server is
now using. That is exactly the sequence of checks the client would have
made had it been online for each rotation, so accepting the chain grants
nothing a step-by-step client would not have granted. The alternative,
rejecting and demanding manual re-trust, would put a human back in the loop
for precisely the clients (long-sleeping mMTC devices) that miss rotations
most. The chain is bounded at ``MAX_CHAIN_LINKS``: that bounds the
verification work and response size, and a client further behind than that
gets the ordinary TOFU key-change rejection and needs manual re-trust.

Explicitly OUT of scope: certificate authorities, revocation, and recovery
from a compromised old key. If the old private key is compromised, an
attacker can issue a valid rotation to their own key; nothing here can tell
that apart from a genuine rotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROTATION_DOMAIN = b"pqc_auth.key_rotation.v1\x00"

# A server retains, and sends, at most this many most-recent statements.
# ML-DSA-65: each link is ~2.1 KB of statement + 3.3 KB of signature,
# ~10.8 KB hex-encoded on the wire; 4 links keep a response well under
# transport.MAX_RESPONSE_BYTES (64 KB) alongside the ~11 KB signed answer.
MAX_CHAIN_LINKS = 4

ROTATIONS_FILENAME = "rotations.json"

_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")
_MAX_FIELD_BYTES = 16 * 1024  # generous for any ML-DSA public key; bounds a hostile length prefix


class RotationFormatError(ValueError):
    """A rotation statement's bytes are not a valid encoding."""


@dataclass(frozen=True)
class RotationStatement:
    server_id: str
    old_pubkey_hash: bytes
    new_pubkey: bytes
    epoch: int
    issued_at: int


@dataclass(frozen=True)
class SignedRotation:
    """A statement's exact encoded bytes plus the old key's signature over
    them. The bytes are kept, not re-derived, so what is verified, stored
    and audited is always exactly what was signed."""

    encoded: bytes
    signature: bytes

    @property
    def statement(self) -> RotationStatement:
        return decode_statement(self.encoded)

    def to_wire(self) -> dict:
        return {"statement": self.encoded.hex(), "signature": self.signature.hex()}

    @classmethod
    def from_wire(cls, item) -> "SignedRotation":
        if not isinstance(item, dict):
            raise RotationFormatError("rotation entry is not an object")
        try:
            return cls(bytes.fromhex(item["statement"]), bytes.fromhex(item["signature"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RotationFormatError(f"rotation entry fields: {exc}") from None


def pubkey_hash(public_key: bytes) -> bytes:
    return hashlib.sha256(public_key).digest()


def encode_statement(statement: RotationStatement) -> bytes:
    server_id = statement.server_id.encode("utf-8")
    for name, value in (("server_id", server_id), ("new_pubkey", statement.new_pubkey)):
        if not 0 < len(value) <= _MAX_FIELD_BYTES:
            raise ValueError(f"{name} length {len(value)} out of range")
    if len(statement.old_pubkey_hash) != 32:
        raise ValueError("old_pubkey_hash must be a 32-byte sha256 digest")
    if not 0 < statement.epoch < 2**64 or not 0 <= statement.issued_at < 2**64:
        raise ValueError("epoch must be in [1, 2**64) and issued_at in [0, 2**64)")
    return b"".join((
        ROTATION_DOMAIN,
        _U32.pack(len(server_id)), server_id,
        _U32.pack(32), statement.old_pubkey_hash,
        _U32.pack(len(statement.new_pubkey)), statement.new_pubkey,
        _U64.pack(statement.epoch),
        _U64.pack(statement.issued_at),
    ))


def decode_statement(data: bytes) -> RotationStatement:
    """Strict inverse of ``encode_statement``: rejects anything that
    encode_statement could not have produced."""
    if not data.startswith(ROTATION_DOMAIN):
        raise RotationFormatError("wrong domain tag")
    offset = len(ROTATION_DOMAIN)

    def take(n: int) -> bytes:
        nonlocal offset
        if n > len(data) - offset:
            raise RotationFormatError("truncated statement")
        chunk = data[offset:offset + n]
        offset += n
        return chunk

    def field() -> bytes:
        (length,) = _U32.unpack(take(4))
        if not 0 < length <= _MAX_FIELD_BYTES:
            raise RotationFormatError(f"field length {length} out of range")
        return take(length)

    try:
        server_id = field().decode("utf-8")
    except UnicodeDecodeError:
        raise RotationFormatError("server_id is not utf-8") from None
    old_hash = field()
    if len(old_hash) != 32:
        raise RotationFormatError("old_pubkey_hash is not 32 bytes")
    new_pubkey = field()
    (epoch,) = _U64.unpack(take(8))
    (issued_at,) = _U64.unpack(take(8))
    if offset != len(data):
        raise RotationFormatError("trailing bytes after statement")
    if epoch == 0:
        raise RotationFormatError("epoch 0 is reserved for a never-rotated key")
    return RotationStatement(server_id, old_hash, new_pubkey, epoch, issued_at)


def issue_rotation(old_signer, new_public_key: bytes, server_id: str, epoch: int,
                   issued_at: int | None = None) -> SignedRotation:
    """Sign a statement endorsing ``new_public_key`` with ``old_signer``.

    ``old_signer`` is any Signer (OqsDilithiumSigner in production). Its
    secret key lives inside the signer object -- liboqs's Signature holds it
    and ``sign()`` takes no key argument -- so the caller hands over the
    old signer itself, never raw secret bytes."""
    statement = RotationStatement(
        server_id=server_id,
        old_pubkey_hash=pubkey_hash(old_signer.public_key),
        new_pubkey=new_public_key,
        epoch=epoch,
        issued_at=int(time.time()) if issued_at is None else issued_at,
    )
    encoded = encode_statement(statement)
    return SignedRotation(encoded, old_signer.sign(encoded))


# -- client-side verification -------------------------------------------------


@dataclass(frozen=True)
class RotationCheck:
    accepted: bool
    reason: str = ""
    new_epoch: int | None = None
    links: tuple[SignedRotation, ...] = ()  # the links actually applied, oldest first


def verify_rotation_chain(
    *,
    pinned_key: bytes,
    pinned_epoch: int,
    presented_key: bytes,
    server_id: str,
    rotations: Iterable[SignedRotation],
    verify_fn: Callable[[bytes, bytes, bytes], bool],
) -> RotationCheck:
    """Decide whether ``rotations`` carry the trust in ``pinned_key`` (at
    ``pinned_epoch``) over to ``presented_key``. Pure: touches no store.

    Links with epoch <= ``pinned_epoch`` are ones this client has already
    applied (or that predate its pin) and are skipped. The remaining links
    must form one contiguous chain: the first names ``pinned_key`` as its
    old key, each later link names the previous link's new key, epochs
    strictly increase, every link's signature verifies under its old key
    with ``verify_fn``, every link is for ``server_id``, and the last link
    endorses ``presented_key``. Any deviation rejects the whole chain.

    Cheap structural checks run before any signature is verified, and the
    first failing check is the reported reason.
    """
    try:
        links = [(link, link.statement) for link in rotations]
    except RotationFormatError as exc:
        return RotationCheck(False, f"malformed rotation statement: {exc}")
    if not links:
        return RotationCheck(False, "no rotation statement")
    fresh = [(link, st) for link, st in links if st.epoch > pinned_epoch]
    if not fresh:
        return RotationCheck(
            False, f"stale epoch: every rotation statement has epoch <= stored epoch {pinned_epoch}"
        )
    if len(fresh) > MAX_CHAIN_LINKS:
        return RotationCheck(False, f"chain of {len(fresh)} links exceeds MAX_CHAIN_LINKS={MAX_CHAIN_LINKS}")

    current_key, current_epoch = pinned_key, pinned_epoch
    for link, st in fresh:
        if st.server_id != server_id:
            return RotationCheck(False, f"statement is for server_id {st.server_id!r}, not {server_id!r}")
        if st.epoch <= current_epoch:
            return RotationCheck(False, f"stale epoch: statement epoch {st.epoch} <= {current_epoch}")
        if st.old_pubkey_hash != pubkey_hash(current_key):
            return RotationCheck(False, f"epoch {st.epoch} statement was not issued by the currently trusted key")
        if not verify_fn(link.encoded, link.signature, current_key):
            return RotationCheck(False, f"epoch {st.epoch} statement signature does not verify under the trusted key")
        current_key, current_epoch = st.new_pubkey, st.epoch
    if current_key != presented_key:
        return RotationCheck(False, "rotation chain does not end at the key the server presented")
    return RotationCheck(True, new_epoch=current_epoch, links=tuple(link for link, _ in fresh))


# -- server-side persistence --------------------------------------------------


def load_rotations(key_path: str | Path) -> list[SignedRotation]:
    """The retained statements in ``key_path`` (oldest first); [] if none."""
    path = Path(key_path) / ROTATIONS_FILENAME
    if not path.exists():
        return []
    return [SignedRotation.from_wire(item) for item in json.loads(path.read_text())]


def next_epoch(rotations: list[SignedRotation]) -> int:
    return rotations[-1].statement.epoch + 1 if rotations else 1


def rotate_persisted_key(key_path: str | Path, server_id: str, algorithm: str = "ML-DSA-65",
                         issued_at: int | None = None) -> SignedRotation:
    """Rotate an ``OqsDilithiumSigner`` key directory in place.

    Loads the current key from ``key_path``, generates a new keypair, has the
    OLD signer sign the statement endorsing the new public key, then
    replaces the key files with the new ones and appends the statement to
    ``rotations.json`` (keeping the last ``MAX_CHAIN_LINKS``).

    The old secret key is kept only as long as needed to sign: it exists in
    the loaded ``oqs.Signature`` for the duration of this call, and its file
    is overwritten by the new secret key. Nothing else copies it.

    Run this with the server stopped, then restart the server: a running
    ReauthServer holds its signer in memory and keeps using the old key.

    Writes use a temp file plus ``os.replace`` per file, so each file is
    either old or new, but the three files are not replaced as one atomic
    unit. A crash between replacements leaves the directory inconsistent
    and needs an operator. Plaintext on disk, as everywhere in pqc_auth.
    """
    import oqs  # type: ignore[import-not-found]

    from pqc_auth.dilithium import OqsDilithiumSigner

    key_dir = Path(key_path)
    if not (key_dir / "secret_key.bin").exists():
        raise FileNotFoundError(f"no key to rotate in {key_dir}")
    old_signer = OqsDilithiumSigner(algorithm=algorithm, key_path=str(key_dir))
    rotations = load_rotations(key_dir)

    with oqs.Signature(algorithm) as new_sig:
        new_public_key = new_sig.generate_keypair()
        new_secret_key = new_sig.export_secret_key()

    signed = issue_rotation(old_signer, new_public_key, server_id, next_epoch(rotations), issued_at)
    del old_signer  # the old secret now exists only in the file about to be replaced
    retained = (rotations + [signed])[-MAX_CHAIN_LINKS:]

    _atomic_write(key_dir / ROTATIONS_FILENAME, json.dumps([r.to_wire() for r in retained], indent=2).encode())
    _atomic_write(key_dir / "public_key.bin", new_public_key)
    _atomic_write(key_dir / "secret_key.bin", new_secret_key)
    return signed


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pqc_auth.key_rotation")
    sub = parser.add_subparsers(dest="command", required=True)
    rotate = sub.add_parser("rotate", help="rotate a persisted OqsDilithiumSigner key_path (server stopped)")
    rotate.add_argument("--key-path", required=True)
    rotate.add_argument("--server-id", required=True, help="must equal the server's --server-id")
    args = parser.parse_args(argv)
    signed = rotate_persisted_key(args.key_path, args.server_id)
    st = signed.statement
    print(json.dumps({"server_id": st.server_id, "epoch": st.epoch, "issued_at": st.issued_at,
                      "old_pubkey_sha256": st.old_pubkey_hash.hex(),
                      "new_pubkey_sha256": pubkey_hash(st.new_pubkey).hex()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
