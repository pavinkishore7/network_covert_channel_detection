"""Optional liboqs-backed Dilithium signing adapter.

No classical fallback is silently used: absent liboqs means the production
backend is unavailable, which prevents accidental PQC overclaims.
"""

from __future__ import annotations

from pathlib import Path


# liboqs renamed Dilithium to ML-DSA per FIPS 204; ML-DSA-65 is Dilithium3's equivalent security category.
def verify_with_public_key(message: bytes, signature: bytes, public_key: bytes, algorithm: str = "ML-DSA-65") -> bool:
    """Verify a signature against an EXPLICIT public key, independent of any
    signer instance's own keypair.

    This is what a real, separate verifying party needs (e.g. the client
    role in pqc_auth.transport): it never has to hold or construct an
    OqsDilithiumSigner capable of signing, only the public key bytes it
    received. Requires liboqs, same as OqsDilithiumSigner.
    """
    try:
        import oqs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("liboqs-python is required for Dilithium verification") from exc
    with oqs.Signature(algorithm) as verifier:
        return bool(verifier.verify(message, signature, public_key))


class OqsDilithiumSigner:
    """liboqs-backed signer, optionally with a persisted keypair.

    ``key_path``, if given, is a DIRECTORY (created if missing) holding two
    files: ``secret_key.bin`` and ``public_key.bin``. Both are required to
    reconstruct signing capability -- liboqs-python's ``oqs.Signature``
    constructor can reload a Signature able to sign again from an exported
    secret key (``secret_key=`` kwarg), but it has no API to re-derive the
    matching public key from a secret key alone, so the public key must be
    saved alongside it.

    Key storage here is PLAINTEXT ON DISK. This is acceptable only for this
    project's demo/review purposes -- there is no encryption at rest, no
    access control beyond whatever OS file permissions the process happens
    to have, and no key rotation. Do not treat this as a production
    key-management approach.
    """

    # liboqs renamed Dilithium to ML-DSA per FIPS 204; ML-DSA-65 is Dilithium3's equivalent security category.
    def __init__(self, algorithm: str = "ML-DSA-65", key_path: str | None = None):
        try:
            import oqs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("liboqs-python is required for Dilithium signing") from exc
        self._oqs = oqs
        self.algorithm = algorithm

        if key_path is None:
            self._signer = oqs.Signature(algorithm)
            self.public_key = self._signer.generate_keypair()
            return

        key_dir = Path(key_path)
        secret_key_file = key_dir / "secret_key.bin"
        public_key_file = key_dir / "public_key.bin"
        if secret_key_file.exists() and public_key_file.exists():
            secret_key = secret_key_file.read_bytes()
            self.public_key = public_key_file.read_bytes()
            self._signer = oqs.Signature(algorithm, secret_key)
        else:
            key_dir.mkdir(parents=True, exist_ok=True)
            self._signer = oqs.Signature(algorithm)
            self.public_key = self._signer.generate_keypair()
            secret_key_file.write_bytes(self._signer.export_secret_key())
            public_key_file.write_bytes(self.public_key)

    def sign(self, message: bytes) -> bytes:
        return self._signer.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify against THIS instance's own public key. Kept as-is for
        backward compatibility with existing callers/tests — this is now a
        thin wrapper around verify_with_public_key(), not a separate
        implementation, but its signature and behavior are unchanged."""
        return verify_with_public_key(message, signature, self.public_key, self.algorithm)
