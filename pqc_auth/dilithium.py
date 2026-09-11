"""Optional liboqs-backed Dilithium signing adapter.

No classical fallback is silently used: absent liboqs means the production
backend is unavailable, which prevents accidental PQC overclaims.
"""

from __future__ import annotations


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
    # liboqs renamed Dilithium to ML-DSA per FIPS 204; ML-DSA-65 is Dilithium3's equivalent security category.
    def __init__(self, algorithm: str = "ML-DSA-65"):
        try:
            import oqs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("liboqs-python is required for Dilithium signing") from exc
        self._oqs = oqs
        self.algorithm = algorithm
        self._signer = oqs.Signature(algorithm)
        self.public_key = self._signer.generate_keypair()

    def sign(self, message: bytes) -> bytes:
        return self._signer.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify against THIS instance's own public key. Kept as-is for
        backward compatibility with existing callers/tests — this is now a
        thin wrapper around verify_with_public_key(), not a separate
        implementation, but its signature and behavior are unchanged."""
        return verify_with_public_key(message, signature, self.public_key, self.algorithm)
