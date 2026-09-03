"""Optional liboqs-backed Dilithium signing adapter.

No classical fallback is silently used: absent liboqs means the production
backend is unavailable, which prevents accidental PQC overclaims.
"""

from __future__ import annotations


class OqsDilithiumSigner:
    def __init__(self, algorithm: str = "Dilithium3"):
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
        with self._oqs.Signature(self.algorithm) as verifier:
            return bool(verifier.verify(message, signature, self.public_key))
