"""Deterministic fake signer for tests ONLY.

NOT real post-quantum cryptography, NOT a production substitute. This is a
plain HMAC-SHA256 keyed with a fixed test key — trivially forgeable by
anyone who reads this file. It exists solely so pqc_auth.reauth tests don't
require liboqs to be built (see docs/PQC_SETUP.md).

Matches the spirit of pqc_auth/dilithium.py's "no classical fallback is
silently used" principle: this class lives in tests/, not pqc_auth/, so
there is no import path by which production code could pick it up by
accident. Do not move it into pqc_auth/ and do not import it from anything
outside tests/.
"""

from __future__ import annotations

import hashlib
import hmac


class FakeSigner:
    """Deterministic HMAC-SHA256 signer. Test-only — see module docstring.

    Named ``FakeSigner`` rather than ``TestFakeSigner`` deliberately: a
    ``Test*`` prefix makes pytest try to collect this as a test class
    (it has an ``__init__``, which pytest then warns about). Living in
    ``tests/fake_signer.py`` and this docstring are what mark it test-only.
    """

    _DEFAULT_TEST_KEY = b"pqc-auth-tests-only-not-a-secret-not-real-crypto"

    def __init__(self, key: bytes | None = None):
        self._key = self._DEFAULT_TEST_KEY if key is None else key

    @property
    def public_key(self) -> bytes:
        """NOT a real public key -- HMAC is symmetric, so this is the same
        secret key used to sign. Named ``public_key`` only so this class
        mirrors OqsDilithiumSigner's ``.public_key`` attribute shape for
        code (like pqc_auth.transport) that treats signer backends
        uniformly. Handing this out is exactly as unsafe as it sounds for
        real crypto -- acceptable here only because FakeSigner never signs
        or verifies anything outside a test process."""
        return self._key

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        expected = self.sign(message)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def verify_with_public_key(message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Test-fixture simplification, not a claim that HMAC is
        asymmetric: mirrors pqc_auth.dilithium.verify_with_public_key's
        call shape, (message, signature, public_key) -> bool, so a
        verifying party can be written once and tested against either
        backend. Here ``public_key`` is really the shared HMAC key -- there
        is no public/private key separation for a symmetric MAC. This
        staticmethod takes no signer instance, matching the
        verify-with-only-a-public-key contract real callers rely on.
        """
        expected = hmac.new(public_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)
