"""Tests for pqc_auth/dilithium.py.

Requires liboqs (see docs/PQC_SETUP.md) — a real C library build, not a
plain pip install. Gated with pytest.importorskip so this file is
automatically SKIPPED, not failed, in environments without it. Do not treat
a skip here as a pass: check the pytest summary line for "skipped" vs.
"passed" before reporting these as verified.
"""

import pytest

oqs = pytest.importorskip("oqs")

from pqc_auth.dilithium import OqsDilithiumSigner  # noqa: E402


class TestKeypairGeneration:
    def test_public_key_has_expected_type_and_length(self):
        signer = OqsDilithiumSigner()
        expected_length = oqs.Signature(signer.algorithm).details["length_public_key"]
        assert isinstance(signer.public_key, bytes)
        assert len(signer.public_key) == expected_length


class TestSignVerifyRoundTrip:
    def test_signed_message_verifies_against_correct_public_key(self):
        signer = OqsDilithiumSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature = signer.sign(message)
        assert signer.verify(message, signature) is True

    def test_tampered_message_fails_verification(self):
        signer = OqsDilithiumSigner()
        message = bytearray(b"slice re-auth challenge: URLLC t=0")
        signature = signer.sign(bytes(message))
        message[0] ^= 0xFF  # flip a single byte
        assert signer.verify(bytes(message), signature) is False

    def test_tampered_signature_fails_verification(self):
        signer = OqsDilithiumSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature = bytearray(signer.sign(message))
        signature[0] ^= 0xFF  # flip a single byte
        assert signer.verify(message, bytes(signature)) is False

    def test_different_signers_public_key_fails_verification(self):
        signer_a = OqsDilithiumSigner()
        signer_b = OqsDilithiumSigner()
        message = b"slice re-auth challenge: URLLC t=0"
        signature_from_a = signer_a.sign(message)
        # signer_b.verify() checks against signer_b's own public key, not
        # signer_a's -- a signature made under a's private key must not
        # validate against a different keypair.
        assert signer_b.verify(message, signature_from_a) is False
