"""Runnable demo: ``python -m pqc_auth.demo``

Starts a real ``ReauthServer`` and ``ReauthClient`` (an actual TCP socket
connection, not two objects calling each other's methods directly) and
walks through a few re-auth cycles, printing a readable trace:

  1. Periodic re-auth on eMBB (no detector alert).
  2. A simulated detector alert on URLLC firing re-auth early, and an
     immediate second alert correctly suppressed by ``alert_cooldown_seconds``.
  3. A deliberately tampered signature and a deliberately replayed valid
     message, each independently rejected by the CLIENT's own verification
     -- not by the server refusing to answer (the server has no idea either
     of these checks is happening; they are purely client-side).

Auto-detects ``oqs``: uses a real ``OqsDilithiumSigner`` if liboqs is
installed and importable, otherwise falls back to a small HMAC-based
demo-only signer so this demo runs everywhere without liboqs. Prints
clearly which backend is in use.

Note on that fallback: it is intentionally NOT ``tests/fake_signer.py``.
``pqc_auth/`` (production code) does not import anything from ``tests/``
-- see that module's own docstring for why (so there's no import path by
which a real deployment could pick up the test fixture by accident). This
demo instead carries its own small, separately-defined HMAC stand-in,
``_DemoFakeSigner``, duplicating a few lines rather than crossing that
boundary. It is clearly labeled, used only here, and the backend in use is
always printed, so there's no ambiguity about which signer is actually
running.

What this demo does NOT claim: it is not a production deployment, not a
benchmark, and not proof the crypto is used correctly under real adversarial
network conditions beyond the two rejection cases shown here. See
pqc_auth/README.md for the fuller list of what's not done yet.
"""

from __future__ import annotations

import hashlib
import hmac

from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer


class _DemoFakeSigner:
    """HMAC-SHA256 stand-in used ONLY by this demo when oqs isn't
    installed. NOT real cryptography, NOT tests/fake_signer.py, NOT for
    production use -- see module docstring above."""

    _KEY = b"pqc-auth-demo-only-not-a-secret-not-real-crypto"

    def __init__(self):
        self.public_key = self._KEY  # symmetric: the "public key" is the shared secret

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._KEY, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(message), signature)

    @staticmethod
    def verify_with_public_key(message: bytes, signature: bytes, public_key: bytes) -> bool:
        expected = hmac.new(public_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


def _make_signer():
    try:
        import oqs  # noqa: F401
    except ImportError:
        return _DemoFakeSigner(), _DemoFakeSigner.verify_with_public_key, "_DemoFakeSigner (oqs not installed -- HMAC stand-in)"
    from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

    return OqsDilithiumSigner(), verify_with_public_key, "OqsDilithiumSigner (Dilithium3, real liboqs)"


def _print_result(label: str, result) -> None:
    if not result.due:
        print(f"  [{label}] due=False")
        return
    print(
        f"  [{label}] due=True reason={result.reason.value} "
        f"trusted={result.trusted} replay_rejected={result.rejected_as_replay}"
    )


def main() -> None:
    signer, verify_fn, backend_label = _make_signer()
    print(f"pqc_auth demo -- signer backend: {backend_label}\n")

    controller = DualTriggerReauthController(signer=signer)
    server = ReauthServer(controller)
    server.start()
    client = ReauthClient(server.host, server.port, verify_fn=verify_fn)
    print(f"Server listening on {server.host}:{server.port}\n")

    try:
        print("1) Periodic re-auth on eMBB (no detector alert)")
        _print_result("eMBB  t=0", client.request_reauth("eMBB", 0))

        print("\n2) Detector alert on URLLC fires re-auth early; cooldown suppresses the immediate repeat")
        _print_result("URLLC t=0  periodic          ", client.request_reauth("URLLC", 0))
        _print_result("URLLC t=5  detector alert    ", client.request_reauth("URLLC", 5, detector_alert=True))
        _print_result("URLLC t=8  alert (in cooldown)", client.request_reauth("URLLC", 8, detector_alert=True))

        print("\n3) Tampering and replay are both independently rejected by the CLIENT")
        captured = client._send_request("mMTC", 0, detector_alert=False)

        # Check the tampered copy FIRST, before the genuine nonce is ever
        # marked as seen -- otherwise this would be rejected as a replay
        # (same nonce, already seen) rather than on the signature check
        # it's meant to demonstrate.
        tampered = dict(captured)
        tampered_sig = bytearray(bytes.fromhex(tampered["signature"]))
        tampered_sig[0] ^= 0xFF
        tampered["signature"] = tampered_sig.hex()
        _print_result("mMTC  t=0  tampered signature ", client.process_response(tampered, now=0))

        _print_result("mMTC  t=1  genuine response   ", client.process_response(captured, now=1))
        _print_result("mMTC  t=2  replay of t=1 msg  ", client.process_response(captured, now=2))
    finally:
        server.stop()

    print("\nDone.")


if __name__ == "__main__":
    main()
