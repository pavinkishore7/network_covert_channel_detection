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
  4. A public-key pinning rejection: a SECOND, entirely different signer
     (its own genuine keypair) produces a response claiming to come from
     the pinned identity, with a signature that is genuinely valid under
     ITS OWN public key. The client rejects it purely because that public
     key isn't the pinned one -- distinct from steps 3's tamper/replay
     cases, which reject responses that DO carry the pinned key.

The client in every scenario above is constructed with
``expected_public_key`` pinned to the real signer's own public key,
simulating a key obtained once through some trusted out-of-band channel
(see pqc_auth/README.md for exactly what that does and does not solve).

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

Leaves state behind under ``pqc_auth/.demo_state/`` (gitignored,
PLAINTEXT, demo-only -- see pqc_auth/README.md): an ``oqs_key/`` directory
so the real signer has a stable identity across runs instead of a fresh
keypair every time, and an append-only ``audit_log.jsonl`` that every run
adds to (both backends log to the same file, distinguished by each
record's ``backend`` field, so it's clear which entries are HMAC-backed --
NOT post-quantum -- versus real ML-DSA-65 ones if this demo is ever run
both with and without oqs on the same machine). After the trace, this demo
runs pqc_auth.audit_verify's own independent check against that log and
prints its PASS/FAIL summary as the last thing it prints.

What this demo does NOT claim: it is not a production deployment, not a
benchmark, and not proof the crypto is used correctly under real adversarial
network conditions beyond the two rejection cases shown here. See
pqc_auth/README.md for the fuller list of what's not done yet.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from pqc_auth.audit_verify import format_report, verify_log
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer
from pqc_auth.trust_store import TrustStore

DEMO_STATE_DIR = Path(__file__).parent / ".demo_state"
DEMO_KEY_DIR = DEMO_STATE_DIR / "oqs_key"
DEMO_AUDIT_LOG_PATH = DEMO_STATE_DIR / "audit_log.jsonl"
# Unlike DEMO_KEY_DIR/DEMO_AUDIT_LOG_PATH, this is reset at the start of
# every demo run (see _demo_tofu_pinning) rather than persisted -- the
# point of this section is to show FIRST CONTACT happening, which would
# otherwise only be observable the very first time this demo is ever run.
DEMO_TOFU_TRUST_STORE_PATH = DEMO_STATE_DIR / "tofu_trust_store.json"
DEMO_TOFU_SERVER_ID = "pqc_auth-demo-server"


class _DemoFakeSigner:
    """HMAC-SHA256 stand-in used ONLY by this demo when oqs isn't
    installed. NOT real cryptography, NOT tests/fake_signer.py, NOT for
    production use -- see module docstring above."""

    _KEY = b"pqc-auth-demo-only-not-a-secret-not-real-crypto"

    def __init__(self, key: bytes | None = None):
        # key is overridable ONLY so this demo can construct a second,
        # genuinely different signer identity for the pinning-rejection
        # scenario below -- every other use of this class relies on the
        # fixed default key so its "public key" (= shared secret) is
        # stable across a run.
        self._key = self._KEY if key is None else key
        self.public_key = self._key  # symmetric: the "public key" is the shared secret

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()

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
        label = "_DemoFakeSigner (oqs not installed -- HMAC stand-in, NOT post-quantum)"
        return _DemoFakeSigner(), _DemoFakeSigner.verify_with_public_key, label
    from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

    signer = OqsDilithiumSigner(key_path=str(DEMO_KEY_DIR))
    return signer, verify_with_public_key, "OqsDilithiumSigner (ML-DSA-65, real liboqs, persisted key_path)"


def _make_impersonator_signer(backend_label: str):
    """A second signer with its own genuine, DIFFERENT keypair, for the
    pinning-rejection demonstration -- matching whichever backend
    _make_signer() picked, so the demo compares like with like (both
    real oqs signers, or both HMAC stand-ins)."""
    if backend_label.startswith("OqsDilithiumSigner"):
        from pqc_auth.dilithium import OqsDilithiumSigner

        return OqsDilithiumSigner()  # no key_path -- fresh, ephemeral keypair, deliberately unrelated to DEMO_KEY_DIR
    return _DemoFakeSigner(key=b"a-second-demo-identity-different-key-entirely")


def _print_result(label: str, result) -> None:
    if not result.due:
        print(f"  [{label}] due=False")
        return
    print(
        f"  [{label}] due=True reason={result.reason.value} "
        f"trusted={result.trusted} replay_rejected={result.rejected_as_replay} "
        f"pinned_key_mismatch={result.pinned_key_mismatch} "
        f"trust_store_key_changed={result.trust_store_key_changed}"
    )


def _demo_tofu_pinning(server: ReauthServer, signer, verify_fn, backend_label: str) -> None:
    """Trust-on-first-use (TOFU) pinning -- a DIFFERENT mechanism from the
    expected_public_key pinning demonstrated in step 4 above. That client
    was told the server's key in advance (caller-asserted). This one has
    NEVER been told the key: it learns and pins whatever key it sees on
    the first response for a given server_id, then holds it fixed after
    that. See pqc_auth/trust_store.py and pqc_auth/README.md for what this
    does and does NOT solve -- most importantly, an attacker already
    present on this very first connection would be indistinguishable from
    a legitimate one; nothing here proves the first key seen is correct.
    """
    print("\n5) Trust-on-first-use (TOFU) pinning -- distinct from step 4's expected_public_key:")
    print("   this client has never been told the server's key in advance. It learns whichever")
    print("   key it sees on first contact, persists it, and pins to it from then on.")

    if DEMO_TOFU_TRUST_STORE_PATH.exists():
        DEMO_TOFU_TRUST_STORE_PATH.unlink()  # reset every run so first contact is always observable

    tofu_client = ReauthClient(
        server.host, server.port, verify_fn=verify_fn,
        trust_store_path=str(DEMO_TOFU_TRUST_STORE_PATH), server_id=DEMO_TOFU_SERVER_ID,
        audit_log_path=str(DEMO_AUDIT_LOG_PATH),
    )
    # detector_alert=True bypasses mMTC's 300s periodic interval (mMTC was
    # already touched earlier in this same demo run, at t=0, by step 3's
    # _send_request -- relying on periodic timing here could show
    # due=False depending on what ran before this function). The two calls
    # are 100s apart, comfortably clearing the 10s alert_cooldown_seconds
    # between them.
    _print_result("TOFU  t=100 first contact              ", tofu_client.request_reauth("mMTC", 100, detector_alert=True))
    _print_result("TOFU  t=200 same server again           ", tofu_client.request_reauth("mMTC", 200, detector_alert=True))

    print("\n   Simulating the server's identity actually changing (a second, genuinely different signer,")
    print("   answering under the SAME server_id):")
    impersonator = _make_impersonator_signer(backend_label)
    changed_controller = DualTriggerReauthController(signer=impersonator)
    changed_server = ReauthServer(changed_controller)
    changed_server.start()
    try:
        client_against_changed_server = ReauthClient(
            changed_server.host, changed_server.port, verify_fn=verify_fn,
            trust_store_path=str(DEMO_TOFU_TRUST_STORE_PATH), server_id=DEMO_TOFU_SERVER_ID,
            audit_log_path=str(DEMO_AUDIT_LOG_PATH),
        )
        _print_result(
            "TOFU  t=0   changed identity (rejected) ", client_against_changed_server.request_reauth("mMTC", 0)
        )

        print("\n   An operator explicitly accepts the rotation -- force_retrust() is never called automatically:")
        TrustStore(DEMO_TOFU_TRUST_STORE_PATH).force_retrust(DEMO_TOFU_SERVER_ID, impersonator.public_key)
        client_after_retrust = ReauthClient(
            changed_server.host, changed_server.port, verify_fn=verify_fn,
            trust_store_path=str(DEMO_TOFU_TRUST_STORE_PATH), server_id=DEMO_TOFU_SERVER_ID,
            audit_log_path=str(DEMO_AUDIT_LOG_PATH),
        )
        _print_result(
            "TOFU  t=1   after explicit re-trust     ",
            client_after_retrust.request_reauth("mMTC", 1, detector_alert=True),
        )
    finally:
        changed_server.stop()


def main() -> None:
    signer, verify_fn, backend_label = _make_signer()
    print(f"pqc_auth demo -- signer backend: {backend_label}\n")

    controller = DualTriggerReauthController(signer=signer)
    server = ReauthServer(controller)
    server.start()
    client = ReauthClient(
        server.host,
        server.port,
        verify_fn=verify_fn,
        # Pinned to the real signer's own public key -- simulating a key
        # obtained once through a trusted out-of-band channel (see
        # pqc_auth/README.md for exactly what this does and does not
        # solve). Every scenario below, including the pre-existing ones,
        # now runs against a pinned client.
        expected_public_key=signer.public_key,
        audit_log_path=str(DEMO_AUDIT_LOG_PATH),
    )
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

        print("\n4) A SECOND, different signer's genuinely valid signature is rejected by PINNING")
        impersonator = _make_impersonator_signer(backend_label)
        # detector_alert=True so this is due regardless of eMBB's periodic
        # schedule (already used at t=0 in step 1, and not due again for
        # 90s) -- the alert trigger has its own, much shorter cooldown.
        genuine = client._send_request("eMBB", 20, detector_alert=True)
        nonce = bytes.fromhex(genuine["nonce"])
        impersonated = dict(genuine)
        impersonated["signature"] = impersonator.sign(nonce).hex()
        impersonated["public_key"] = impersonator.public_key.hex()
        # Prove first that this is NOT just another invalid-signature case:
        # the impersonator's own signature over the SAME nonce genuinely
        # verifies under the impersonator's OWN public key.
        impersonator_signature_is_genuinely_valid = impersonator.verify(nonce, bytes.fromhex(impersonated["signature"]))
        print(
            f"     (impersonator's signature over this nonce verifies under "
            f"ITS OWN key: {impersonator_signature_is_genuinely_valid} -- "
            f"this is a real signature, not a broken one)"
        )
        _print_result("eMBB  t=20 impersonator's key ", client.process_response(impersonated, now=20))

        _demo_tofu_pinning(server, signer, verify_fn, backend_label)
    finally:
        server.stop()

    print("\nDone.")

    print(f"\nIndependently auditing {DEMO_AUDIT_LOG_PATH} ...\n")
    print(format_report(verify_log(DEMO_AUDIT_LOG_PATH)))


if __name__ == "__main__":
    main()
