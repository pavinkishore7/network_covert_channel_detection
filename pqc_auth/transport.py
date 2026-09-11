"""A minimal, real two-party re-auth transport over a local TCP socket.

This is deliberately small: one request/response exchange per call, no
connection pooling, no framing beyond a newline-terminated JSON line. It
exists to prove the challenge-response actually works between two separate
parties over a real socket (not two objects calling each other's methods in
the same process) and that the client's verification is genuinely
independent of the server.

Roles:
  - ReauthServer holds a Signer capable of signing (FakeSigner or
    OqsDilithiumSigner) plus a DualTriggerReauthController. On a request
    for a given slice_type, it calls controller.reauth() and, if due,
    sends back (reason, nonce, signature, public_key).
  - ReauthClient holds ONLY a public key and a PublicKeyVerifier-shaped
    verify function -- never a Signer, never anything with a sign()
    method. It is constructed with that verify function directly by
    whoever is wiring things up (tests, the demo); pqc_auth.transport
    itself never imports FakeSigner or picks a backend by name, which
    keeps this module signer-agnostic and keeps tests/fake_signer.py out
    of any production import path.

Critically: the server's ReauthOutcome.verified field (from pqc_auth.reauth)
is NEVER sent to the client and NEVER used by the client to decide
anything. That field only says "the server's own sign+verify round trip
worked on its end" -- it is not a substitute for the client independently
verifying the signature against the public key with its own verify_fn.

Replay protection: the client tracks nonces it has already accepted and
rejects a repeat, even if the signature is still cryptographically valid
(a captured, still-valid (nonce, signature, public_key) tuple must not work
twice). See ReauthClient's docstring for the expiry-window reasoning.

What this does NOT do (see pqc_auth/README.md for the fuller list):
  - No production-grade connection handling: no retries, no timeouts beyond
    a plain socket timeout, no TLS-equivalent transport security -- the
    signature is the only integrity guarantee, there is no confidentiality
    or anti-tampering on the request itself (a request only carries
    slice_type/now/detector_alert, none of which are secret).
  - No multi-client or multi-server topology; one server, sequential
    connections, localhost only.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass, field
from typing import Callable

from pqc_auth.audit_log import AuditLogger
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason

# How long the client remembers a nonce it has accepted, in the same units
# as the `now` timestamps passed to reauth()/request_reauth(). Chosen to
# comfortably exceed the longest configured slice interval (mMTC = 300s in
# DEFAULT_POLICIES) so a legitimately-recent nonce is never forgotten
# mid-cycle, while still bounding memory for a long-running client rather
# than remembering every nonce forever.
DEFAULT_REPLAY_WINDOW_SECONDS = 300


def _recv_line(sock: socket.socket) -> bytes | None:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if chunk.endswith(b"\n"):
            break
    if not chunks:
        return None
    return b"".join(chunks).rstrip(b"\n")


def _send_line(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(payload + b"\n")


class ReauthServer:
    """Listens on a local TCP socket and answers re-auth requests for a
    controller that has a signing-capable Signer configured."""

    def __init__(self, controller: DualTriggerReauthController, host: str = "127.0.0.1", port: int = 0):
        if controller.signer is None:
            raise ValueError("ReauthServer requires a controller with a signer configured")
        self.controller = controller
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.host = host
        self.port = self._sock.getsockname()[1]
        self._sock.listen(4)
        self._sock.settimeout(0.5)  # let _serve_forever notice _running is False and exit
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def backend_name(self) -> str:
        """Human-readable label for the configured signer, for logs/demo output only."""
        return type(self.controller.signer).__name__

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sock.close()

    def _serve_forever(self) -> None:
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                self._handle_connection(conn)

    def _handle_connection(self, conn: socket.socket) -> None:
        raw = _recv_line(conn)
        if not raw:
            return
        request = json.loads(raw)
        outcome = self.controller.reauth(
            request["slice_type"], request["now"], detector_alert=request.get("detector_alert", False)
        )
        if outcome is None:
            response = {"due": False}
        else:
            signer = self.controller.signer
            response = {
                "due": True,
                "reason": outcome.reason.value,
                "nonce": outcome.nonce.hex(),
                "signature": outcome.signature.hex(),
                "public_key": signer.public_key.hex(),
                # Label only -- the client's trust decision never depends
                # on this string, it's informational (logging/demo output).
                "backend": self.backend_name,
                # Echoed back only for the client's own audit log record --
                # not used in any trust decision.
                "slice_type": request["slice_type"],
            }
        _send_line(conn, json.dumps(response).encode())


@dataclass
class ClientVerificationResult:
    due: bool
    reason: ReauthReason | None = None
    trusted: bool | None = None
    rejected_as_replay: bool = False
    backend: str | None = None


class ReauthClient:
    """Holds ONLY a public key and a verify function -- never a Signer, and
    structurally cannot end up with signing capability, since verify_fn is
    a bare callable (see pqc_auth.reauth.PublicKeyVerifier), not an object
    that could also expose sign().

    Never trusts the server's own ReauthOutcome.verified field -- that
    field isn't even sent over the wire (see ReauthServer._handle_connection).
    Every trust decision here is made by calling verify_fn independently.
    """

    def __init__(
        self,
        host: str,
        port: int,
        verify_fn: Callable[[bytes, bytes, bytes], bool],
        replay_window_seconds: float = DEFAULT_REPLAY_WINDOW_SECONDS,
        timeout: float = 5.0,
        audit_log_path: str | None = None,
    ):
        self.host = host
        self.port = port
        self._verify_fn = verify_fn
        self._replay_window_seconds = replay_window_seconds
        self._timeout = timeout
        self._seen_nonces: dict[bytes, float] = {}
        # None by default so existing tests/callers get no file side
        # effects; set audit_log_path to append a record for every real
        # client-side verification (see pqc_auth/audit_log.py).
        self._audit_logger = AuditLogger(audit_log_path) if audit_log_path is not None else None

    def request_reauth(self, slice_type: str, now: float, detector_alert: bool = False) -> ClientVerificationResult:
        """Send a real request over a real socket, then independently verify the response."""
        response = self._send_request(slice_type, now, detector_alert)
        return self.process_response(response, now)

    def _send_request(self, slice_type: str, now: float, detector_alert: bool) -> dict:
        sock = socket.create_connection((self.host, self.port), timeout=self._timeout)
        with sock:
            _send_line(sock, json.dumps({"slice_type": slice_type, "now": now, "detector_alert": detector_alert}).encode())
            raw = _recv_line(sock)
        if raw is None:
            raise ConnectionError("no response from ReauthServer")
        return json.loads(raw)

    def process_response(self, response: dict, now: float) -> ClientVerificationResult:
        """Verify a (possibly captured/replayed) server response independently.

        Exposed separately from request_reauth() so a captured response
        dict can be re-submitted to test replay rejection without needing
        the server to somehow resend an identical nonce over the wire a
        second time.
        """
        if not response.get("due"):
            return ClientVerificationResult(due=False)

        reason = ReauthReason(response["reason"])
        nonce = bytes.fromhex(response["nonce"])
        signature = bytes.fromhex(response["signature"])
        public_key = bytes.fromhex(response["public_key"])
        backend = response.get("backend")

        self._prune_expired(now)
        if nonce in self._seen_nonces:
            self._log_verification(response, reason, nonce, signature, public_key, backend, trusted=False, rejected_as_replay=True)
            return ClientVerificationResult(due=True, reason=reason, trusted=False, rejected_as_replay=True, backend=backend)

        trusted = self._verify_fn(nonce, signature, public_key)
        if trusted:
            self._seen_nonces[nonce] = now
        self._log_verification(response, reason, nonce, signature, public_key, backend, trusted=trusted, rejected_as_replay=False)
        return ClientVerificationResult(due=True, reason=reason, trusted=trusted, rejected_as_replay=False, backend=backend)

    def _log_verification(
        self,
        response: dict,
        reason: ReauthReason,
        nonce: bytes,
        signature: bytes,
        public_key: bytes,
        backend: str | None,
        *,
        trusted: bool,
        rejected_as_replay: bool,
    ) -> None:
        if self._audit_logger is None:
            return
        self._audit_logger.log(
            slice_type=response.get("slice_type", ""),
            reason=reason.value,
            nonce=nonce,
            signature=signature,
            public_key=public_key,
            backend=backend or "",
            trusted=trusted,
            rejected_as_replay=rejected_as_replay,
        )

    def _prune_expired(self, now: float) -> None:
        cutoff = now - self._replay_window_seconds
        expired = [nonce for nonce, seen_at in self._seen_nonces.items() if seen_at < cutoff]
        for nonce in expired:
            del self._seen_nonces[nonce]
