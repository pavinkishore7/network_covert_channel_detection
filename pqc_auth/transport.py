"""A minimal, real two-party re-auth transport over TCP.

One request/response exchange per connection, framed as one newline-
terminated JSON line each way. It exists to prove the challenge-response
actually works between two separate parties over a real socket (not two
objects calling each other's methods in the same process) and that the
client's verification is genuinely independent of the server.

Roles:
  - ReauthServer holds a Signer capable of signing (FakeSigner or
    OqsDilithiumSigner) plus a DualTriggerReauthController. On a request
    for a given slice_type, it calls controller.reauth() and, if due,
    sends back a signed payload bound to the client's challenge.
  - ReauthClient holds ONLY a public key and a PublicKeyVerifier-shaped
    verify function -- never a Signer, never anything with a sign()
    method. It is constructed with that verify function directly by
    whoever is wiring things up (tests, the demo); pqc_auth.transport
    itself never imports FakeSigner or picks a backend by name, which
    keeps this module signer-agnostic and keeps tests/fake_signer.py out
    of any production import path.

Wire format (version 2 -- version 1 is gone, see below):

  request   {"v": 2, "slice_type": str, "now": number, "detector_alert": bool,
             "client_challenge": hex}           # >= 16 random bytes, fresh per request
  response  {"v": 2, "due": false}                                   (not due)
            {"v": 2, "due": true, "payload": str, "signature": hex,
             "public_key": hex, "backend": str}                      (due)
            {"v": 2, "error": code, "detail": str}                   (request refused)
  A signed response may also carry ``"rotations": [{"statement": hex,
  "signature": hex}, ...]`` -- the server's recent key-rotation statements,
  oldest first (see pqc_auth/key_rotation.py). They sit outside ``payload``
  on purpose: each statement is authenticated by the OLD key's signature,
  which is the whole point; having the new key sign them as well would
  prove nothing about continuity.

  ``payload`` is the canonical JSON (``canonical_payload()``: sorted keys,
  no whitespace, ASCII-only, no NaN/Infinity) of
      {"v": 2, "server_id", "slice_type", "reason", "client_challenge",
       "server_nonce", "request_now", "issued_at"}
  and the signature covers ``SIGNATURE_DOMAIN + payload.encode("ascii")``
  (``signed_message()``). The payload travels as the exact string that was
  signed, so the client verifies those bytes rather than re-serializing
  parsed values (no float/key-order round-trip to get wrong).

Why version 2 (the root cause it fixes): in version 1 the request carried
no client randomness and the server signed only its own nonce
(``controller._challenge()``). Nothing tied a response to the request it
answered, so the client's only replay defence was an in-memory set of
nonces it had already accepted -- replay DETECTION, not challenge-
response. A client process that restarted (empty set) accepted one
recorded response as genuine; integration/README.md shows that happening
on the real namespace topology. Now the client sends a fresh random
challenge per request and rejects any response whose signed payload does
not carry that exact challenge (``challenge_mismatch``), before the
response can be accepted. The seen-nonce set is kept as defence in depth
only. Version-1 requests are refused by the server (``unsupported_version``)
and version-1 responses are rejected by the client (``malformed_response``);
there is no compatibility path.

Critically: the server's ReauthOutcome.verified field (from pqc_auth.reauth)
is NEVER sent to the client and NEVER used by the client to decide
anything. That field only says "the server's own sign+verify round trip
worked on its end" -- it is not a substitute for the client independently
verifying the signature against the public key with its own verify_fn.

Public-key pinning: if ReauthClient is constructed with expected_public_key,
a response carrying any other public_key is rejected BEFORE verify_fn is
even called -- see ReauthClient's docstring and process_response() for why
that ordering matters (a signature can be genuinely valid under the wrong
key, so checking crypto validity first would ask the wrong question).

Trust-on-first-use (TOFU) pinning: if ReauthClient is constructed with
trust_store_path + server_id instead of expected_public_key, it learns and
persists whichever key it sees on the FIRST response for that server_id
(see pqc_auth/trust_store.py) -- but only once the challenge matches and
verify_fn has confirmed that first response's signature is genuinely valid
under that key; a first-contact response that fails either check is never
persisted, and leaves the trust store empty for this server_id so a later,
genuine first response can still be accepted normally. Once a key is
learned, the client pins to it on every later response, with the same
before-verify_fn ordering and the same "don't mark the nonce as seen"
behavior on a mismatch as explicit pinning. See pqc_auth/trust_store.py's
module docstring for what TOFU does and does NOT solve -- it is not a
substitute for real key distribution, and an attacker present on the very
first connection to a never-before-seen server_id, WITH a validly signed
response, is indistinguishable from a legitimate first contact.

Signed key rotation: when a TOFU client sees a key other than its pinned
one, it no longer rejects outright if the response carries a rotation
chain that verifies from the pinned key (and its stored epoch) to the
presented key -- see pqc_auth/key_rotation.py for the rules. The new key is
persisted only after the challenge and verify_fn checks pass under it,
exactly like first contact. With no chain, or one that fails any check, the
rejection is the unchanged REJECTED_TOFU_KEY_CHANGED. An explicit
expected_public_key is never overridden by a rotation statement.

Server robustness: each accepted connection is handled on its own thread,
up to ``max_connections`` at once; beyond that a new connection is closed
immediately and logged. Every connection has a read timeout, so an idle
or slow peer is dropped rather than holding a slot forever. Requests are
size-capped before parsing, and any malformed request (oversized,
truncated, non-UTF-8, invalid JSON, wrong types, unknown slice type,
unsupported version) gets an error response or a closed connection and a
log line -- it never reaches the serve loop.

What this does NOT do (see pqc_auth/README.md for the fuller list):
  - No retries, and no TLS-equivalent transport security -- the signature
    is the only integrity guarantee. Requests are not authenticated, and a
    ``{"due": false}`` answer is not signed, so an on-path attacker can
    still suppress a re-auth (make it look "not due") or drop traffic; it
    can no longer make a client accept a stale response.
  - Host/port default to loopback; binding elsewhere (e.g. inside a
    network namespace, see the CLI below and integration/auth_over_topology.py)
    is the caller's choice.

Command-line entry point (so a server or client can run as its own process,
e.g. inside a network namespace via ``ip netns exec <ns> ...``)::

    python -m pqc_auth.transport serve --bind ADDR --port P --key-path DIR [--audit-log FILE]
    python -m pqc_auth.transport request --server ADDR --port P --slice-type S \\
        (--expected-pubkey-file FILE | --trust-store FILE --server-id ID) [...]

Both use the real ``OqsDilithiumSigner``/``verify_with_public_key`` and
therefore need liboqs. See ``main()`` for every flag.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import random
import secrets
import signal
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from pqc_auth.audit_log import AuditLogger
from pqc_auth.key_rotation import (
    MAX_CHAIN_LINKS, RotationCheck, RotationFormatError, SignedRotation, verify_rotation_chain,
)
from pqc_auth.outcomes import OutcomeCategory, RequestOutcome
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason
from pqc_auth.trust_store import TrustStore, TrustStoreError

WIRE_VERSION = 2

# Prefixed to every signed payload so a signature made for this protocol
# can't be confused with a signature over the same bytes in any other
# context. pqc_auth/audit_verify.py repeats this literal on purpose (it
# must not import this module); tests/test_transport_binding.py checks the
# two stay equal.
SIGNATURE_DOMAIN = b"pqc_auth.reauth.v2\x00"

CLIENT_CHALLENGE_BYTES = 32
MIN_CLIENT_CHALLENGE_BYTES = 16
MAX_CLIENT_CHALLENGE_BYTES = 64

# A well-formed request is ~200 bytes; anything past this is refused
# before any decoding or JSON parsing happens.
MAX_REQUEST_BYTES = 1024
# ML-DSA-65: 3309-byte signature + 1952-byte public key, hex-encoded, plus
# the payload -- about 11 KB. Generous headroom, still bounded.
MAX_RESPONSE_BYTES = 64 * 1024

DEFAULT_SERVER_ID = "reauth-server"
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_CONNECTIONS_PER_PEER = 4  # see ReauthServer's docstring
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5.0

# Client connection handling -- see ReauthClient's docstring for why.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 3.0
DEFAULT_READ_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.25
DEFAULT_BACKOFF_MAX_SECONDS = 2.0

# How long the client remembers a server nonce it has accepted, in the same
# units as the `now` timestamps passed to reauth()/request_reauth(). No
# longer the primary replay defence (the per-request client challenge is --
# see the module docstring); kept as defence in depth, sized to comfortably
# exceed the longest configured slice interval (mMTC = 300s in
# DEFAULT_POLICIES) while bounding memory for a long-running client.
DEFAULT_REPLAY_WINDOW_SECONDS = 300


def canonical_payload(fields: dict) -> str:
    """The one serialization used for signed payloads."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def signed_message(payload: str) -> bytes:
    """The exact bytes a response signature covers."""
    return SIGNATURE_DOMAIN + payload.encode("ascii")


class TruncatedMessage(ConnectionError):
    """The peer closed the connection partway through a line."""


class MessageTooLarge(ConnectionError):
    """A line exceeded the receiver's size cap before its newline arrived."""


def _recv_line(sock: socket.socket, max_bytes: int, deadline: float | None = None) -> bytes | None:
    """One newline-terminated line, without the newline. None if the peer
    closed without sending anything. Raises MessageTooLarge once more than
    ``max_bytes`` arrive without a newline, TruncatedMessage if the peer
    closes mid-line, and lets socket.timeout / OSError propagate.
    ``deadline`` (time.monotonic()), if given, bounds the whole line rather
    than each recv() call."""
    buf = bytearray()
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("read deadline exceeded")
            sock.settimeout(remaining)
        chunk = sock.recv(4096)
        if not chunk:
            if buf:
                raise TruncatedMessage(f"peer closed after {len(buf)} bytes without a newline")
            return None
        buf += chunk
        newline = buf.find(b"\n")
        if newline != -1:
            if newline > max_bytes:
                raise MessageTooLarge(f"line of {newline} bytes exceeds {max_bytes}")
            return bytes(buf[:newline])
        if len(buf) > max_bytes:
            raise MessageTooLarge(f"more than {max_bytes} bytes without a newline")


def _send_line(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(payload + b"\n")


class _RequestError(Exception):
    def __init__(self, code: str, detail: str = "", challenge: bytes | None = None):
        super().__init__(code)
        self.code = code
        self.detail = detail
        # The client challenge, if one could be extracted before the request
        # was found invalid -- lets the refusal be signed and bound to it.
        self.challenge = challenge


def _extract_challenge(request: dict) -> bytes | None:
    challenge_hex = request.get("client_challenge")
    try:
        challenge = bytes.fromhex(challenge_hex) if isinstance(challenge_hex, str) else None
    except ValueError:
        return None
    if challenge is None or not (MIN_CLIENT_CHALLENGE_BYTES <= len(challenge) <= MAX_CLIENT_CHALLENGE_BYTES):
        return None
    return challenge


def _parse_request(raw: bytes, known_slices) -> dict:
    """Validate one request line completely before anything acts on it."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _RequestError("invalid_utf8", str(exc)) from None
    try:
        request = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _RequestError("invalid_json", str(exc)) from None
    if not isinstance(request, dict):
        raise _RequestError("invalid_request", "request must be a JSON object")
    challenge = _extract_challenge(request)
    if request.get("v") != WIRE_VERSION:
        raise _RequestError("unsupported_version", f"expected v={WIRE_VERSION}, got {request.get('v')!r}", challenge)
    if challenge is None:
        raise _RequestError(
            "invalid_client_challenge",
            f"need {MIN_CLIENT_CHALLENGE_BYTES}-{MAX_CLIENT_CHALLENGE_BYTES} bytes as hex",
        )
    slice_type = request.get("slice_type")
    if not isinstance(slice_type, str) or slice_type not in known_slices:
        raise _RequestError("unknown_slice_type", repr(slice_type)[:64], challenge)
    now = request.get("now")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise _RequestError("invalid_now", repr(now)[:64], challenge)
    detector_alert = request.get("detector_alert", False)
    if not isinstance(detector_alert, bool):
        raise _RequestError("invalid_detector_alert", repr(detector_alert)[:64], challenge)
    return {"slice_type": slice_type, "now": float(now), "detector_alert": detector_alert, "client_challenge": challenge}


def _status_payload(
    *,
    status: str,
    server_id: str,
    client_challenge: bytes,
    server_nonce: bytes,
    slice_type: str | None = None,
    reason: str | None = None,
    error_code: str | None = None,
    request_now: float | None = None,
) -> str:
    """Every signed payload has the same keys, whatever its status."""
    return canonical_payload({
        "v": WIRE_VERSION,
        "status": status,
        "server_id": server_id,
        "slice_type": slice_type,
        "reason": reason,
        "error_code": error_code,
        "client_challenge": client_challenge.hex(),
        "server_nonce": server_nonce.hex(),
        "request_now": request_now,
        "issued_at": time.time(),
    })


class ReauthServer:
    """Listens on a TCP socket and answers re-auth requests for a
    controller that has a signing-capable Signer configured.

    Every answer to a request that carries a valid client challenge is
    signed and bound to that challenge -- "re-auth performed" (status
    ``due``), "not due yet" (``not_due``) AND refusals (``error``). An
    unsigned "not due" would let an on-path attacker answer every PERIODIC
    request with a forged "not due" and silently switch periodic
    re-authentication off; see pqc_auth/README.md. Only requests from which
    no valid challenge can even be extracted (garbage, oversized, truncated)
    get an unsigned error: no legitimate client sends those, so a signature
    would protect nobody and would only let junk traffic make the server
    sign.

    Concurrency model: one thread per accepted connection, capped by a
    semaphore at ``max_connections``. Chosen over selectors/asyncio because
    everything a connection does is blocking and short -- one recv, one
    controller.reauth() whose signer is a blocking C call (liboqs), one
    send -- so an event loop would still need worker threads for signing,
    and the thread-per-connection version keeps the existing synchronous
    code. Every connection has a read timeout (``connection_timeout``).

    Two caps, checked in this order when a connection is accepted:
      - ``max_connections_per_peer`` (default 4) concurrent connections per
        source IP. A legitimate client has exactly one request in flight
        (retries are sequential), so 4 leaves room for a few client
        processes sharing one address while guaranteeing that one address
        can hold at most 4 of the 32 global slots -- exhausting the server
        takes at least 8 distinct addresses. This is per IP on TCP: it stops
        a single peer, including an off-path one (TCP needs a completed
        handshake, so the source address can't simply be spoofed), but NOT
        an attacker who controls many addresses.
      - ``max_connections`` (default 32) in total.
    A connection over either cap is closed at once and logged
    (``rejected_per_peer_limit`` / ``rejected_at_capacity``), never queued.

    ``controller`` may be shared with other callers; its scheduling state is
    protected by the controller's own lock (see DualTriggerReauthController),
    so two simultaneous requests for one slice cannot both fire.
    """

    def __init__(
        self,
        controller: DualTriggerReauthController,
        host: str = "127.0.0.1",
        port: int = 0,
        served_log_path: str | None = None,
        server_id: str = DEFAULT_SERVER_ID,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_connections_per_peer: int = DEFAULT_MAX_CONNECTIONS_PER_PEER,
        rotations: list[SignedRotation] | None = None,
    ):
        if controller.signer is None:
            raise ValueError("ReauthServer requires a controller with a signer configured")
        if max_connections < 1 or max_connections_per_peer < 1:
            raise ValueError("connection limits must be at least 1")
        self.controller = controller
        self.server_id = server_id
        # Rotation statements attached to every signed response, oldest
        # first (pqc_auth/key_rotation.py). Only the last MAX_CHAIN_LINKS
        # are sent; a client further behind needs manual re-trust.
        self._rotations_wire = [r.to_wire() for r in (rotations or [])[-MAX_CHAIN_LINKS:]]
        self._max_request_bytes = max_request_bytes
        self._connection_timeout = connection_timeout
        self._slots = threading.BoundedSemaphore(max_connections)
        self._max_per_peer = max_connections_per_peer
        self._active_by_peer: Counter[str] = Counter()
        self._peer_lock = threading.Lock()
        # Optional plain-JSONL record of every connection this server
        # handled: served requests (with measured sign/verify time), refused
        # requests (with the error code), timeouts, and connections dropped
        # at a cap. NOT hash-chained and NOT an input to
        # pqc_auth/audit_verify.py -- that tool audits the CLIENT's
        # verification log, which is where trust decisions are made. None =
        # no file side effects.
        self._served_log_path = Path(served_log_path) if served_log_path is not None else None
        self._log_lock = threading.Lock()
        # In-memory tally of the same events, by name (e.g. "served",
        # "rejected_at_capacity", "error:invalid_json"), for tests/operators.
        self.stats: Counter[str] = Counter()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.host = host
        self.port = self._sock.getsockname()[1]
        self._sock.listen(max(16, max_connections))
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
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    break  # stop() closed the socket: normal shutdown
                raise  # e.g. EMFILE: genuinely unexpected; let the thread die visibly
            peer_ip = addr[0]
            with self._peer_lock:
                over_peer_limit = self._active_by_peer[peer_ip] >= self._max_per_peer
                if not over_peer_limit:
                    self._active_by_peer[peer_ip] += 1
            if over_peer_limit:
                self._record(addr, "rejected_per_peer_limit")
                conn.close()
                continue
            if not self._slots.acquire(blocking=False):
                self._release_peer(peer_ip)
                self._record(addr, "rejected_at_capacity")
                conn.close()
                continue
            threading.Thread(target=self._connection_worker, args=(conn, addr), daemon=True).start()

    def _release_peer(self, peer_ip: str) -> None:
        with self._peer_lock:
            self._active_by_peer[peer_ip] -= 1
            if self._active_by_peer[peer_ip] <= 0:
                del self._active_by_peer[peer_ip]

    def _connection_worker(self, conn: socket.socket, addr) -> None:
        try:
            with conn:
                conn.settimeout(self._connection_timeout)
                self._handle_connection(conn, addr)
        except Exception as exc:  # noqa: BLE001 - one connection must never take the server down
            self._record(addr, "error", code="internal_error", detail=f"{type(exc).__name__}: {exc}"[:200])
        finally:
            self._slots.release()
            self._release_peer(addr[0])

    def _handle_connection(self, conn: socket.socket, addr) -> None:
        try:
            raw = _recv_line(conn, self._max_request_bytes)
        except socket.timeout:
            self._record(addr, "error", code="read_timeout")
            return
        except MessageTooLarge as exc:
            self._refuse(conn, addr, "request_too_large", str(exc))
            return
        except TruncatedMessage as exc:
            self._record(addr, "error", code="truncated_request", detail=str(exc))
            return
        except OSError as exc:
            self._record(addr, "error", code="connection_error", detail=f"{type(exc).__name__}: {exc}"[:200])
            return
        if raw is None:
            self._record(addr, "error", code="empty_connection")
            return
        try:
            request = _parse_request(raw, self.controller.policies)
        except _RequestError as exc:
            self._refuse(conn, addr, exc.code, exc.detail, challenge=exc.challenge)
            return

        payload_holder: list[str] = []

        def bind_payload(reason: ReauthReason, nonce: bytes) -> bytes:
            payload = _status_payload(
                status="due", server_id=self.server_id, client_challenge=request["client_challenge"],
                server_nonce=nonce, slice_type=request["slice_type"], reason=reason.value,
                request_now=request["now"],
            )
            payload_holder.append(payload)
            return signed_message(payload)

        outcome = self.controller.reauth(
            request["slice_type"], request["now"], detector_alert=request["detector_alert"], sign_message=bind_payload
        )
        if outcome is not None:
            payload, signature = payload_holder[0], outcome.signature
            sign_ms, verify_ms, nonce = outcome.sign_ms, outcome.verify_ms, outcome.nonce
        else:
            nonce = secrets.token_bytes(16)
            payload = _status_payload(
                status="not_due", server_id=self.server_id, client_challenge=request["client_challenge"],
                server_nonce=nonce, slice_type=request["slice_type"], request_now=request["now"],
            )
            signature, sign_ms = self._sign(payload)
            verify_ms = None
        _send_line(conn, json.dumps(self._signed_response(payload, signature)).encode())
        self._record(
            addr, "served",
            slice_type=request["slice_type"], now=request["now"], detector_alert=request["detector_alert"],
            due=outcome is not None,
            reason=outcome.reason.value if outcome is not None else None,
            server_nonce=nonce.hex(),
            client_challenge=request["client_challenge"].hex(),
            sign_ms=sign_ms,
            verify_ms=verify_ms,
        )

    def _sign(self, payload: str) -> tuple[bytes, float]:
        started = time.perf_counter()
        signature = self.controller.signer.sign(signed_message(payload))
        return signature, (time.perf_counter() - started) * 1000.0

    def _signed_response(self, payload: str, signature: bytes) -> dict:
        response = {
            "v": WIRE_VERSION,
            "payload": payload,
            "signature": signature.hex(),
            "public_key": self.controller.signer.public_key.hex(),
            # Label only -- the client's trust decision never depends on
            # this string, it's informational (logging/demo output).
            "backend": self.backend_name,
        }
        if self._rotations_wire:
            response["rotations"] = self._rotations_wire
        return response

    def _refuse(self, conn: socket.socket, addr, code: str, detail: str = "", challenge: bytes | None = None) -> None:
        if challenge is not None:
            payload = _status_payload(
                status="error", server_id=self.server_id, client_challenge=challenge,
                server_nonce=secrets.token_bytes(16), error_code=code,
            )
            signature, sign_ms = self._sign(payload)
            response = self._signed_response(payload, signature)
        else:
            response, sign_ms = {"v": WIRE_VERSION, "error": code, "detail": detail[:200]}, None
        self._record(addr, "error", code=code, detail=detail[:200], signed=challenge is not None, sign_ms=sign_ms)
        try:
            _send_line(conn, json.dumps(response).encode())
        except OSError:
            pass  # the peer may already be gone; the refusal is logged either way

    def _record(self, addr, event: str, **fields) -> None:
        key = f"error:{fields['code']}" if event == "error" else event
        with self._log_lock:
            self.stats[key] += 1
            if self._served_log_path is None:
                return
            try:
                peer = "%s:%d" % tuple(addr[:2])
            except (TypeError, ValueError):
                peer = None
            record = {"timestamp": time.time(), "peer": peer, "event": event, **fields}
            with self._served_log_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class AttemptRecord:
    """One network attempt made by ``request_reauth()``."""

    attempt: int
    outcome: RequestOutcome
    elapsed_ms: float
    challenge: str  # hex; fresh for every attempt
    detail: str = ""
    audit_seq: int | None = None


@dataclass
class ClientVerificationResult:
    """How one request ended. ``outcome`` is authoritative; the boolean
    flags are kept for existing callers and mirror it.

    ``due`` is what a parseable response CLAIMED (status "due"), whether or
    not it was then accepted -- check ``trusted``/``outcome`` before acting
    on it. ``trusted`` is True when the server's answer was authenticated (a
    verified ``due``, ``not_due`` or refusal), False when a response was
    rejected, and None when no response arrived at all (transport failure).
    """

    due: bool
    reason: ReauthReason | None = None
    trusted: bool | None = None
    rejected_as_replay: bool = False
    pinned_key_mismatch: bool = False
    trust_store_key_changed: bool = False
    challenge_mismatch: bool = False
    malformed_response: bool = False
    backend: str | None = None
    outcome: RequestOutcome | None = None
    status: str | None = None  # what the response claimed: "due" / "not_due" / "error"
    error_code: str | None = None  # for an authenticated refusal
    detail: str = ""
    audit_seq: int | None = None  # this attempt's audit-log record
    attempts: list[AttemptRecord] = field(default_factory=list)
    key_rotation_accepted: bool = False  # the TOFU pin moved along a verified rotation chain
    trusted_epoch: int | None = None  # the pin's rotation epoch after this response (TOFU only)

    @property
    def category(self) -> OutcomeCategory | None:
        return self.outcome.category if self.outcome is not None else None


@dataclass(frozen=True)
class _ParsedResponse:
    payload: str
    fields: dict
    status: str
    reason: ReauthReason | None
    error_code: str | None
    client_challenge: bytes
    server_nonce: bytes
    signature: bytes
    public_key: bytes
    backend: str | None
    rotations: object = None  # raw "rotations" list; only parsed on a TOFU key change


def _parse_response(response) -> _ParsedResponse | None:
    """Structural parse of a signed response. None if anything is off."""
    try:
        payload = response["payload"]
        fields = json.loads(payload)
        if not isinstance(fields, dict) or fields.get("v") != WIRE_VERSION:
            return None
        status = fields["status"]
        if status not in ("due", "not_due", "error"):
            return None
        reason = ReauthReason(fields["reason"]) if status == "due" else None
        return _ParsedResponse(
            payload=payload,
            fields=fields,
            status=status,
            reason=reason,
            error_code=fields.get("error_code"),
            client_challenge=bytes.fromhex(fields["client_challenge"]),
            server_nonce=bytes.fromhex(fields["server_nonce"]),
            signature=bytes.fromhex(response["signature"]),
            public_key=bytes.fromhex(response["public_key"]),
            backend=response.get("backend"),
            rotations=response.get("rotations"),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class _MalformedLine(Exception):
    """A complete response line arrived but is not a JSON object."""


class ReauthClient:
    """Holds ONLY a public key and a verify function -- never a Signer, and
    structurally cannot end up with signing capability, since verify_fn is
    a bare callable (see pqc_auth.reauth.PublicKeyVerifier), not an object
    that could also expose sign().

    Never trusts the server's own ReauthOutcome.verified field -- that
    field isn't even sent over the wire (see ReauthServer._handle_connection).
    Every trust decision here is made by calling verify_fn independently.

    Every request carries a fresh random challenge (CLIENT_CHALLENGE_BYTES
    from ``secrets``), and a response -- whatever its status -- is only
    acceptable if its signed payload carries that exact challenge. See
    process_response().

    Connection handling (request_reauth()):
      - ``connect_timeout`` (default 3.0 s) bounds the TCP handshake. On the
        namespace topology a connect normally completes in tens of ms; a
        lost SYN is retransmitted after Linux's 1 s initial RTO, and 3 s
        lets that one retransmission complete before giving up (the next
        SYN would only go out at ~3 s).
      - ``read_timeout`` (default 5.0 s) bounds the whole wait for the
        response line, as one deadline across recv() calls (so a peer
        dripping a byte at a time can't stretch it). Measured server-side
        signing is under 2 ms and the slowest RTTs seen on the topology were
        ~1 s TCP-retransmit tails; 5 s is the same value the transport used
        before and matches ReauthServer's own idle timeout.
      - ``timeout``, if given (the older single knob), sets both.
      - Up to ``max_attempts`` (default 3) attempts, but ONLY after a
        TRANSPORT_FAILURE (timeout / connection failed). Exponential
        backoff with full jitter between attempts: sleep a uniform random
        time in [0, min(backoff_max, backoff_base * 2**(n-1))], so many
        clients hitting the same outage don't retry in lockstep. Every
        attempt uses a NEW connection and a FRESH challenge -- never a
        reused one: the challenge is what binds a response to one specific
        request. If a retry reused it, a response to the earlier attempt
        (e.g. one delayed or captured on the path while that attempt was
        timing out) would be acceptable as the answer to the retry; a fresh
        challenge makes every attempt's answer unique to that attempt.
      - An AUTH_FAILURE or REFUSED outcome is NEVER retried. A forged or
        replayed response is a security event, not a flaky network:
        retrying would hide it behind a later success and give an attacker
        as many tries as the retry budget.
      - Every attempt, not just the last, is written to the audit log with
        its outcome and attempt number.

    ``expected_public_key``, if set, pins the client to one specific
    identity obtained through some trusted out-of-band channel (e.g.
    read once from the server's own persisted key_path at deployment
    time). It closes a gap that verify_fn alone cannot: a signature can be
    perfectly valid and still be signed by the WRONG keypair -- verify_fn
    only ever answers "is this a genuine signature under THIS public_key",
    never "is this public_key the one I actually trust". See
    process_response() for how a mismatch is handled, and pqc_auth/README.md
    for what pinning does and does not solve.

    ``trust_store_path`` + ``server_id``, given together instead of
    ``expected_public_key``, switch to trust-on-first-use (TOFU): the
    client learns and persists whichever key it sees on the first response
    for that ``server_id`` that actually verifies (see
    pqc_auth/trust_store.py) -- an unverified first response is never
    persisted -- then pins to the learned key thereafter.

    Precedence when both are given: ``expected_public_key`` wins outright
    and the trust store is not consulted at all. Reasoning: an explicit
    ``expected_public_key`` is a stronger, caller-asserted guarantee --
    someone already obtained that exact key through a channel they trust
    -- whereas TOFU-learned trust is, by construction, only ever as good
    as whatever showed up first over the wire. A caller who supplies both
    has already done the stronger thing; falling back to the weaker
    mechanism underneath it would silently discard that guarantee. The same
    holds for signed rotation: a rotation statement only ever moves a
    TOFU-learned pin, never an explicit ``expected_public_key`` (the trust
    store, where rotation lives, isn't even constructed in that case).
    """

    def __init__(
        self,
        host: str,
        port: int,
        verify_fn: Callable[[bytes, bytes, bytes], bool],
        expected_public_key: bytes | None = None,
        trust_store_path: str | None = None,
        server_id: str | None = None,
        replay_window_seconds: float = DEFAULT_REPLAY_WINDOW_SECONDS,
        timeout: float | None = None,
        audit_log_path: str | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max: float = DEFAULT_BACKOFF_MAX_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        if (trust_store_path is None) != (server_id is None):
            raise ValueError("trust_store_path and server_id must be given together, or not at all")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.host = host
        self.port = port
        self._verify_fn = verify_fn
        self._expected_public_key = expected_public_key
        self._server_id = server_id
        # Only constructed (and only ever consulted) when expected_public_key
        # is NOT set -- see the precedence note in this class's docstring.
        self._trust_store = (
            TrustStore(trust_store_path) if trust_store_path is not None and expected_public_key is None else None
        )
        self._replay_window_seconds = replay_window_seconds
        self.connect_timeout = timeout if timeout is not None else connect_timeout
        self.read_timeout = timeout if timeout is not None else read_timeout
        self.max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._seen_nonces: dict[bytes, float] = {}
        self.last_response: dict | None = None
        # None by default so existing tests/callers get no file side
        # effects; set audit_log_path to append a record for every attempt
        # (see pqc_auth/audit_log.py).
        self.audit_logger = AuditLogger(audit_log_path) if audit_log_path is not None else None

    # -- request path -------------------------------------------------------

    def request_reauth(self, slice_type: str, now: float, detector_alert: bool = False) -> ClientVerificationResult:
        """Send a real request with a fresh challenge over a real socket,
        then independently verify the response against that challenge.
        Retries transport failures only (see the class docstring). Never
        raises for a network problem: every way a request can end is a
        RequestOutcome on the returned result."""
        request_id = secrets.token_hex(8)
        attempts: list[AttemptRecord] = []
        result: ClientVerificationResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                self._sleep(self.backoff_delay(attempt - 1))
            challenge = secrets.token_bytes(CLIENT_CHALLENGE_BYTES)
            started = time.perf_counter()
            try:
                response, _ = self._send_request(slice_type, now, detector_alert, client_challenge=challenge)
            except TimeoutError as exc:
                result = self._transport_failure(RequestOutcome.TIMEOUT, exc, slice_type, now, attempt, request_id)
            except _MalformedLine as exc:
                result = self._unauthenticated(RequestOutcome.MALFORMED_RESPONSE, str(exc), slice_type, now, attempt, request_id)
            except MessageTooLarge as exc:
                result = self._unauthenticated(RequestOutcome.MALFORMED_RESPONSE, str(exc), slice_type, now, attempt, request_id)
            except OSError as exc:  # refused, reset, unreachable, peer closed (ConnectionError/TruncatedMessage)
                result = self._transport_failure(RequestOutcome.CONNECTION_FAILED, exc, slice_type, now, attempt, request_id)
            else:
                self.last_response = response  # raw, for callers that capture it (the CLI's --capture-response)
                result = self.process_response(
                    response, now, expected_challenge=challenge,
                    slice_type=slice_type, attempt=attempt, request_id=request_id,
                )
            attempts.append(AttemptRecord(
                attempt=attempt, outcome=result.outcome, elapsed_ms=(time.perf_counter() - started) * 1000.0,
                challenge=challenge.hex(), detail=result.detail, audit_seq=result.audit_seq,
            ))
            if result.outcome.category is not OutcomeCategory.TRANSPORT_FAILURE:
                break
        result.attempts = attempts
        return result

    def backoff_delay(self, retry_number: int) -> float:
        """Full-jitter exponential backoff before retry ``retry_number`` (1-based)."""
        ceiling = min(self._backoff_max, self._backoff_base * (2 ** (retry_number - 1)))
        return self._rng.uniform(0.0, ceiling)

    def _send_request(
        self, slice_type: str, now: float, detector_alert: bool, client_challenge: bytes | None = None
    ) -> tuple[dict, bytes]:
        """One attempt: new connection, one request, one response line.
        Returns (response, the challenge this request carried); a new
        challenge is generated unless a test passes one explicitly. Raises
        OSError subclasses (TimeoutError, ConnectionError, ...) for
        transport problems and _MalformedLine for a non-JSON-object line."""
        challenge = client_challenge if client_challenge is not None else secrets.token_bytes(CLIENT_CHALLENGE_BYTES)
        request = {
            "v": WIRE_VERSION, "slice_type": slice_type, "now": now,
            "detector_alert": detector_alert, "client_challenge": challenge.hex(),
        }
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        with sock:
            sock.settimeout(self.read_timeout)
            _send_line(sock, json.dumps(request).encode())
            raw = _recv_line(sock, MAX_RESPONSE_BYTES, deadline=time.monotonic() + self.read_timeout)
        if raw is None:
            raise ConnectionError("peer closed the connection without responding")
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _MalformedLine(f"response is not JSON: {exc}") from None
        if not isinstance(response, dict):
            raise _MalformedLine("response is not a JSON object")
        return response, challenge

    # -- verification ---------------------------------------------------------

    def process_response(
        self,
        response: dict,
        now: float,
        *,
        expected_challenge: bytes,
        slice_type: str | None = None,
        attempt: int = 1,
        request_id: str | None = None,
    ) -> ClientVerificationResult:
        """Verify a (possibly captured/replayed) server response independently.

        ``expected_challenge`` is the challenge THIS client sent in the
        request this response claims to answer. ``slice_type`` is the slice
        that was requested (recorded in the audit log; defaults to what the
        response claims). Exposed separately from request_reauth() so tests
        can hand in a captured response.

        Check order, each one returning immediately on failure:
          1. structure/version       -> MALFORMED_RESPONSE, or
                                        REJECTED_UNSIGNED_STATUS for a
                                        "not due"/error answer with no
                                        signed payload
          2. explicit pin            -> REJECTED_PINNED_KEY
          3. TOFU stored key         -> REJECTED_TOFU_KEY_CHANGED, unless
                                        the response carries a rotation
                                        chain that verifies from the pinned
                                        key to the presented one
          4. client challenge        -> REJECTED_CHALLENGE_MISMATCH
          5. seen server nonce       -> REJECTED_REPLAY (defence in depth)
          6. verify_fn over signed_message(payload) -> REJECTED_SIGNATURE
          then by the payload's status: VERIFIED / NOT_DUE / SERVER_REFUSED.
        The same checks apply whatever the response's status: a "not due"
        or a refusal must be exactly as authentic as "re-auth performed",
        or an attacker could forge them to suppress re-authentication.
        Identity (2, 3) comes first: "not the key I trust" is the more
        fundamental rejection. The challenge (4) is checked before verify_fn
        for the same reason the key checks are: a replayed response carries
        a perfectly valid signature, so crypto validity is the wrong question
        -- the question is "is this an answer to MY request". None of 2-6
        mark the nonce as seen or persist a TOFU key, and an accepted
        rotation chain is only persisted once 4-6 have passed under the new
        key.
        """
        if not isinstance(response, dict) or response.get("v") != WIRE_VERSION:
            return self._unauthenticated(RequestOutcome.MALFORMED_RESPONSE, "not a wire-v2 response",
                                         slice_type, now, attempt, request_id)
        if "payload" not in response:
            if response.get("due") is False or "error" in response:
                claimed = "not_due" if response.get("due") is False else "error"
                return self._unauthenticated(
                    RequestOutcome.REJECTED_UNSIGNED_STATUS, f"unsigned {claimed} status", slice_type, now,
                    attempt, request_id, status=claimed,
                )
            return self._unauthenticated(RequestOutcome.MALFORMED_RESPONSE, "no signed payload",
                                         slice_type, now, attempt, request_id)

        parsed = _parse_response(response)
        if parsed is None:
            return self._unauthenticated(RequestOutcome.MALFORMED_RESPONSE, "unparseable signed payload",
                                         slice_type, now, attempt, request_id)
        slice_type = slice_type if slice_type is not None else str(parsed.fields.get("slice_type") or "")
        context = (parsed, expected_challenge, slice_type, now, attempt, request_id)

        if self._expected_public_key is not None and parsed.public_key != self._expected_public_key:
            # Pinning rejection -- deliberately independent of verify_fn: a
            # signature under the wrong key may well be genuine (an
            # impersonator's own keypair, a rotated server).
            return self._reject(RequestOutcome.REJECTED_PINNED_KEY, *context, pinned_key_mismatch=True)

        first_contact = False
        rotation = None  # a verified RotationCheck, persisted only if everything below passes
        stored_key = stored_epoch = None
        if self._trust_store is not None:
            stored_key = self._trust_store.get_trusted_key(self._server_id)
            stored_epoch = self._trust_store.get_epoch(self._server_id)
            if stored_key is None:
                # First contact: the identity check passes trivially this
                # once, but the challenge and verify_fn below still must.
                # Persisting is deferred until both have -- persisting here
                # would let a forged or replayed first packet poison the
                # store permanently (force_retrust() is never automatic).
                first_contact = True
            elif stored_key != parsed.public_key:
                rotation = self._check_rotation(parsed, stored_key, stored_epoch)
                if not rotation.accepted:
                    # Unchanged TOFU key-change rejection; the reason the
                    # chain failed (or "no rotation statement") is recorded.
                    return self._reject(RequestOutcome.REJECTED_TOFU_KEY_CHANGED, *context,
                                        trust_store_key_changed=True, rotation_rejected_reason=rotation.reason)

        if not hmac.compare_digest(parsed.client_challenge, expected_challenge):
            # Not an answer to this request: a recorded response from an
            # earlier exchange (possibly genuinely signed by the trusted
            # key), or one signed over someone else's challenge. The
            # primary replay defence; it needs no memory, so it holds across
            # client restarts.
            return self._reject(RequestOutcome.REJECTED_CHALLENGE_MISMATCH, *context, challenge_mismatch=True)

        self._prune_expired(now)
        if parsed.server_nonce in self._seen_nonces:
            # Defence in depth only: with a fresh challenge per request, a
            # response reaching this point with an already-seen nonce means
            # the server reused a nonce for a new challenge.
            return self._reject(RequestOutcome.REJECTED_REPLAY, *context, rejected_as_replay=True)

        if not self._verify_fn(signed_message(parsed.payload), parsed.signature, parsed.public_key):
            return self._reject(RequestOutcome.REJECTED_SIGNATURE, *context)

        trusted_epoch = stored_epoch
        if first_contact:
            # Only now -- challenge matched AND signature genuinely valid
            # under this key -- is it safe to learn it.
            trusted_epoch = self._first_contact_epoch(parsed)
            self._trust_store.trust_first_contact(self._server_id, parsed.public_key, epoch=trusted_epoch)
        elif rotation is not None:
            # Same deferral for a rotation: the chain verified from the old
            # pin, and now this response has proven it answers MY challenge
            # under the new key. accept_rotation() re-checks the pin and
            # epoch against the file (compare-and-set).
            try:
                self._trust_store.accept_rotation(self._server_id, stored_key, parsed.public_key, rotation.new_epoch)
            except TrustStoreError as exc:
                return self._reject(RequestOutcome.REJECTED_TOFU_KEY_CHANGED, *context,
                                    trust_store_key_changed=True, rotation_rejected_reason=str(exc))
            trusted_epoch = rotation.new_epoch
        self._seen_nonces[parsed.server_nonce] = now
        outcome = {
            "due": RequestOutcome.VERIFIED, "not_due": RequestOutcome.NOT_DUE, "error": RequestOutcome.SERVER_REFUSED,
        }[parsed.status]
        rotation_fields = {}
        if rotation is not None:
            rotation_fields = dict(
                key_rotation_accepted=True, rotation_statements=[link.to_wire() for link in rotation.links],
                previous_public_key=stored_key, previous_epoch=stored_epoch, new_epoch=rotation.new_epoch,
            )
        seq = self._log_verification(outcome, *context, trusted=True, **rotation_fields)
        return ClientVerificationResult(
            due=parsed.status == "due", reason=parsed.reason, trusted=True, backend=parsed.backend,
            outcome=outcome, status=parsed.status, error_code=parsed.error_code, audit_seq=seq,
            key_rotation_accepted=rotation is not None, trusted_epoch=trusted_epoch,
        )

    def _rotations_of(self, parsed: _ParsedResponse) -> list[SignedRotation]:
        if parsed.rotations is None:
            return []
        if not isinstance(parsed.rotations, list):
            raise RotationFormatError("rotations is not a list")
        return [SignedRotation.from_wire(item) for item in parsed.rotations]

    def _check_rotation(self, parsed: _ParsedResponse, stored_key: bytes, stored_epoch: int):
        try:
            rotations = self._rotations_of(parsed)
        except RotationFormatError as exc:
            return RotationCheck(False, f"malformed rotations field: {exc}")
        return verify_rotation_chain(
            pinned_key=stored_key, pinned_epoch=stored_epoch, presented_key=parsed.public_key,
            server_id=str(parsed.fields.get("server_id")), rotations=rotations, verify_fn=self._verify_fn,
        )

    def _first_contact_epoch(self, parsed: _ParsedResponse) -> int:
        """The epoch to store with a first-contact key: the highest epoch
        among attached statements endorsing that key, else 0. Exactly as
        unverified as the first-contact key itself (TOFU); it matters only
        so that a client first meeting a server that once rotated BACK to an
        earlier key (e.g. restored from backup) doesn't start at epoch 0 and
        accept a replay of an older statement from that same key."""
        try:
            statements = [r.statement for r in self._rotations_of(parsed)]
        except RotationFormatError:
            return 0
        return max((st.epoch for st in statements if st.new_pubkey == parsed.public_key), default=0)

    def _reject(self, outcome, parsed, expected_challenge, slice_type, now, attempt, request_id,
                rotation_rejected_reason: str | None = None, **flag):
        seq = self._log_verification(outcome, parsed, expected_challenge, slice_type, now, attempt, request_id,
                                     trusted=False, rotation_rejected_reason=rotation_rejected_reason, **flag)
        return ClientVerificationResult(
            due=parsed.status == "due", reason=parsed.reason, trusted=False, backend=parsed.backend, outcome=outcome,
            status=parsed.status, error_code=parsed.error_code, audit_seq=seq, detail=rotation_rejected_reason or "",
            **flag,
        )

    def _unauthenticated(self, outcome, detail, slice_type, now, attempt, request_id, status=None):
        seq = None
        if self.audit_logger is not None:
            seq = self.audit_logger.log_event(
                "unauthenticated_response", slice_type=slice_type or "", outcome=outcome.value, now=now,
                attempt=attempt, request_id=request_id, trusted=False, status=status, detail=detail[:200],
            )["seq"]
        return ClientVerificationResult(
            due=False, trusted=False, outcome=outcome, status=status, detail=detail,
            malformed_response=outcome is RequestOutcome.MALFORMED_RESPONSE, audit_seq=seq,
        )

    def _transport_failure(self, outcome, exc, slice_type, now, attempt, request_id):
        detail = f"{type(exc).__name__}: {exc}"[:200]
        seq = None
        if self.audit_logger is not None:
            seq = self.audit_logger.log_event(
                "transport_failure", slice_type=slice_type, outcome=outcome.value, now=now,
                attempt=attempt, request_id=request_id, detail=detail,
            )["seq"]
        return ClientVerificationResult(due=False, trusted=None, outcome=outcome, detail=detail, audit_seq=seq)

    def _log_verification(
        self,
        outcome: RequestOutcome,
        parsed: _ParsedResponse,
        expected_challenge: bytes,
        slice_type: str,
        now: float,
        attempt: int,
        request_id: str | None,
        *,
        trusted: bool,
        rejected_as_replay: bool = False,
        pinned_key_mismatch: bool = False,
        trust_store_key_changed: bool = False,
        challenge_mismatch: bool = False,
        **rotation_fields,
    ) -> int | None:
        if self.audit_logger is None:
            return None
        record = self.audit_logger.log(
            slice_type=slice_type,
            reason=parsed.reason.value if parsed.reason is not None else "",
            nonce=parsed.server_nonce,
            signature=parsed.signature,
            public_key=parsed.public_key,
            backend=parsed.backend or "",
            trusted=trusted,
            rejected_as_replay=rejected_as_replay,
            pinned_key_mismatch=pinned_key_mismatch,
            trust_store_key_changed=trust_store_key_changed,
            challenge_mismatch=challenge_mismatch,
            signed_payload=parsed.payload,
            expected_challenge=expected_challenge,
            outcome=outcome.value,
            status=parsed.status,
            now=now,
            attempt=attempt,
            request_id=request_id,
            **rotation_fields,
        )
        return record["seq"]

    def _prune_expired(self, now: float) -> None:
        cutoff = now - self._replay_window_seconds
        expired = [nonce for nonce, seen_at in self._seen_nonces.items() if seen_at < cutoff]
        for nonce in expired:
            del self._seen_nonces[nonce]


# -- command-line entry point ----------------------------------------------


def _serve_main(args: argparse.Namespace) -> int:
    from pqc_auth.dilithium import OqsDilithiumSigner

    from pqc_auth.key_rotation import load_rotations

    signer = OqsDilithiumSigner(key_path=args.key_path)
    server = ReauthServer(
        DualTriggerReauthController(signer=signer), host=args.bind, port=args.port, served_log_path=args.audit_log,
        server_id=args.server_id, max_connections=args.max_connections, connection_timeout=args.connection_timeout,
        max_connections_per_peer=args.max_connections_per_peer, rotations=load_rotations(args.key_path),
    )

    stop = threading.Event()

    def _on_signal(signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    server.start()
    # One machine-readable line so whoever launched this process (e.g.
    # integration/auth_over_topology.py) knows the socket is listening
    # before it points any client at it.
    ready = {"host": server.host, "port": server.port, "backend": server.backend_name, "key_path": args.key_path,
             "server_id": server.server_id}
    print("READY " + json.dumps(ready), flush=True)
    try:
        while not stop.wait(0.5):
            if server._thread is not None and not server._thread.is_alive():
                # The accept loop died. Per-connection work runs on worker
                # threads behind a catch-all (see _connection_worker), so
                # malformed or hostile requests cannot reach this. What
                # can: accept() itself failing while the server is meant
                # to be running (e.g. EMFILE/ENFILE, out of descriptors)
                # or a bug in the accept loop. Exit visibly rather than
                # keep a listening socket that will never answer.
                print("SERVER_THREAD_DIED", flush=True)
                return 1
    finally:
        server.stop()
    return 0


def _parse_target(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {value!r}")
    return host, int(port)


def _request_main(args: argparse.Namespace) -> int:
    import oqs  # noqa: F401  # type: ignore[import-not-found]

    from pqc_auth.dilithium import verify_with_public_key
    from pqc_auth.failure_policy import DEFAULT_SLICE_POLICIES, ReauthSupervisor, SliceFailurePolicy

    # verify_with_public_key imports oqs lazily on its first call; importing
    # it here first keeps that one-time module load (~0.9 s measured on the
    # dev machine) out of the first request's rtt_ms.
    expected_public_key = (
        Path(args.expected_pubkey_file).read_bytes() if args.expected_pubkey_file is not None else None
    )
    client = ReauthClient(
        args.server,
        args.port,
        verify_fn=verify_with_public_key,
        expected_public_key=expected_public_key,
        trust_store_path=args.trust_store,
        server_id=args.server_id,
        timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        max_attempts=args.max_attempts,
        audit_log_path=args.audit_log,
    )
    policies = dict(DEFAULT_SLICE_POLICIES)
    if args.fail_closed:
        policies[args.slice_type] = SliceFailurePolicy(fail_closed=True)
    supervisor = ReauthSupervisor(client, policies=policies)

    # Every request is made by this ONE client (one seen-nonce set) and ONE
    # supervisor (one policy history), and every request -- --then
    # follow-ups included -- carries its own fresh challenge, so a --then to
    # an endpoint replaying an earlier response is judged against the
    # challenge of THAT request.
    targets = [(args.server, args.port)] * args.count + list(args.then)
    exit_code = 0
    for index, (host, port) in enumerate(targets):
        if index > 0 and args.interval > 0:
            time.sleep(args.interval)
        client.host, client.port = host, port
        now = args.now + index * args.now_step
        started = time.perf_counter()
        supervised = supervisor.reauth(args.slice_type, now, detector_alert=args.detector_alert)
        record = {"index": index, "target": f"{host}:{port}", "slice_type": args.slice_type, "now": now,
                  "total_ms": (time.perf_counter() - started) * 1000.0, **_describe_supervised(supervised)}
        if index == 0 and args.capture_response is not None and client.last_response is not None:
            Path(args.capture_response).write_text(json.dumps(client.last_response))
        if supervised.result.category is OutcomeCategory.TRANSPORT_FAILURE:
            exit_code = 1
        print(json.dumps(record, sort_keys=True), flush=True)
    return exit_code


def _describe_supervised(supervised) -> dict:
    """One JSON-able summary of a supervised request (and its escalation)."""
    result, decision = supervised.result, supervised.decision
    fields = asdict(result)
    fields.pop("attempts")
    fields["reason"] = result.reason.value if result.reason is not None else None
    fields["outcome"] = result.outcome.value
    last = result.attempts[-1]
    out = {
        "outcome": result.outcome.value,
        "category": result.category.value,
        "attempts": [
            {"attempt": a.attempt, "outcome": a.outcome.value, "elapsed_ms": round(a.elapsed_ms, 3),
             "challenge": a.challenge[:16], "detail": a.detail, "audit_seq": a.audit_seq}
            for a in result.attempts
        ],
        "result": fields,
        "policy": {"action": decision.action.value, "rule": decision.rule, "reason": decision.reason,
                   "escalated": decision.escalate, "escalation_suppressed": decision.escalation_suppressed,
                   "quarantined": decision.quarantined},
        # kept for the integration orchestrator's older summary code
        "error": ({"type": result.outcome.value, "message": result.detail}
                  if result.category is OutcomeCategory.TRANSPORT_FAILURE else None),
    }
    if result.category is OutcomeCategory.TRANSPORT_FAILURE:
        out["elapsed_ms"] = sum(a.elapsed_ms for a in result.attempts)
    else:
        out["rtt_ms"] = last.elapsed_ms  # the attempt that got the answer
    if supervised.escalation is not None:
        out["escalation"] = _describe_supervised(supervised.escalation)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pqc_auth.transport")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run a ReauthServer (real OqsDilithiumSigner) until SIGTERM/SIGINT")
    serve.add_argument("--bind", default="127.0.0.1", help="address to bind (default: loopback)")
    serve.add_argument("--port", type=int, default=0, help="port to bind (default 0 = ephemeral)")
    serve.add_argument("--key-path", required=True, help="OqsDilithiumSigner key_path directory (created/persisted)")
    serve.add_argument(
        "--audit-log",
        default=None,
        help="server-side served-request log (plain JSONL, not hash-chained; audit_verify runs on the CLIENT log)",
    )
    serve.add_argument("--server-id", default=DEFAULT_SERVER_ID, help="identity string placed in every signed payload")
    serve.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS)
    serve.add_argument("--max-connections-per-peer", type=int, default=DEFAULT_MAX_CONNECTIONS_PER_PEER)
    serve.add_argument("--connection-timeout", type=float, default=DEFAULT_CONNECTION_TIMEOUT_SECONDS,
                       help="seconds an accepted connection may stay silent before it is dropped")

    req = sub.add_parser("request", help="run a ReauthClient: request, independently verify, print one JSON line each")
    req.add_argument("--server", default="127.0.0.1", help="server address (default: loopback)")
    req.add_argument("--port", type=int, required=True)
    req.add_argument("--slice-type", required=True)
    trust = req.add_mutually_exclusive_group(required=True)
    trust.add_argument("--expected-pubkey-file", help="explicit pin: raw public key bytes obtained out of band")
    trust.add_argument("--trust-store", help="TOFU trust store path (requires --server-id)")
    req.add_argument("--server-id", help="TOFU server identity (requires --trust-store)")
    req.add_argument("--audit-log", default=None, help="client verification audit log (hash-chained JSONL)")
    req.add_argument("--now", type=float, default=None, help="logical time of the first request (default: time.time())")
    req.add_argument("--now-step", type=float, default=0.0, help="added to --now for each successive request")
    req.add_argument("--count", type=int, default=1, help="requests to --server/--port with this one client")
    req.add_argument("--interval", type=float, default=0.0, help="real seconds to sleep between requests")
    req.add_argument("--detector-alert", action="store_true")
    req.add_argument("--timeout", type=float, default=None, help="sets both connect and read timeout (older single knob)")
    req.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT_SECONDS)
    req.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_SECONDS)
    req.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                     help="attempts per request; only transport failures are retried")
    req.add_argument("--fail-closed", action="store_true",
                     help="configure this slice fail-closed (transport failures can quarantine it)")
    req.add_argument(
        "--then",
        type=_parse_target,
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="after --count requests, send one more request per HOST:PORT with the SAME client instance",
    )
    req.add_argument("--capture-response", default=None, help="write the first raw response JSON to this file")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve_main(args)
    if args.trust_store is not None and args.server_id is None:
        parser.error("--trust-store requires --server-id")
    if args.now is None:
        args.now = time.time()
    return _request_main(args)


if __name__ == "__main__":
    sys.exit(main())
