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
import secrets
import signal
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from pqc_auth.audit_log import AuditLogger
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason
from pqc_auth.trust_store import TrustStore

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
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 5.0

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


class ReauthRequestError(RuntimeError):
    """The server refused the request (its ``{"error": code}`` response)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _recv_line(sock: socket.socket, max_bytes: int) -> bytes | None:
    """One newline-terminated line, without the newline. None if the peer
    closed without sending anything. Raises MessageTooLarge once more than
    ``max_bytes`` arrive without a newline, TruncatedMessage if the peer
    closes mid-line, and lets socket.timeout / OSError propagate."""
    buf = bytearray()
    while True:
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
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.detail = detail


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
    if request.get("v") != WIRE_VERSION:
        raise _RequestError("unsupported_version", f"expected v={WIRE_VERSION}, got {request.get('v')!r}")
    slice_type = request.get("slice_type")
    if not isinstance(slice_type, str) or slice_type not in known_slices:
        raise _RequestError("unknown_slice_type", repr(slice_type)[:64])
    now = request.get("now")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise _RequestError("invalid_now", repr(now)[:64])
    detector_alert = request.get("detector_alert", False)
    if not isinstance(detector_alert, bool):
        raise _RequestError("invalid_detector_alert", repr(detector_alert)[:64])
    challenge_hex = request.get("client_challenge")
    try:
        challenge = bytes.fromhex(challenge_hex) if isinstance(challenge_hex, str) else None
    except ValueError:
        challenge = None
    if challenge is None or not (MIN_CLIENT_CHALLENGE_BYTES <= len(challenge) <= MAX_CLIENT_CHALLENGE_BYTES):
        raise _RequestError(
            "invalid_client_challenge",
            f"need {MIN_CLIENT_CHALLENGE_BYTES}-{MAX_CLIENT_CHALLENGE_BYTES} bytes as hex",
        )
    return {"slice_type": slice_type, "now": float(now), "detector_alert": detector_alert, "client_challenge": challenge}


class ReauthServer:
    """Listens on a TCP socket and answers re-auth requests for a
    controller that has a signing-capable Signer configured.

    Concurrency model: one thread per accepted connection, capped by a
    semaphore at ``max_connections``. Chosen over selectors/asyncio because
    everything a connection does is blocking and short -- one recv, one
    controller.reauth() whose signer is a blocking C call (liboqs), one
    send -- so an event loop would still need worker threads for signing,
    and the thread-per-connection version keeps the existing synchronous
    code. The cap plus ``connection_timeout`` bound what an attacker can
    hold: at most ``max_connections`` threads, each for at most
    ``connection_timeout`` seconds of silence. A connection arriving while
    every slot is taken is closed at once (and counted/logged), never
    queued behind the others.

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
    ):
        if controller.signer is None:
            raise ValueError("ReauthServer requires a controller with a signer configured")
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        self.controller = controller
        self.server_id = server_id
        self._max_request_bytes = max_request_bytes
        self._connection_timeout = connection_timeout
        self._slots = threading.BoundedSemaphore(max_connections)
        # Optional plain-JSONL record of every connection this server
        # handled: served requests (with measured sign/verify time), refused
        # requests (with the error code), timeouts, and connections dropped
        # at capacity. NOT hash-chained and NOT an input to
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
            if not self._slots.acquire(blocking=False):
                self._record(addr, "rejected_at_capacity")
                conn.close()
                continue
            threading.Thread(target=self._connection_worker, args=(conn, addr), daemon=True).start()

    def _connection_worker(self, conn: socket.socket, addr) -> None:
        try:
            with conn:
                conn.settimeout(self._connection_timeout)
                self._handle_connection(conn, addr)
        except Exception as exc:  # noqa: BLE001 - one connection must never take the server down
            self._record(addr, "error", code="internal_error", detail=f"{type(exc).__name__}: {exc}"[:200])
        finally:
            self._slots.release()

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
            self._refuse(conn, addr, exc.code, exc.detail)
            return

        payload_holder: list[str] = []

        def bind_payload(reason: ReauthReason, nonce: bytes) -> bytes:
            payload = canonical_payload({
                "v": WIRE_VERSION,
                "server_id": self.server_id,
                "slice_type": request["slice_type"],
                "reason": reason.value,
                "client_challenge": request["client_challenge"].hex(),
                "server_nonce": nonce.hex(),
                "request_now": request["now"],
                "issued_at": time.time(),
            })
            payload_holder.append(payload)
            return signed_message(payload)

        outcome = self.controller.reauth(
            request["slice_type"], request["now"], detector_alert=request["detector_alert"], sign_message=bind_payload
        )
        if outcome is None:
            response = {"v": WIRE_VERSION, "due": False}
        else:
            response = {
                "v": WIRE_VERSION,
                "due": True,
                "payload": payload_holder[0],
                "signature": outcome.signature.hex(),
                "public_key": self.controller.signer.public_key.hex(),
                # Label only -- the client's trust decision never depends
                # on this string, it's informational (logging/demo output).
                "backend": self.backend_name,
            }
        _send_line(conn, json.dumps(response).encode())
        self._record(
            addr, "served",
            slice_type=request["slice_type"], now=request["now"], detector_alert=request["detector_alert"],
            due=outcome is not None,
            reason=outcome.reason.value if outcome is not None else None,
            server_nonce=outcome.nonce.hex() if outcome is not None else None,
            client_challenge=request["client_challenge"].hex(),
            sign_ms=outcome.sign_ms if outcome is not None else None,
            verify_ms=outcome.verify_ms if outcome is not None else None,
        )

    def _refuse(self, conn: socket.socket, addr, code: str, detail: str = "") -> None:
        self._record(addr, "error", code=code, detail=detail[:200])
        try:
            _send_line(conn, json.dumps({"v": WIRE_VERSION, "error": code, "detail": detail[:200]}).encode())
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
class ClientVerificationResult:
    due: bool
    reason: ReauthReason | None = None
    trusted: bool | None = None
    rejected_as_replay: bool = False
    pinned_key_mismatch: bool = False
    trust_store_key_changed: bool = False
    challenge_mismatch: bool = False
    malformed_response: bool = False
    backend: str | None = None


@dataclass(frozen=True)
class _ParsedResponse:
    payload: str
    fields: dict
    reason: ReauthReason
    client_challenge: bytes
    server_nonce: bytes
    signature: bytes
    public_key: bytes
    backend: str | None


def _parse_response(response) -> _ParsedResponse | None:
    """Structural parse of a due=true response. None if anything is off."""
    try:
        payload = response["payload"]
        fields = json.loads(payload)
        if not isinstance(fields, dict) or fields.get("v") != WIRE_VERSION:
            return None
        return _ParsedResponse(
            payload=payload,
            fields=fields,
            reason=ReauthReason(fields["reason"]),
            client_challenge=bytes.fromhex(fields["client_challenge"]),
            server_nonce=bytes.fromhex(fields["server_nonce"]),
            signature=bytes.fromhex(response["signature"]),
            public_key=bytes.fromhex(response["public_key"]),
            backend=response.get("backend"),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class ReauthClient:
    """Holds ONLY a public key and a verify function -- never a Signer, and
    structurally cannot end up with signing capability, since verify_fn is
    a bare callable (see pqc_auth.reauth.PublicKeyVerifier), not an object
    that could also expose sign().

    Never trusts the server's own ReauthOutcome.verified field -- that
    field isn't even sent over the wire (see ReauthServer._handle_connection).
    Every trust decision here is made by calling verify_fn independently.

    Every request carries a fresh random challenge (CLIENT_CHALLENGE_BYTES
    from ``secrets``), and a response is only acceptable if its signed
    payload carries that exact challenge -- see process_response().

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
    mechanism underneath it would silently discard that guarantee.
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
        timeout: float = 5.0,
        audit_log_path: str | None = None,
    ):
        if (trust_store_path is None) != (server_id is None):
            raise ValueError("trust_store_path and server_id must be given together, or not at all")
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
        self._timeout = timeout
        self._seen_nonces: dict[bytes, float] = {}
        # None by default so existing tests/callers get no file side
        # effects; set audit_log_path to append a record for every real
        # client-side verification (see pqc_auth/audit_log.py).
        self._audit_logger = AuditLogger(audit_log_path) if audit_log_path is not None else None

    def request_reauth(self, slice_type: str, now: float, detector_alert: bool = False) -> ClientVerificationResult:
        """Send a real request with a fresh challenge over a real socket,
        then independently verify the response against that challenge."""
        response, challenge = self._send_request(slice_type, now, detector_alert)
        return self.process_response(response, now, expected_challenge=challenge)

    def _send_request(
        self, slice_type: str, now: float, detector_alert: bool, client_challenge: bytes | None = None
    ) -> tuple[dict, bytes]:
        """Returns (response, the challenge this request carried). A new
        challenge is generated for every call, retries included, unless a
        test passes one explicitly. Raises ReauthRequestError if the server
        refused the request."""
        challenge = client_challenge if client_challenge is not None else secrets.token_bytes(CLIENT_CHALLENGE_BYTES)
        request = {
            "v": WIRE_VERSION, "slice_type": slice_type, "now": now,
            "detector_alert": detector_alert, "client_challenge": challenge.hex(),
        }
        sock = socket.create_connection((self.host, self.port), timeout=self._timeout)
        with sock:
            _send_line(sock, json.dumps(request).encode())
            raw = _recv_line(sock, MAX_RESPONSE_BYTES)
        if raw is None:
            raise ConnectionError("no response from ReauthServer")
        response = json.loads(raw)
        if isinstance(response, dict) and "error" in response:
            raise ReauthRequestError(str(response["error"]), str(response.get("detail", "")))
        return response, challenge

    def process_response(self, response: dict, now: float, *, expected_challenge: bytes) -> ClientVerificationResult:
        """Verify a (possibly captured/replayed) server response independently.

        ``expected_challenge`` is the challenge THIS client sent in the
        request this response claims to answer. Exposed separately from
        request_reauth() so tests can hand in a captured response.

        Check order, each one returning immediately on failure:
          1. structure/version -> malformed_response (nothing signed to audit)
          2. explicit pin      -> pinned_key_mismatch
          3. TOFU stored key   -> trust_store_key_changed
          4. client challenge  -> challenge_mismatch
          5. seen server nonce -> rejected_as_replay (defence in depth)
          6. verify_fn over signed_message(payload)
        Identity (2, 3) comes first: "not the key I trust" is the more
        fundamental rejection and keeps the existing pin/TOFU records
        unchanged. The challenge (4) is checked before verify_fn for the
        same reason the key checks are: a replayed response carries a
        perfectly valid signature, so crypto validity is the wrong question
        -- the question is "is this an answer to MY request". None of 2-5
        mark the nonce as seen or persist a TOFU key.
        """
        if not isinstance(response, dict) or response.get("v") != WIRE_VERSION:
            return ClientVerificationResult(due=True, trusted=False, malformed_response=True)
        if response.get("due") is not True:
            return ClientVerificationResult(due=False)

        parsed = _parse_response(response)
        if parsed is None:
            return ClientVerificationResult(due=True, trusted=False, malformed_response=True, backend=response.get("backend"))

        def reject(**flag) -> ClientVerificationResult:
            self._log_verification(parsed, expected_challenge, trusted=False, **flag)
            return ClientVerificationResult(
                due=True, reason=parsed.reason, trusted=False, backend=parsed.backend, **flag
            )

        if self._expected_public_key is not None and parsed.public_key != self._expected_public_key:
            # Pinning rejection -- deliberately independent of verify_fn: a
            # signature under the wrong key may well be genuine (an
            # impersonator's own keypair, a rotated server). The nonce is
            # NOT marked as seen.
            return reject(pinned_key_mismatch=True)

        first_contact = False
        if self._trust_store is not None:
            stored_key = self._trust_store.get_trusted_key(self._server_id)
            if stored_key is None:
                # First contact: the identity check passes trivially this
                # once, but the challenge and verify_fn below still must.
                # Persisting is deferred until both have -- persisting here
                # would let a forged or replayed first packet poison the
                # store permanently (force_retrust() is never automatic).
                first_contact = True
            elif stored_key != parsed.public_key:
                # TOFU-detected key change: same category, same ordering
                # and same "don't consume the nonce" rule as pinning.
                return reject(trust_store_key_changed=True)

        if not hmac.compare_digest(parsed.client_challenge, expected_challenge):
            # Not an answer to this request: a recorded response from an
            # earlier exchange (possibly genuinely signed by the trusted
            # key), or one signed over someone else's challenge. This is
            # the primary replay defence and needs no memory of past
            # nonces, so it holds across client restarts.
            return reject(challenge_mismatch=True)

        self._prune_expired(now)
        if parsed.server_nonce in self._seen_nonces:
            # Defence in depth only: with a fresh challenge per request, a
            # response reaching this point with an already-seen nonce means
            # the server reused a nonce for a new challenge.
            return reject(rejected_as_replay=True)

        trusted = bool(self._verify_fn(signed_message(parsed.payload), parsed.signature, parsed.public_key))
        if trusted:
            self._seen_nonces[parsed.server_nonce] = now
            if first_contact:
                # Only now -- challenge matched AND signature genuinely
                # valid under this key -- is it safe to learn it.
                self._trust_store.trust_first_contact(self._server_id, parsed.public_key)
        self._log_verification(parsed, expected_challenge, trusted=trusted)
        return ClientVerificationResult(due=True, reason=parsed.reason, trusted=trusted, backend=parsed.backend)

    def _log_verification(
        self,
        parsed: _ParsedResponse,
        expected_challenge: bytes,
        *,
        trusted: bool,
        rejected_as_replay: bool = False,
        pinned_key_mismatch: bool = False,
        trust_store_key_changed: bool = False,
        challenge_mismatch: bool = False,
    ) -> None:
        if self._audit_logger is None:
            return
        self._audit_logger.log(
            slice_type=str(parsed.fields.get("slice_type", "")),
            reason=parsed.reason.value,
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
        )

    def _prune_expired(self, now: float) -> None:
        cutoff = now - self._replay_window_seconds
        expired = [nonce for nonce, seen_at in self._seen_nonces.items() if seen_at < cutoff]
        for nonce in expired:
            del self._seen_nonces[nonce]


# -- command-line entry point ----------------------------------------------


def _serve_main(args: argparse.Namespace) -> int:
    from pqc_auth.dilithium import OqsDilithiumSigner

    signer = OqsDilithiumSigner(key_path=args.key_path)
    server = ReauthServer(
        DualTriggerReauthController(signer=signer), host=args.bind, port=args.port, served_log_path=args.audit_log,
        server_id=args.server_id, max_connections=args.max_connections, connection_timeout=args.connection_timeout,
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
        audit_log_path=args.audit_log,
    )

    # Every request is made by this ONE client instance (one seen-nonce
    # set), and every request -- --then follow-ups included -- carries its
    # own fresh challenge, so a --then to an endpoint replaying an earlier
    # response is judged against the challenge of THAT request.
    targets = [(args.server, args.port)] * args.count + list(args.then)
    exit_code = 0
    for index, (host, port) in enumerate(targets):
        if index > 0 and args.interval > 0:
            time.sleep(args.interval)
        client.host, client.port = host, port
        now = args.now + index * args.now_step
        record = {"index": index, "target": f"{host}:{port}", "slice_type": args.slice_type, "now": now}
        started = time.perf_counter()
        try:
            response, challenge = client._send_request(args.slice_type, now, args.detector_alert)
            if index == 0 and args.capture_response is not None:
                Path(args.capture_response).write_text(json.dumps(response))
            result = client.process_response(response, now, expected_challenge=challenge)
        except Exception as exc:  # noqa: BLE001 - record what ReauthClient raises, verbatim
            record.update(
                error={"type": type(exc).__name__, "message": str(exc)},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            exit_code = 1
        else:
            fields = asdict(result)
            fields["reason"] = result.reason.value if result.reason is not None else None
            record.update(result=fields, error=None, rtt_ms=(time.perf_counter() - started) * 1000.0)
        print(json.dumps(record, sort_keys=True), flush=True)
    return exit_code


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
    req.add_argument("--timeout", type=float, default=5.0, help="ReauthClient socket timeout (default: its own 5.0)")
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
