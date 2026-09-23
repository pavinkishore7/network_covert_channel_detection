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

Public-key pinning: if ReauthClient is constructed with expected_public_key,
a response carrying any other public_key is rejected BEFORE verify_fn is
even called -- see ReauthClient's docstring and process_response() for why
that ordering matters (a signature can be genuinely valid under the wrong
key, so checking crypto validity first would ask the wrong question).

Trust-on-first-use (TOFU) pinning: if ReauthClient is constructed with
trust_store_path + server_id instead of expected_public_key, it learns and
persists whichever key it sees on the FIRST response for that server_id
(see pqc_auth/trust_store.py) -- but only once verify_fn has confirmed
that first response's signature is genuinely valid under that key; a
first-contact response that fails verification is never persisted, and
leaves the trust store empty for this server_id so a later, genuine first
response can still be accepted normally. Once a key is learned, the client
pins to it on every later response, with the same before-verify_fn
ordering and the same "don't mark the nonce as seen" behavior on a
mismatch as explicit pinning. See pqc_auth/trust_store.py's module
docstring for what TOFU does and does NOT solve -- it is not a substitute
for real key distribution, and an attacker present on the very first
connection to a never-before-seen server_id, WITH a validly signed
response, is indistinguishable from a legitimate first contact.

What this does NOT do (see pqc_auth/README.md for the fuller list):
  - No production-grade connection handling: no retries, no timeouts beyond
    a plain socket timeout, no TLS-equivalent transport security -- the
    signature is the only integrity guarantee, there is no confidentiality
    or anti-tampering on the request itself (a request only carries
    slice_type/now/detector_alert, none of which are secret).
  - No concurrent connection handling: one server thread, connections
    served strictly one at a time. Host/port default to loopback; binding
    elsewhere (e.g. inside a network namespace, see the CLI below and
    integration/auth_over_topology.py) is the caller's choice.

Command-line entry point (so a server or client can run as its own process,
e.g. inside a network namespace via ``ip netns exec <ns> ...``)::

    python -m pqc_auth.transport serve --bind ADDR --port P --key-path DIR [--audit-log FILE]
    python -m pqc_auth.transport request --server ADDR --port P --slice-type S \
        (--expected-pubkey-file FILE | --trust-store FILE --server-id ID) [...]

Both use the real ``OqsDilithiumSigner``/``verify_with_public_key`` and
therefore need liboqs. See ``main()`` for every flag.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from pqc_auth.audit_log import AuditLogger
from pqc_auth.reauth import DualTriggerReauthController, ReauthReason
from pqc_auth.trust_store import TrustStore

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

    def __init__(
        self,
        controller: DualTriggerReauthController,
        host: str = "127.0.0.1",
        port: int = 0,
        served_log_path: str | None = None,
    ):
        if controller.signer is None:
            raise ValueError("ReauthServer requires a controller with a signer configured")
        self.controller = controller
        # Optional plain-JSONL record of every request this server answered
        # (peer, slice_type, now, due, reason, nonce). NOT hash-chained and
        # NOT an input to pqc_auth/audit_verify.py -- that tool audits the
        # CLIENT's verification log, which is where trust decisions are
        # made. This one only lets an operator correlate "what the server
        # signed" with "what a client accepted". None = no file side effects.
        self._served_log_path = Path(served_log_path) if served_log_path is not None else None
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
        if self._served_log_path is not None:
            self._log_served(conn, request, response)

    def _log_served(self, conn: socket.socket, request: dict, response: dict) -> None:
        try:
            peer = "%s:%d" % conn.getpeername()[:2]
        except OSError:
            peer = None
        record = {
            "timestamp": time.time(),
            "peer": peer,
            "slice_type": request["slice_type"],
            "now": request["now"],
            "detector_alert": request.get("detector_alert", False),
            "due": response["due"],
            "reason": response.get("reason"),
            "nonce": response.get("nonce"),
        }
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
    backend: str | None = None


class ReauthClient:
    """Holds ONLY a public key and a verify function -- never a Signer, and
    structurally cannot end up with signing capability, since verify_fn is
    a bare callable (see pqc_auth.reauth.PublicKeyVerifier), not an object
    that could also expose sign().

    Never trusts the server's own ReauthOutcome.verified field -- that
    field isn't even sent over the wire (see ReauthServer._handle_connection).
    Every trust decision here is made by calling verify_fn independently.

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

        if self._expected_public_key is not None and public_key != self._expected_public_key:
            # Pinning rejection -- checked BEFORE verify_fn is ever called,
            # and deliberately independent of it. The whole point is that
            # public_key itself is not the one this client trusts; whether
            # a signature under that wrong key would itself verify is
            # irrelevant (it very well might: this could be a real
            # signature from a real, just-not-expected, keypair -- e.g. an
            # impersonator with their own genuine keypair, or a
            # misconfigured/rotated server). Calling verify_fn here would
            # answer a question nobody asked. The nonce is NOT marked as
            # seen: this response was never accepted, so a later response
            # for the same nonce from the CORRECTLY pinned key must still
            # be able to be accepted on its own merits.
            self._log_verification(
                response, reason, nonce, signature, public_key, backend,
                trusted=False, rejected_as_replay=False, pinned_key_mismatch=True,
            )
            return ClientVerificationResult(
                due=True, reason=reason, trusted=False, rejected_as_replay=False,
                pinned_key_mismatch=True, backend=backend,
            )

        first_contact = False
        if self._trust_store is not None:
            stored_key = self._trust_store.get_trusted_key(self._server_id)
            if stored_key is None:
                # First contact for this server_id: "I don't yet have an
                # opinion on whether this key is the right one" only means
                # the identity check passes trivially this one time -- it
                # does NOT mean the signature stops needing to verify via
                # verify_fn like every other response. Persisting the key
                # is deferred until AFTER verify_fn confirms it below (see
                # the trusted branch further down): persisting it here,
                # unconditionally, would let an unsigned/forged first
                # packet permanently poison the trust store for this
                # server_id before the real server's genuine first
                # response ever arrives, with no automatic recovery since
                # force_retrust() is never called automatically.
                first_contact = True
            elif stored_key != public_key:
                # TOFU-detected key change -- structurally the same "wrong
                # key" rejection as explicit pinning above, for the same
                # reasons: checked BEFORE verify_fn (a signature can be
                # genuinely valid under the changed key, so crypto validity
                # is the wrong question), and the nonce is NOT marked as
                # seen (this response was never accepted, so a later
                # response under the ORIGINALLY-trusted key must still be
                # acceptable on its own merits).
                self._log_verification(
                    response, reason, nonce, signature, public_key, backend,
                    trusted=False, rejected_as_replay=False, trust_store_key_changed=True,
                )
                return ClientVerificationResult(
                    due=True, reason=reason, trusted=False, rejected_as_replay=False,
                    trust_store_key_changed=True, backend=backend,
                )
            # else: stored_key == public_key -- matches, fall through.

        self._prune_expired(now)
        if nonce in self._seen_nonces:
            self._log_verification(response, reason, nonce, signature, public_key, backend, trusted=False, rejected_as_replay=True)
            return ClientVerificationResult(due=True, reason=reason, trusted=False, rejected_as_replay=True, backend=backend)

        trusted = self._verify_fn(nonce, signature, public_key)
        if trusted:
            self._seen_nonces[nonce] = now
            if first_contact:
                # Only now, with a genuinely valid signature under this
                # public_key confirmed, is it safe to learn and persist
                # it as this server_id's trusted identity.
                self._trust_store.trust_first_contact(self._server_id, public_key)
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
        pinned_key_mismatch: bool = False,
        trust_store_key_changed: bool = False,
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
            pinned_key_mismatch=pinned_key_mismatch,
            trust_store_key_changed=trust_store_key_changed,
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
        DualTriggerReauthController(signer=signer), host=args.bind, port=args.port, served_log_path=args.audit_log
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
    ready = {"host": server.host, "port": server.port, "backend": server.backend_name, "key_path": args.key_path}
    print("READY " + json.dumps(ready), flush=True)
    try:
        while not stop.wait(0.5):
            if server._thread is not None and not server._thread.is_alive():
                # The serve thread died (e.g. an unhandled exception while
                # handling one connection). Exit visibly rather than keep a
                # listening socket that will never answer.
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

    # Every request is made by this ONE client instance, so its in-memory
    # replay window (seen nonces) is shared across them -- which is what
    # lets a --then follow-up to a replaying endpoint be judged against
    # nonces this same client already accepted.
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
            response = client._send_request(args.slice_type, now, args.detector_alert)
            if index == 0 and args.capture_response is not None:
                Path(args.capture_response).write_text(json.dumps(response))
            result = client.process_response(response, now)
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
