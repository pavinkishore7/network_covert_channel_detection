"""Challenge binding, server concurrency, and malformed-input handling in
pqc_auth/transport.py (wire version 2). Real sockets, FakeSigner, no root,
no liboqs.
"""

from __future__ import annotations

import hashlib
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from pqc_auth import audit_verify
from pqc_auth.audit_verify import verify_log
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import (
    SIGNATURE_DOMAIN,
    ReauthClient,
    ReauthServer,
    canonical_payload,
    signed_message,
)
from pqc_auth.outcomes import RequestOutcome
from pqc_auth.trust_store import TrustStore

from tests.fake_signer import FakeSigner


class ReplayResponder:
    """An attacker endpoint: answers every request with fixed bytes."""

    def __init__(self, response: dict):
        self._line = json.dumps(response).encode() + b"\n"
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                conn.recv(4096)
                conn.sendall(self._line)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._sock.close()


def _error_code(reply: bytes) -> str:
    """A refusal is signed (code inside the payload) when the request carried
    a usable challenge, and a bare unsigned error otherwise."""
    response = json.loads(reply)
    if "payload" in response:
        return json.loads(response["payload"])["error_code"]
    return response["error"]


def _raw_exchange(port: int, payload: bytes, *, shutdown_write: bool = False, timeout: float = 3.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(payload)
        if shutdown_write:
            s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            try:
                chunk = s.recv(4096)
            except (ConnectionResetError, socket.timeout):
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


class _ServerCase(unittest.TestCase):
    server_kwargs: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.audit_log = self.dir / "client_audit.jsonl"
        self.served_log = self.dir / "served.jsonl"
        self.signer = FakeSigner()
        self.controller = DualTriggerReauthController(signer=self.signer)
        self.server = ReauthServer(self.controller, served_log_path=str(self.served_log), **self.server_kwargs)
        self.server.start()
        self.addCleanup(self.server.stop)

    def client(self, port: int | None = None, **kwargs) -> ReauthClient:
        kwargs.setdefault("expected_public_key", self.signer.public_key)
        kwargs.setdefault("audit_log_path", str(self.audit_log))
        return ReauthClient("127.0.0.1", port or self.server.port, verify_fn=FakeSigner.verify_with_public_key, **kwargs)

    def assertServesValidRequest(self, slice_type="eMBB", now=10_000.0):
        result = self.client(timeout=3).request_reauth(slice_type, now)
        self.assertTrue(result.trusted, result)


class ChallengeBindingTests(_ServerCase):
    def test_fresh_client_given_a_recorded_valid_response_rejects_it(self):
        """The exact P1 case ('replay, fresh client process') that used to
        come back TRUSTED."""
        recorded, _ = self.client()._send_request("URLLC", 0, detector_alert=False)
        responder = ReplayResponder(recorded)
        self.addCleanup(responder.stop)

        fresh = self.client(port=responder.port)  # new instance: empty seen-nonce set
        result = fresh.request_reauth("URLLC", 1)

        self.assertFalse(result.trusted)
        self.assertTrue(result.challenge_mismatch)
        self.assertFalse(result.rejected_as_replay)
        self.assertEqual(fresh._seen_nonces, {})
        self.assertTrue(verify_log(self.audit_log).all_clean)

    def test_fresh_tofu_client_does_not_learn_a_key_from_a_replayed_first_contact(self):
        recorded, _ = self.client()._send_request("URLLC", 0, detector_alert=False)
        responder = ReplayResponder(recorded)
        self.addCleanup(responder.stop)
        store = self.dir / "trust.json"
        tofu = self.client(port=responder.port, expected_public_key=None, trust_store_path=str(store), server_id="s")
        self.assertTrue(tofu.request_reauth("URLLC", 1).challenge_mismatch)
        self.assertIsNone(TrustStore(store).get_trusted_key("s"))

    def test_response_signed_over_a_different_challenge_is_rejected(self):
        # A genuine response to SOMEONE ELSE's request (challenge X),
        # delivered to a client that sent challenge Y.
        other_client = self.client()
        response, challenge_x = other_client._send_request("URLLC", 0, detector_alert=False)
        challenge_y = bytes(b ^ 0xFF for b in challenge_x)
        victim = self.client()
        result = victim.process_response(response, now=0, expected_challenge=challenge_y)
        self.assertFalse(result.trusted)
        self.assertTrue(result.challenge_mismatch)
        self.assertEqual(victim._seen_nonces, {})

    def test_editing_the_challenge_in_the_payload_breaks_the_signature(self):
        response, challenge_x = self.client()._send_request("URLLC", 0, detector_alert=False)
        challenge_y = bytes(32)
        fields = json.loads(response["payload"])
        fields["client_challenge"] = challenge_y.hex()
        forged = {**response, "payload": canonical_payload(fields)}  # signature still over challenge X
        victim = self.client()
        result = victim.process_response(forged, now=0, expected_challenge=challenge_y)
        self.assertFalse(result.challenge_mismatch)  # the challenge now "matches" ...
        self.assertFalse(result.trusted)  # ... but the signature doesn't cover it
        self.assertEqual(victim._seen_nonces, {})
        self.assertTrue(verify_log(self.audit_log).all_clean)

    def test_every_request_carries_a_new_random_challenge(self):
        client = self.client()
        seen = set()
        for i in range(5):
            _, challenge = client._send_request("mMTC", 1000.0 * i, detector_alert=False)
            self.assertGreaterEqual(len(challenge), 16)
            seen.add(challenge)
        self.assertEqual(len(seen), 5)

    def test_signed_payload_carries_the_bound_fields(self):
        response, challenge = self.client()._send_request("eMBB", 7, detector_alert=True)
        fields = json.loads(response["payload"])
        self.assertEqual(response["payload"], canonical_payload(fields))
        self.assertEqual(fields["client_challenge"], challenge.hex())
        self.assertEqual(fields["server_id"], self.server.server_id)
        self.assertEqual(fields["slice_type"], "eMBB")
        self.assertEqual(fields["reason"], "detector_alert")
        self.assertEqual(fields["request_now"], 7)
        self.assertIn("server_nonce", fields)
        self.assertIn("issued_at", fields)
        self.assertTrue(FakeSigner.verify_with_public_key(
            signed_message(response["payload"]), bytes.fromhex(response["signature"]), self.signer.public_key))

    def test_old_wire_format_response_is_not_accepted(self):
        response, challenge = self.client()._send_request("URLLC", 0, detector_alert=False)
        nonce = b"v1-style-nonce"
        v1 = {"due": True, "reason": "periodic", "nonce": nonce.hex(),
              "signature": self.signer.sign(nonce).hex(), "public_key": self.signer.public_key.hex(),
              "backend": "FakeSigner", "slice_type": "URLLC"}
        result = self.client().process_response(v1, now=0, expected_challenge=challenge)
        self.assertFalse(result.trusted)
        self.assertTrue(result.malformed_response)

    def test_audit_verify_uses_the_same_domain_tag(self):
        self.assertEqual(audit_verify._SIGNATURE_DOMAIN_V2, SIGNATURE_DOMAIN)


class ConcurrencyTests(_ServerCase):
    server_kwargs = {"connection_timeout": 5.0}

    def test_idle_connection_does_not_block_other_clients(self):
        idle = socket.create_connection(("127.0.0.1", self.server.port))
        self.addCleanup(idle.close)
        time.sleep(0.1)
        started = time.monotonic()
        self.assertServesValidRequest()  # client timeout is 3s < server's 5s idle timeout
        self.assertLess(time.monotonic() - started, 3.0)

    def test_concurrent_requests_for_one_slice_fire_exactly_once(self):
        n = 16
        barrier = threading.Barrier(n)
        results = [None] * n

        def worker(i):
            client = self.client(timeout=5)
            barrier.wait()
            results[i] = client.request_reauth("URLLC", 500.0)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        due = [r for r in results if r is not None and r.due]
        self.assertEqual(len(due), 1, results)
        self.assertTrue(due[0].trusted)
        self.assertEqual(sum(1 for r in results if r is not None and not r.due), n - 1)
        self.assertEqual(self.controller._last_reauth["URLLC"], 500.0)
        served = [json.loads(l) for l in self.served_log.read_text().splitlines()]
        self.assertEqual(sum(1 for r in served if r["event"] == "served" and r["due"]), 1)

    def test_controller_due_is_atomic_under_threads(self):
        # Widen the read-modify-write window in due() so the race is real:
        # measured with this same setup, removing the controller's lock
        # makes all 32 threads fire; with it, exactly one does.
        class SlowReadDict(dict):
            def get(self, *args):
                value = super().get(*args)
                time.sleep(0.01)
                return value

        controller = DualTriggerReauthController()
        controller._last_reauth = SlowReadDict()
        n = 32
        barrier = threading.Barrier(n)
        fired = []

        def worker():
            barrier.wait()
            if controller.due("mMTC", 42.0) is not None:
                fired.append(1)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(fired), 1)


class CapacityAndTimeoutTests(_ServerCase):
    server_kwargs = {"max_connections": 2, "connection_timeout": 0.5}

    def test_connections_beyond_the_cap_are_closed_immediately_and_idle_ones_time_out(self):
        idle = [socket.create_connection(("127.0.0.1", self.server.port)) for _ in range(2)]
        for s in idle:
            self.addCleanup(s.close)
        time.sleep(0.1)
        with socket.create_connection(("127.0.0.1", self.server.port), timeout=2) as extra:
            started = time.monotonic()
            self.assertEqual(extra.recv(10), b"")  # closed by the server, not queued
            self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(self.server.stats["rejected_at_capacity"], 1)

        time.sleep(0.8)  # past the 0.5s idle timeout
        self.assertEqual(self.server.stats["error:read_timeout"], 2)
        self.assertServesValidRequest()


class MalformedInputTests(_ServerCase):
    CASES = [
        ("oversized, no newline", b"x" * 5000, "request_too_large"),
        ("oversized line", b'{"v": 2, "pad": "' + b"y" * 2000 + b'"}\n', "request_too_large"),
        ("non-utf8", b"\xff\xfe\xfd\n", "invalid_utf8"),
        ("invalid json", b"{not json\n", "invalid_json"),
        ("json array", b"[1, 2, 3]\n", "invalid_request"),
        ("v1 request", b'{"slice_type": "URLLC", "now": 0}\n', "unsupported_version"),
        ("unknown slice", json.dumps({"v": 2, "slice_type": "6G", "now": 0, "client_challenge": "00" * 32}).encode() + b"\n",
         "unknown_slice_type"),
        ("NaN now", b'{"v": 2, "slice_type": "URLLC", "now": NaN, "client_challenge": "' + b"00" * 32 + b'"}\n', "invalid_now"),
        ("short challenge", json.dumps({"v": 2, "slice_type": "URLLC", "now": 0, "client_challenge": "00" * 4}).encode() + b"\n",
         "invalid_client_challenge"),
        ("bool detector_alert", json.dumps({"v": 2, "slice_type": "URLLC", "now": 0, "detector_alert": "yes",
                                            "client_challenge": "00" * 32}).encode() + b"\n", "invalid_detector_alert"),
    ]

    def test_each_malformed_request_is_refused_and_the_server_keeps_serving(self):
        for label, payload, code in self.CASES:
            with self.subTest(label):
                reply = _raw_exchange(self.server.port, payload)
                self.assertEqual(_error_code(reply), code, reply[:200])
                self.assertTrue(self.server._thread.is_alive())
                self.assertEqual(self.server.stats[f"error:{code}"] >= 1, True)
        self.assertServesValidRequest()

    def test_truncated_request_is_logged_and_the_server_keeps_serving(self):
        reply = _raw_exchange(self.server.port, b'{"v": 2, "slice_ty', shutdown_write=True)
        self.assertEqual(reply, b"")
        time.sleep(0.1)
        self.assertEqual(self.server.stats["error:truncated_request"], 1)
        self.assertServesValidRequest()

    def test_client_reports_an_authenticated_server_refusal(self):
        result = self.client().request_reauth("not-a-slice", 0)
        self.assertEqual(result.outcome, RequestOutcome.SERVER_REFUSED)
        self.assertEqual(result.error_code, "unknown_slice_type")
        self.assertTrue(result.trusted)  # the refusal itself was signed and bound to our challenge
        self.assertEqual(len(result.attempts), 1)  # a refusal is never retried

    def test_served_log_records_errors_and_measured_sign_verify_time(self):
        _raw_exchange(self.server.port, b"{bad\n")
        self.assertServesValidRequest()
        records = [json.loads(l) for l in self.served_log.read_text().splitlines()]
        self.assertTrue(any(r["event"] == "error" and r["code"] == "invalid_json" for r in records))
        served = [r for r in records if r["event"] == "served" and r["due"]]
        self.assertTrue(served)
        self.assertGreaterEqual(served[-1]["sign_ms"], 0.0)
        self.assertGreaterEqual(served[-1]["verify_ms"], 0.0)


class ChallengeAuditTests(_ServerCase):
    """audit_verify on logs containing challenge-mismatch records."""

    def _log_with_replay_and_normal_records(self):
        recorded, _ = self.client()._send_request("URLLC", 0, detector_alert=False)
        responder = ReplayResponder(recorded)
        self.addCleanup(responder.stop)
        self.client().request_reauth("eMBB", 0)  # trusted
        self.assertTrue(self.client(port=responder.port).request_reauth("URLLC", 1).challenge_mismatch)
        return [json.loads(l) for l in self.audit_log.read_text().splitlines()]

    def test_log_with_a_replayed_genuine_response_passes(self):
        records = self._log_with_replay_and_normal_records()
        mismatch = [r for r in records if r.get("challenge_mismatch")]
        self.assertEqual(len(mismatch), 1)
        # The replayed response's signature IS genuinely valid -- which is
        # why audit_verify must not assert "crypto valid => trusted" here.
        self.assertTrue(FakeSigner.verify_with_public_key(
            signed_message(mismatch[0]["signed_payload"]), bytes.fromhex(mismatch[0]["signature"]),
            bytes.fromhex(mismatch[0]["public_key"])))
        result = verify_log(self.audit_log)
        self.assertTrue(result.all_clean, [r.reasons for r in result.records])

    def _rewrite(self, records, index, edit) -> Path:
        """Edit one record and recompute its hash and the chain after it, so
        only the trust-outcome consistency check can object."""
        records = [dict(r) for r in records]
        edit(records[index])
        path = self.dir / "edited.jsonl"
        prev = "0" * 64
        lines = []
        for r in records:
            r["prev_hash"] = prev
            body = {k: v for k, v in r.items() if k != "record_hash"}
            r["record_hash"] = hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            line = json.dumps(r, sort_keys=True, separators=(",", ":"))
            lines.append(line)
            prev = hashlib.sha256(line.encode()).hexdigest()
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_mismatch_record_whose_challenges_actually_match_fails(self):
        records = self._log_with_replay_and_normal_records()
        index = next(i for i, r in enumerate(records) if r.get("challenge_mismatch"))

        def claim_false_mismatch(r):
            r["expected_challenge"] = json.loads(r["signed_payload"])["client_challenge"]

        result = verify_log(self._rewrite(records, index, claim_false_mismatch))
        reasons = result.records[index].reasons
        self.assertFalse(result.all_clean)
        self.assertTrue(any("EQUALS the expected challenge" in r for r in reasons), reasons)

    def test_trusted_record_with_a_mismatched_challenge_fails(self):
        records = self._log_with_replay_and_normal_records()
        index = next(i for i, r in enumerate(records) if r.get("trusted"))

        def swap_expected(r):
            r["expected_challenge"] = "ab" * 32

        result = verify_log(self._rewrite(records, index, swap_expected))
        self.assertTrue(any("client_challenge differs" in r for r in result.records[index].reasons))

    def test_mismatch_record_that_also_claims_trusted_fails(self):
        records = self._log_with_replay_and_normal_records()
        index = next(i for i, r in enumerate(records) if r.get("challenge_mismatch"))
        result = verify_log(self._rewrite(records, index, lambda r: r.update(trusted=True)))
        self.assertTrue(any("also claims trusted" in r for r in result.records[index].reasons))


if __name__ == "__main__":
    unittest.main()
