"""Outcome taxonomy, client retries, the failure policy, and their audit
records. FakeSigner, fake servers, loopback sockets; no root, no liboqs.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from pqc_auth import audit_verify
from pqc_auth.audit_verify import verify_log
from pqc_auth.failure_policy import (
    DEFAULT_SLICE_POLICIES,
    DecisionEvent,
    OutcomeEvent,
    PolicyAction,
    ReauthSupervisor,
    SliceFailurePolicy,
    decide,
    is_quarantined,
)
from pqc_auth.outcomes import OutcomeCategory, RequestOutcome
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer

from tests.fake_signer import FakeSigner

O = RequestOutcome


# -- fake peers -----------------------------------------------------------------


class FakePeer:
    """A TCP endpoint whose behaviour per connection is ``handler(conn, request_line)``.
    Records every request line it receives."""

    def __init__(self, handler, host="127.0.0.1"):
        self.handler = handler
        self.requests: list[dict] = []
        self._sock = socket.socket()
        self._sock.bind((host, 0))
        self._sock.listen(16)
        self._sock.settimeout(0.1)
        self.host, self.port = self._sock.getsockname()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn:
            conn.settimeout(10)
            try:
                line = conn.makefile("rb").readline()
                try:
                    self.requests.append(json.loads(line))
                except ValueError:
                    pass
                self.handler(conn, line)
            except OSError:
                pass

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._sock.close()


def relay_to(server: ReauthServer, mutate=None):
    """Handler: forward to a real ReauthServer, optionally mutate the response dict."""

    def handler(conn, line):
        with socket.create_connection((server.host, server.port), timeout=5) as up:
            up.sendall(line)
            response = json.loads(up.makefile("rb").readline())
        if mutate is not None:
            response = mutate(response)
        conn.sendall(json.dumps(response).encode() + b"\n")

    return handler


def forge_signature(response: dict) -> dict:
    sig = bytearray(bytes.fromhex(response["signature"]))
    sig[0] ^= 0xFF
    return {**response, "signature": sig.hex()}


def hang(conn, line):
    time.sleep(3)


def close_mid_response(conn, line):
    conn.sendall(b'{"v": 2, "payload": "{\\"stat')


def send_garbage(conn, line):
    conn.sendall(b"this is not json\n")


def unsigned_not_due(conn, line):
    conn.sendall(b'{"v": 2, "due": false}\n')


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "audit.jsonl"
        self.signer = FakeSigner()
        self.controller = DualTriggerReauthController(signer=self.signer)
        self.server = ReauthServer(self.controller)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.sleeps: list[float] = []

    def peer(self, handler, **kw) -> FakePeer:
        peer = FakePeer(handler, **kw)
        self.addCleanup(peer.stop)
        return peer

    def client(self, host, port, **kw) -> ReauthClient:
        kw.setdefault("expected_public_key", self.signer.public_key)
        kw.setdefault("audit_log_path", str(self.log))
        kw.setdefault("read_timeout", 0.3)
        kw.setdefault("connect_timeout", 0.3)
        kw.setdefault("sleep", self.sleeps.append)
        kw.setdefault("rng", random.Random(7))
        return ReauthClient(host, port, verify_fn=FakeSigner.verify_with_public_key, **kw)

    def records(self) -> list[dict]:
        return [json.loads(l) for l in self.log.read_text().splitlines()]

    def assertAuditClean(self):
        result = verify_log(self.log)
        self.assertTrue(result.all_clean, [(r.seq, r.reasons) for r in result.records if not r.ok])


# -- taxonomy -----------------------------------------------------------------------


class TaxonomyTests(unittest.TestCase):
    def test_every_outcome_has_exactly_one_category_and_crypto_is_separate_from_transport(self):
        by_category = {c: {o for o in RequestOutcome if o.category is c} for c in OutcomeCategory}
        self.assertEqual(by_category[OutcomeCategory.TRANSPORT_FAILURE], {O.TIMEOUT, O.CONNECTION_FAILED})
        self.assertTrue(by_category[OutcomeCategory.AUTH_FAILURE].isdisjoint(by_category[OutcomeCategory.TRANSPORT_FAILURE]))
        self.assertEqual(sum(len(v) for v in by_category.values()), len(RequestOutcome))

    def test_audit_verify_literals_match_the_enum(self):
        def values(cat):
            return {o.value for o in RequestOutcome if o.category is cat}

        self.assertEqual(audit_verify._OK_OUTCOMES, values(OutcomeCategory.OK))
        self.assertEqual(audit_verify._AUTH_FAILURE_OUTCOMES, values(OutcomeCategory.AUTH_FAILURE))
        self.assertEqual(audit_verify._TRANSPORT_FAILURE_OUTCOMES, values(OutcomeCategory.TRANSPORT_FAILURE))
        self.assertEqual(audit_verify._REFUSED_OUTCOMES, values(OutcomeCategory.REFUSED))


# -- the pure policy ----------------------------------------------------------------


def outcomes(*pairs):
    return [OutcomeEvent(now=t, outcome=o, audit_seq=i) for i, (t, o) in enumerate(pairs)]


class DecideTests(unittest.TestCase):
    def test_ok_is_none(self):
        self.assertIs(decide("URLLC", outcomes((0, O.VERIFIED)), 0).action, PolicyAction.NONE)

    def test_any_auth_failure_escalates_immediately(self):
        for outcome in (O.REJECTED_SIGNATURE, O.REJECTED_UNSIGNED_STATUS, O.MALFORMED_RESPONSE, O.REJECTED_PINNED_KEY):
            d = decide("eMBB", outcomes((0, outcome)), 0)
            self.assertIs(d.action, PolicyAction.ESCALATE_TO_DETECTOR_ALERT, outcome)
            self.assertTrue(d.escalate)

    def test_second_auth_failure_in_window_quarantines(self):
        history = outcomes((0, O.REJECTED_SIGNATURE)) + [DecisionEvent(0, PolicyAction.ESCALATE_TO_DETECTOR_ALERT, True)]
        history.append(OutcomeEvent(0, O.REJECTED_SIGNATURE, 5))
        d = decide("eMBB", history, 0)
        self.assertIs(d.action, PolicyAction.QUARANTINE_FLAG)
        self.assertTrue(d.escalation_suppressed)  # cooldown: the escalation just happened
        self.assertEqual(d.trigger_seqs, (0, 5))

    def test_auth_failures_outside_the_window_do_not_quarantine(self):
        history = outcomes((0, O.REJECTED_SIGNATURE), (1000, O.REJECTED_SIGNATURE))
        self.assertIs(decide("eMBB", history, 1000).action, PolicyAction.ESCALATE_TO_DETECTOR_ALERT)

    def test_transport_failures_alert_at_threshold_and_never_quarantine_when_fail_open(self):
        history = outcomes(*[(t, O.TIMEOUT) for t in range(2)])
        self.assertIs(decide("URLLC", history, 2).action, PolicyAction.NONE)
        # an on-path attacker that only drops: arbitrarily many failures
        history = outcomes(*[(t, O.TIMEOUT if t % 2 else O.CONNECTION_FAILED) for t in range(300)])
        d = decide("URLLC", history, 300)
        self.assertIs(d.action, PolicyAction.ALERT)
        self.assertFalse(d.quarantined)
        self.assertFalse(d.escalate)  # transport failures never escalate

    def test_fail_closed_slice_quarantines_on_sustained_transport_failure(self):
        policy = SliceFailurePolicy(fail_closed=True)
        history = outcomes(*[(t, O.TIMEOUT) for t in range(6)])
        d = decide("URLLC", history, 6, policy)
        self.assertIs(d.action, PolicyAction.QUARANTINE_FLAG)
        self.assertEqual(d.rule, "transport_failures_to_quarantine_fail_closed")

    def test_an_intervening_success_resets_the_transport_run(self):
        history = outcomes((0, O.TIMEOUT), (1, O.TIMEOUT), (2, O.VERIFIED), (3, O.TIMEOUT))
        self.assertIs(decide("URLLC", history, 3).action, PolicyAction.NONE)

    def test_refusal_alerts_without_escalating(self):
        d = decide("URLLC", outcomes((0, O.SERVER_REFUSED)), 0)
        self.assertIs(d.action, PolicyAction.ALERT)
        self.assertFalse(d.escalate)

    def test_authenticated_answer_clears_quarantine(self):
        history = outcomes((0, O.REJECTED_SIGNATURE)) + [DecisionEvent(0, PolicyAction.QUARANTINE_FLAG)]
        self.assertTrue(is_quarantined(history))
        history.append(OutcomeEvent(30, O.NOT_DUE, 9))
        d = decide("URLLC", history, 30)
        self.assertEqual(d.rule, "quarantine_cleared")
        self.assertFalse(is_quarantined(history))

    def test_defaults_are_fail_open_everywhere(self):
        self.assertTrue(all(not p.fail_closed for p in DEFAULT_SLICE_POLICIES.values()))


# -- the client's retry behaviour -------------------------------------------------


class RetryTests(_Case):
    def test_hanging_server_times_out_retries_with_distinct_challenges_then_alerts(self):
        peer = self.peer(hang)
        client = self.client(peer.host, peer.port)
        supervisor = ReauthSupervisor(client)
        sup = supervisor.reauth("URLLC", 0)
        result = sup.result
        self.assertEqual(result.outcome, O.TIMEOUT)
        self.assertEqual([a.outcome for a in result.attempts], [O.TIMEOUT] * 3)
        sent = [r["client_challenge"] for r in peer.requests]
        self.assertEqual(len(sent), 3)
        self.assertEqual(len(set(sent)), 3)  # a fresh challenge on every attempt
        self.assertEqual([a.challenge for a in result.attempts], sent)
        self.assertEqual(len(self.sleeps), 2)  # backoff between attempts, not after the last
        self.assertLessEqual(self.sleeps[0], 0.25)
        self.assertLessEqual(self.sleeps[1], 0.5)
        self.assertIs(sup.decision.action, PolicyAction.ALERT)
        self.assertEqual(sup.decision.rule, "transport_failures_to_alert")
        # every attempt was logged, then the decision citing all three
        recs = self.records()
        self.assertEqual([r["record_type"] for r in recs], ["transport_failure"] * 3 + ["policy_decision"])
        self.assertEqual([r["attempt"] for r in recs[:3]], [1, 2, 3])
        self.assertEqual(recs[3]["trigger_seqs"], [0, 1, 2])
        self.assertAuditClean()

    def test_server_closing_mid_response_is_a_clean_connection_failure(self):
        peer = self.peer(close_mid_response)
        result = self.client(peer.host, peer.port).request_reauth("URLLC", 0)  # must not raise
        self.assertEqual(result.outcome, O.CONNECTION_FAILED)
        self.assertIn("TruncatedMessage", result.attempts[0].detail)
        self.assertEqual(len(result.attempts), 3)
        self.assertAuditClean()

    def test_refused_connection_is_a_clean_connection_failure(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        result = self.client("127.0.0.1", port).request_reauth("URLLC", 0)
        self.assertEqual(result.outcome, O.CONNECTION_FAILED)
        self.assertIsNone(result.trusted)

    def test_garbage_line_is_malformed_and_not_retried(self):
        peer = self.peer(send_garbage)
        result = self.client(peer.host, peer.port).request_reauth("URLLC", 0)
        self.assertEqual(result.outcome, O.MALFORMED_RESPONSE)
        self.assertEqual(len(peer.requests), 1)
        self.assertAuditClean()

    def test_forged_response_is_rejected_never_retried_and_escalated(self):
        peer = self.peer(relay_to(self.server, mutate=forge_signature))
        alerts = []
        supervisor = ReauthSupervisor(self.client(peer.host, peer.port), on_alert=lambda s, d: alerts.append(d))
        sup = supervisor.reauth("eMBB", 0)
        self.assertEqual(sup.result.outcome, O.REJECTED_SIGNATURE)
        self.assertEqual(len(sup.result.attempts), 1)  # zero retries
        self.assertIs(sup.decision.action, PolicyAction.ESCALATE_TO_DETECTOR_ALERT)
        self.assertIsNotNone(sup.escalation)  # the escalation re-auth actually went out ...
        self.assertTrue(peer.requests[1]["detector_alert"])  # ... through the detector-alert path
        self.assertEqual(len(peer.requests), 2)  # original + one escalation, nothing else
        self.assertEqual(sup.escalation.result.outcome, O.REJECTED_SIGNATURE)
        self.assertIs(sup.escalation.decision.action, PolicyAction.QUARANTINE_FLAG)
        self.assertTrue(supervisor.is_quarantined("eMBB"))
        self.assertEqual([d.action for d in alerts], [PolicyAction.ESCALATE_TO_DETECTOR_ALERT, PolicyAction.QUARANTINE_FLAG])
        self.assertAuditClean()

    def test_backoff_is_full_jitter_under_a_capped_exponential(self):
        client = self.client("127.0.0.1", 1, backoff_base=0.25, backoff_max=2.0)
        for n, ceiling in [(1, 0.25), (2, 0.5), (3, 1.0), (4, 2.0), (9, 2.0)]:
            for _ in range(50):
                self.assertTrue(0.0 <= client.backoff_delay(n) <= ceiling)


class SignedStatusTests(_Case):
    """Requirement A: "not due" and refusals are signed, and an unsigned or
    unbound one is an authentication failure."""

    def test_real_not_due_is_signed_and_accepted(self):
        client = self.client(self.server.host, self.server.port)
        client.request_reauth("URLLC", 0)
        result = client.request_reauth("URLLC", 1)
        self.assertEqual(result.outcome, O.NOT_DUE)
        self.assertTrue(result.trusted)
        self.assertAuditClean()

    def test_unsigned_not_due_is_rejected_escalated_and_not_retried(self):
        peer = self.peer(unsigned_not_due)
        supervisor = ReauthSupervisor(self.client(peer.host, peer.port))
        sup = supervisor.reauth("URLLC", 0)
        self.assertEqual(sup.result.outcome, O.REJECTED_UNSIGNED_STATUS)
        self.assertEqual(len(sup.result.attempts), 1)
        self.assertIs(sup.decision.action, PolicyAction.ESCALATE_TO_DETECTOR_ALERT)
        self.assertAuditClean()

    def test_a_captured_signed_not_due_replayed_to_a_later_request_is_rejected(self):
        client = self.client(self.server.host, self.server.port)
        client.request_reauth("URLLC", 0)
        captured, _ = client._send_request("URLLC", 1, detector_alert=False)  # a genuine signed "not due"
        self.assertEqual(json.loads(captured["payload"])["status"], "not_due")
        peer = self.peer(lambda conn, line: conn.sendall(json.dumps(captured).encode() + b"\n"))
        result = self.client(peer.host, peer.port).request_reauth("URLLC", 40)
        self.assertEqual(result.outcome, O.REJECTED_CHALLENGE_MISMATCH)
        self.assertEqual(result.category, OutcomeCategory.AUTH_FAILURE)
        self.assertAuditClean()

    def test_forged_not_due_signature_is_rejected(self):
        client = self.client(self.server.host, self.server.port)
        client.request_reauth("URLLC", 0)
        peer = self.peer(relay_to(self.server, mutate=forge_signature))
        result = self.client(peer.host, peer.port).request_reauth("URLLC", 1)
        self.assertEqual(result.outcome, O.REJECTED_SIGNATURE)
        self.assertEqual(result.status, "not_due")
        self.assertAuditClean()


class StormGuardTests(_Case):
    def test_a_failure_burst_escalates_at_most_once_per_cooldown(self):
        peer = self.peer(relay_to(self.server, mutate=forge_signature))
        supervisor = ReauthSupervisor(self.client(peer.host, peer.port))
        cooldown = supervisor.policy_for("eMBB").escalation_cooldown_s
        burst = 20
        for t in range(burst):  # one failure per logical second, for 20 s
            supervisor.reauth("eMBB", float(t))
        escalated = [e for e in supervisor.history["eMBB"] if isinstance(e, DecisionEvent) and e.escalated]
        self.assertEqual(len(escalated), 1)  # 20 s < one 30 s cooldown
        self.assertLessEqual(len(peer.requests), burst + burst // cooldown + 1)
        detector_alert_requests = [r for r in peer.requests if r["detector_alert"]]
        self.assertEqual(len(detector_alert_requests), 1)
        self.assertAuditClean()


class DualTriggerIntactTests(_Case):
    def test_periodic_reauth_still_fires_on_schedule_while_quarantined(self):
        forging = {"on": True}
        peer = self.peer(relay_to(self.server, mutate=lambda r: forge_signature(r) if forging["on"] else r))
        supervisor = ReauthSupervisor(self.client(peer.host, peer.port))
        interval = self.controller.policies["URLLC"].interval_seconds

        fired_at = []
        for now in range(0, 4 * interval + 1):
            # the live loop's own question: would a PERIODIC re-auth fire now?
            if self.controller.reauth("URLLC", float(now), dry_run=True) is None:
                continue
            fired_at.append(now)
            if now == 3 * interval:
                forging["on"] = False  # the attacker goes away
            supervisor.reauth("URLLC", float(now))
            if now < 3 * interval:
                self.assertTrue(supervisor.is_quarantined("URLLC") or now == 0)

        # periodic kept firing on the controller's own schedule throughout
        self.assertEqual(fired_at, [0, interval, 2 * interval, 3 * interval, 4 * interval])
        self.assertEqual(self.server.stats["served"] >= 5, True)
        # and the first authenticated periodic answer lifted the quarantine
        self.assertFalse(supervisor.is_quarantined("URLLC"))
        self.assertAuditClean()


class PerPeerLimitTests(unittest.TestCase):
    """Requirement B."""

    def test_one_address_cannot_take_more_than_its_share(self):
        signer = FakeSigner()
        server = ReauthServer(DualTriggerReauthController(signer=signer), max_connections_per_peer=2,
                              connection_timeout=5.0)
        server.start()
        self.addCleanup(server.stop)
        held = [socket.create_connection(("127.0.0.1", server.port)) for _ in range(2)]
        for s in held:
            self.addCleanup(s.close)
        time.sleep(0.1)
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as third:
            self.assertEqual(third.recv(10), b"")  # closed at once
        self.assertEqual(server.stats["rejected_per_peer_limit"], 1)

        # a different source address is unaffected (Linux routes all of 127/8 to lo)
        other = socket.socket()
        other.bind(("127.0.0.2", 0))
        other.settimeout(3)
        other.connect(("127.0.0.1", server.port))
        time.sleep(0.1)
        self.assertEqual(server.stats["rejected_per_peer_limit"], 1)  # 127.0.0.2 was accepted
        other.close()
        client = ReauthClient("127.0.0.1", server.port, verify_fn=FakeSigner.verify_with_public_key,
                              expected_public_key=signer.public_key)
        # this client also comes from 127.0.0.1, which is at its limit -> refused at connect level
        self.assertEqual(client.request_reauth("URLLC", 0).outcome, O.CONNECTION_FAILED)
        for s in held:
            s.close()
        time.sleep(0.2)
        self.assertEqual(client.request_reauth("URLLC", 0).outcome, O.VERIFIED)


# -- audit negatives ------------------------------------------------------------------


def rewrite(records: list[dict], path: Path, index: int, edit) -> Path:
    """Edit one record and re-chain everything, so only the record-level
    consistency checks can object."""
    records = [dict(r) for r in records]
    edit(records[index])
    prev, lines = "0" * 64, []
    for r in records:
        r["prev_hash"] = prev
        body = {k: v for k, v in r.items() if k != "record_hash"}
        r["record_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        line = json.dumps(r, sort_keys=True, separators=(",", ":"))
        lines.append(line)
        prev = hashlib.sha256(line.encode()).hexdigest()
    path.write_text("\n".join(lines) + "\n")
    return path


class AuditNegativeTests(_Case):
    def _mixed_log(self):
        """verified, not_due, a forged response (escalation + quarantine), and a timeout run."""
        client = self.client(self.server.host, self.server.port)
        sup = ReauthSupervisor(client)
        sup.reauth("URLLC", 0)
        sup.reauth("URLLC", 1)
        forger = self.peer(relay_to(self.server, mutate=forge_signature))
        ReauthSupervisor(self.client(forger.host, forger.port)).reauth("eMBB", 0)
        hung = self.peer(hang)
        ReauthSupervisor(self.client(hung.host, hung.port)).reauth("mMTC", 0)
        self.assertAuditClean()
        return self.records()

    def _fails(self, records, index, edit, expect: str):
        result = verify_log(rewrite(records, Path(self.tmp.name) / "edited.jsonl", index, edit))
        reasons = result.records[index].reasons
        self.assertFalse(result.all_clean)
        self.assertTrue(any(expect in r for r in reasons), reasons)

    def test_verified_record_carrying_a_crypto_failure_flag_fails(self):
        records = self._mixed_log()
        i = next(i for i, r in enumerate(records) if r.get("outcome") == "verified")
        self._fails(records, i, lambda r: r.update(pinned_key_mismatch=True), "pinned_key_mismatch")

    def test_rejection_relabelled_verified_fails(self):
        records = self._mixed_log()
        i = next(i for i, r in enumerate(records) if r.get("outcome") == "rejected_signature")
        self._fails(records, i, lambda r: r.update(outcome="verified", trusted=True), "signature check mismatch")

    def test_transport_failure_claiming_trust_fails(self):
        records = self._mixed_log()
        i = next(i for i, r in enumerate(records) if r.get("record_type") == "transport_failure")
        self._fails(records, i, lambda r: r.update(trusted=True), "claims trusted=True")

    def test_quarantine_justified_only_by_transport_failures_on_a_fail_open_slice_fails(self):
        records = self._mixed_log()
        i = next(i for i, r in enumerate(records) if r.get("record_type") == "policy_decision"
                 and r["slice_type"] == "mMTC")
        self._fails(records, i, lambda r: r.update(action="quarantine_flag",
                                                   rule="transport_failures_to_quarantine_fail_closed"),
                    "not fail_closed")

    def test_quarantine_citing_an_authenticated_record_fails(self):
        records = self._mixed_log()
        verified_seq = next(r["seq"] for r in records if r.get("outcome") == "verified")
        i = next(i for i, r in enumerate(records) if r.get("action") == "quarantine_flag")

        def cite_verified(r):
            r.update(trigger_seqs=[verified_seq], slice_type="URLLC", threshold=1)

        self._fails(records, i, cite_verified, "auth failures")

    def test_decision_citing_a_later_record_fails(self):
        records = self._mixed_log()
        i = next(i for i, r in enumerate(records) if r.get("record_type") == "policy_decision")
        self._fails(records, i, lambda r: r.update(trigger_seqs=[len(records) + 5]), "not an earlier record")

    def test_unsigned_status_record_carrying_a_signature_fails(self):
        peer = self.peer(unsigned_not_due)
        ReauthSupervisor(self.client(peer.host, peer.port)).reauth("URLLC", 0)
        records = self.records()
        i = next(i for i, r in enumerate(records) if r.get("record_type") == "unauthenticated_response")
        self._fails(records, i, lambda r: r.update(signature="ab" * 8), "only a signed response could produce")


if __name__ == "__main__":
    unittest.main()
