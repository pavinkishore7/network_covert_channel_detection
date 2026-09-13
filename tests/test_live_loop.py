"""Tests for pqc_auth/live_loop.py.

Trains+calibrates the detector and streams the FULL held-out split exactly
ONCE for the whole file (setUpClass), since that's the expensive part
(TensorFlow import + a few training epochs); every test method below then
just inspects the resulting trace/audit report, which is fast. Forces the
non-oqs FakeSigner path explicitly via signer_override, matching this
project's "the non-oqs path is never allowed to skip" test convention (see
pqc_auth/README.md) regardless of whether liboqs happens to be installed
wherever this runs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pqc_auth.live_loop import run_live_loop
from tests.fake_signer import FakeSigner


class LiveLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.log_path = Path(cls._tmpdir.name) / "audit_log.jsonl"
        cls.trace, cls.report = run_live_loop(
            ticks=None,  # the whole deterministic held-out split -- guarantees anomalous windows exist
            interval_seconds=0.0,  # no artificial sleep: this must run fast, not slow
            audit_log_path=cls.log_path,
            quiet=True,
            signer_override=(FakeSigner(), FakeSigner.verify_with_public_key, "FakeSigner"),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_trace_covers_the_whole_held_out_split_and_is_nonempty(self):
        self.assertGreater(len(self.trace), 0)

    def test_at_least_one_detector_triggered_reauth_actually_fires(self):
        alert_fires = [e for e in self.trace if e["reauth_fired"] and e["reauth_reason"] == "detector_alert"]
        self.assertGreater(
            len(alert_fires), 0,
            "expected at least one genuinely detector-triggered reauth somewhere in the "
            "deterministic held-out stream -- if this ever fails, the detector's own "
            "predict_anomaly-equivalent call never flagged a single window as anomalous",
        )

    def test_a_tick_with_no_detector_anomaly_never_fires_a_detector_alert_reauth(self):
        clean_ticks = [e for e in self.trace if not e["detector_predicted_anomaly"]]
        self.assertTrue(clean_ticks, "expected at least one tick the detector did not flag")
        for entry in clean_ticks:
            self.assertNotEqual(
                entry["reauth_reason"], "detector_alert",
                f"tick {entry['tick']}: detector did not flag this window, but reauth fired as detector_alert anyway",
            )

    def test_ground_truth_label_never_leaks_into_the_reauth_trigger(self):
        # Structural check: a tick where the detector said "clean" but the
        # ground truth says "attack" must behave exactly like any other
        # detector-said-clean tick (no detector_alert reauth) -- if the loop
        # were (incorrectly) peeking at the label to decide, this is
        # precisely the case where that would show up as a spurious alert.
        leaking = [
            e for e in self.trace
            if not e["detector_predicted_anomaly"] and e["ground_truth_anomalous"] and e["reauth_reason"] == "detector_alert"
        ]
        self.assertEqual(leaking, [])

    def test_resulting_audit_log_passes_the_independent_auditor_cleanly(self):
        self.assertTrue(self.report.all_clean, [r.reasons for r in self.report.records if not r.ok])
        self.assertGreater(len(self.report.records), 0)
        # Every logged record must be backed by an actual reauth-firing tick.
        fired_ticks = sum(1 for e in self.trace if e["reauth_fired"])
        self.assertEqual(len(self.report.records), fired_ticks)

    def test_every_logged_record_reflects_a_real_independent_client_verification(self):
        fired_entries = [e for e in self.trace if e["reauth_fired"]]
        self.assertTrue(fired_entries)
        for entry in fired_entries:
            self.assertIsNotNone(entry["trusted"])
            self.assertTrue(entry["trusted"], f"tick {entry['tick']}: a real, correctly-pinned reauth should be trusted")
            self.assertFalse(entry["rejected_as_replay"])
            self.assertFalse(entry["pinned_key_mismatch"])


if __name__ == "__main__":
    unittest.main()
