"""Runnable continuous detector-to-reauth loop.

    python -m pqc_auth.live_loop [--ticks N] [--interval-seconds S]

What this actually does:
  1. Trains and calibrates ONE ``detector.autoencoder_detector.AutoencoderDetector``
     (the settled ``cnn_preset`` architecture) using this project's EXISTING
     fast train+calibrate protocol -- ``detector/generate_frozen_dataset.py``'s
     ``build_dataset()``, ``detector/evaluate_structured_dae.py``'s
     ``matched_split()``/``target_masks_for_rows()``, and the same
     per-SNR-band 95th-percentile calibration
     ``detector/evaluate_cnn_autoencoder.py`` uses -- reused directly, not
     reimplemented, just run at ONE SNR band's 200 scenarios instead of all
     seven (``matched_split()`` requires exactly 200 scenarios/600 rows per
     band, so this is the smallest scale that protocol supports) and a
     handful of epochs instead of a full run. See ``train_and_calibrate()``.
  2. Streams that detector's own EXISTING held-out (matched_split "test")
     split, one window per tick, at ``--interval-seconds`` real-world
     pacing. For each window it calls the detector's OWN
     ``reconstruction_error()``/threshold comparison (the same mechanism
     ``AutoencoderDetector.predict_anomaly()`` uses, applied per-window with
     the eMBB target ``region_mask`` the calibration protocol requires) --
     see ``predict_window()``.
  3. Feeds that per-tick ``{"eMBB": bool}`` anomaly flag to
     ``pqc_auth.orchestration.drive_reauth_from_detector_flags()`` against a
     real, signer-equipped ``DualTriggerReauthController``, and -- for every
     tick that decides re-auth is due -- performs a REAL, independent
     verification round trip over an actual local TCP socket via
     ``pqc_auth.transport.ReauthServer``/``ReauthClient`` (the client pinned
     to the signer's own public key, see pqc_auth/transport.py's
     ``expected_public_key``). Every such attempt is logged to the same
     ``pqc_auth.audit_log.AuditLogger`` JSONL mechanism used elsewhere in
     this project. See ``run_live_loop()`` for exactly how the "decide" and
     "verify over the wire" steps are kept from stepping on each other.
  4. At the end, runs ``pqc_auth.audit_verify`` against the resulting log
     and prints its PASS/FAIL summary.

What this deliberately is NOT (stated plainly, matching this project's
existing habit -- pqc_auth/README.md, docs/DECISIONS.md, docs/NOVELTY.md):
  - This streams EXISTING SIMULATED / held-out windows produced by
    ``detector/generate_frozen_dataset.py``'s protocol, NOT live captured
    network traffic. Real integration with captured traffic depends on the
    separate, currently-unstarted network-layer work in this project.
  - The detector here is trained FAST, for this demo only: a few epochs on
    ONE SNR band's 200 scenarios (comfortably under a minute total,
    including the one-time TensorFlow import -- see the timing printed at
    the start of a run). This is NOT the full 3x Colab-scale training
    described in notebooks/README.md, and its detection-rate/false-alarm
    numbers, if you look at them, are demo-scale artifacts, not a claim
    about the real detector's performance -- do not quote them anywhere.
  - liboqs and TensorFlow, imported in the wrong order in the SAME
    process, crash (a native allocator conflict -- `free(): invalid
    pointer` -- confirmed by isolating it outside this module entirely).
    run_live_loop() constructs the signer BEFORE training the detector for
    exactly this reason; see the comment at that call site before
    reordering anything here.
  - Ground-truth attack labels ARE shown in the printed trace and carried
    on every trace entry (``ground_truth_anomalous``), but strictly for
    demo transparency and this module's own tests -- see
    ``predict_window()`` and ``run_live_loop()``: the actual re-auth
    trigger decision passed to ``drive_reauth_from_detector_flags()`` uses
    ONLY ``detector_predicted_anomaly``, the detector's own call, never the
    label. ``tests/test_live_loop.py`` checks this structurally.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import time
from pathlib import Path

import numpy as np

from pqc_auth.audit_verify import LogVerificationResult, format_report, verify_log
from pqc_auth.orchestration import drive_reauth_from_detector_flags
from pqc_auth.reauth import DualTriggerReauthController
from pqc_auth.transport import ReauthClient, ReauthServer

LIVE_LOOP_STATE_DIR = Path(__file__).parent / ".live_loop_state"
LIVE_LOOP_AUDIT_LOG_PATH = LIVE_LOOP_STATE_DIR / "audit_log.jsonl"

# The dataset's attacker/detector protocol only ever targets eMBB's
# subcarrier allocation (see detector/evaluate_structured_dae.py's
# target_masks_for_rows() docstring) -- URLLC/mMTC are simulated as part of
# the combined grid but are not what this detector scores, so this is the
# only slice this loop has a genuine detector verdict for.
MONITORED_SLICE = "eMBB"

TRAIN_SNR_DB = 10
TRAIN_SCENARIOS = 200  # matched_split() requires exactly 200 scenarios (600 rows) per SNR band -- see its docstring
TRAIN_EPOCHS = 3  # fast, demo-scale -- see module docstring
TRAIN_SEED = 2026

# Simulated re-auth clock advance per tick, DECOUPLED from
# --interval-seconds (which only paces real wall-clock sleeping between
# ticks). Without this, a small --interval-seconds chosen to keep the real
# run fast would also starve DualTriggerReauthController's policy clock
# (eMBB's periodic interval is 90s, its alert cooldown 10s), so the demo
# would show essentially one firing and then nothing. This keeps the
# re-auth cadence meaningful regardless of how fast you choose to watch it
# stream.
LOGICAL_TICK_SECONDS = 2.0


class _DemoLoopSigner:
    """HMAC-SHA256 stand-in, used only when oqs isn't installed. NOT real
    cryptography, NOT production use. A separate, small, locally-defined
    class for the same reason pqc_auth/demo.py has its own
    ``_DemoFakeSigner`` rather than importing tests/fake_signer.py:
    pqc_auth/ (production code) never imports anything from tests/."""

    _KEY = b"pqc-auth-live-loop-demo-only-not-a-secret-not-real-crypto"

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
        return _DemoLoopSigner(), _DemoLoopSigner.verify_with_public_key, "_DemoLoopSigner (oqs not installed -- HMAC stand-in, NOT post-quantum)"
    from pqc_auth.dilithium import OqsDilithiumSigner, verify_with_public_key

    return OqsDilithiumSigner(), verify_with_public_key, "OqsDilithiumSigner (ML-DSA-65, real liboqs)"


def train_and_calibrate(epochs: int = TRAIN_EPOCHS, seed: int = TRAIN_SEED):
    """Fast train+calibrate path -- reuses this project's EXISTING
    detector/generate_frozen_dataset.py + detector/evaluate_cnn_autoencoder.py
    protocol unchanged, just at demo scale.

    Returns ``(model, X, y, snr, test_rows, test_masks, threshold)``:
      - ``model``: a fitted+calibrated ``AutoencoderDetector.cnn_preset``.
      - ``X, y, snr``: the full generated single-band dataset (label 0/1/2 =
        clean/non_adaptive/adaptive; see generate_frozen_dataset.py).
      - ``test_rows``: row indices of matched_split()'s held-out test split
        (30 scenarios' worth, in original [clean, non_adaptive, adaptive]
        triplet order -- this is what run_live_loop() streams).
      - ``test_masks``: target_masks_for_rows(test_rows) -- the eMBB
        subcarrier mask for each of those rows, required to score them the
        same way they were calibrated.
      - ``threshold``: the single calibrated threshold (95th percentile of
        clean train+valid reconstruction error, scored over the eMBB
        region) -- the per-SNR-band calibration protocol collapses to one
        threshold here because this demo trains on exactly one SNR band,
        not because the mechanism differs from the multi-band case in
        detector/evaluate_cnn_autoencoder.py.
    """
    import detector.generate_frozen_dataset as gfd
    from detector.autoencoder_detector import AutoencoderDetector
    from detector.evaluate_structured_dae import matched_split, target_masks_for_rows

    # Reuse build_dataset() itself (not a rewritten copy of its loop) at a
    # smaller scale: one SNR band instead of all seven. Restored
    # immediately after, since this module is otherwise stateless.
    original_snr_levels, original_scenarios = gfd.SNR_LEVELS, gfd.SCENARIOS_PER_SNR
    try:
        gfd.SNR_LEVELS = np.array([TRAIN_SNR_DB])
        gfd.SCENARIOS_PER_SNR = TRAIN_SCENARIOS
        X, y, snr = gfd.build_dataset()
    finally:
        gfd.SNR_LEVELS, gfd.SCENARIOS_PER_SNR = original_snr_levels, original_scenarios

    train, valid, test = matched_split(y, snr)
    model = AutoencoderDetector.cnn_preset(X.shape[1:], seed=seed)
    model.fit(X[train][y[train] == 0], epochs=epochs, batch_size=8, verbose=0)

    calib = np.concatenate([train, valid])
    calib_masks = target_masks_for_rows(calib)
    calib_scores = model.reconstruction_error(X[calib], mode="mean", region_mask=calib_masks)
    threshold = float(np.percentile(calib_scores[y[calib] == 0], 95))

    test_masks = target_masks_for_rows(test)
    return model, X, y, snr, test, test_masks, threshold


def predict_window(model, grid: np.ndarray, mask: np.ndarray, threshold: float) -> tuple[bool, float]:
    """The detector's OWN anomaly call for one streamed window: reconstruction
    error over the eMBB target region only (matching how ``threshold`` was
    calibrated in train_and_calibrate()), compared to that threshold.
    Mathematically the single-band-collapsed equivalent of
    ``AutoencoderDetector.predict_anomaly()`` applied one window at a time
    with a region_mask, since predict_anomaly() itself has no region_mask
    parameter. Returns ``(predicted_anomaly, score)`` -- NOT told or given
    any ground-truth label."""
    score = float(model.reconstruction_error(grid[np.newaxis], mode="mean", region_mask=mask[np.newaxis])[0])
    return score > threshold, score


def run_live_loop(
    ticks: int | None = None,
    interval_seconds: float = 0.0,
    audit_log_path: str | Path = LIVE_LOOP_AUDIT_LOG_PATH,
    epochs: int = TRAIN_EPOCHS,
    quiet: bool = False,
    signer_override: tuple | None = None,
) -> tuple[list[dict], LogVerificationResult]:
    """Run the live detector-to-reauth loop for real.

    ``signer_override``, if given, is ``(signer, verify_fn, backend_label)``
    and replaces ``_make_signer()``'s auto-detection -- used by
    tests/test_live_loop.py to force the non-oqs FakeSigner path
    deterministically, matching this project's "the non-oqs path is never
    allowed to skip" test convention (see pqc_auth/README.md), independent
    of whether liboqs happens to be installed wherever tests run.

    Two DualTriggerReauthController instances share one signer and are
    driven in lockstep, on purpose (see the inline comment at the loop
    below) -- one decides (in-process, via
    drive_reauth_from_detector_flags(), exactly as
    pqc_auth/orchestration.py already does elsewhere), the other backs the
    real ReauthServer a real ReauthClient actually talks to over a real
    socket. This is what lets EVERY due re-auth get a genuine, independent,
    over-the-wire client verification without a second call to the SAME
    controller instance for the SAME (slice_type, now) silently returning
    "not due" the second time.

    Returns ``(trace, audit_report)``: ``trace`` is one dict per tick (see
    the fields built below); ``audit_report`` is
    ``pqc_auth.audit_verify.verify_log()``'s result on the log this run
    wrote to (also printed, unless quiet).
    """

    def _p(*args) -> None:
        if not quiet:
            print(*args)

    # Signer constructed BEFORE the detector is trained/calibrated -- this
    # order is load-bearing, not stylistic. liboqs (imported here when
    # available) and TensorFlow (imported inside train_and_calibrate(), via
    # detector.autoencoder_detector) both install native allocator/runtime
    # state on first import, and importing them in the OTHER order (as an
    # earlier version of this function did) reproducibly crashed the
    # process with `free(): invalid pointer` the first time oqs signed
    # anything -- confirmed as an import-order issue, not a bug in either
    # library's own logic, by reproducing it in isolation (import
    # tensorflow then oqs -> crash; import oqs then tensorflow -> fine).
    # Nothing before this module ever needed both liboqs and TensorFlow in
    # the same process (pqc_auth/demo.py never imports TensorFlow;
    # detector/*.py never imports oqs), so this is the first place that
    # conflict could show up.
    if signer_override is not None:
        signer, verify_fn, backend_label = signer_override
    else:
        signer, verify_fn, backend_label = _make_signer()
    _p(f"Signer backend: {backend_label}")

    _p("Training/calibrating AutoencoderDetector (fast demo-scale path -- see pqc_auth/live_loop.py's module docstring)...")
    t0 = time.time()
    model, X, y, snr, test_rows, test_masks, threshold = train_and_calibrate(epochs=epochs)
    train_seconds = time.time() - t0
    _p(f"  done in {train_seconds:.1f}s -- {len(test_rows)} held-out windows available, threshold={threshold:.6f}\n")

    if ticks is None:
        ticks = len(test_rows)
    ticks = min(ticks, len(test_rows))

    # Two separate controller instances sharing ONE signer, kept in
    # lockstep by construction: decision_controller.reauth() (via
    # drive_reauth_from_detector_flags) is called every tick and decides
    # whether re-auth is due; server_controller (wrapped in the real
    # ReauthServer) is only ever contacted -- over the real socket, via
    # client.request_reauth() -- on the SAME ticks, with the SAME
    # (slice_type, now, detector_alert). due()'s scheduling state
    # (_last_reauth/_last_alert) only mutates on a call that actually
    # fires, so the two controllers' histories of firing events are
    # identical at every point in the run, and server_controller
    # independently re-derives "due" (and re-signs, fresh) for real,
    # rather than trusting decision_controller's already-computed outcome.
    decision_controller = DualTriggerReauthController(signer=signer)
    server_controller = DualTriggerReauthController(signer=signer)
    server = ReauthServer(server_controller)
    server.start()
    client = ReauthClient(
        server.host,
        server.port,
        verify_fn=verify_fn,
        expected_public_key=signer.public_key,
        audit_log_path=str(audit_log_path),
    )
    _p(f"Server listening on {server.host}:{server.port}\n")

    trace: list[dict] = []
    try:
        for tick in range(ticks):
            row = int(test_rows[tick])
            grid = X[row]
            mask = test_masks[tick]
            ground_truth_anomalous = bool(y[row] != 0)  # DISPLAY/TEST ONLY -- never fed into the decision below
            now = tick * LOGICAL_TICK_SECONDS

            predicted_anomaly, score = predict_window(model, grid, mask, threshold)

            # The ONLY thing that drives the re-auth decision: the
            # detector's own prediction for this tick's slice.
            decisions = drive_reauth_from_detector_flags({MONITORED_SLICE: predicted_anomaly}, decision_controller, now)

            reauth_fired = False
            fired_reason = None
            client_result = None
            for decision in decisions:
                reauth_fired = True
                fired_reason = decision.outcome.reason.value
                client_result = client.request_reauth(decision.slice_type, now, detector_alert=predicted_anomaly)

            entry = {
                "tick": tick,
                "row": row,
                "slice_type": MONITORED_SLICE,
                "score": score,
                "threshold": threshold,
                "detector_predicted_anomaly": predicted_anomaly,
                "ground_truth_anomalous": ground_truth_anomalous,
                "reauth_fired": reauth_fired,
                "reauth_reason": fired_reason,
                "trusted": client_result.trusted if client_result is not None else None,
                "rejected_as_replay": client_result.rejected_as_replay if client_result is not None else None,
                "pinned_key_mismatch": client_result.pinned_key_mismatch if client_result is not None else None,
            }
            trace.append(entry)

            reauth_desc = f"fired({fired_reason})" if reauth_fired else "no"
            trust_desc = (
                f"trusted={client_result.trusted} replay_rejected={client_result.rejected_as_replay} "
                f"pinned_key_mismatch={client_result.pinned_key_mismatch}"
                if client_result is not None
                else ""
            )
            _p(
                f"tick {tick:>3} row={row:>3} slice={MONITORED_SLICE} "
                f"detector={'ANOMALY' if predicted_anomaly else 'clean  '} "
                f"(score={score:.4f} vs threshold={threshold:.4f})  "
                f"ground_truth={'attack' if ground_truth_anomalous else 'clean '} [shown for demo transparency only, NOT used above]  "
                f"reauth={reauth_desc}  {trust_desc}"
            )

            if interval_seconds > 0:
                time.sleep(interval_seconds)
    finally:
        server.stop()

    _p(f"\nIndependently auditing {audit_log_path} ...\n")
    report = verify_log(audit_log_path)
    _p(format_report(report))
    return trace, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticks", type=int, default=None, help="number of held-out windows to stream (default: the whole test split)")
    parser.add_argument("--interval-seconds", type=float, default=0.0, help="real wall-clock seconds to sleep between ticks (default: 0, no sleep)")
    args = parser.parse_args(argv)

    LIVE_LOOP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _, report = run_live_loop(ticks=args.ticks, interval_seconds=args.interval_seconds)
    return 0 if report.all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
