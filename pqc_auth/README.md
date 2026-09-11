# pqc_auth

Dual-trigger slice re-authentication, with an optional CRYSTALS-Dilithium
signer and a minimal real two-party transport.

## Design, layer by layer

**`reauth.py`** — the scheduling policy, independent of any crypto.
`DualTriggerReauthController.due(slice_type, now, detector_alert)` decides
whether a slice is due for re-auth: periodically (`SlicePolicy.interval_seconds`)
or immediately on a detector alert (subject to `alert_cooldown_seconds` so a
noisy detector can't trigger re-auth on every tick). `due()` never touches a
signer and is unchanged since before this file existed.

`reauth()` is the signer-aware layer on top: it calls `due()` internally
and, only when a `signer` was configured on the controller, additionally
signs a fresh per-call challenge (timestamp + slice_type + random token)
and verifies its own signature, returning a `ReauthOutcome` with the real
verification result. This proves the signing round trip works, but it is
**not** two-party authentication — the same object signs and verifies.

Two `typing.Protocol`s describe the two roles:
- `Signer` — `sign()` + `verify()`, satisfied structurally by both
  `dilithium.OqsDilithiumSigner` and `tests/fake_signer.py::FakeSigner`.
- `PublicKeyVerifier` — `(message, signature, public_key) -> bool`, a bare
  callable shape with no signer instance behind it at all. This is what a
  genuinely separate verifying party uses.

**`dilithium.py`** — the optional real backend. `OqsDilithiumSigner` wraps
liboqs (via the `oqs` Python package); if `oqs` can't be imported, its
constructor raises rather than silently falling back to weaker crypto — see
its module docstring. `verify_with_public_key(message, signature,
public_key, algorithm)` is a free function for a party that only has public
key bytes, never a `Signer`. `OqsDilithiumSigner.verify()` is now a thin
wrapper around it.

**`orchestration.py`** — `drive_reauth_from_detector_flags(anomaly_by_slice,
controller, now)` bridges a `{slice_type: bool}` anomaly mapping (e.g. from
`detector.autoencoder_detector.AutoencoderDetector.predict_anomaly()`) to
per-slice `reauth()` calls. This is the only place detector output and
re-auth decisions are connected; nothing here reimplements `due()`'s
scheduling logic.

**`transport.py`** — a real two-party channel over a local TCP socket.
`ReauthServer` holds a signing-capable `Signer` and a controller; on a
request it calls `reauth()` and sends back `(reason, nonce, signature,
public_key)` — **not** `ReauthOutcome.verified`, which never crosses the
wire at all. `ReauthClient` holds only a public key and a bare
`PublicKeyVerifier`-shaped `verify_fn`, injected by whoever constructs it —
`transport.py` never imports a concrete signer backend itself, which is
what keeps it signer-agnostic. The client independently verifies every
response; it never trusts anything the server claims about its own
verification.

Replay protection lives entirely on the client: `ReauthClient` remembers
accepted nonces for `DEFAULT_REPLAY_WINDOW_SECONDS` (300s, chosen to exceed
`mMTC`'s 300s periodic interval — the longest configured — so a legitimate
nonce is never pruned mid-cycle, while still bounding memory for a
long-running client) and rejects a repeated nonce even when its signature
is still cryptographically valid.

**`demo.py`** — `python -m pqc_auth.demo` (see below).

## What's tested vs. what requires liboqs

liboqs (the `oqs` Python package, backed by a compiled C library — see
`docs/PQC_SETUP.md`) is not assumed to be installed. Everything using
`OqsDilithiumSigner` or `dilithium.verify_with_public_key` is gated with
`pytest.importorskip("oqs")` and reports as **skipped**, not passed or
failed, when it isn't available:

- `tests/test_dilithium.py` — keypair generation, sign/verify round trip,
  tampered message, tampered signature, wrong signer's public key, and the
  explicit-public-key verify path.
- `tests/test_transport.py::DilithiumTransportTests` — the same transport
  round trip as the FakeSigner tests below, with a real signer.

**Not gated, and must always pass without liboqs** (this is the point of
`FakeSigner`): `tests/test_reauth_signer.py` (the `Signer`/`PublicKeyVerifier`
protocol conformance and reauth()'s sign+verify round trip),
`tests/test_detector_reauth_orchestration.py` (the detector→reauth glue),
and `tests/test_transport.py::FakeSignerTransportTests` — these open real
local TCP sockets and specifically prove the two-party round trip and
replay rejection work in any environment, liboqs or not.

`tests/fake_signer.py::FakeSigner` is deterministic HMAC-SHA256, explicitly
documented as test-only, and lives in `tests/` — `pqc_auth/` does not
import it. `pqc_auth/demo.py`'s fallback signer (`_DemoFakeSigner`) is a
separate, small, locally-defined stand-in for the same reason: keeping
`tests/` out of any production import path was worth a few duplicated
lines of HMAC logic rather than crossing that boundary.

Run `pytest tests/ -v -rs` to see the current pass/skip counts; check the
summary line for "skipped" explicitly rather than assuming a green run
means liboqs tests actually ran.

## Running the demo

```bash
python -m pqc_auth.demo
```

Starts a `ReauthServer` and `ReauthClient` on `127.0.0.1` (an OS-assigned
port) and prints a trace of: periodic re-auth, a detector alert firing
re-auth early with the immediate repeat suppressed by cooldown, and a
tampered signature followed by a replay of a genuine message — both
rejected independently by the client. It auto-detects `oqs` and prints
which signer backend it actually used; no setup is required either way.

## What's still not done

Stated plainly, matching this project's own habit (`docs/DECISIONS.md`,
`docs/NOVELTY.md`) of not overclaiming:

- **No key persistence across restarts.** Every `OqsDilithiumSigner` or
  `FakeSigner`/`_DemoFakeSigner` generates or holds its key material only
  in memory for the life of the process. Restart the server and it has a
  new keypair; nothing here handles key distribution to already-connected
  clients or re-establishing trust after a restart.
- **No key rotation policy.** A signer's key is fixed for its lifetime;
  there is no rotation schedule, no revocation, and no way to signal "this
  public key is no longer valid" to a client.
- **The transport is localhost-only** and has not been run over the real
  network-namespace topology being built separately in this project. It
  proves the challenge-response and replay-rejection logic work over an
  actual socket, not that they work across the slices' real network
  boundaries.
- **No production-grade connection handling.** No retries on a dropped
  connection, no reconnection logic, no timeout tuning beyond a flat
  per-call socket timeout, and no transport-layer security beyond the
  signature itself — there is no TLS-equivalent confidentiality or
  anti-tampering on the request side (which only carries `slice_type`,
  `now`, and a boolean, none of them secret), and a network-level attacker
  can still see and drop traffic even though they can't forge or replay a
  valid response.
- **No real-time integration with a running detector process** (this was
  already true of `orchestration.py` before this round of work, and still
  is): `drive_reauth_from_detector_flags()` takes anomaly flags you've
  already computed; it doesn't poll or subscribe to a detector.
- **No handling of what a client does after a rejection.** `ReauthClient`
  correctly reports `trusted=False` or `rejected_as_replay=True`, but there
  is no retry policy, alerting, or lockout behavior built on top of that
  result yet — that decision is left to the caller.
