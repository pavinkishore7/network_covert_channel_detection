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
verification. It can optionally be pinned to one specific
`expected_public_key` — see "Public-key pinning" below.

Replay protection lives entirely on the client: `ReauthClient` remembers
accepted nonces for `DEFAULT_REPLAY_WINDOW_SECONDS` (300s, chosen to exceed
`mMTC`'s 300s periodic interval — the longest configured — so a legitimate
nonce is never pruned mid-cycle, while still bounding memory for a
long-running client) and rejects a repeated nonce even when its signature
is still cryptographically valid.

**Public-key pinning.** `ReauthClient(..., expected_public_key=...)` pins
the client to one specific identity. In `process_response()`, if the
response's `public_key` doesn't match `expected_public_key` byte-for-byte,
the response is rejected as `pinned_key_mismatch=True` — and this check
happens **before** `verify_fn` is even called. That ordering is
deliberate: a signature can be perfectly, genuinely valid and still be
signed by the wrong keypair (a second signer with its own real identity,
impersonating the expected one), so checking cryptographic validity first
would answer a question nobody asked — the point of pinning is that the
key itself is wrong, independent of whether a signature under that wrong
key would itself verify. A pinning rejection does not mark the nonce as
seen (it was never accepted) and does not call `verify_fn` at all.

What pinning solves: without it, `ReauthClient` trusts *any* public key
that arrives over the wire with a self-consistent signature — verify_fn
only ever checks "is this signature genuine under the key attached to
it", never "is this the key I actually meant to trust". Pinning closes
that gap, but **only if `expected_public_key` was itself obtained through
some trusted channel to begin with**. This project still ships **no
certificate authority or signed-key-distribution scheme of any kind**:
nothing here proves that the bytes handed to `expected_public_key=` are
the real server's key rather than an attacker's, or verifies a
certificate chain. In the demo (`pqc_auth/demo.py`), the client is pinned
to `signer.public_key` — i.e. the same process's own key, obtained
in-process, which only proves the pinning *mechanism* works; it is not a
demonstration of secure key distribution, because there is no separate,
independent channel involved. Rotating a pinned key, or re-establishing
trust after a server's identity legitimately changes, is not handled by
`expected_public_key` at all — see "What's still not done" below, and the
trust-on-first-use subsection immediately below for the one thing this
project does offer for a client that was never told the key in advance.

### Trust-on-first-use (TOFU) pinning

`ReauthClient(..., trust_store_path=..., server_id=...)` is a second,
distinct mechanism (see `pqc_auth/trust_store.py`) for a client that has
**never** been told the server's key in advance — the gap the previous
paragraph left explicitly unsolved. Instead of a caller asserting the
expected key up front, the client learns whichever key it sees on the
**first** response for a given `server_id`, persists it to a plaintext
JSON file (`{server_id: hex public key}`, same plaintext-on-disk,
demo-grade convention as `dilithium.py`'s `key_path`), and pins to that
learned key on every later response — checked before `verify_fn`, and
never marking the nonce as seen on a mismatch, for the identical reasons
as explicit pinning above. If both `expected_public_key` and
`trust_store_path`/`server_id` are given, `expected_public_key` wins
outright and the trust store isn't consulted: it's a stronger,
caller-asserted guarantee, and silently falling back to a weaker
mechanism underneath it would discard that.

**(a) What this solves that plain `expected_public_key` pinning didn't:**
there is now at least one code path for a client that starts with zero
prior knowledge of the server's key to arrive at a trusted one on its own,
the same trust model SSH uses for host keys — rather than requiring every
caller to already have the key from somewhere else.

**(b) What this explicitly does NOT solve:** an attacker already present
on the path during the very **first-ever** connection to a given
`server_id` is indistinguishable from a legitimate first contact. TOFU
turns "do I trust this key" into "is this the same key I saw last time" —
it defends against a server's identity being silently swapped out *after*
a legitimate first connection, not against a hostile first connection.
This is TOFU's own well-known limitation (SSH has the identical gap), not
a shortcut taken in this implementation.

**(c) No safe-rotation path:** a legitimate, intentional key rotation by
the server looks **identical** to an attack under this scheme — both
present a new key under an already-known `server_id`. Accepting either
requires a human to call `TrustStore.force_retrust()` explicitly; nothing
here tries to distinguish "this is probably a planned rotation" from "this
is probably an attack," and no such heuristic is planned. Certificate
authorities, revocation, and rotation-with-continuity are all explicitly
out of scope for this project, not partially-solved.

**`demo.py`** — `python -m pqc_auth.demo` (see below).

**`live_loop.py`** — `python -m pqc_auth.live_loop` (see "Running the live
detector-to-reauth loop" below): a continuous demonstration that a real
`detector.autoencoder_detector.AutoencoderDetector`'s own anomaly
predictions can drive real, independently-verified re-auth over the actual
transport, not just `pqc_auth.orchestration.drive_reauth_from_detector_flags()`
in isolation.

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
re-auth early with the immediate repeat suppressed by cooldown, a
tampered signature followed by a replay of a genuine message — both
rejected independently by the client — a public-key pinning rejection,
where a second, different signer's genuinely valid signature (valid under
its own key) is rejected purely because it isn't the pinned key, and a
separate trust-on-first-use demonstration: a client with no prior
knowledge of the server's key learns it on first contact, stays trusting
across a second connection, rejects a simulated identity change under the
same `server_id`, and accepts that new key only after an explicit
re-trust call. The TOFU section resets its own trust-store file at the
start of every run (unlike the persisted signer key / audit log) so first
contact is observable every time, not just the very first run ever. It
auto-detects `oqs` and prints which signer backend it actually used; no
setup is required either way.

## Running the live detector-to-reauth loop

```bash
python -m pqc_auth.live_loop [--ticks N] [--interval-seconds S]
```

Trains and calibrates one `AutoencoderDetector.cnn_preset` using this
project's existing fast train+calibrate protocol
(`detector/generate_frozen_dataset.py` + `detector/evaluate_structured_dae.py`'s
`matched_split()`/`target_masks_for_rows()` + the per-SNR-band calibration
`detector/evaluate_cnn_autoencoder.py` uses — reused directly, at ONE SNR
band's 200 scenarios instead of all seven and a handful of epochs, not
reimplemented), then streams that detector's own held-out test split one
window per tick. Each tick's window is scored with the detector's OWN
reconstruction-error/threshold call — never the ground-truth label — and
that verdict alone drives `pqc_auth.orchestration.drive_reauth_from_detector_flags()`
against a real, signer-equipped `DualTriggerReauthController`; every tick
that decides re-auth is due gets a real, independent verification round
trip over an actual local TCP socket via `ReauthServer`/`ReauthClient`
(pinned to the signer's own public key), logged to the same
`pqc_auth.audit_log` mechanism as everything else here. At the end it runs
`pqc_auth.audit_verify` against that log and prints its PASS/FAIL summary.

Stated plainly, matching this project's own habit of not overclaiming:

- **This streams EXISTING SIMULATED / held-out windows, not live captured
  network traffic.** The windows come from
  `detector/generate_frozen_dataset.py`'s protocol (OFDM interference
  grids with simulated attacker injections), replayed one at a time. Real
  integration with captured traffic depends on the separate,
  currently-unstarted network-layer work in this project.
- **The detector is trained FAST, for this demo only** — a few epochs on
  one SNR band's 200 scenarios (a few tens of seconds total, including the
  one-time TensorFlow import), not the full 3x Colab-scale run described
  in `notebooks/README.md`. Its detection-rate/false-alarm numbers, if you
  look at them in a run's trace, are demo-scale artifacts and must not be
  quoted as representative of the real detector's performance.
- **Ground-truth attack labels never influence the reauth decision.** They
  are printed in the trace and carried on every trace entry purely for
  demo transparency and this module's own tests
  (`tests/test_live_loop.py` checks this structurally) — the trigger fed
  to `drive_reauth_from_detector_flags()` is only ever the detector's own
  prediction.
- **liboqs and TensorFlow must be imported in a specific order in the same
  process, or the process crashes.** Confirmed by isolating it: importing
  `tensorflow` before `oqs` and then using `oqs` to sign anything crashes
  with `free(): invalid pointer` (a native allocator conflict, not a bug
  in either library's own logic); importing `oqs` first avoids it
  entirely. `run_live_loop()` constructs the signer before training the
  detector for exactly this reason — this is the first place in this
  project that needed both liboqs and TensorFlow in one process at all
  (`pqc_auth/demo.py` never imports TensorFlow; `detector/*.py` never
  imports `oqs`), so nothing before this surfaced the conflict.

## What's still not done

Stated plainly, matching this project's own habit (`docs/DECISIONS.md`,
`docs/NOVELTY.md`) of not overclaiming:

- **Key persistence exists now, but only as plaintext files, and only for
  `OqsDilithiumSigner`.** Passing `key_path` to `OqsDilithiumSigner.__init__`
  saves the exported secret key and public key to `secret_key.bin` /
  `public_key.bin` under that directory and reloads them on the next
  construction, so the same identity survives a process restart. This is
  PLAINTEXT ON DISK: no encryption at rest, no access control beyond OS
  file permissions, and no rotation -- acceptable only for this project's
  demo/review purposes, not a production key-management approach. Without
  `key_path` (the default), behavior is unchanged: a fresh in-memory-only
  keypair every construction. `FakeSigner`/`_DemoFakeSigner` were not
  touched -- `FakeSigner` already uses a fixed default HMAC key
  (`_DEFAULT_TEST_KEY`), so it was already trivially "persistent" across
  runs with no code change needed. This bullet is about the SERVER's own
  identity surviving a restart; see the trust-on-first-use subsection
  above for the separate question of how a CLIENT that was never told the
  key in advance can arrive at one.
- **No automatic key rotation policy.** A signer's key is fixed for its
  lifetime; there is no rotation schedule and no way to signal "this
  public key is no longer valid" to a client. `TrustStore.force_retrust()`
  (see the trust-on-first-use subsection above) lets an operator
  *manually* accept a new key for an already-known `server_id`, but there
  is no automatic distinction between a legitimate rotation and an
  attacker's key, and no revocation mechanism of any kind.
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
- **No key-distribution or certificate infrastructure behind pinning.**
  `expected_public_key` only compares bytes the caller already has; it does
  not obtain them, verify a certificate chain, handle first-contact trust,
  or support rotating the pinned key without restarting the client with a
  new value.
