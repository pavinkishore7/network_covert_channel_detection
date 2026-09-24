# integration: pqc_auth over the namespace topology

`auth_over_topology.py` runs the `pqc_auth` re-auth exchange across the real
Phase 1 topology (`network_covert_channel/topology.py`). Before this, the
`ReauthServer` and `ReauthClient` had only talked over loopback inside one
process. Here the server and every client are **separate processes in
separate network namespaces**. They share nothing in memory, and the only
traffic between them crosses the bridge and each slice's tc shaping.

## Topology

| namespace  | address      | role                                                              |
|------------|--------------|-------------------------------------------------------------------|
| `ns-core`  | 10.200.0.2   | real `ReauthServer`, `OqsDilithiumSigner` with a persisted `key_path` |
| `ns-rogue` | 10.200.0.3   | adversary: rogue `ReauthServer` (own keypair) on :7000, replay responder on :7001 |
| `ns-urllc` / `ns-embb` / `ns-mmtc` | .11 / .12 / .13 | clients (Phase 1 slice namespaces, unchanged, tc-shaped) |

`topology.py` has no "core" namespace: in Phase 1 the only peer was the
bridge address in the root namespace. This module adds `ns-core` and
`ns-rogue` to the same bridge, using the same command shape, but **without**
tc shaping. They sit on the backhaul side, and the per-slice shaping stays
on the slice veths where Phase 1 put it. `topology.py` itself is unchanged
apart from `netns_privilege_skip_reason()`. That function is the existing
privilege probe from `tests/test_network_live_integration.py`, moved there
so both live tests share one check.

## How a client learns the server's key

Server and client no longer share an in-memory signer, so:

- **Explicit pin:** after the server starts and writes `keys/core/public_key.bin`,
  the orchestrator copies it to `oob/core_public_key.bin`, and clients get
  `--expected-pubkey-file`. **This file on the same disk only stands in for a
  real trusted channel** such as provisioning, a signed config, or a human
  comparing fingerprints. It exercises the client side of pinning. It is not
  key distribution.
- **TOFU:** clients get `--trust-store trust/<slice>.json --server-id core-reauth`
  and learn the key from the first verified response on the wire. The run
  checks that each learned key equals the out-of-band copy.

Every slice runs one batch of each, so the split is exactly half pinned and
half TOFU.

## Running it

```
python -m integration.auth_over_topology plan            # print every command; no privileges needed
sudo venv/bin/python -m integration.auth_over_topology run --report report.json

# without root: an unprivileged user+net namespace (real kernel netns/veth/bridge/tc,
# CAP_NET_ADMIN scoped to that user namespace), with a private /run for `ip netns`:
unshare --user --map-root-user --mount --net sh -c \
    'mount -t tmpfs tmpfs /run && venv/bin/python -m integration.auth_over_topology run --report report.json'
```

Integration test (excluded by default, skips with the probe's reason when it lacks the privileges it needs):
`pytest tests/test_auth_over_topology_integration.py -m integration -v -rs`, wrapped in `sudo` or in the
`unshare` line above.

Teardown always runs (`try/finally`). The orchestrator stops every process
it started, deletes every namespace and link the config describes
(best-effort, blind to how far setup got), then lists namespaces again and
reports anything left over.

## What was run, and where (2026-09-23)

The environment was WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2), with no
root and no passwordless sudo. `ip netns add` fails with
`mkdir /run/netns failed: Permission denied`. **Unprivileged user namespaces
are enabled**, so every run below used the `unshare` form. The namespaces,
veths, bridge and netem/tbf qdiscs are real kernel objects. The only
difference from `sudo` is that `CAP_NET_ADMIN` applies to the user
namespace, not to the host. Runs used the default 10 requests per (slice,
trust mode) batch.

### First run: wire version 1 (the auth-over-topology branch)

Legit batches and the rogue-server and link-failure outcomes were the same
as in the current run below. The replay results were not:

- **Replay, same client process:** rejected as `rejected_as_replay`.
- **Replay, fresh client process: accepted (`trusted=True`).** The request
  carried no client randomness. The server signed only its own nonce. The
  client's only replay defence was an in-memory set of nonces it had
  already accepted, so a restarted client accepted one recorded response
  as genuine. That was replay detection, not challenge-response. The
  transport-hardening branch fixed it (wire version 2, see
  `pqc_auth/transport.py`).

### Second run: wire version 2 (the transport-hardening branch)

- **Legit batches:** 60/60 trusted (10 per slice per mode). Every TOFU store
  learned exactly the out-of-band key.
- **Replay:** the replay responder in ns-rogue serves a response that a
  URLLC client captured from the real server.
  - **Same client process:** `REJECTED (challenge_mismatch)`.
  - **Fresh client process:** `REJECTED (challenge_mismatch)`. Before the
    fix, this case returned TRUSTED.
  - Each request now carries a fresh 32-byte challenge, and the server
    signs a payload that contains it. A recorded response carries the old
    challenge, so it fails the challenge check before the signature is even
    considered, and that check needs no memory of earlier nonces.
- **Rogue server** (a real `ReauthServer` with a different ML-DSA-65 keypair,
  answering a client that already trusts the real server):
  - pinned client → **`pinned_key_mismatch`**
  - TOFU client → **`trust_store_key_changed`**
  - These key checks run before the challenge check, so this path is
    unchanged from version 1.
- **Idle attacker:** ns-rogue opens 8 TCP connections to the core server and
  never sends a byte.
  - While they were held, one pinned client per slice was served and
    trusted. Those requests ran from t=+0.00 s to t=+3.54 s after the
    attacker reported its connections open.
  - The server dropped all 8 connections at t≈10.01 s after the attacker
    started. That matches its `--connection-timeout 10` in this plan (5 s
    by default), and the server log records 8 `read_timeout` events.
  - The attack uses **fewer connections than the server's cap (32)**. See
    the limitations below for what happens when an attacker fills the cap.
- **Malformed requests from ns-rogue** (9 cases). Each got an error response
  or a closed connection. The core server process was still running
  afterwards, and one request per slice was then served and trusted. The
  server log counts one event each for:
  - `invalid_utf8`
  - `invalid_json`
  - `invalid_request` (a JSON array)
  - `unsupported_version` (a v1 request)
  - `unknown_slice_type`
  - `invalid_now` (NaN)
  - `invalid_client_challenge` (4 bytes)
  - `truncated_request` (peer half-closes mid-line; closed without a reply)
  - `request_too_large` (5000 bytes, no newline). This one reached the
    prober as `ConnectionResetError`. The server stops reading at the cap
    and closes with unread bytes still queued, so the kernel sends a RST,
    and that can arrive before the error line.
  - The log shows no `internal_error`.
- **Link failure mid-run:** unchanged from version 1. `blackhole` gives
  `TimeoutError` after ~5.0 s per faulted request, `link_down` gives an
  immediate `OSError` errno 101, and the client recovers on the next
  request after restore. The client still has no retry policy.
- `pqc_auth.audit_verify` on the shared client log: **79 records checked, 0
  failures, OVERALL: PASS**. That includes the challenge-mismatch records,
  whose replayed signatures are genuinely valid. The audit branch that
  handles them is explained in `pqc_auth/audit_verify.py`.
- **Teardown:** no namespaces were left inside the user namespace.
  Afterwards on the host, `ip netns list` was empty, no bridge or veth
  remained, and no `pqc_auth.transport` or orchestrator processes were left.
  The integration test (`-m integration`) also passed in the same setup.

### Current run: signed status answers, per-IP limit, failure policy (the auth-failure-policy branch)

Same topology and user-namespace setup, same plan, with these changes:

- **Legit batches:** 60/60 trusted, and every TOFU store matched the
  out-of-band key.
- **Replay, same process and fresh process:** both
  `REJECTED (rejected_challenge_mismatch)`, as in the second run. The
  failure policy now acts on them. Each rejection → `ESCALATE_TO_DETECTOR_ALERT`,
  then one detector-alert re-auth, which reached the same replaying
  endpoint and was rejected again → `QUARANTINE_FLAG` (2 authentication
  failures in 300 s, further escalation suppressed by the cooldown).
- **Rogue server:** the rejection paths are unchanged (`rejected_pinned_key`
  and `rejected_tofu_key_changed`), and each now gets the same
  escalate-then-quarantine treatment.
- **Idle attacker, 40 connections from ns-rogue.** That is more than the
  server's global cap of 32.
  - The per-IP limit (4) held.
  - Exactly 4 connections were kept open until the server's 10 s read
    timeout (4 `read_timeout` events).
  - The other 36 were closed by the server on accept (36
    `rejected_per_peer_limit`, 0 `rejected_at_capacity`). The attacker
    saw those closes at its first poll, t≈1.02 s after it started, i.e.
    once it had finished opening all 40.
  - URLLC, eMBB and mMTC were each served and trusted from t=+0.00 s to
    t=+3.13 s after the attacker's READY.
  - Without the per-IP limit, 40 connections from one address would have
    filled all 32 slots.
- **Malformed probe:** the server survived, and each slice was served
  afterwards. The two refusals whose request still carried a usable
  challenge are now **signed**: `unknown_slice_type (signed)` and
  `invalid_now (signed)`. The rest are unsigned by design, since there is
  no challenge to bind them to.
- **Link failure, blackhole (100% loss both ways).** Each faulted request
  made **3 attempts**, each `timeout after 3004 ms`, which is the new 3.0 s
  connect timeout. The attempts were separated by jittered backoff, and
  each used a fresh connection and challenge. Then
  `POLICY ALERT (transport_failures_to_alert): 3 consecutive transport failures
  (threshold 3; fail-open: continuing on the last verified session)`.
  The second faulted request reported 6 consecutive failures. The client
  recovered on the first request after the fault was removed. Nothing was
  quarantined, because the slice is fail-open.
- **Link failure, link down.** Each faulted request made 3 attempts, each
  `connection_failed` with `OSError: [Errno 101] Network is unreachable`
  within ~1 ms, then the same ALERT, then recovery.
- **Audit:** `audit_verify` on the client log gave **107 records checked,
  0 failures, OVERALL: PASS**. Those records include every retry attempt
  (as `transport_failure` records) and every policy decision (each citing
  its evidence).
- **Integration test and leak check:** the integration test passed. On
  the host afterwards there were no namespaces, no bridge or veth, and no
  leftover processes.

### Latency: what these numbers are and are not

**Server-side signing cost, measured directly.** `time.perf_counter()`
brackets each `signer.sign()` and `signer.verify()`: inside
`DualTriggerReauthController.reauth()` for a "re-auth performed" answer,
and in `ReauthServer._sign` for a "not due" one. The server logs both.

- **Algorithm:** ML-DSA-65 via liboqs 0.16.0 / liboqs-python 0.16.0.
- **Machine:** 12th Gen Intel Core i5-1235U (12 logical CPUs), WSL2
  kernel 6.6.87.2-microsoft-standard-WSL2, Python 3.12.3, inside the
  rootless user namespace.
- **Sample:** N = 75 signed responses in the current run.

| operation | N  | min      | median   | p95      | max      |
|-----------|----|----------|----------|----------|----------|
| sign      | 75 | 0.145 ms | 0.398 ms | 0.916 ms | 1.365 ms |
| verify    | 75 | 0.060 ms | 0.131 ms | 0.236 ms | 0.345 ms |

Quote these as, e.g., "ML-DSA-65 sign: median 0.40 ms (N=75) on an Intel
i5-1235U under WSL2". The second run measured median sign 0.47 ms and
verify 0.17 ms, also with N=75, on the same machine. The client's own
verify is the same operation but is not separately timed.

**RTT per request**, current run, legit batches only. Measured by the
client process from connect to verified result, for the attempt that got
the answer. liboqs is imported before timing starts.

| slice | n  | min      | median   | p95      | max      |
|-------|----|----------|----------|----------|----------|
| URLLC | 20 | 8.87 ms  | 10.72 ms | 12.77 ms | 14.89 ms |
| eMBB  | 20 | 19.02 ms | 21.43 ms | 30.54 ms | 42.34 ms |
| mMTC  | 20 | 28.55 ms | 32.59 ms | 47.21 ms | 63.34 ms |

**This is veth-in-a-VM latency: signature + transport overhead on a software
topology.** It is not radio latency, not 5G latency, and it must **not** be
read against the 3GPP TS 22.261 URLLC targets as if it showed compliance or
non-compliance.

Most of each RTT is the netem delay that `derive_qos_profile()`
*configures*, a modeling choice rather than a measurement. That delay is
applied on the host-side veth, so it hits both the SYN-ACK and the
response. Signing and verifying are measured above at under 1 ms at the
median. The rest is Python, process and TCP overhead on this machine, and
it is not broken down further.

Earlier runs had long tails: 1090 ms (mMTC) in the first run, 451 ms
(eMBB) in the second. They are consistent with TCP retransmission timers
under each slice's configured netem loss. That explanation is inferred,
not captured.

## Transport behaviour now

**Wire format (version 2).**

- Requests carry `v: 2` and a 32-byte `client_challenge`.
- **Every** answer to a request with a usable challenge is signed. That
  covers `status` `due`, `not_due` and `error`.
- The signature covers `b"pqc_auth.reauth.v2\x00" + payload`.
- The payload keys are the same for every status: `v`, `status`,
  `server_id`, `slice_type`, `reason`, `error_code`, `client_challenge`,
  `server_nonce`, `request_now`, `issued_at`.
- Only requests with no usable challenge get an unsigned refusal.
- There is no v1 compatibility path.

**Client.**

- Connect timeout 3.0 s, and a whole-response read deadline of 5.0 s.
- Up to 3 attempts, for transport failures only.
- Full-jitter backoff, with a fresh connection and a fresh challenge on
  every attempt.
- Every attempt is audit-logged.
- Checks, in order: structure, pin, TOFU key, **challenge**, seen nonce,
  signature.
- Every request ends in one `RequestOutcome`, and none of them raises. See
  pqc_auth/README.md for the outcome and policy tables.

**Server.**

- One thread per connection.
- 4 concurrent connections per source IP and 32 in total. Beyond either
  cap, a connection is closed at once and logged.
- Per-connection read timeout, 1024-byte request cap, full validation.
- The controller's scheduling decision runs under a lock.
- `SERVER_THREAD_DIED` (CLI) fires only if the accept loop itself dies.

**Still not addressed.**

- **Many addresses:** the per-IP limit is per IP on TCP. An attacker with
  8 or more addresses can still fill every slot, for up to the read
  timeout per held connection.
- **Unauthenticated requests:** requests are not authenticated. An on-path
  attacker can still drop traffic. That is now *detected* (TIMEOUT /
  CONNECTION_FAILED → ALERT), but it is not *prevented*.
- **Server state on a client timeout:** a client that times out may have
  had its request processed, and schedule state committed, by the server.
  With retries, the next attempt would then get a signed "not due" (the
  server already fired), which is authenticated and ends the request
  normally. That follows from the code; it was not observed in these
  runs.
