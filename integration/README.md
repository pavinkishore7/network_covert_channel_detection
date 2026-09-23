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

### Current run: wire version 2 (the transport-hardening branch)

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

### Latency: what these numbers are and are not

**Server-side signing cost, measured directly.** `DualTriggerReauthController.reauth()`
brackets `signer.sign()` and `signer.verify()` with `time.perf_counter()`
and the server logs both. ML-DSA-65 (liboqs), 75 due responses in this run:

| operation | min      | median   | p95      | max      |
|-----------|----------|----------|----------|----------|
| sign      | 0.187 ms | 0.471 ms | 0.942 ms | 1.722 ms |
| verify    | 0.067 ms | 0.166 ms | 0.263 ms | 0.360 ms |

The client's verify is the same operation, but it is not separately timed.

**RTT per request**, measured by the client process from connect to verified
result (TCP connect + request + server sign + response + client verify).
liboqs is imported before timing starts. Legit batches only:

| slice | n  | min      | median   | p95      | max       |
|-------|----|----------|----------|----------|-----------|
| URLLC | 20 | 10.36 ms | 11.84 ms | 13.24 ms | 16.03 ms  |
| eMBB  | 20 | 19.01 ms | 22.18 ms | 43.88 ms | 450.93 ms |
| mMTC  | 20 | 27.95 ms | 32.69 ms | 50.70 ms | 256.73 ms |

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

The long tails (e.g. eMBB 450 ms, mMTC 257 ms here; one mMTC request took
1090 ms in the version-1 run) are consistent with TCP retransmission timers
under each slice's configured netem loss (0.75% eMBB, 1.2% mMTC). That
explanation is inferred, not captured.

## Transport behaviour now (wire version 2)

**Wire format.** Requests carry `v: 2` and a 32-byte `client_challenge`. A
due response carries `payload`, the exact canonical-JSON string that was
signed, containing:

- `v`
- `server_id`
- `slice_type`
- `reason`
- `client_challenge`
- `server_nonce`
- `request_now`
- `issued_at`

The signature covers `b"pqc_auth.reauth.v2\x00" + payload`. Version-1
requests are refused (`unsupported_version`) and version-1 responses are
rejected (`malformed_response`). There is no compatibility path.

**Client check order:**

1. Structure/version
2. Explicit pin
3. TOFU key
4. **Challenge** (constant-time compare)
5. Seen server nonce (defence in depth only)
6. Signature

**Server:**

- **Connections:** one thread per connection, capped at 32 by default.
  Beyond the cap, a new connection is closed at once and logged as
  `rejected_at_capacity`.
- **Timeouts:** each connection has a read timeout (5 s by default).
- **Scheduling:** the controller's scheduling decision is made under a lock,
  so two simultaneous requests for one slice fire once.
- **Request limits:** requests are capped at 1024 bytes and fully validated
  before use.
- **`SERVER_THREAD_DIED` (CLI):** now fires only if the accept loop itself
  dies while the server should be running, e.g. `accept()` failing with
  EMFILE, or a bug in that loop. Per-connection work cannot reach it.

**Client failure modes** (still no retries):

- Connection refused → `ConnectionRefusedError`
- Peer closes before sending anything → `ConnectionError`
- Peer closes mid-line → `TruncatedMessage` (a `ConnectionError`)
- Response over 64 KB → `MessageTooLarge`
- Silence → `TimeoutError`
- Server refusal → `ReauthRequestError(code)`

**Still not addressed:**

- **Cap exhaustion:** an attacker holding **all** connection slots (32 by
  default) gets every further connection closed immediately, legitimate
  clients included, for up to the read timeout per held connection. There
  is no per-peer limit.
- **Unauthenticated requests and unsigned "not due" answers:** an on-path
  attacker can still suppress a re-auth by answering `{"due": false}`, or
  can drop traffic. It can no longer make a client accept a stale response.
- **Server state on a client timeout:** a client that times out may have
  had its request processed, and its schedule state committed, by the
  server. The client never sees the result.
