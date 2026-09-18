# network_covert_channel

A **second, independent** covert-channel threat vector for this project,
separate from `covert_channel/`'s PHY-layer OFDM resource-grid channel.
That one perturbs a simulated interference grid in NumPy — no real network
traffic anywhere. This one hides information in the *timing* of real
packets moving through a real Linux network, to be detected later with a
classical statistical/ML method (not the CNN autoencoder — timing series
aren't image-shaped, so forcing that architecture on would be a mismatch).
The two packages are kept structurally separate on purpose: apart from
both being framed around the same three network slices, they share no code
and no threat model.

**Phase 1 scope — this package only. This is topology + traffic generation
+ capture infrastructure. It does NOT include:**
- the covert channel injection itself (encoding bits into inter-packet
  timing),
- a detector,
- wiring into the PQC re-auth handshake in `pqc_auth/`.

Those are separate follow-up phases once this lands and is reviewed —
deliberately, to avoid landing too much new surface (real namespaces, real
subprocess calls, real packet I/O) in one commit.

Note on project scope: `docs/DECISIONS.md` records a 2026-07 decision that
narrowed this project's *novelty claim* to the PHY-layer OFDM channel
specifically, because a March 2026 paper already covers PQC + AI anomaly
detection at the orchestration layer. This package adds a network-layer
timing channel as an additional, explicitly-scoped-as-such threat vector
for the same testbed — it does not change or re-open that novelty-claim
decision, and `docs/NOVELTY.md`/`docs/DECISIONS.md` were deliberately left
untouched by this phase. Anyone writing up results from this package
should keep that distinction (testbed scope vs. novelty claim) explicit.

## Privilege requirement

Building the actual topology (`NetnsTopology.setup()`/`teardown()`) needs
**root or `CAP_NET_ADMIN`**: it creates real Linux network namespaces, veth
pairs, a bridge, and `tc` qdiscs. None of that is simulated or faked — if
you don't have the privilege, the methods that call `ip`/`tc` will fail
(cleanly; see `TopologyCommandError`), not silently no-op.

**What was checked in the session that wrote this package** (2026-09-18,
WSL2 dev environment, confirmed directly, not assumed):
```
$ ip netns add __claude_probe_ns__
mkdir /run/netns failed: Permission denied
$ sudo -n true
sudo: a password is required
```
No root, no passwordless sudo. **Live integration testing has NOT happened
yet** — everything in this package was built and verified only against a
mocked/injected command runner (see `tests/test_network_topology.py`,
`tests/test_network_traffic.py`, `tests/test_network_capture.py`; all 29
of those tests pass with zero privileges and are part of the default
`pytest tests/` run). The one test that touches a real kernel
(`tests/test_network_live_integration.py`) is excluded by default via
`pytest.ini`'s `addopts = -m "not integration"` and was not run this
session — it self-skips with a clear message if it's ever run without the
required privileges, rather than failing confusingly mid-setup.

Also worth knowing if you pick this up on WSL2 specifically: even with
root, WSL2's netns/veth support is not guaranteed to match a native Linux
kernel's — that was not tested either way here. A native Linux host (or a
VM) is the safer bet for the live run below.

## Running the live demo

On a real Linux machine where you have root or `CAP_NET_ADMIN`:

```bash
sudo venv/bin/python -m pytest tests/test_network_live_integration.py -m integration -v
```

This stands up the three-namespace topology (`ns-urllc`, `ns-embb`,
`ns-mmtc`, bridged via `br-ncc0`), generates ~15 seconds of scapy-crafted
traffic on all three slices, captures each slice's host-side veth with
`tshark`, and asserts that the resulting pcaps' inter-arrival-time
distributions come out ordered URLLC (tightest/most uniform) < eMBB <
mMTC (loosest/most bursty) — the property every later phase of this
threat vector (covert channel injection, detector) depends on actually
existing. Requires `tshark` (or adjust the test to use `tcpdump`, see
`network_covert_channel/capture.py`'s `preferred_capture_tool()`).

Always call `NetnsTopology.teardown()` when done (the test does this via
`addCleanup`) — it is a best-effort delete of every namespace/interface
the config describes and is safe to call even after a partial or failed
setup.

## Module map

- **`topology.py`** — `NetnsTopology`: builds/tears down the three
  namespaces + bridge + veth pairs, and derives per-slice `tc` netem/tbf
  QoS parameters (`derive_qos_profile`) from
  `slicing_sim.ofdm_grid.SLICE_PROFILES`'s `subcarrier_frac`/`burstiness`
  — see that function's docstring for exactly how, and why each mapping
  was chosen; these are modeling assumptions, not measured values, the
  same honesty standard `SLICE_PROFILES` itself is held to. Command
  *construction* (`build_*_cmds`, pure functions returning argv lists) is
  split from *execution* (`self._run`, an injectable callable defaulting
  to `subprocess.run`) specifically so every test can assert on the
  constructed commands without needing root or a real interface.
- **`traffic.py`** — per-slice traffic generation. Uses `scapy` (not
  `iperf3`): `iperf3`'s binary wasn't available in the environment this
  was built in and needs root to install via apt, while `scapy` is a
  pure-Python package that installed cleanly via pip and gives direct
  control over per-packet send timing. `generate_inter_packet_gaps` blends
  a "steady" and a "bursty" timing process using each slice's
  `SLICE_PROFILES[...]["burstiness"]` as the mix weight — see the module
  docstring for the full reasoning; this is a modeling choice for
  producing visibly distinguishable per-slice timing, not a validated
  model of real 5G traffic statistics.
- **`capture.py`** — wraps `tshark`/`tcpdump` (whichever is present;
  `preferred_capture_tool()` checks) for live capture, plus
  `parse_pcap_to_dataframe`, which uses `scapy.rdpcap` to turn a pcap into
  a pandas DataFrame of `timestamp`, `size`, `src`, `dst`,
  `inter_arrival_s` per packet. The parser is tested against a synthetic
  pcap built and written with scapy in the test itself — no live capture
  needed to verify it.

## Phase 2 will need

- A way to inject bits into `traffic.py`'s inter-packet gaps (the covert
  encoding itself) without breaking the per-slice timing signature this
  phase established as the baseline "normal."
- A classical/statistical or ML-but-not-CNN detector operating on
  `capture.py`'s parsed timing DataFrame (this is why the DataFrame format
  was kept simple and generic rather than image-shaped).
- Eventually, wiring a detector alert from this vector into
  `pqc_auth.reauth.DualTriggerReauthController`, the same way
  `pqc_auth/live_loop.py` already wires the PHY-layer detector's alerts in.
