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

**Phase 1 (topology + traffic generation + capture infrastructure) is
complete and merged to `master`.** It provides `NetnsTopology`,
`traffic.py`'s per-slice packet generation, and `capture.py`'s
capture/parsing utilities — no covert channel and no detector yet.

**Phase 2 (this package's `covert_injector.py`, `timing_detector.py`, and
`covert_demo.py`) adds the timing covert channel itself and a classical
statistical detector for it.** See "Phase 2: covert channel + classical
detector" below for the design, the actual measured detection accuracy,
and what Phase 2 deliberately still does NOT do (wiring into the PQC
re-auth handshake in `pqc_auth/` — that remains a separate follow-up
phase, to avoid landing too much new surface in one change).

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

The privilege probe itself (try `ip netns add`, then delete it) now lives in
`network_covert_channel.topology.netns_privilege_skip_reason()`, shared by
every live test instead of copied into each one.

**Update (2026-09-23):** on this same WSL2 machine, unprivileged user
namespaces turned out to be enabled.
`unshare --user --map-root-user --mount --net`, plus a tmpfs on `/run`,
gives real `CAP_NET_ADMIN` over real kernel namespaces, veths, bridges and
tc qdiscs, scoped to that user namespace. `integration/auth_over_topology.py`
was run for real this way; see `integration/README.md`. The Phase 1 timing
test above was **not** re-run as part of that work.

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

## Phase 2: covert channel + classical detector

**Scope: `covert_injector.py` (the covert channel), `timing_detector.py`
(the detector), and `covert_demo.py` (end-to-end composition + live
path). Does NOT include wiring a detector alert into
`pqc_auth.reauth.DualTriggerReauthController` — that is Phase 3, see
"Phase 3 will need" below. Nothing in `pqc_auth/` or `detector/` is
touched by Phase 2.**

### The covert channel (`covert_injector.py`)

One hidden bit per packet, encoded as an additive delay on that packet's
inter-packet gap: bit=0 sends the clean gap unperturbed; bit=1 adds a
fixed offset (`offset_s`) on top of whatever `traffic.py`'s
`generate_inter_packet_gaps` already produced. `NonAdaptiveCovertInjector`
uses the same offset everywhere; `AdaptiveCovertInjector` shrinks the
offset when the slice's own local jitter (a causal rolling stdev) is
already high, capped at the non-adaptive offset as a ceiling — mirroring
`covert_channel/attacker.py`'s `NonAdaptiveAttacker`/`AdaptiveAttacker`
split. See the module docstring for the full channel-capacity-vs.
-detectability reasoning (short version: a fixed absolute offset is far
more visible on `URLLC`, whose low burstiness keeps its natural gap
spread tight, than on `mMTC`, whose high burstiness gives it much more
room to hide in).

### The detector (`timing_detector.py`)

`TimingKSDetector` — a two-sample Kolmogorov-Smirnov test
(`scipy.stats.ks_2samp`) comparing an observed inter-packet-gap sample
against a clean baseline sample for the same slice. Chosen deliberately
over the CNN autoencoder (`detector/autoencoder_detector.py`):
inter-packet-gap series are 1-D, not the 2-D image-shaped grids that
detector's Conv2D layers are built for, and using a different method here
is also a legitimate point of comparison for the report — does a
network-layer timing channel need a CNN to see, or does a much simpler
classical test already catch it? KS was chosen over chi-squared (avoids a
binning choice) and a z-score-of-variance score (would miss the
mean-shift this injector's encoding actually produces). Threshold
calibration mirrors `AutoencoderDetector.calibrate()`'s
percentile-of-clean-tail convention: `calibrate()` draws many independent
clean-vs-clean pairs to build a null distribution of the KS D statistic
and sets the threshold at its 95th percentile, so the false-positive rate
is explicit rather than assumed.

`anomaly_by_slice()` returns `dict[str, bool]` — checked (not wired in)
against `pqc_auth.orchestration.drive_reauth_from_detector_flags`'s
expected input shape via a compatibility test
(`tests/test_network_timing_detector.py::OrchestrationCompatibilityTests`)
that calls it directly with `dry_run=True`. The shapes line up with no
adaptation needed; nothing in `pqc_auth/` is imported outside that one
test, and nothing there is modified.

### Measured detection accuracy (synthetic, this session's actual run)

Produced by `python -m network_covert_channel.covert_demo` (300 packets/
trial, 30 independent trials per cell, threshold calibrated at the 95th
percentile of 100 clean-vs-clean trials) — reproducible, not hand-picked:

| Slice | False-alarm rate (clean-vs-clean) | 20ms offset (large) | 4ms offset (moderate) | 0.6ms offset (marginal) |
|-------|-----------------------------------|----------------------|--------------------------|----------------------------|
| URLLC | 6.67%  | 100.00% | 100.00% | 100.00% |
| eMBB  | 10.00% | 100.00% | 100.00% | 83.33%  |
| mMTC  | 6.67%  | 100.00% | 100.00% | 40.00%  |

**The honest finding is the marginal (0.6ms) column, not the large/
moderate ones.** At a large enough offset, the KS detector catches the
covert channel reliably on every slice — that part isn't surprising. What
*is* worth reporting is that the same small absolute offset is
meaningfully less detectable on `mMTC` (burstiness=0.8, wide natural gap
spread swallows a 0.6ms shift) than on `URLLC` (burstiness=0.2, tight
natural spread makes the same shift stand out more). This is not a fixed
property of the detector alone — it is a property of how much cover the
carrier traffic's own natural jitter provides, and it varies by slice.
`tests/test_network_timing_detector.py`'s `DetectionAccuracyTests`
encodes this as an assertion (a different, seed-swept measurement that
also confirms mMTC's marginal-offset detection rate stays below URLLC's),
not just a one-off demo run — the numbers above and the numbers in that
test file's docstring differ slightly (different seeds/trial counts) but
tell the same qualitative story.

A real clean-vs-covert gap-distribution comparison, generated from an
actual run of `covert_demo.plot_clean_vs_covert` against real synthetic
gap arrays at the 4ms offset (not mocked or fabricated):
`results/network_covert_channel_phase2_gap_distributions.png`.

### Live verification status (stated plainly, same standard as Phase 1)

**Live (real-topology) verification did NOT happen for Phase 2, same as
Phase 1.** Privilege check re-verified fresh in this session
(2026-09-22, same WSL2 dev environment):
```
$ id -u
1000
$ ip netns add __ncc_phase2_probe__
mkdir /run/netns failed: Permission denied
$ sudo -n true
sudo: a password is required
$ which tcpdump tshark ip
/usr/sbin/ip
```
No root/CAP_NET_ADMIN, no passwordless sudo, and neither `tshark` nor
`tcpdump` is on PATH. `covert_demo.run_live_demo()` and
`tests/test_network_covert_live_integration.py` are written and ready but
were not exercised against a real kernel this session — the live test
self-skips with the exact message above (via `netns_privileges_available()`)
rather than failing confusingly mid-setup, exactly like Phase 1's live
test already does. Everything reported above is from the synthetic
(in-memory) path only.

## Phase 3 will need

- Wiring a detector alert from `timing_detector.TimingKSDetector.anomaly_by_slice()`
  into `pqc_auth.reauth.DualTriggerReauthController` via
  `pqc_auth.orchestration.drive_reauth_from_detector_flags`, the same way
  `pqc_auth/live_loop.py` already wires the PHY-layer detector's alerts in.
  Phase 2 checked (see above) that the output shape is already compatible;
  Phase 3 is the actual wiring, plus deciding how this vector's alerts
  should interact with the PHY-layer vector's alerts on a slice that both
  could fire on.
- Actual live-topology verification, once run on a host with root/
  CAP_NET_ADMIN and a real Linux kernel (this project's dev environment
  still doesn't have either, per both Phase 1's and Phase 2's privilege
  checks above).
