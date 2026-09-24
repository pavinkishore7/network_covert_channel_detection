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
(the detector), `covert_demo.py` (end-to-end composition + live
path), and `sweep.py` (the detection/false-alarm evidence below). Does NOT include wiring a detector alert into
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

Thresholds are **per slice**: `calibrate(..., slice_type=s)` calibrates on
that slice's own clean gaps, `anomaly_by_slice()` judges each slice by its
own threshold, and an uncalibrated slice raises `SliceNotCalibratedError`
rather than borrowing another slice's. The two-sample KS null depends only
on the sample sizes, not on the gap distribution, so thresholds differ by
packets per window. With time-based windows the slices hold different
packet counts. At 2 s windows (141 URLLC packets vs 75 mMTC packets), a
single threshold calibrated on URLLC flags about 20% of clean mMTC windows;
mMTC's own threshold flags about 3.5%
(`tests/test_network_timing_detector.py::PerSliceThresholdTests`).

`anomaly_by_slice()` returns `dict[str, bool]` — checked (not wired in)
against `pqc_auth.orchestration.drive_reauth_from_detector_flags`'s
expected input shape via a compatibility test
(`tests/test_network_timing_detector.py::OrchestrationCompatibilityTests`)
that calls it directly with `dry_run=True`. The shapes line up with no
adaptation needed; nothing in `pqc_auth/` is imported outside that one
test, and nothing there is modified.

### Measured detection accuracy (synthetic)

Produced by `python -m network_covert_channel.sweep` and written to
`results/phase2_sweep.csv`: all 30 rows, with 95% Clopper–Pearson
intervals for every rate. Setup:

- seed 2026;
- 500 covert windows per cell and 500 fresh clean windows per slice for
  the false-alarm rate;
- the demo's 300-packet window, each window scored against one fixed clean
  baseline window per slice;
- per-slice thresholds at the 95th percentile of 1,000 clean-vs-clean
  pairs.

Offsets are multiples of each slice's own clean-gap standard deviation σ.
"Mean abs perturbation" is what the injector actually applied, averaged
over all packets (bit-0 packets add nothing). Covert bits/s is the covert
window's packet rate: one bit per packet.

| Slice | offset/σ | offset (ms) | detection, non-adaptive % [95% CI] | detection, adaptive % [95% CI] | mean abs perturbation, non-adaptive / adaptive (ms) | covert bits/s, non-adaptive / adaptive |
|---|---|---|---|---|---|---|
| URLLC | 0.1 | 1.02 | 98.8 [97.4, 99.6] | 97.0 [95.1, 98.3] | 0.51 / 0.48 | 68.0 / 68.2 |
| URLLC | 0.25 | 2.56 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 1.28 / 1.20 | 64.9 / 65.1 |
| URLLC | 0.5 | 5.12 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 2.56 / 2.39 | 59.9 / 60.6 |
| URLLC | 1 | 10.24 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 5.09 / 4.76 | 52.0 / 53.0 |
| URLLC | 2 | 20.48 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 10.25 / 9.51 | 40.9 / 42.3 |
| eMBB | 0.1 | 2.57 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 1.29 / 1.20 | 46.4 / 46.2 |
| eMBB | 0.25 | 6.43 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 3.22 / 3.00 | 42.8 / 42.8 |
| eMBB | 0.5 | 12.86 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 6.42 / 6.00 | 37.4 / 38.0 |
| eMBB | 1 | 25.73 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 12.90 / 11.99 | 30.2 / 30.9 |
| eMBB | 2 | 51.45 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 25.72 / 23.98 | 21.7 / 22.5 |
| mMTC | 0.1 | 4.11 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 2.05 / 1.91 | 35.0 / 35.4 |
| mMTC | 0.25 | 10.27 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 5.15 / 4.79 | 31.5 / 32.2 |
| mMTC | 0.5 | 20.54 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 10.19 / 9.58 | 27.2 / 27.8 |
| mMTC | 1 | 41.09 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 20.53 / 19.22 | 21.3 / 21.8 |
| mMTC | 2 | 82.18 | 100.0 [99.3, 100.0] | 100.0 [99.3, 100.0] | 41.09 / 38.22 | 14.8 / 15.5 |

| Slice | σ of clean gaps (ms) | window length (300 packets) | false-alarm rate % [95% CI] |
|---|---|---|---|
| URLLC | 10.24 | 4.26 s | 0.8 [0.2, 2.0] |
| eMBB | 25.73 | 6.18 s | 1.6 [0.7, 3.1] |
| mMTC | 41.09 | 8.00 s | 1.6 [0.7, 3.1] |

**Reading the numbers.**

- **Detection is at least 90% at the smallest offset tested, 0.1σ, on
  every slice** (URLLC 1.0 ms, eMBB 2.6 ms, mMTC 4.1 ms), for both
  injectors. The lower CI bound is also at or above 95% at 0.1σ.
- **This sweep does not locate where detection breaks down**; that lies
  below 0.1σ. The earlier 30-trial table's 0.6 ms "marginal" offset was
  about 0.015σ on mMTC, which is consistent with the 40% detection it
  showed there.
- **The adaptive injector barely lowers its perturbation.** Its mean
  absolute perturbation is 93–94% of the non-adaptive one at every offset
  and on every slice, and its detection rate is the same within the CIs.
  So explanation (a) holds: **the adaptive variant is weak**. This sweep
  does not show that the detector is robust to a genuinely adaptive
  attacker.
- **The measured false-alarm rate (0.8–1.6%) is below the nominal 5%.**
  The KS statistic is discrete (steps of 1/300), the decision uses a
  strict `>`, and every window is compared against one fixed baseline
  window. The measured rate is the one to use.

A real clean-vs-covert gap-distribution comparison, generated from an
actual run of `covert_demo.plot_clean_vs_covert` against real synthetic
gap arrays at the 4ms offset (not mocked or fabricated):
`results/network_covert_channel_phase2_gap_distributions.png`.

### Limits

- **KS tests only the marginal gap distribution.** This injector adds a
  delay to about half the gaps, which changes that distribution, so KS is
  well matched to this attacker. A distribution-preserving timing channel
  would leave the marginal distribution unchanged and would evade KS by
  construction. One example is a channel that encodes bits by reordering
  gaps, or by resampling them from the slice's own clean distribution.
  This is a known limitation of first-order statistical tests. It is **not**
  something tested here.
- **Expected spurious DETECTOR_ALERTs** = measured false-alarm rate ×
  windows per second, per slice. Window length is 300 packets × the mean
  clean gap. This is before the controller's 10 s alert cooldown and P2's
  30 s escalation cooldown; at these rates neither cooldown binds.

  | Slice | window length | windows/s | spurious alerts/s (95% CI upper) | per hour (95% CI upper) |
  |---|---|---|---|---|
  | URLLC | 4.26 s | 0.235 | 0.0019 (0.0048) | 6.8 (17.2) |
  | eMBB | 6.18 s | 0.162 | 0.0026 (0.0051) | 9.3 (18.2) |
  | mMTC | 8.00 s | 0.125 | 0.0020 (0.0039) | 7.2 (14.1) |

- **Never run on the live topology.** The rootless
  `unshare --user --map-root-user --mount --net` setup used for P1/P1.5
  does give `CAP_NET_ADMIN`: inside it, `netns_privileges_available()`
  returns True. But this machine has neither `tcpdump` nor `tshark`, and
  installing either needs root, which is unavailable (no passwordless
  sudo). `tests/test_network_covert_live_integration.py` therefore skips
  with "requires tshark or tcpdump on PATH". Every number above comes from
  the synthetic, in-memory path.

## Phase 3 will need

- Wiring a detector alert from `timing_detector.TimingKSDetector.anomaly_by_slice()`
  into `pqc_auth.reauth.DualTriggerReauthController` via
  `pqc_auth.orchestration.drive_reauth_from_detector_flags`, the same way
  `pqc_auth/live_loop.py` already wires the PHY-layer detector's alerts in.
  Phase 2 checked (see above) that the output shape is already compatible;
  Phase 3 is the actual wiring, plus deciding how this vector's alerts
  should interact with the PHY-layer vector's alerts on a slice that both
  could fire on.
- Actual live-topology verification, on a host with a capture tool
  (`tcpdump`/`tshark`); see "Limits" above. Privilege is not the blocker:
  the rootless `unshare` setup provides it.
