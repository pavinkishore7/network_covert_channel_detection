# Cross-Slice Covert Channel Detection with Post-Quantum Authentication

UG Project — EC22711, Phase I. Department of ECE, SVCE Chennai.
Supervisor: Mrs. Stella Mercy M.

**Team**
| Name | Role | Reg No |
|---|---|---|
| Pavin Kishore N | Attack & Detection Lead | 2127230701098 |
| Poojasree V | Authentication Lead | 2127230701100 |
| Priyadharshini R | Cryptography Lead | 2127230701109 |

## What this is
Detects OFDM subcarrier-interference-based covert channels crossing
network slice boundaries, using a CNN Autoencoder research detector and a
clean-calibrated adaptive residual benchmark guard, backed by an optional
CRYSTALS-Dilithium post-quantum slice authentication with a dual-trigger
re-authentication scheme (periodic slice-aware timer + detector-triggered).

Scope note: this targets the **physical-layer** covert channel gap, not the
orchestration-layer AI/PQC work already published in 2026. See
`docs/NOVELTY.md` for why that distinction matters — cite it if a reviewer
asks "isn't this already done."

## Repo structure
```
slicing_sim/      Network slicing + OFDM resource allocation simulation
covert_channel/   NonAdaptiveAttacker (baseline) + AdaptiveAttacker (sqrt-law-bounded, shaped)
detector/         CNN Autoencoder anomaly detector (TensorFlow/Keras) — see docs/DECISIONS.md
pqc_auth/         Optional CRYSTALS-Dilithium adapter + dual-trigger re-auth policy
dashboard/        Streamlit app — visualizes sim/attacker output live
monitoring/       Prometheus exporter + Grafana provisioning (untested end-to-end, see below)
results/          Output plots, CSVs, benchmark numbers — versioned, not overwritten
docs/             Design decisions, setup guides, meeting notes
tests/            pytest unit tests — 4 files, 14 passing tests
```

## What's actually implemented vs. still a stub
- **Working, smoke-tested:** `slicing_sim/ofdm_grid.py`, `covert_channel/attacker.py`
  (both non-adaptive and adaptive), `dashboard/app.py`.
- **Trained and evaluated:** `CNNAutoencoderDetector` (`detector/cnn_autoencoder.py`)
  and `StructuredDAE` (`detector/dae_autoencoder.py`) have both been trained on the
  frozen dataset (`detector/generate_frozen_dataset.py`, 4200 rows spanning 7 SNR
  levels x 3 scenario classes) and evaluated across all 7 SNR levels x 2 attacker
  types (non-adaptive, adaptive) with region-masked reconstruction-error scoring
  and per-SNR calibrated thresholds. Results: `results/cnn_autoencoder_results.csv`,
  `results/structured_dae_results.csv`.
- **Written but NOT run end-to-end:** `docker-compose.yml`, `monitoring/exporter.py`
  inside Docker, Grafana provisioning. No Docker available in the environment
  this was built in — test locally before relying on it for a demo.
- **Policy implemented, crypto runtime optional:** `pqc_auth/` includes the
  dual-trigger policy and an `oqs` Dilithium adapter. It requires liboqs before
  signing can run.
- **Benchmark:** `python -m detector.run_adaptive_benchmark` creates held-out
  multi-impairment simulation results in `results/adaptive_benchmark_v1.csv`.
- **PHY slicing benchmark:** `python -m slicing_sim.run_mixed_numerology_benchmark`
  reports guard-band and cancellation effects on ISBI, SINR and QPSK BER.
- **CNN-family benchmark:** `python -m detector.run_convolutional_autoencoder_benchmark`
  runs the CPU-compatible linear convolutional patch autoencoder. The TensorFlow
  CNN remains a separate architecture requiring its runtime dependency.

## Running the dashboard locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Running the full monitoring stack (Docker, untested by me — verify it works)
```bash
docker-compose up --build
# Streamlit:   http://localhost:8501
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000  (anonymous viewer access enabled)
```

## Setup
```bash
git clone <this repo url>
cd 5g-covert-channel-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Before installing `liboqs-python`, read `docs/PQC_SETUP.md`.** It will not
work with a bare `pip install` — there's a C library build step first.

## Workflow rules (read before your first commit)
1. **Never push directly to `main`.** Branch per feature: `git checkout -b detector/cnn-lstm-v1`
2. **One PR per logical change**, not one giant PR at the end of the week.
3. **Commit messages describe what changed, not "update"**: `fix: correct SNR scaling in OFDM sim`
4. **Results go in `results/`, named with a version**: `results/detection_accuracy_v1.csv`, not overwritten in place.
5. **Design decisions go in `docs/DECISIONS.md`**, not buried in a chat thread. If you argued about something for more than 10 minutes, write down what you decided and why.

## Known open issues (be honest about these when presenting)
- Adaptive-detector metrics are simulation-only and must be reported with their
  configured false-positive rate; they do not establish field robustness.
- Square-root law covert-channel bound is derived under **AWGN**; real multipath 5G channels aren't validated yet.
- Phase 1 = design + simulation only. Empirical numbers (see
  `results/cnn_autoencoder_results.csv` / `results/structured_dae_results.csv`) are
  simulation-only, not field-validated: ROC-AUC against the non-adaptive attacker
  climbs from ~0.53 at 0dB to ~0.99 at 30dB, while against the adaptive attacker it
  stays much lower, ~0.52 at 0dB to ~0.74 at 30dB — the headline finding of Phase 1.
