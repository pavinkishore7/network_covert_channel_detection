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
network slice boundaries, using a CNN+LSTM anomaly detector, backed by
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
pqc_auth/         CRYSTALS-Dilithium signing + dual-trigger re-auth logic (not yet implemented)
dashboard/        Streamlit app — visualizes sim/attacker output live
monitoring/       Prometheus exporter + Grafana provisioning (untested end-to-end, see below)
results/          Output plots, CSVs, benchmark numbers — versioned, not overwritten
docs/             Design decisions, setup guides, meeting notes
tests/            pytest unit tests (not yet written)
```

## What's actually implemented vs. still a stub
- **Working, smoke-tested:** `slicing_sim/ofdm_grid.py`, `covert_channel/attacker.py`
  (both non-adaptive and adaptive), `detector/cnn_autoencoder.py` (architecture
  runs end-to-end; not trained on real data volume yet), `dashboard/app.py`.
- **Written but NOT run end-to-end:** `docker-compose.yml`, `monitoring/exporter.py`
  inside Docker, Grafana provisioning. No Docker available in the environment
  this was built in — test locally before relying on it for a demo.
- **Not started:** `pqc_auth/` (CRYSTALS-Dilithium signing, dual-trigger re-auth
  logic), `tests/`.

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
- CNN autoencoder detection assumes a **non-adaptive attacker** — evasion under the adaptive/interference-shaping attacker model is unresolved.
- Square-root law covert-channel bound is derived under **AWGN**; real multipath 5G channels aren't validated yet.
- Phase 1 = design + simulation only. No empirical detection-accuracy numbers exist yet — don't present projected numbers as measured.
