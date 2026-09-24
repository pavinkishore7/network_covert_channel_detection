# Physics-informed multi-scale scan detector

Code: `detector/scan_detector.py`. Evaluation: `python -m detector.generate_frozen_dataset && python -m detector.evaluate_scan_detector && python scripts/make_report_figures.py` (about 10 minutes on a laptop CPU; no GPU or TensorFlow needed).

## Method
1. **Residual per cell.**
   - *Level residual* (blind): squared distance to the nearest power a clean cell can take (0, or the allocated band 0.8–1.0).
   - *Allocation-aware residual*: (observed − scheduled power)², for an observer that knows the schedule (the gNB/scheduler).
2. **Multi-scale scan.** For 16 generic window sizes (1, 2, 4, 8 symbols × 8, 16, 32, 64 subcarriers), take the maximum window-mean over every position. Standardise each scale on clean data; the score is the largest z-score.
3. **Calibration.** 2,000 fresh clean grids per SNR, threshold at the 95th percentile. The detector needs no training.

## Results
200 test scenarios per class per SNR; 95% bootstrap CIs. Full tables: `results/report/tables.md`.
CSVs are rounded to 6 decimals; reproduction is checked to ±1e-6 because some bootstrap CI bounds fall exactly on a rounding tie.
- **Non-adaptive attacker:** AUC 0.965 at 10 dB and 1.000 from 15 dB. The CNN-AE baseline gets 0.649 and 0.787.
- **Adaptive attacker:** 0.812 at 15 dB and 1.000 from 20 dB. The baseline gets 0.572 and 0.604.
- **Observing more frames** (adaptive attacker, 15 dB, allocation residual): 0.80 at 1 frame → 0.89 at 2 → 0.97 at 4 → 1.00 at 16.
- **Unseen attacker:** a logistic model trained on clean vs non-adaptive only reaches 0.74 at 15 dB and 0.99 at 20 dB on the adaptive attacker.
- **Band-limited stress test** (an attacker that stays inside the legal power band): the level residual is blind by construction (0.500). The allocation-aware residual catches it from 25 dB (0.979).
- **Floor:** below about 10 dB no variant separates the adaptive attacker at 1 frame, because the noise exceeds its ≤0.3 perturbation.

## Limits
- Simulation only, AWGN profile.
- The allocation-aware residual needs the scheduled per-cell power.
- The level residual is evaded by in-band attackers.
- Frame aggregation assumes the attacker stays active across frames.
- The CNN-AE baseline row is the region-masked energy score; it matches the trained CNN-AE at r = 0.99999999 (see docs/DECISIONS.md, 2026-09-24).

## Throughput vs detectability (`detector/evaluate_throughput.py`, `results/throughput_detectability.csv`)
Payload of 8–256 covert bits per 200-symbol frame; 200 test scenarios per class; allocation-aware scan detector. Conversion to bit/s assumes 30 kHz SCS (frame ≈ 7.14 ms, so 32 bits ≈ 4.5 kbit/s and 256 bits ≈ 35.8 kbit/s).
- **Non-adaptive attacker:** more bits means more detectable. At 10 dB, AUC is 0.61 at 8 bits and 1.00 at 64 bits.
- **Adaptive attacker at 15 dB:** detectability peaks at 32 bits (0.84) and then falls: 0.62 at 128 bits, 0.56 at 256 bits. Its per-symbol magnitude shrinks as 1/√n once the 0.3 cap stops binding, so larger payloads get quieter per symbol. This is the square-root-law trade-off made visible. At ≥ 20 dB it is caught at every payload (AUC ≥ 0.97).
- Figure: `results/report/fig_phy_throughput.png`.

## Fully-convolutional CNN (TensorFlow) (`detector/cnn_scan_detector.py`, `detector/evaluate_cnn_scan.py`)
**Architecture.** Five 2-D conv layers with BatchNorm and ReLU (3×3, dilated 3×3, 1×9, 3×3, 1×9), then a 1×1 per-cell evidence map, then log-sum-exp pooling to a single score. There is no dense layer, so the model is translation-invariant and runs on any grid size.

**Inputs.** Raw grid and |level deviation|; the allocation-aware variant adds |observed − scheduled|.

**Training.** Supervised, on 32×64 crops: the positive crop contains the burst; the negatives are a clean crop and a crop of the attacked grid away from the burst. Settings: 600 scenarios per SNR, 15 epochs, Adam 1e-3, fresh seeds (6,000,000+). Tested with TensorFlow 2.21 on CPU; about 20 minutes per model. Models are saved to `results/models/` (gitignored).

**Results** (`results/cnn_scan_results.csv`; 200 frozen test scenarios per class per SNR; threshold from 600 fresh clean grids):

| Adaptive attacker, ROC-AUC | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|
| CNN-AE (old baseline) | 0.55 | 0.57 | 0.60 | 0.66 | 0.70 |
| CNN, blind | 0.52 | 0.58 | 0.96 | 1.00 | 1.00 |
| CNN, allocation-aware | 0.52 | 0.67 | 0.99 | 1.00 | 1.00 |
| Scan, allocation-aware | 0.54 | 0.81 | 1.00 | 1.00 | 1.00 |

- **Non-adaptive attacker:** the CNN reaches AUC ≥ 0.99 from 15 dB.
- **Unseen attacker:** trained on clean vs non-adaptive only, the allocation-aware CNN scores 0.98 at 20 dB on the adaptive attacker.
- **Band-limited stress test:** the allocation-aware CNN reaches 0.97 at 30 dB and 0.72 at 25 dB. The scan detector reaches 1.00 and 0.98.
- **Limitation (payload size):** trained on 32-bit payloads only, the CNN degrades on 256-bit payloads (0.62–0.83 at 25–30 dB), while the scan detector stays at 1.00. It has learned the burst length.
- **Honest summary:** the learned model now works and is about 0.4 AUC better than the CNN-AE at 20 dB. It does not beat the physics-informed scan detector.
