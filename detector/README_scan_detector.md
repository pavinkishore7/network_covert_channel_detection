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
