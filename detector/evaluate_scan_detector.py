"""Evaluate the physics-informed multi-scale scan detector.

Outputs (all in results/):
  scan_detector_results.csv      per SNR x residual x attacker: ROC-AUC [95% bootstrap CI],
                                 FPR / detection at the calibrated threshold [Clopper-Pearson CI],
                                 detection rate at exactly 5% FPR
  scan_baseline_energy.csv       the CNN-AE baseline (see note below) on the same rows
  scan_aggregation_results.csv   adaptive attacker: AUC vs number of frames L
  scan_generalization_results.csv logistic regression trained on clean vs NON-adaptive only,
                                 tested on the unseen adaptive attacker

Protocol:
  * Calibration: CALIB_PER_SNR fresh clean grids per SNR, seeds from CALIB_SEED_BASE (disjoint
    from the frozen dataset's seeds 0..1399). The detector needs no training.
  * Test: all 200 frozen-dataset scenarios per SNR (clean vs each attacker), regenerated
    from their seeds so the scheduled power is available; the regenerated grids are checked
    against results/dataset_X.npy when that file exists.
  * Baseline: the trained CNN-AE and structured-DAE scores in results/*_errors.npz correlate
    with plain region-masked input energy at r > 0.9999999 (both autoencoders collapsed to a
    near-constant reconstruction), so the baseline row here is that region-masked energy
    score, evaluated on the same 200 scenarios. See docs/DECISIONS.md 2026-09-24.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from covert_channel.attacker import (
    AdaptiveAttacker,
    AttackerConfig,
    BandLimitedAdaptiveAttacker,
    NonAdaptiveAttacker,
)
from detector.scan_detector import ScanDetector, allocation_residual, level_residual, multiscale_features
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

SNR_LEVELS = tuple(range(0, 35, 5))
SCENARIOS_PER_SNR = 200
CALIB_PER_SNR = 2000
CALIB_SEED_BASE = 1_000_000
AGG_SEED_BASE = 2_000_000
AGG_LS = (1, 2, 4, 8, 16)
AGG_GROUPS = 100
ATTACKERS = {
    "non_adaptive": NonAdaptiveAttacker,
    "adaptive": AdaptiveAttacker,
    "band_limited_adaptive": BandLimitedAdaptiveAttacker,
}


def scenario(seed: int, snr: float, attacker: str | None):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=float(snr), seed=seed))
    alloc = sim.allocate_slices()
    scheduled = sum(a.power for a in alloc.values())
    clean = sim.combined_interference_grid(alloc)
    mask = alloc["eMBB"].subcarrier_mask
    if attacker is None:
        return clean, scheduled, mask
    return ATTACKERS[attacker](AttackerConfig(seed=seed)).inject(clean, mask), scheduled, mask


def residuals(grid, scheduled):
    return {"level": level_residual(grid), "allocation": allocation_residual(grid, scheduled)}


def fast_auc(neg: np.ndarray, pos: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def boot_ci(neg, pos, rng, n=1000):
    vals = [fast_auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cp(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(method="exact")
    return float(ci.low), float(ci.high)


def det_at_fpr(neg, pos, fpr=0.05):
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--calib", type=int, default=CALIB_PER_SNR)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.results_dir
    frozen = out / "dataset_X.npy"
    frozen_X = np.load(frozen, mmap_mode="r") if frozen.exists() else None

    # Normalisation used by the CNN-AE/DAE evaluators (AutoencoderDetector.fit locks
    # mu_/sigma_ on the clean TRAINING grids of all SNRs: scenarios 0..139 per SNR).
    train_clean = []
    for si, snr in enumerate(SNR_LEVELS):
        for k in range(140):
            train_clean.append(scenario(si * SCENARIOS_PER_SNR + k, snr, None)[0])
    train_clean = np.stack(train_clean)
    ae_mu, ae_sigma = float(train_clean.mean()), float(train_clean.std() + 1e-8)
    del train_clean

    rows, base_rows, agg_rows, gen_rows = [], [], [], []
    gen_train_F, gen_train_y, gen_test = [], [], []
    scenario_seed = 0
    for si, snr in enumerate(SNR_LEVELS):
        # --- calibration on fresh clean grids
        cal = {"level": [], "allocation": []}
        for k in range(args.calib):
            g, sch, _ = scenario(CALIB_SEED_BASE + si * args.calib + k, snr, None)
            for name, r in residuals(g, sch).items():
                cal[name].append(r)
        dets = {}
        for name in cal:
            d = ScanDetector()
            d.calibrate(np.stack(cal[name]))
            dets[name] = d
        cal_feats = multiscale_features(np.stack(cal["level"]))
        mu_f, sd_f = np.log(cal_feats + 1e-12).mean(0), np.log(cal_feats + 1e-12).std(0) + 1e-9
        del cal

        # --- test on the frozen scenarios (regenerated)
        scores = {n: {a: [] for a in ["clean", *ATTACKERS]} for n in dets}
        energy = {a: [] for a in ["clean", *ATTACKERS]}
        feats = {a: [] for a in ["clean", *ATTACKERS]}
        for k in range(SCENARIOS_PER_SNR):
            seed = scenario_seed + k
            for kind in ["clean", *ATTACKERS]:
                g, sch, mask = scenario(seed, snr, None if kind == "clean" else kind)
                if frozen_X is not None and kind in ("clean", "non_adaptive", "adaptive"):
                    row = 3 * seed + {"clean": 0, "non_adaptive": 1, "adaptive": 2}[kind]
                    if not np.array_equal(np.asarray(frozen_X[row]), g.astype(np.float32)):
                        raise SystemExit(f"regenerated grid for row {row} does not match results/dataset_X.npy; "
                                         "regenerate the frozen dataset with the current code first")
                for name, r in residuals(g, sch).items():
                    scores[name][kind].append(dets[name].score(r[None])[0])
                energy[kind].append(float(np.square((g[mask] - ae_mu) / ae_sigma).mean()))
                feats[kind].append(multiscale_features(level_residual(g)[None])[0])
        scenario_seed += SCENARIOS_PER_SNR

        for name in dets:
            neg = np.asarray(scores[name]["clean"])
            thr = dets[name].threshold_
            for atk in ATTACKERS:
                pos = np.asarray(scores[name][atk])
                lo, hi = boot_ci(neg, pos, rng)
                fp, tp = int((neg > thr).sum()), int((pos > thr).sum())
                rows.append({"snr": snr, "residual": name, "attack": atk, "n_clean": len(neg), "n_attack": len(pos),
                             "roc_auc": fast_auc(neg, pos), "auc_ci_low": lo, "auc_ci_high": hi,
                             "threshold": thr, "fpr": fp / len(neg), "fpr_ci_low": cp(fp, len(neg))[0],
                             "fpr_ci_high": cp(fp, len(neg))[1], "detection_rate": tp / len(pos),
                             "det_ci_low": cp(tp, len(pos))[0], "det_ci_high": cp(tp, len(pos))[1],
                             "detection_at_5pct_fpr": det_at_fpr(neg, pos)})
        neg = np.asarray(energy["clean"])
        for atk in ATTACKERS:
            pos = np.asarray(energy[atk])
            lo, hi = boot_ci(neg, pos, rng)
            base_rows.append({"snr": snr, "detector": "cnn_ae_equivalent_masked_energy", "attack": atk,
                              "roc_auc": fast_auc(neg, pos), "auc_ci_low": lo, "auc_ci_high": hi,
                              "detection_at_5pct_fpr": det_at_fpr(neg, pos)})

        # --- generalization features (standardised with clean calibration stats)
        z = {k: (np.log(np.asarray(v) + 1e-12) - mu_f) / sd_f for k, v in feats.items()}
        tr = slice(0, 140)
        te = slice(140, 200)
        gen_train_F += [z["clean"][tr], z["non_adaptive"][tr]]
        gen_train_y += [np.zeros(140), np.ones(140)]
        gen_test.append((snr, z["clean"][te], z["non_adaptive"][te], z["adaptive"][te], z["band_limited_adaptive"][te]))

        # --- temporal aggregation (adaptive attacker active in every one of L frames)
        maxL = max(AGG_LS)
        need = AGG_GROUPS * maxL
        agg = {"level": {"clean": [], "adaptive": []}, "allocation": {"clean": [], "adaptive": []}}
        for k in range(need):
            base = AGG_SEED_BASE + si * need * 2 + k
            for kind, seed in (("clean", base), ("adaptive", base + need)):
                g, sch, _ = scenario(seed, snr, None if kind == "clean" else "adaptive")
                for name, r in residuals(g, sch).items():
                    agg[name][kind].append(dets[name].score(r[None])[0])
        for name in agg:
            c = np.asarray(agg[name]["clean"])
            a = np.asarray(agg[name]["adaptive"])
            for L in AGG_LS:
                cg = c[: AGG_GROUPS * L].reshape(AGG_GROUPS, L).mean(1)
                ag = a[: AGG_GROUPS * L].reshape(AGG_GROUPS, L).mean(1)
                lo, hi = boot_ci(cg, ag, rng)
                agg_rows.append({"snr": snr, "residual": name, "attack": "adaptive", "frames_L": L,
                                 "groups": AGG_GROUPS, "roc_auc": fast_auc(cg, ag), "auc_ci_low": lo,
                                 "auc_ci_high": hi, "detection_at_5pct_fpr": det_at_fpr(cg, ag)})
        print(f"SNR {snr} done", flush=True)

    clf = LogisticRegression(max_iter=5000).fit(np.vstack(gen_train_F), np.concatenate(gen_train_y))
    for snr, c, n, a, b in gen_test:
        pc = clf.predict_proba(c)[:, 1]
        for name, x in (("non_adaptive (seen)", n), ("adaptive (unseen)", a), ("band_limited_adaptive (unseen)", b)):
            px = clf.predict_proba(x)[:, 1]
            lo, hi = boot_ci(pc, px, rng)
            gen_rows.append({"snr": snr, "features": "level multiscale (17)", "trained_on": "clean vs non_adaptive",
                             "tested_on": name, "n_per_class": len(pc), "roc_auc": fast_auc(pc, px),
                             "auc_ci_low": lo, "auc_ci_high": hi})

    # Rounding makes the CSVs reproducible across scipy/numpy versions (they differed by <=3.4e-13 between scipy 1.17.1 and 1.18.0).
    pd.DataFrame(rows).round(6).to_csv(out / "scan_detector_results.csv", index=False)
    pd.DataFrame(base_rows).round(6).to_csv(out / "scan_baseline_energy.csv", index=False)
    pd.DataFrame(agg_rows).round(6).to_csv(out / "scan_aggregation_results.csv", index=False)
    pd.DataFrame(gen_rows).round(6).to_csv(out / "scan_generalization_results.csv", index=False)
    print(pd.DataFrame(rows)[["snr", "residual", "attack", "roc_auc", "auc_ci_low", "auc_ci_high", "fpr",
                              "detection_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
