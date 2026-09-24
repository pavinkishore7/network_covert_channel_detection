"""Train and evaluate the fully-convolutional CNN detector (TensorFlow).

Two models, both trained on fresh seeds (disjoint from the frozen dataset and from every
calibration/test seed used elsewhere):
  * "cnn_seen":      trained on clean vs {non-adaptive, adaptive} with 32-bit payloads;
  * "cnn_nonadapt":  trained on clean vs non-adaptive ONLY (unseen-attacker test).
Each has a blind variant (grid + level residual) and an allocation-aware variant
(+ observed-minus-scheduled residual).

Evaluation (per SNR): threshold from CALIB fresh clean grids at 5% FPR; test on the 200
frozen-dataset scenarios per SNR (clean vs non-adaptive, adaptive, band-limited adaptive),
plus fresh scenarios with adaptive payloads of 8, 128 and 256 bits (never trained on).
Outputs results/cnn_scan_results.csv and saves models to results/models/ (gitignored).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from covert_channel.attacker import (AdaptiveAttacker, AttackerConfig, BandLimitedAdaptiveAttacker,
                                     NonAdaptiveAttacker)
from detector.cnn_scan_detector import build_inputs, build_model, crop_examples, logit_scores
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

SNR_LEVELS = tuple(range(0, 35, 5))
TRAIN_SEED_BASE = 6_000_000
CALIB_SEED_BASE = 7_000_000
EXTRA_SEED_BASE = 8_000_000
FROZEN_PER_SNR = 200
ATK = {"non_adaptive": NonAdaptiveAttacker, "adaptive": AdaptiveAttacker,
       "band_limited_adaptive": BandLimitedAdaptiveAttacker}


def scen(seed, snr):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=float(snr), seed=seed))
    al = sim.allocate_slices()
    return sim.combined_interference_grid(al), sum(a.power for a in al.values()), al["eMBB"].subcarrier_mask


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def boot(neg, pos, rng, n=1000):
    v = [auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def make_training_set(attackers, per_snr, allocation, rng):
    X, y = [], []
    seed = TRAIN_SEED_BASE
    for snr in SNR_LEVELS:
        for _ in range(per_snr):
            g, sch, mask = scen(seed, snr)
            name = attackers[int(rng.integers(0, len(attackers)))]
            a = ATK[name](AttackerConfig(seed=seed)).inject(g, mask)
            xs, ys = crop_examples(g, a, rng, sch if allocation else None)
            X += xs
            y += ys
            seed += 1
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--train-per-snr", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--calib", type=int, default=600)
    ap.add_argument("--extra", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    models_dir = args.results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows, hist_rows = [], []
    configs = [("cnn_seen", ["non_adaptive", "adaptive"]), ("cnn_nonadapt", ["non_adaptive"])]
    for model_name, train_atk in configs:
        for allocation in (False, True):
            tag = f"{model_name}_{'allocation' if allocation else 'blind'}"
            Xtr, ytr = make_training_set(train_atk, args.train_per_snr, allocation, np.random.default_rng(args.seed))
            model = build_model(Xtr.shape[-1], seed=args.seed)
            perm = np.random.default_rng(args.seed).permutation(len(ytr))
            h = model.fit(Xtr[perm], ytr[perm], epochs=args.epochs, batch_size=64, validation_split=0.1, verbose=2)
            for ep, (l, vl, va) in enumerate(zip(h.history["loss"], h.history["val_loss"], h.history["val_auc"]), 1):
                hist_rows.append({"model": tag, "epoch": ep, "loss": l, "val_loss": vl, "val_auc_crops": va})
            model.save(models_dir / f"{tag}.keras")
            del Xtr, ytr
            for si, snr in enumerate(SNR_LEVELS):
                cal = []
                for k in range(args.calib):
                    g, sch, _ = scen(CALIB_SEED_BASE + si * args.calib + k, snr)
                    cal.append(build_inputs(g, sch if allocation else None)[0])
                thr = float(np.quantile(logit_scores(model, np.stack(cal)), 0.95))
                del cal
                inputs = {k: [] for k in ["clean", *ATK]}
                for k in range(FROZEN_PER_SNR):
                    seed = si * FROZEN_PER_SNR + k           # frozen-dataset scenario seeds
                    g, sch, mask = scen(seed, snr)
                    s = sch if allocation else None
                    inputs["clean"].append(build_inputs(g, s)[0])
                    for name, cls in ATK.items():
                        inputs[name].append(build_inputs(cls(AttackerConfig(seed=seed)).inject(g, mask), s)[0])
                sc = {k: logit_scores(model, np.stack(v)) for k, v in inputs.items()}
                del inputs
                for name in ATK:
                    lo, hi = boot(sc["clean"], sc[name], rng)
                    rows.append({"model": tag, "trained_on": "+".join(train_atk), "snr": snr, "attack": name,
                                 "bits_per_frame": 32, "seen_in_training": name in train_atk,
                                 "roc_auc": round(auc(sc["clean"], sc[name]), 6), "auc_ci_low": round(lo, 6),
                                 "auc_ci_high": round(hi, 6), "fpr": round(float((sc["clean"] > thr).mean()), 6),
                                 "detection_rate": round(float((sc[name] > thr).mean()), 6), "n_per_class": FROZEN_PER_SNR})
                for n_bits in (8, 128, 256):
                    xc, xa = [], []
                    for k in range(args.extra):
                        seed = EXTRA_SEED_BASE + si * args.extra + k
                        g, sch, mask = scen(seed, snr)
                        s = sch if allocation else None
                        xc.append(build_inputs(g, s)[0])
                        xa.append(build_inputs(AdaptiveAttacker(AttackerConfig(seed=seed, n_covert_bits=n_bits)).inject(g, mask), s)[0])
                    c, a = logit_scores(model, np.stack(xc)), logit_scores(model, np.stack(xa))
                    lo, hi = boot(c, a, rng)
                    rows.append({"model": tag, "trained_on": "+".join(train_atk), "snr": snr, "attack": "adaptive",
                                 "bits_per_frame": n_bits, "seen_in_training": False, "roc_auc": round(auc(c, a), 6),
                                 "auc_ci_low": round(lo, 6), "auc_ci_high": round(hi, 6),
                                 "fpr": round(float((c > thr).mean()), 6), "detection_rate": round(float((a > thr).mean()), 6),
                                 "n_per_class": args.extra})
                print(f"{tag} SNR {snr} done", flush=True)
    pd.DataFrame(rows).to_csv(args.results_dir / "cnn_scan_results.csv", index=False)
    pd.DataFrame(hist_rows).round(6).to_csv(args.results_dir / "cnn_scan_training_history.csv", index=False)
    df = pd.DataFrame(rows)
    print(df[df.bits_per_frame == 32].pivot_table(index=["model", "attack"], columns="snr", values="roc_auc").round(3))


if __name__ == "__main__":
    main()
