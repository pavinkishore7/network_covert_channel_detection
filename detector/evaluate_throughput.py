"""Covert throughput vs detectability for the PHY covert channel.

For each covert payload size (bits per 200-symbol frame) and attacker class, measure how
detectable the channel is at each SNR. This answers "how many bits can the attacker send
before it gets caught?", the throughput half of the Phase 1 novelty claim.

The attackers are unchanged (covert_channel/attacker.py); only AttackerConfig.n_covert_bits
varies. The adaptive attacker's per-symbol magnitude follows its own square-root-law rule
(base / sqrt(n_bits), capped at 60% of the non-adaptive magnitude), so larger payloads
automatically get quieter per symbol, which is exactly the trade-off being mapped.

Throughput in bit/s assumes eMBB numerology mu=1 (30 kHz subcarrier spacing, 0.5 ms slot of
14 OFDM symbols -> 35.7 us per symbol), so a 200-symbol frame lasts ~7.14 ms. This is a
modelling assumption for converting bits/frame to bit/s, stated in the output CSV.

Output: results/throughput_detectability.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from covert_channel.attacker import AdaptiveAttacker, AttackerConfig, NonAdaptiveAttacker
from detector.scan_detector import ScanDetector, allocation_residual, level_residual
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

SNR_LEVELS = (5, 10, 15, 20, 25, 30)
PAYLOADS = (8, 16, 32, 64, 128, 256)
SYMBOL_S = 0.5e-3 / 14            # 30 kHz SCS, normal CP
FRAME_S = 200 * SYMBOL_S          # ~7.14 ms
CALIB_SEED_BASE = 4_000_000
TEST_SEED_BASE = 5_000_000
ATTACKERS = {"non_adaptive": NonAdaptiveAttacker, "adaptive": AdaptiveAttacker}


def _scenario(seed, snr):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=float(snr), seed=seed))
    alloc = sim.allocate_slices()
    return sim.combined_interference_grid(alloc), sum(a.power for a in alloc.values()), alloc["eMBB"].subcarrier_mask


def _auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def _boot(neg, pos, rng, n=500):
    v = [_auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--calib", type=int, default=1000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for si, snr in enumerate(SNR_LEVELS):
        cal = {"level": [], "allocation": []}
        for k in range(args.calib):
            g, sch, _ = _scenario(CALIB_SEED_BASE + si * args.calib + k, snr)
            cal["level"].append(level_residual(g))
            cal["allocation"].append(allocation_residual(g, sch))
        dets = {}
        for name, maps in cal.items():
            dets[name] = ScanDetector()
            dets[name].calibrate(np.stack(maps))
        del cal
        clean = {"level": [], "allocation": []}
        base = TEST_SEED_BASE + si * args.test
        for k in range(args.test):
            g, sch, _ = _scenario(base + k, snr)
            clean["level"].append(dets["level"].score(level_residual(g)[None])[0])
            clean["allocation"].append(dets["allocation"].score(allocation_residual(g, sch)[None])[0])
        for atk_name, atk in ATTACKERS.items():
            for n_bits in PAYLOADS:
                sc = {"level": [], "allocation": []}
                mags = []
                for k in range(args.test):
                    seed = base + k
                    g, sch, mask = _scenario(seed, snr)
                    a = atk(AttackerConfig(seed=seed, n_covert_bits=n_bits)).inject(g, mask)
                    changed = a != g
                    mags.append(float(np.abs(a - g)[changed].mean()) if changed.any() else 0.0)
                    sc["level"].append(dets["level"].score(level_residual(a)[None])[0])
                    sc["allocation"].append(dets["allocation"].score(allocation_residual(a, sch)[None])[0])
                for res in ("level", "allocation"):
                    neg, pos = np.asarray(clean[res]), np.asarray(sc[res])
                    lo, hi = _boot(neg, pos, rng)
                    thr = dets[res].threshold_
                    rows.append({"snr": snr, "attack": atk_name, "residual": res, "bits_per_frame": n_bits,
                                 "throughput_bps": round(n_bits / FRAME_S, 1),
                                 "mean_abs_perturbation": round(float(np.mean(mags)), 6),
                                 "roc_auc": round(_auc(neg, pos), 6), "auc_ci_low": round(lo, 6), "auc_ci_high": round(hi, 6),
                                 "detection_rate": round(float((pos > thr).mean()), 6),
                                 "fpr": round(float((neg > thr).mean()), 6), "n_per_class": args.test})
        print(f"SNR {snr} done", flush=True)
    df = pd.DataFrame(rows)
    out = args.results_dir / "throughput_detectability.csv"
    with out.open("w") as fh:
        fh.write(f"# throughput_bps assumes 30 kHz SCS: frame = 200 symbols = {FRAME_S * 1e3:.3f} ms\n")
        df.to_csv(fh, index=False)
    print(df[df.residual == "allocation"].pivot_table(index=["attack", "bits_per_frame"], columns="snr", values="roc_auc").round(3))


if __name__ == "__main__":
    main()
