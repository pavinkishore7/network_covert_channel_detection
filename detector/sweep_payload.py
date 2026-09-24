"""Payload-size sweep: does the scan detector's result depend on the attack footprint?

Question an examiner asks: "the attacker perturbs only 32 of 12,800 cells (0.25%); would
the result hold for a larger or smaller payload?" This script answers it by re-running the
scan-detector evaluation with n_covert_bits in PAYLOADS, at SNR_LEVELS, for all three
attackers, with EVERYTHING else identical to detector/evaluate_scan_detector.py:
same calibration seeds (2,000 fresh clean grids per SNR), same 200 test scenarios per SNR
(same seeds), same residuals, same scan detector. The clean scores therefore do not depend
on n, and the n = 32 rows reproduce results/scan_detector_results.csv exactly (checked at
the end of the run when that file exists).

Also recorded per row, so the result can be explained rather than only reported:
  mean_abs_perturbation   mean |attacked - clean| over the cells the attacker changed
  total_perturbation_energy  sum of (attacked - clean)^2 per grid, averaged over scenarios
Under the square-root law the adaptive attacker's per-cell magnitude falls as 1/sqrt(n),
so its total energy stays roughly constant until the 0.3 cap binds (small n).

Output: results/scan_payload_sweep.csv (floats rounded to 6 decimals).
Figure: python scripts/make_report_figures.py writes results/report/fig_phy_payload_sweep.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from covert_channel.attacker import AttackerConfig
from detector.evaluate_scan_detector import (ATTACKERS, CALIB_PER_SNR, CALIB_SEED_BASE, SCENARIOS_PER_SNR,
                                             SNR_LEVELS as ALL_SNRS, boot_ci, cp, fast_auc, residuals)
from detector.scan_detector import ScanDetector
from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig

PAYLOADS = (8, 32, 128, 512, 2048)
SNR_LEVELS = (10, 15, 20)
RESIDUALS = ("allocation", "level")


def _clean(seed: int, snr: float):
    sim = NetworkSlicingSimulator(OFDMGridConfig(snr_db=float(snr), seed=seed))
    alloc = sim.allocate_slices()
    return sim.combined_interference_grid(alloc), sum(a.power for a in alloc.values()), alloc["eMBB"].subcarrier_mask


def run(calib: int = CALIB_PER_SNR, seed: int = 2026, snrs=SNR_LEVELS, payloads=PAYLOADS,
        scenarios: int = SCENARIOS_PER_SNR) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for snr in snrs:
        si = ALL_SNRS.index(snr)  # same seed layout as evaluate_scan_detector.py
        cal = {r: [] for r in RESIDUALS}
        for k in range(calib):
            g, sch, _ = _clean(CALIB_SEED_BASE + si * calib + k, snr)
            for name, r in residuals(g, sch).items():
                cal[name].append(r)
        dets = {}
        for name in RESIDUALS:
            dets[name] = ScanDetector()
            dets[name].calibrate(np.stack(cal[name]))
        del cal

        clean_scores = {r: [] for r in RESIDUALS}
        att = {(a, n): {"scores": {r: [] for r in RESIDUALS}, "mag": [], "energy": [], "cells": []}
               for a in ATTACKERS for n in payloads}
        for k in range(scenarios):
            sseed = si * SCENARIOS_PER_SNR + k
            g, sch, mask = _clean(sseed, snr)
            for name, r in residuals(g, sch).items():
                clean_scores[name].append(dets[name].score(r[None])[0])
            for a, cls in ATTACKERS.items():
                for n in payloads:
                    x = cls(AttackerConfig(seed=sseed, n_covert_bits=n)).inject(g, mask)
                    d = x - g
                    changed = d != 0
                    cell = att[(a, n)]
                    cell["mag"].append(float(np.abs(d[changed]).mean()) if changed.any() else 0.0)
                    cell["energy"].append(float(np.square(d).sum()))
                    cell["cells"].append(int(changed.sum()))
                    for name, r in residuals(x, sch).items():
                        cell["scores"][name].append(dets[name].score(r[None])[0])

        for name in RESIDUALS:
            neg = np.asarray(clean_scores[name])
            thr = dets[name].threshold_
            fp = int((neg > thr).sum())
            for a in ATTACKERS:
                for n in payloads:
                    cell = att[(a, n)]
                    pos = np.asarray(cell["scores"][name])
                    lo, hi = boot_ci(neg, pos, rng)
                    tp = int((pos > thr).sum())
                    rows.append({"snr": snr, "residual": name, "attack": a, "n_covert_bits": n,
                                 "fraction_of_grid": n / 12800, "mean_cells_changed": float(np.mean(cell["cells"])),
                                 "mean_abs_perturbation": float(np.mean(cell["mag"])),
                                 "total_perturbation_energy": float(np.mean(cell["energy"])),
                                 "roc_auc": fast_auc(neg, pos), "auc_ci_low": lo, "auc_ci_high": hi,
                                 "fpr": fp / len(neg), "detection_rate": tp / len(pos),
                                 "det_ci_low": cp(tp, len(pos))[0], "det_ci_high": cp(tp, len(pos))[1],
                                 "n_clean": len(neg), "n_attack": len(pos)})
        print(f"SNR {snr} done", flush=True)
    return pd.DataFrame(rows)


def check_against_main(df: pd.DataFrame, main_csv: Path) -> list[str]:
    """n = 32 must reproduce the main evaluation's AUC, FPR and detection rate (CIs use a different RNG stream)."""
    if not main_csv.exists():
        return [f"{main_csv} not found; reproduction check skipped"]
    main = pd.read_csv(main_csv)
    bad = []
    for _, r in df[df.n_covert_bits == 32].iterrows():
        m = main[(main.snr == r.snr) & (main.residual == r.residual) & (main.attack == r.attack)]
        if m.empty:
            continue
        for col in ("roc_auc", "fpr", "detection_rate"):
            if abs(float(m[col].iloc[0]) - float(r[col])) > 1e-6:
                bad.append(f"snr={r.snr} {r.residual} {r.attack} {col}: main {m[col].iloc[0]} vs sweep {r[col]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m detector.sweep_payload")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--calib", type=int, default=CALIB_PER_SNR)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    df = run(args.calib, args.seed).round(6)
    out = args.results_dir / "scan_payload_sweep.csv"
    df.to_csv(out, index=False)
    view = df.pivot_table(index=["residual", "attack", "n_covert_bits"], columns="snr", values="roc_auc")
    print(view.round(3).to_string())
    problems = check_against_main(df, args.results_dir / "scan_detector_results.csv") if args.calib == CALIB_PER_SNR else ["non-default --calib; reproduction check skipped"]
    print("reproduction check (n=32 vs scan_detector_results.csv):", "PASS" if not problems else "\n  " + "\n  ".join(problems))
    print(f"wrote {out} ({len(df)} rows) in {time.perf_counter() - t0:.0f} s")
    return 0 if not problems or problems[0].endswith("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
