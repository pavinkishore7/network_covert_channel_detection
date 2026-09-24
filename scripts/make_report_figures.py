"""Regenerate the PHY-detector report figures from the CSVs in results/.

Reads only results/scan_*.csv (written by detector/evaluate_scan_detector.py and
detector/sweep_payload.py) and
writes 300-dpi PNGs to results/report/. Styles are distinguishable in grayscale
(line style + marker), and every AUC point carries its 95% bootstrap CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results")
OUT = R / "report"


def _err(df):
    return [df.roc_auc - df.auc_ci_low, df.auc_ci_high - df.roc_auc]


def fig_auc_vs_snr():
    scan = pd.read_csv(R / "scan_detector_results.csv")
    base = pd.read_csv(R / "scan_baseline_energy.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        (base[base.attack == "non_adaptive"], "CNN-AE baseline, non-adaptive", "k", "--", "o"),
        (base[base.attack == "adaptive"], "CNN-AE baseline, adaptive", "0.5", "--", "s"),
        (scan[(scan.residual == "allocation") & (scan.attack == "non_adaptive")], "Scan detector, non-adaptive", "k", "-", "o"),
        (scan[(scan.residual == "allocation") & (scan.attack == "adaptive")], "Scan detector, adaptive", "0.5", "-", "s"),
        (scan[(scan.residual == "allocation") & (scan.attack == "band_limited_adaptive")], "Scan detector, band-limited adaptive (stress test)", "0.3", ":", "^"),
    ]
    for df, label, c, ls, m in series:
        ax.errorbar(df.snr, df.roc_auc, yerr=_err(df), label=label, color=c, ls=ls, marker=m, capsize=3, ms=5)
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlabel="SNR (dB)", ylabel="ROC-AUC (95% CI)", ylim=(0.4, 1.02),
           title="PHY covert-channel detection vs SNR (200 test scenarios per class per SNR)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_auc_vs_snr.png", dpi=300)


def fig_residual_comparison():
    scan = pd.read_csv(R / "scan_detector_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (res, atk), c, ls, m in [(("level", "adaptive"), "k", "--", "s"), (("allocation", "adaptive"), "k", "-", "s"),
                                 (("level", "band_limited_adaptive"), "0.5", "--", "^"),
                                 (("allocation", "band_limited_adaptive"), "0.5", "-", "^")]:
        df = scan[(scan.residual == res) & (scan.attack == atk)]
        ax.errorbar(df.snr, df.roc_auc, yerr=_err(df), label=f"{res} residual, {atk.replace('_', ' ')}",
                    color=c, ls=ls, marker=m, capsize=3, ms=5)
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlabel="SNR (dB)", ylabel="ROC-AUC (95% CI)", ylim=(0.4, 1.02),
           title="Blind (power-level) vs allocation-aware residual")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_residual_comparison.png", dpi=300)


def fig_aggregation():
    agg = pd.read_csv(R / "scan_aggregation_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for snr, m in [(10, "o"), (15, "s"), (20, "^")]:
        for res, ls in [("allocation", "-"), ("level", "--")]:
            df = agg[(agg.snr == snr) & (agg.residual == res)]
            ax.errorbar(df.frames_L, df.roc_auc, yerr=_err(df), label=f"{snr} dB, {res}", ls=ls, marker=m,
                        color="k" if res == "allocation" else "0.5", capsize=3, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlabel="Frames observed, L (attacker active in every frame)", ylabel="ROC-AUC (95% CI)", ylim=(0.4, 1.02),
           title="Adaptive attacker: detection vs observation length (100 groups per point)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_aggregation.png", dpi=300)


def fig_generalization():
    g = pd.read_csv(R / "scan_generalization_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c, m in [("non_adaptive (seen)", "k", "o"), ("adaptive (unseen)", "0.5", "s")]:
        df = g[g.tested_on == name]
        ax.errorbar(df.snr, df.roc_auc, yerr=_err(df), label=name, color=c, marker=m, capsize=3, ms=5)
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlabel="SNR (dB)", ylabel="ROC-AUC (95% CI)", ylim=(0.35, 1.02),
           title="Trained on clean vs non-adaptive only; tested on the unseen adaptive attacker")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_generalization.png", dpi=300)


def fig_payload_sweep():
    """AUC vs payload size (n_covert_bits), allocation-aware residual; one panel per SNR."""
    path = R / "scan_payload_sweep.csv"
    if not path.exists():
        print("skip fig_phy_payload_sweep.png: run python -m detector.sweep_payload first")
        return
    d = pd.read_csv(path)
    d = d[d.residual == "allocation"]
    snrs = sorted(d.snr.unique())
    fig, axes = plt.subplots(1, len(snrs), figsize=(3.2 * len(snrs), 3.8), sharey=True)
    for ax, snr in zip(np.atleast_1d(axes), snrs):
        for atk, label, c, ls, m in [("non_adaptive", "non-adaptive", "k", "-", "o"),
                                     ("adaptive", "adaptive (sqrt-law)", "0.5", "-", "s"),
                                     ("band_limited_adaptive", "band-limited adaptive", "0.3", ":", "^")]:
            df = d[(d.snr == snr) & (d.attack == atk)].sort_values("n_covert_bits")
            ax.errorbar(df.n_covert_bits, df.roc_auc, yerr=_err(df), label=label, color=c, ls=ls, marker=m, capsize=3, ms=5)
        ax.axvline(32, color="0.8", lw=0.8, ls="--")
        ax.axhline(0.5, color="0.7", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="covert symbols per grid, n (log scale)", title=f"{snr} dB", ylim=(0.4, 1.02))
    np.atleast_1d(axes)[0].set_ylabel("ROC-AUC (95% CI)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Scan detector (allocation-aware) vs payload size; dashed line = n = 32 used elsewhere", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_payload_sweep.png", dpi=300)


def tables_md():
    scan = pd.read_csv(R / "scan_detector_results.csv")
    base = pd.read_csv(R / "scan_baseline_energy.csv")
    lines = ["# Report numbers: PHY detector (source CSV named in each table)", "",
             "## ROC-AUC [95% CI], allocation-aware scan vs CNN-AE baseline (results/scan_detector_results.csv, results/scan_baseline_energy.csv)", "",
             "| SNR | Baseline non-adaptive | Baseline adaptive | Scan non-adaptive | Scan adaptive | Scan band-limited | Scan FPR at threshold |",
             "|---|---|---|---|---|---|---|"]
    f = lambda r: f"{r.roc_auc:.3f} [{r.auc_ci_low:.2f}, {r.auc_ci_high:.2f}]"
    for snr in sorted(scan.snr.unique()):
        b = base[base.snr == snr].set_index("attack")
        s = scan[(scan.snr == snr) & (scan.residual == "allocation")].set_index("attack")
        lines.append(f"| {snr} | {f(b.loc['non_adaptive'])} | {f(b.loc['adaptive'])} | {f(s.loc['non_adaptive'])} | "
                     f"{f(s.loc['adaptive'])} | {f(s.loc['band_limited_adaptive'])} | {s.loc['adaptive'].fpr:.3f} |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_auc_vs_snr()
    fig_residual_comparison()
    fig_aggregation()
    fig_generalization()
    fig_payload_sweep()
    tables_md()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
