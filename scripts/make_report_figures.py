"""Regenerate the PHY-detector report figures from the CSVs in results/.

Reads only results/scan_*.csv (written by detector/evaluate_scan_detector.py) and
writes 300-dpi PNGs to results/report/. Styles are distinguishable in grayscale
(line style + marker), and every AUC point carries its 95% bootstrap CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


def fig_net_detection_vs_offset():
    p = R / "phase2_sweep_extended.csv"
    if not p.exists():
        print("skip network sweep (no CSV)")
        return
    d = pd.read_csv(p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    styles = {"URLLC": ("k", "o"), "eMBB": ("0.45", "s"), "mMTC": ("0.7", "^")}
    for sl, (c, m) in styles.items():
        for inj, ls in (("non_adaptive", "-"), ("adaptive", "--")):
            g = d[(d.part == "a_offsets") & (d.slice == sl) & (d.injector == inj)].sort_values("offset_over_sigma")
            a1.errorbar(g.offset_over_sigma, g.detection_rate, yerr=[g.detection_rate - g.det_ci_low, g.det_ci_high - g.detection_rate],
                        color=c, marker=m, ls=ls, capsize=2, ms=4, label=f"{sl}, {inj.replace('_', '-')}")
        for k, ls in ((0.05, "--"), (0.1, "-")):
            g = d[(d.part == "b_jitter") & (d.slice == sl) & (d.offset_over_sigma == k)].sort_values("steady_jitter_frac")
            a2.errorbar(g.steady_jitter_frac, g.detection_rate, yerr=[g.detection_rate - g.det_ci_low, g.det_ci_high - g.detection_rate],
                        color=c, marker=m, ls=ls, capsize=2, ms=4, label=f"{sl}, {k}σ")
    a1.set_xscale("log")
    a1.set(xlabel="covert delay offset / σ (overall gap std, log scale)", ylabel="detection rate (95% CI)",
           title="Default traffic (steady jitter 5%)", ylim=(-0.03, 1.03))
    a2.set_xscale("log")
    a2.set_xticks([0.05, 0.1, 0.2, 0.5], ["5%", "10%", "20%", "50%"])
    a2.set(xlabel="steady-component jitter (std / base gap)", title="Same absolute delay, noisier traffic (non-adaptive)")
    a1.legend(fontsize=7, loc="upper left")
    a2.legend(fontsize=7, loc="lower left")
    fig.suptitle("KS timing-channel detection: 500 windows of 300 packets per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_net_detection_vs_offset.png", dpi=300)


def fig_throughput():
    p = R / "throughput_detectability.csv"
    if not p.exists():
        print("skip throughput (no CSV)")
        return
    df = pd.read_csv(p, comment="#")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, atk in zip(axes, ["non_adaptive", "adaptive"]):
        for snr, m, c in [(10, "o", "0.6"), (15, "s", "0.3"), (20, "^", "k")]:
            d = df[(df.attack == atk) & (df.residual == "allocation") & (df.snr == snr)].sort_values("bits_per_frame")
            ax.errorbar(d.bits_per_frame, d.roc_auc, yerr=_err(d), label=f"{snr} dB", marker=m, color=c, capsize=3, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256], ["8", "16", "32", "64", "128", "256"])
        ax.axhline(0.5, color="0.7", lw=0.8)
        top = ax.secondary_xaxis("top", functions=(lambda b: b / 7.142857, lambda k: k * 7.142857))
        top.set_xlabel("covert throughput (kbit/s, 30 kHz SCS)")
        ax.set(xlabel="covert bits per 200-symbol frame", title=atk.replace("_", "-") + " attacker", ylim=(0.4, 1.02))
    axes[0].set_ylabel("ROC-AUC (95% CI)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Throughput vs detectability (allocation-aware scan detector, 200 per class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_throughput.png", dpi=300)


def fig_cnn():
    p = R / "cnn_scan_results.csv"
    if not p.exists():
        print("skip cnn (no CSV)")
        return
    cnn = pd.read_csv(p)
    scan = pd.read_csv(R / "scan_detector_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, res, atk, lab, c, ls, m in [
        ("cnn_seen_allocation", None, "adaptive", "CNN (allocation-aware), adaptive", "k", "-", "s"),
        ("cnn_seen_blind", None, "adaptive", "CNN (blind), adaptive", "0.5", "-", "s"),
        (None, "allocation", "adaptive", "Scan (allocation-aware), adaptive", "k", "--", "o"),
        (None, "level", "adaptive", "Scan (blind), adaptive", "0.5", "--", "o"),
        ("cnn_nonadapt_blind", None, "adaptive", "CNN (blind) trained w/o adaptive", "0.3", ":", "^")]:
        d = (cnn[(cnn.model == model) & (cnn.attack == atk) & (cnn.bits_per_frame == 32)] if model
             else scan[(scan.residual == res) & (scan.attack == atk)]).sort_values("snr")
        ax.errorbar(d.snr, d.roc_auc, yerr=_err(d), label=lab, color=c, ls=ls, marker=m, capsize=3, ms=5)
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set(xlabel="SNR (dB)", ylabel="ROC-AUC (95% CI)", ylim=(0.4, 1.02), title="Fully-convolutional CNN vs scan detector (adaptive attacker)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phy_cnn_vs_scan.png", dpi=300)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_auc_vs_snr()
    fig_residual_comparison()
    fig_aggregation()
    fig_generalization()
    fig_net_detection_vs_offset()
    fig_throughput()
    fig_cnn()
    tables_md()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
