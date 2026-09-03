import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("results")

X_PATH = RESULTS_DIR / "dataset_smoke_X.npy"
Y_PATH = RESULTS_DIR / "dataset_smoke_y.npy"
OUTPUT_PATH = RESULTS_DIR / "sanity_check.png"


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)

assert X.ndim == 3, f"Expected 3-D X, got {X.shape}"
assert X.shape[1:] == (200, 64), f"Unexpected grid shape: {X.shape}"
assert len(X) == len(y), "X and y lengths do not match"

clean = X[y == 0]
nonadaptive = X[y == 1]
adaptive = X[y == 2]

assert len(clean) > 0
assert len(nonadaptive) > 0
assert len(adaptive) > 0


# ---------------------------------------------------------------------
# Select corresponding smoke-test samples
# ---------------------------------------------------------------------

clean_sample = clean[0]
nonadaptive_sample = nonadaptive[0]
adaptive_sample = adaptive[0]


# ---------------------------------------------------------------------
# Calculate signed and absolute perturbations
# ---------------------------------------------------------------------

nonadaptive_diff = nonadaptive_sample - clean_sample
adaptive_diff = adaptive_sample - clean_sample

nonadaptive_abs = np.abs(nonadaptive_diff)
adaptive_abs = np.abs(adaptive_diff)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

nonadaptive_mad = float(np.mean(nonadaptive_abs))
adaptive_mad = float(np.mean(adaptive_abs))

nonadaptive_max = float(np.max(nonadaptive_abs))
adaptive_max = float(np.max(adaptive_abs))

nonadaptive_changed = int(np.count_nonzero(nonadaptive_diff))
adaptive_changed = int(np.count_nonzero(adaptive_diff))


print("=== Dataset Sanity Check ===")
print(f"X shape              : {X.shape}")
print(f"Clean samples        : {len(clean)}")
print(f"Non-adaptive samples : {len(nonadaptive)}")
print(f"Adaptive samples     : {len(adaptive)}")

print("\nPerturbation statistics:")
print(f"Non-adaptive MAD     : {nonadaptive_mad:.10f}")
print(f"Adaptive MAD         : {adaptive_mad:.10f}")
print(f"Non-adaptive max     : {nonadaptive_max:.10f}")
print(f"Adaptive max         : {adaptive_max:.10f}")
print(f"Non-adaptive changed : {nonadaptive_changed}")
print(f"Adaptive changed     : {adaptive_changed}")


# ---------------------------------------------------------------------
# Sanity assertions
# ---------------------------------------------------------------------

assert nonadaptive_mad > 0, "Non-adaptive attack produced no perturbation."
assert adaptive_mad > 0, "Adaptive attack produced no perturbation."
assert nonadaptive_max > 0, "Non-adaptive maximum is zero."
assert adaptive_max > 0, "Adaptive maximum is zero."


# ---------------------------------------------------------------------
# COMMON SCALE
#
# Signed difference plots:
#       -0.5 ........ 0 ........ +0.5
#
# Absolute magnitude plots:
#        0 ....................... 0.5
#
# Both attackers therefore use EXACTLY the same visual scale.
# ---------------------------------------------------------------------

common_signed_max = max(
    nonadaptive_max,
    adaptive_max,
)

common_abs_max = max(
    nonadaptive_max,
    adaptive_max,
)


# ---------------------------------------------------------------------
# Figure
#
# 1. Clean grid
# 2. Non-adaptive − clean
# 3. Adaptive − clean
# 4. |Non-adaptive − clean|
# 5. |Adaptive − clean|
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    5,
    figsize=(24, 5.5),
    constrained_layout=True,
)


# ---------------------------------------------------------------------
# Panel 1 — Clean grid
# ---------------------------------------------------------------------

axes[0].imshow(
    clean_sample,
    aspect="auto",
)

axes[0].set_title("Clean Grid")
axes[0].set_xlabel("Subcarrier")
axes[0].set_ylabel("OFDM Symbol")


# ---------------------------------------------------------------------
# Panel 2 — Signed non-adaptive perturbation
# ---------------------------------------------------------------------

nonadaptive_signed_image = axes[1].imshow(
    nonadaptive_diff,
    aspect="auto",
    vmin=-common_signed_max,
    vmax=common_signed_max,
)

axes[1].set_title(
    f"Non-Adaptive − Clean\n"
    f"MAD = {nonadaptive_mad:.5f}, "
    f"max = {nonadaptive_max:.3f}"
)

axes[1].set_xlabel("Subcarrier")
axes[1].set_ylabel("OFDM Symbol")


# ---------------------------------------------------------------------
# Panel 3 — Signed adaptive perturbation
# ---------------------------------------------------------------------

axes[2].imshow(
    adaptive_diff,
    aspect="auto",
    vmin=-common_signed_max,
    vmax=common_signed_max,
)

axes[2].set_title(
    f"Adaptive − Clean\n"
    f"MAD = {adaptive_mad:.5f}, "
    f"max = {adaptive_max:.3f}"
)

axes[2].set_xlabel("Subcarrier")
axes[2].set_ylabel("OFDM Symbol")


# Shared colorbar for signed perturbations
signed_colorbar = fig.colorbar(
    nonadaptive_signed_image,
    ax=[axes[1], axes[2]],
    shrink=0.85,
)

signed_colorbar.set_label(
    "Signed perturbation (Attacked − Clean)"
)


# ---------------------------------------------------------------------
# Panel 4 — Absolute non-adaptive magnitude
# ---------------------------------------------------------------------

nonadaptive_abs_image = axes[3].imshow(
    nonadaptive_abs,
    aspect="auto",
    vmin=0.0,
    vmax=common_abs_max,
)

axes[3].set_title(
    f"|Non-Adaptive − Clean|\n"
    f"MAD = {nonadaptive_mad:.5f}"
)

axes[3].set_xlabel("Subcarrier")
axes[3].set_ylabel("OFDM Symbol")


# ---------------------------------------------------------------------
# Panel 5 — Absolute adaptive magnitude
# ---------------------------------------------------------------------

axes[4].imshow(
    adaptive_abs,
    aspect="auto",
    vmin=0.0,
    vmax=common_abs_max,
)

axes[4].set_title(
    f"|Adaptive − Clean|\n"
    f"MAD = {adaptive_mad:.5f}"
)

axes[4].set_xlabel("Subcarrier")
axes[4].set_ylabel("OFDM Symbol")


# Shared colorbar for absolute magnitude plots
absolute_colorbar = fig.colorbar(
    nonadaptive_abs_image,
    ax=[axes[3], axes[4]],
    shrink=0.85,
)

absolute_colorbar.set_label(
    "Absolute perturbation magnitude"
)


# ---------------------------------------------------------------------
# Overall title
# ---------------------------------------------------------------------

fig.suptitle(
    "Covert-Channel Attack Sanity Check — Common Scales",
    fontsize=15,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


print(f"\nSaved visualization: {OUTPUT_PATH}")
print("Sanity check PASSED.")
print(
    "Signed and absolute perturbation panels use common scales "
    "for direct attacker comparison."
)
