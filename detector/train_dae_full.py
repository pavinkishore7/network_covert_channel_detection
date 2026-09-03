import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from detector.denoising_autoencoder import DenoisingAutoencoderDetector

X = np.load("results/dataset_X.npy")
y = np.load("results/dataset_y.npy")
snr = np.load("results/dataset_snr.npy")

LABEL_CLEAN = 0
LABEL_NON_ADAPTIVE = 1
LABEL_ADAPTIVE = 2

results = []

for snr_val in sorted(np.unique(snr)):

    mask = snr == snr_val

    clean = X[mask & (y == LABEL_CLEAN)]
    non_adaptive = X[mask & (y == LABEL_NON_ADAPTIVE)]
    adaptive = X[mask & (y == LABEL_ADAPTIVE)]

    print(f"\nSNR {snr_val:.0f} dB")
    print(f"  Clean        : {len(clean)}")
    print(f"  Non-adaptive : {len(non_adaptive)}")
    print(f"  Adaptive     : {len(adaptive)}")

    if len(clean) < 20:
        print("  Skipping: too few clean samples")
        continue

    split = int(0.7 * len(clean))

    clean_train = clean[:split]
    clean_val = clean[split:]

    det = DenoisingAutoencoderDetector(
        input_shape=X.shape[1:]
    )

    det.fit(
        clean_train,
        snr_db=float(snr_val),
        epochs=20
    )

    det.calibrate(clean_val)

    clean_scores = det.reconstruction_error(clean_val)
    non_scores = det.reconstruction_error(non_adaptive)
    adapt_scores = det.reconstruction_error(adaptive)

    y_non = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(non_scores))
    ])

    scores_non = np.concatenate([
        clean_scores,
        non_scores
    ])

    y_adapt = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adapt_scores))
    ])

    scores_adapt = np.concatenate([
        clean_scores,
        adapt_scores
    ])

    auc_non = roc_auc_score(y_non, scores_non)
    auc_adapt = roc_auc_score(y_adapt, scores_adapt)

    print(f"  Non-adaptive AUC : {auc_non:.4f}")
    print(f"  Adaptive AUC     : {auc_adapt:.4f}")

    results.append({
        "snr": snr_val,
        "auc_non_adaptive": auc_non,
        "auc_adaptive": auc_adapt
    })

df = pd.DataFrame(results)

df.to_csv(
    "results/dae_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("DAE RESULTS")
print("=" * 60)
print(df.to_string(index=False))
print("\nSaved: results/dae_results.csv")
