from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from detector.dae_autoencoder import DAEAutoencoderDetector


RESULTS_DIR = Path("results")

X_PATH = RESULTS_DIR / "dataset_X.npy"
Y_PATH = RESULTS_DIR / "dataset_y.npy"
SNR_PATH = RESULTS_DIR / "dataset_snr.npy"

OUTPUT_CSV = RESULTS_DIR / "dae_results.csv"

SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]

SAMPLES_PER_SNR = 200

TRAIN_COUNT = 140
VAL_COUNT = 30
TEST_COUNT = 30

CLASS_CLEAN = 0
CLASS_NON_ADAPTIVE = 1
CLASS_ADAPTIVE = 2

EPOCHS = 20
BATCH_SIZE = 8

PERCENTILE = 95.0
MODE = "topk"


def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load():

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    snr = np.load(SNR_PATH)

    assert X.shape == (4200, 200, 64)
    assert y.shape == (4200,)
    assert snr.shape == (4200,)

    return X, y, snr


def split(y, snr):

    train = []
    val = []
    test = []

    for s_idx, s in enumerate(SNR_VALUES):

        start = (
            s_idx
            * SAMPLES_PER_SNR
            * 3
        )

        for i in range(SAMPLES_PER_SNR):

            base = start + i * 3

            ids = [
                base,
                base + 1,
                base + 2,
            ]

            assert [
                int(y[j]) for j in ids
            ] == [0, 1, 2]

            assert all(
                float(snr[j]) == float(s)
                for j in ids
            )

            if i < TRAIN_COUNT:
                train.extend(ids)

            elif i < TRAIN_COUNT + VAL_COUNT:
                val.extend(ids)

            else:
                test.extend(ids)

    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)

    return {
        "train_clean": train[
            y[train] == CLASS_CLEAN
        ],
        "val_clean": val[
            y[val] == CLASS_CLEAN
        ],
        "test_clean": test[
            y[test] == CLASS_CLEAN
        ],
        "test_nonadaptive": test[
            y[test] == CLASS_NON_ADAPTIVE
        ],
        "test_adaptive": test[
            y[test] == CLASS_ADAPTIVE
        ],
    }


def metrics(clean, attack, threshold):

    truth = np.concatenate([
        np.zeros(len(clean)),
        np.ones(len(attack)),
    ])

    scores = np.concatenate([
        clean,
        attack,
    ])

    pred = (scores > threshold).astype(int)

    return {
        "detection_rate":
            float(pred[len(clean):].mean()),

        "false_positive_rate":
            float(pred[:len(clean)].mean()),

        "precision":
            float(
                precision_score(
                    truth,
                    pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    truth,
                    pred,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    truth,
                    scores,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    truth,
                    scores,
                )
            ),

        "threshold": float(threshold),

        "clean_mean_error":
            float(clean.mean()),

        "attack_mean_error":
            float(attack.mean()),
    }


def main():

    seed_everything()

    X, y, snr = load()

    split_data = split(y, snr)

    print("=" * 70)
    print("DAE COVERT-CHANNEL DETECTOR")
    print("=" * 70)

    print(
        f"Training clean: "
        f"{len(split_data['train_clean'])}"
    )

    print(
        f"Validation clean: "
        f"{len(split_data['val_clean'])}"
    )

    print(
        f"Test clean: "
        f"{len(split_data['test_clean'])}"
    )

    print(
        f"Test non-adaptive: "
        f"{len(split_data['test_nonadaptive'])}"
    )

    print(
        f"Test adaptive: "
        f"{len(split_data['test_adaptive'])}"
    )

    # ----------------------------------------------------------
    # ONE DAE trained on ALL clean training SNRs
    # ----------------------------------------------------------

    train_clean = X[
        split_data["train_clean"]
    ]

    val_clean = X[
        split_data["val_clean"]
    ]

    detector = DAEAutoencoderDetector(
        input_shape=X.shape[1:],
        latent_dim=16,
        noise_std=0.10,
    )

    print("\nTraining DAE...")
    print("Corruption std = 0.10 × clean-data std")

    detector.fit(
        train_clean,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ----------------------------------------------------------
    # CLEAN validation calibration
    # ----------------------------------------------------------

    threshold = detector.calibrate(
        val_clean,
        percentile=PERCENTILE,
        mode=MODE,
    )

    print(
        f"\nThreshold ({MODE}, {PERCENTILE}th percentile): "
        f"{threshold:.6f}"
    )

    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------

    test_clean = X[
        split_data["test_clean"]
    ]

    test_non = X[
        split_data["test_nonadaptive"]
    ]

    test_adapt = X[
        split_data["test_adaptive"]
    ]

    clean_scores = detector.reconstruction_error(
        test_clean,
        mode=MODE,
    )

    non_scores = detector.reconstruction_error(
        test_non,
        mode=MODE,
    )

    adapt_scores = detector.reconstruction_error(
        test_adapt,
        mode=MODE,
    )

    non_metrics = metrics(
        clean_scores,
        non_scores,
        threshold,
    )

    adapt_metrics = metrics(
        clean_scores,
        adapt_scores,
        threshold,
    )

    print("\n" + "=" * 70)
    print("OVERALL DAE RESULTS")
    print("=" * 70)

    print("\nNon-adaptive attacker:")
    print(
        f"  Detection rate : "
        f"{non_metrics['detection_rate']:.2%}"
    )
    print(
        f"  False-positive : "
        f"{non_metrics['false_positive_rate']:.2%}"
    )
    print(
        f"  ROC-AUC        : "
        f"{non_metrics['roc_auc']:.4f}"
    )
    print(
        f"  PR-AUC         : "
        f"{non_metrics['pr_auc']:.4f}"
    )

    print("\nAdaptive attacker:")
    print(
        f"  Detection rate : "
        f"{adapt_metrics['detection_rate']:.2%}"
    )
    print(
        f"  False-positive : "
        f"{adapt_metrics['false_positive_rate']:.2%}"
    )
    print(
        f"  ROC-AUC        : "
        f"{adapt_metrics['roc_auc']:.4f}"
    )
    print(
        f"  PR-AUC         : "
        f"{adapt_metrics['pr_auc']:.4f}"
    )

    # ----------------------------------------------------------
    # Per-SNR
    # ----------------------------------------------------------

    rows = []

    print("\n" + "=" * 70)
    print("PER-SNR RESULTS")
    print("=" * 70)

    print(
        f"{'SNR':>5} "
        f"{'Non-Adapt DR':>15} "
        f"{'Non-Adapt AUC':>15} "
        f"{'Adaptive DR':>15} "
        f"{'Adaptive AUC':>15}"
    )

    for s in SNR_VALUES:

        clean_idx = split_data[
            "test_clean"
        ][snr[
            split_data["test_clean"]
        ] == s]

        non_idx = split_data[
            "test_nonadaptive"
        ][snr[
            split_data["test_nonadaptive"]
        ] == s]

        adapt_idx = split_data[
            "test_adaptive"
        ][snr[
            split_data["test_adaptive"]
        ] == s]

        c = detector.reconstruction_error(
            X[clean_idx],
            mode=MODE,
        )

        n = detector.reconstruction_error(
            X[non_idx],
            mode=MODE,
        )

        a = detector.reconstruction_error(
            X[adapt_idx],
            mode=MODE,
        )

        nm = metrics(c, n, threshold)
        am = metrics(c, a, threshold)

        print(
            f"{s:>5} "
            f"{nm['detection_rate']:>14.2%} "
            f"{nm['roc_auc']:>15.4f} "
            f"{am['detection_rate']:>14.2%} "
            f"{am['roc_auc']:>15.4f}"
        )

        rows.append({
            "snr_db": s,
            "nonadaptive_detection_rate":
                nm["detection_rate"],
            "nonadaptive_auc":
                nm["roc_auc"],
            "nonadaptive_pr_auc":
                nm["pr_auc"],
            "adaptive_detection_rate":
                am["detection_rate"],
            "adaptive_auc":
                am["roc_auc"],
            "adaptive_pr_auc":
                am["pr_auc"],
            "false_positive_rate":
                am["false_positive_rate"],
            "threshold":
                threshold,
        })

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
