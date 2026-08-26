"""Run one predeclared structured-DAE experiment on the frozen dataset.

This script intentionally has no test-set model selection.  Its train/valid/
test indices are based on matched simulation triplets, never class-wise rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def matched_split(y: np.ndarray, snr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices for 140/30/30 matched triplets at every SNR."""
    train, valid, test = [], [], []
    for value in np.unique(snr):
        rows = np.flatnonzero(snr == value)
        if len(rows) != 600 or not np.array_equal(y[rows].reshape(-1, 3), np.tile([0, 1, 2], (200, 1))):
            raise ValueError(f"SNR {value}: expected 200 ordered [clean, non-adaptive, adaptive] triplets")
        triplets = rows.reshape(200, 3)
        train.extend(triplets[:140].ravel())
        valid.extend(triplets[140:170].ravel())
        test.extend(triplets[170:].ravel())
    return np.asarray(train), np.asarray(valid), np.asarray(test)


def _metrics(clean_scores: np.ndarray, attack_scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attack_scores))]
    scores = np.r_[clean_scores, attack_scores]
    return {"fpr": float((clean_scores > threshold).mean()), "detection_rate": float((attack_scores > threshold).mean()),
            "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def main() -> None:
    # Keep the split utility importable for protocol tests even on systems
    # that have not installed the TensorFlow experiment dependency yet.
    from detector.dae_autoencoder import StructuredDAE

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--mask-probability", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    np.random.seed(args.seed)
    paths = {name: args.results_dir / f"dataset_{name}.npy" for name in ("X", "y", "snr")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Frozen dataset missing: " + ", ".join(missing))
    X, y, snr = (np.load(paths[name]) for name in ("X", "y", "snr"))
    if X.shape != (4200, 200, 64) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Unexpected frozen dataset shape or labels")
    train, valid, test = matched_split(y, snr)
    model = StructuredDAE(X.shape[1:], seed=args.seed)
    model.fit(X[train][y[train] == 0], epochs=args.epochs, batch_size=args.batch_size,
              mask_probability=args.mask_probability, verbose=2)
    threshold = model.calibrate(X[valid][y[valid] == 0], mode="topk", top_fraction=0.01)
    scores = model.reconstruction_error(X[test], mode="topk", top_fraction=0.01)
    test_y, test_snr = y[test], snr[test]
    rows = []
    for snr_value in ["overall", *np.unique(test_snr).tolist()]:
        select = np.ones(len(test), dtype=bool) if snr_value == "overall" else test_snr == snr_value
        clean = scores[select & (test_y == 0)]
        for label, name in ((1, "non_adaptive"), (2, "adaptive")):
            row = {"snr": snr_value, "attack": name, **_metrics(clean, scores[select & (test_y == label)], threshold)}
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(args.results_dir / "structured_dae_results.csv", index=False)
    np.savez(args.results_dir / "structured_dae_errors.npz", scores=scores, y=test_y, snr=test_snr, threshold=threshold)
    (args.results_dir / "structured_dae_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(output.to_string(index=False))
    print(f"\nThreshold (clean validation 95th percentile): {threshold:.6f}")


if __name__ == "__main__":
    main()
