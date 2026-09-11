# notebooks/

## phase1_colab_training.ipynb

Runs a larger-scale training + evaluation pass than the repo's committed
4,200-row frozen dataset: regenerates the dataset at 3x scale (600
scenarios/SNR instead of 200 — 12,600 rows, n=90 per test/calibration band
instead of n=30), using the repo's own `detector/generate_frozen_dataset.py`
generation logic unmodified (only the scenario count is overridden).

It then runs two pipelines at this larger scale, for both the `cnn` and
`structured_dae` presets of `detector/autoencoder_detector.py`:

- **Primary result:** the existing, verified clean-only methodology (train
  on clean rows only, calibrate thresholds from clean-only calibration
  scores) — same protocol as `evaluate_cnn_autoencoder.py` /
  `evaluate_structured_dae.py`, just at 3x scale.
- **Ablation:** a label-free variant that never uses `y` for training-data
  selection or calibration, to see how much performance is lost without
  known-clean labels.

The two are compared against a **pre-committed tolerance rule**, decided
before looking at the results: the label-free ablation is only kept as a
reported finding if avg ROC-AUC drop ≤ 0.03 and max drop ≤ 0.07 across all
14 SNR x attacker combinations, per architecture. Otherwise it's reported as
a documented limitation, not a finding.

**Requirements:** must be run in Google Colab with a GPU runtime (Runtime →
Change runtime type → GPU; a T4 is enough). It clones the repo fresh and
installs TensorFlow/scikit-learn/pandas/numpy/matplotlib itself — it isn't
meant to be run against this local checkout.

**Branch caveat (from the notebook's first cell):** it clones
`phase1-consolidated`, **not** `master`, because as of when it was written
`master` was still missing the `AdaptiveAttacker` perturbation-ceiling fix.
Change `BRANCH` in the notebook to `"master"` only after confirming
[PR #1](https://github.com/pavinkishore7/network_covert_channel_detection/pull/1)
has actually been merged — don't assume it has.

This notebook has not yet been executed. Results from an actual Colab run
will come back as a separate follow-up once the output has been reviewed.
