# Novelty and evidence boundary

This project does **not** claim to be the first system to combine AI, PQC and
network slicing. Its contribution is a reproducible simulation study of an
OFDM resource-grid covert-channel threat model with: (1) a shaped adaptive
attacker, (2) clean-calibrated adaptive residual detection using slice
telemetry, and (3) a dual-trigger re-authentication policy designed for an
optional Dilithium backend.

The evidence is limited to reproducible OFDM impairment simulation (AWGN,
frequency-selective multipath/fading, Doppler-like temporal variation, CFO
leakage, phase noise and impulsive interference). The benchmark CSV reports held-out
FPR, recall, ROC-AUC and PR-AUC; it is not evidence of field performance,
5G conformance, a deployed cryptographic system, or resistance to arbitrary
adaptive adversaries. Dilithium signing requires liboqs-python at runtime.

Before external submission, add a cited related-work comparison here and
validate against measured channel traces or standards-compliant channel models.

Channel-impairment design reference: 3GPP TR 38.901 defines tapped-delay-line
and clustered-delay-line channel-model families. This repository uses compact
approximations for repeatable tests; it does not claim TR 38.901 conformance.
