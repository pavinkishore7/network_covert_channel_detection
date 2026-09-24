# Decisions Log

Format: Date — Decision — Why — Alternatives rejected

---

**2026-09-24 — PHY detector: physics-informed multi-scale scan detector; CNN-AE kept as baseline**
Why: the trained CNN-AE and structured-DAE scores correlate with plain eMBB-masked input energy at r = 0.99999999 (both collapsed to a near-constant reconstruction), so the old results were an energy detector and the two "methods" were one. The attack is 32 of 12,800 cells; mean pooling dilutes it ~400x. New detector: per-cell residual (blind power-level or allocation-aware) + max over a generic 16-window scan. Evaluated in detector/evaluate_scan_detector.py.
Alternatives rejected: tuning the scan window to the attack's burst shape; weakening the attacker.

**2026-09-24 — Attacker placement: random burst position (default); legacy "first" kept**
Why: both attackers always wrote their 32 symbols into the first eMBB cells in row-major order, so every attack sat at the start of the grid — a modelling artifact. Now a contiguous burst at a random position; size and magnitude unchanged. placement="first" reproduces the old frozen dataset bit-for-bit (checked). results/cnn_autoencoder_*, structured_dae_*, adaptive_benchmark_v1* predate this and use the legacy placement.
Also added BandLimitedAdaptiveAttacker, a STRONGER stress-test attacker that stays inside the legal power band.

**2026-09-24 — adaptive_benchmark_v1.csv superseded**
Why: produced by detector/legacy/adaptive_residual.py with a different configuration (64 covert bits, fading channel profiles, SNR 8–24 dB), not the frozen-dataset protocol; its adaptive > non-adaptive AUC is not comparable with the frozen-dataset results. Kept for history; don't cite.

**2026-09 — Adaptive residual guard added for reproducible benchmark evaluation**
Why: TensorFlow CNN experiments are not portable to every reviewer environment,
and the prior smoke test had no held-out adaptive result. The clean-calibrated
NumPy guard reports FPR/recall/ROC-AUC/PR-AUC with slice telemetry. It is a
benchmark guard, not a replacement for the CNN Autoencoder research model.

---

**2026-07 — Dual-trigger re-authentication retained (timer + detector-triggered)**
Why: removing the timer trigger collapses the architecture into single-layer
defense (detection-only), which kills the defense-in-depth novelty claim —
a missed detection would equal a breach with no independent backstop.
Alternative rejected: detector-only trigger (simpler, but no independence
between the two failure modes).

**2026-07 — Slice-aware timer intervals (not uniform)**
Why: uniform re-auth interval either over-taxes mMTC (too frequent) or
under-protects URLLC (too infrequent). Tighter interval for URLLC, relaxed
for mMTC resolves the mMTC overhead objection without weakening URLLC.
Alternative rejected: single fixed interval across all slice types.

**2026-07 — Novelty scope narrowed to PHY-layer covert channel**
Why: literature check surfaced a March 2026 paper combining PQC + AI
anomaly detection at the orchestration layer. Broad "first to combine PQC +
AI + slicing" claim is already contradicted. Narrowed to OFDM-interference
covert channels specifically at the physical layer, which that paper does
not cover.
Alternative rejected: keep broad claim and argue it's "different enough" —
too risky under adversarial questioning.

**2026-07 — Detector architecture locked to CNN Autoencoder (not CNN+LSTM)**
Why: deck had inconsistent naming across slides (Methodology + Expected
Outcomes deliverables said "CNN Autoencoder"; Novelty + Expected Outcomes
metrics table said "CNN+LSTM"). Team decision: CNN Autoencoder only.
Action item: fix slides 8 and 12 to match — currently still say CNN+LSTM.
Alternative rejected: hybrid CNN+LSTM autoencoder — more architecturally
complex, not worth the added risk for the accuracy gain it might give.

**2026-07 — Adaptive attacker implemented, detector evaluated against it**
Why: deck claimed "Adaptive ML Detection" detects the adaptive attacker,
but this wasn't actually implemented or tested — an overclaim. Built both
NonAdaptiveAttacker (fixed perturbation, baseline) and AdaptiveAttacker
(sqrt-law-bounded, interference-shaped) in covert_channel/attacker.py, so
the detector's performance against each can be measured and reported
separately. Do not claim "detects adaptive attacks" until there's an
actual ROC/AUC number against AdaptiveAttacker output — smoke test alone
(5 epochs, 30 samples) showed 10%/10%, i.e. no discrimination yet.

---

Add new entries above this line. Keep each entry under 5 lines — if you
can't compress it, you haven't finished deciding.
