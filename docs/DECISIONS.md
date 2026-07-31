# Decisions Log

Format: Date — Decision — Why — Alternatives rejected

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
