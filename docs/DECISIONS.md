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

---

Add new entries above this line. Keep each entry under 5 lines — if you
can't compress it, you haven't finished deciding.
