"""Write reproducible mixed-numerology ISBI/SINR/BER results."""

from __future__ import annotations

import csv
from pathlib import Path

from slicing_sim.mixed_numerology import DEFAULT_NUMEROLOGIES, MixedNumerologyRAN, Numerology


def main() -> None:
    rows = []
    for guard in (0, 2):
        numerologies = {
            name: Numerology(config.subcarrier_spacing_khz, config.allocated_subcarriers, guard)
            for name, config in DEFAULT_NUMEROLOGIES.items()
        }
        for cancellation in (0.0, 0.75):
            rows.extend(MixedNumerologyRAN(seed=2026, numerologies=numerologies).evaluate(cancellation_efficiency=cancellation))
    output = Path("results/mixed_numerology_phy_v1.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
