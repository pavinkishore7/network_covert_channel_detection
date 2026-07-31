"""
Prometheus metrics exporter.

Grafana needs a time-series backend to query — it doesn't visualize
anything on its own. This exporter runs simulation cycles periodically and
exposes the resulting metrics on :8000/metrics for Prometheus to scrape,
which Grafana then queries. This is the minimum real stack for "Grafana"
to mean something rather than just being an installed, empty app.

Run standalone: python3 monitoring/exporter.py
Or via docker-compose (see docker-compose.yml) alongside prometheus + grafana.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import start_http_server, Gauge
import numpy as np

from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig, SLICE_TYPES
from covert_channel.attacker import AdaptiveAttacker, NonAdaptiveAttacker, AttackerConfig

# --- Metric definitions ---
GRID_MEAN_POWER = Gauge("ofdm_grid_mean_power", "Mean interference power across the resource grid")
SLICE_OCCUPANCY = Gauge("slice_subcarrier_occupancy", "Fraction of subcarriers occupied", ["slice_type"])
PERTURBATION_MAGNITUDE = Gauge("covert_perturbation_magnitude",
                                "Mean absolute perturbation vs clean grid", ["attacker_type"])
SIMULATION_CYCLES = Gauge("simulation_cycles_total", "Number of simulation cycles run")


def run_cycle(seed: int):
    cfg = OFDMGridConfig(seed=seed)
    sim = NetworkSlicingSimulator(cfg)
    allocations = sim.allocate_slices()
    clean_grid = sim.combined_interference_grid(allocations)

    GRID_MEAN_POWER.set(float(clean_grid.mean()))
    for slice_type in SLICE_TYPES:
        occ = float(allocations[slice_type].subcarrier_mask.mean())
        SLICE_OCCUPANCY.labels(slice_type=slice_type).set(occ)

    target_mask = allocations["eMBB"].subcarrier_mask

    naive = NonAdaptiveAttacker(AttackerConfig(seed=seed))
    naive_grid = naive.inject(clean_grid, target_mask)
    PERTURBATION_MAGNITUDE.labels(attacker_type="non_adaptive").set(
        float(np.mean(np.abs(naive_grid - clean_grid))))

    adaptive = AdaptiveAttacker(AttackerConfig(seed=seed))
    adaptive_grid = adaptive.inject(clean_grid, target_mask)
    PERTURBATION_MAGNITUDE.labels(attacker_type="adaptive").set(
        float(np.mean(np.abs(adaptive_grid - clean_grid))))


if __name__ == "__main__":
    start_http_server(8000)
    print("Metrics exporter running on :8000/metrics")
    cycle = 0
    while True:
        run_cycle(seed=cycle)
        cycle += 1
        SIMULATION_CYCLES.set(cycle)
        time.sleep(15)
