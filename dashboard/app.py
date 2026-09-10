"""
Streamlit dashboard — visual attack/detection status view for the tools slide.

Run with: streamlit run dashboard/app.py
(from the repo root, with PYTHONPATH set or after `pip install -e .`)

This is a REAL working dashboard against the actual simulation/attacker/
detector code, not a mockup. It intentionally does NOT do live continuous
training — that belongs offline (see notebooks or a training script once
you have real data volume). This dashboard visualizes one run at a time
so you can demo the pipeline end-to-end in a review.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from slicing_sim.ofdm_grid import NetworkSlicingSimulator, OFDMGridConfig, SLICE_TYPES
from covert_channel.attacker import NonAdaptiveAttacker, AdaptiveAttacker, AttackerConfig

st.set_page_config(page_title="Cross-Slice Covert Channel Dashboard", layout="wide")
st.title("Cross-Slice Covert Channel Detection — Live Simulation")
st.caption(
    "Attack and Detection Lead: Pavin Kishore N | Authentication Lead: Poojasree V | "
    "Cryptography Lead: Priyadharshini R"
)

with st.sidebar:
    st.header("Simulation Config")
    n_subcarriers = st.slider("Subcarriers", 16, 128, 64, step=16)
    n_symbols = st.slider("OFDM symbols", 50, 400, 200, step=50)
    snr_db = st.slider("SNR (dB)", 0, 30, 20)
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Attacker")
    attacker_mode = st.radio("Attacker type", ["None (clean)", "Non-adaptive", "Adaptive"])
    target_slice = st.selectbox("Target slice", SLICE_TYPES, index=1)

if st.button("Run simulation", type="primary"):
    cfg = OFDMGridConfig(n_subcarriers=n_subcarriers, n_symbols=n_symbols,
                          snr_db=snr_db, seed=int(seed))
    sim = NetworkSlicingSimulator(cfg)
    allocations = sim.allocate_slices()
    clean_grid = sim.combined_interference_grid(allocations)

    grid = clean_grid
    if attacker_mode == "Non-adaptive":
        atk = NonAdaptiveAttacker(AttackerConfig(seed=int(seed), target_slice=target_slice))
        grid = atk.inject(clean_grid, allocations[target_slice].subcarrier_mask)
    elif attacker_mode == "Adaptive":
        atk = AdaptiveAttacker(AttackerConfig(seed=int(seed), target_slice=target_slice))
        grid = atk.inject(clean_grid, allocations[target_slice].subcarrier_mask)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-slice subcarrier occupancy")
        for st_type in SLICE_TYPES:
            occ = allocations[st_type].subcarrier_mask.mean()
            st.metric(st_type, f"{occ:.1%}")

    with col2:
        st.subheader("Grid stats")
        st.metric("Mean interference power", f"{grid.mean():.4f}")
        st.metric("Perturbation vs clean (mean abs delta)",
                   f"{np.mean(np.abs(grid - clean_grid)):.6f}")

    st.subheader("Resource grid heatmap (symbols x subcarriers)")
    fig = go.Figure(data=go.Heatmap(z=grid.T, colorscale="Viridis"))
    fig.update_layout(xaxis_title="OFDM symbol (time)", yaxis_title="Subcarrier index",
                       height=400)
    st.plotly_chart(fig, use_container_width=True)

    if attacker_mode != "None (clean)":
        st.subheader("Perturbation map (attacked - clean)")
        delta = grid - clean_grid
        fig2 = go.Figure(data=go.Heatmap(z=delta.T, colorscale="RdBu", zmid=0))
        fig2.update_layout(xaxis_title="OFDM symbol (time)", yaxis_title="Subcarrier index",
                            height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Detector reconstruction-error view requires a trained model (see "
        "detector/autoencoder_detector.py). Train and save a model, then load it "
        "here — not wired up yet since training needs real data volume, "
        "not this dashboard's single-run scope."
    )
else:
    st.write("Configure a run in the sidebar and click **Run simulation**.")
