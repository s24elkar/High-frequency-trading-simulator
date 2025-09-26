"""Streamlit UI for exploring Hawkes simulations interactively."""

from __future__ import annotations

import functools
from typing import Callable, Dict

import numpy as np
import streamlit as st

try:
    from .bridge_utils import ensure_bridge_path
    from .kernels import ExpKernel, PowerLawKernel
    from .timeline_dashboard import (
        plot_timeline,
        simulate_exp_timeline,
        simulate_powerlaw_timeline,
    )
except (ImportError, ValueError):
    import pathlib
    import sys

    _pkg_dir = pathlib.Path(__file__).resolve().parent
    if str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

    from bridge_utils import ensure_bridge_path
    from kernels import ExpKernel, PowerLawKernel
    from timeline_dashboard import (
        plot_timeline,
        simulate_exp_timeline,
        simulate_powerlaw_timeline,
    )


ensure_bridge_path()


def _exp_controls() -> Dict[str, float]:
    alpha = st.sidebar.slider("alpha", 0.01, 2.0, 0.8, 0.01)
    beta = st.sidebar.slider("beta", 0.1, 5.0, 1.2, 0.01)
    return {"alpha": alpha, "beta": beta}


def _powerlaw_controls() -> Dict[str, float]:
    alpha = st.sidebar.slider("alpha", 0.01, 1.0, 0.12, 0.01)
    c = st.sidebar.slider("c", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("gamma", 1.01, 3.5, 1.4, 0.01)
    return {"alpha": alpha, "c": c, "gamma": gamma}


def _mark_sampler_controls() -> Callable[[np.random.Generator], float]:
    choice = st.sidebar.selectbox("Mark distribution", ("LogNormal", "Exponential", "Deterministic"))
    if choice == "LogNormal":
        mean = st.sidebar.slider("log-mean", -1.0, 1.0, 0.0, 0.05)
        sigma = st.sidebar.slider("log-sigma", 0.1, 1.5, 0.5, 0.05)

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.lognormal(mean, sigma))

    elif choice == "Exponential":
        scale = st.sidebar.slider("scale", 0.1, 5.0, 1.0, 0.1)

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.exponential(scale))

    else:
        value = st.sidebar.slider("value", 0.01, 5.0, 1.0, 0.01)

        def sampler(rng: np.random.Generator) -> float:
            return float(value)

    sampler.__doc__ = f"Sampler: {choice}"
    return sampler


def main() -> None:
    st.set_page_config(page_title="Hawkes Simulator", layout="wide")
    st.title("High-Frequency Order Flow Simulator")
    st.markdown(
        "Interactively explore Hawkes process simulations using the native C++ core."
    )

    kernel_choice = st.sidebar.selectbox("Kernel", ("Exponential", "Power-law"))
    mu = st.sidebar.slider("Base intensity μ", 0.01, 2.0, 0.2, 0.01)
    horizon = st.sidebar.slider("Horizon T", 10.0, 1000.0, 200.0, 10.0)
    bins = st.sidebar.slider("Timeline bins", 20, 200, 100, 5)
    seed = st.sidebar.number_input("Seed", value=12345, step=1)
    mark_sampler = _mark_sampler_controls()

    if kernel_choice == "Exponential":
        params = _exp_controls()
        kernel = ExpKernel(**params)
        timeline = simulate_exp_timeline(mu, kernel, horizon, mark_sampler, seed, bins)
        summary = {
            "kernel": kernel_choice,
            "alpha": params["alpha"],
            "beta": params["beta"],
            "branching_ratio": kernel.branching_ratio(np.mean(timeline.marks) if timeline.marks.size else 0.0),
        }
    else:
        params = _powerlaw_controls()
        kernel = PowerLawKernel(**params)
        timeline = simulate_powerlaw_timeline(mu, kernel, horizon, mark_sampler, seed, bins)
        summary = {
            "kernel": kernel_choice,
            **params,
        }

    col_plot, col_data = st.columns((2, 1))
    with col_plot:
        fig = plot_timeline(timeline, title=f"{kernel_choice} Hawkes Simulation")
        st.pyplot(fig)

    with col_data:
        st.subheader("Summary")
        st.json(summary)
        st.metric("Events", int(timeline.times.size))
        st.metric("Average mark", float(np.mean(timeline.marks) if timeline.marks.size else 0.0))
        st.metric("Max intensity", float(timeline.intensity_grid.max() if timeline.intensity_grid.size else mu))
        st.download_button(
            "Download events (CSV)",
            data="time,mark\n" + "\n".join(f"{t:.6f},{m:.6f}" for t, m in zip(timeline.times, timeline.marks)),
            file_name="hawkes_events.csv",
            mime="text/csv",
        )

    st.caption(
        "Build artefacts must exist under `build/lib` or be specified via the `HFT_HAWKES_BRIDGE` environment variable."
    )


if __name__ == "__main__":
    main()
