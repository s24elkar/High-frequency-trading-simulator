"""Streamlit UI for exploring Hawkes simulations interactively."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
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


DEFAULTS: Dict[str, object] = {
    "kernel_choice": "Exponential",
    "mu": 0.2,
    "horizon": 200.0,
    "bins": 100,
    "seed": 12345,
    "exp_alpha": 0.8,
    "exp_beta": 1.2,
    "power_alpha": 0.12,
    "power_c": 0.1,
    "power_gamma": 1.4,
    "mark_choice": "LogNormal",
    "mark_lognorm_mean": 0.0,
    "mark_lognorm_sigma": 0.5,
    "mark_exponential_scale": 1.0,
    "mark_value": 1.0,
    "compare_enabled": False,
    "compare_kernel": "Power-law",
    "compare_exp_alpha": 0.6,
    "compare_exp_beta": 0.9,
    "compare_power_alpha": 0.18,
    "compare_power_c": 0.08,
    "compare_power_gamma": 1.6,
}


PRESET_DESCRIPTIONS: Dict[str, str] = {
    "Calm market": "Low background activity with quickly fading memory — a sleepy tape.",
    "Frenzy": "High base flow and long memory that create persistent bursts of trading.",
    "Flash crash": "Moderate base flow but strong self-excitation for sudden cascades.",
}


PRESET_SCENARIOS: Dict[str, Dict[str, object]] = {
    "Custom": {},
    "Calm market": {
        "kernel_choice": "Exponential",
        "mu": 0.08,
        "horizon": 300.0,
        "bins": 120,
        "exp_alpha": 0.4,
        "exp_beta": 1.5,
        "mark_choice": "Deterministic",
        "mark_value": 0.7,
    },
    "Frenzy": {
        "kernel_choice": "Power-law",
        "mu": 0.35,
        "horizon": 200.0,
        "bins": 100,
        "power_alpha": 0.4,
        "power_c": 0.05,
        "power_gamma": 1.25,
        "mark_choice": "LogNormal",
        "mark_lognorm_mean": 0.3,
        "mark_lognorm_sigma": 0.9,
    },
    "Flash crash": {
        "kernel_choice": "Exponential",
        "mu": 0.15,
        "horizon": 120.0,
        "bins": 80,
        "exp_alpha": 1.3,
        "exp_beta": 0.7,
        "mark_choice": "Exponential",
        "mark_exponential_scale": 1.6,
    },
}


def _init_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _apply_preset(values: Dict[str, object]) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.experimental_rerun()


def _exp_controls(key_prefix: str = "") -> Dict[str, float]:
    alpha = st.sidebar.slider(
        "α — self-excitation strength",
        0.01,
        2.0,
        value=float(st.session_state.get(f"{key_prefix}exp_alpha", DEFAULTS["exp_alpha"])),
        step=0.01,
        key=f"{key_prefix}exp_alpha",
        help="Higher α means each trade boosts the intensity of follow-on trades more strongly.",
    )
    beta = st.sidebar.slider(
        "β — memory decay",
        0.1,
        5.0,
        value=float(st.session_state.get(f"{key_prefix}exp_beta", DEFAULTS["exp_beta"])),
        step=0.01,
        key=f"{key_prefix}exp_beta",
        help="Controls how quickly the excitement fades. Larger β means the market cools off faster.",
    )
    return {"alpha": alpha, "beta": beta}


def _powerlaw_controls(key_prefix: str = "") -> Dict[str, float]:
    alpha = st.sidebar.slider(
        "α — clustering strength",
        0.01,
        1.0,
        value=float(st.session_state.get(f"{key_prefix}power_alpha", DEFAULTS["power_alpha"])),
        step=0.01,
        key=f"{key_prefix}power_alpha",
        help="Higher α makes bursts of activity more intense.",
    )
    c = st.sidebar.slider(
        "c — short-term cushion",
        0.01,
        1.0,
        value=float(st.session_state.get(f"{key_prefix}power_c", DEFAULTS["power_c"])),
        step=0.01,
        key=f"{key_prefix}power_c",
        help="Acts like a time offset preventing the intensity from blowing up immediately after an event.",
    )
    gamma = st.sidebar.slider(
        "γ — memory tail",
        1.01,
        3.5,
        value=float(st.session_state.get(f"{key_prefix}power_gamma", DEFAULTS["power_gamma"])),
        step=0.01,
        key=f"{key_prefix}power_gamma",
        help="Lower γ keeps memory longer, creating drawn-out cascades of activity.",
    )
    return {"alpha": alpha, "c": c, "gamma": gamma}


def _mark_sampler_controls() -> Callable[[np.random.Generator], float]:
    choice = st.sidebar.selectbox(
        "Order size distribution",
        ("LogNormal", "Exponential", "Deterministic"),
        key="mark_choice",
        help="Choose how trade sizes (marks) are drawn when the process jumps.",
    )
    if choice == "LogNormal":
        mean = st.sidebar.slider(
            "Log-mean",
            -1.0,
            1.0,
            value=float(st.session_state.get("mark_lognorm_mean", DEFAULTS["mark_lognorm_mean"])),
            step=0.05,
            key="mark_lognorm_mean",
            help="Sets the typical log-size of trades during bursts.",
        )
        sigma = st.sidebar.slider(
            "Log-sigma",
            0.1,
            1.5,
            value=float(st.session_state.get("mark_lognorm_sigma", DEFAULTS["mark_lognorm_sigma"])),
            step=0.05,
            key="mark_lognorm_sigma",
            help="Controls how varied the trade sizes are (higher means more spread).",
        )

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.lognormal(mean, sigma))

    elif choice == "Exponential":
        scale = st.sidebar.slider(
            "Scale",
            0.1,
            5.0,
            value=float(st.session_state.get("mark_exponential_scale", DEFAULTS["mark_exponential_scale"])),
            step=0.1,
            key="mark_exponential_scale",
            help="Average order size. Higher scale = larger trades on average.",
        )

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.exponential(scale))

    else:
        value = st.sidebar.slider(
            "Fixed value",
            0.01,
            5.0,
            value=float(st.session_state.get("mark_value", DEFAULTS["mark_value"])),
            step=0.01,
            key="mark_value",
            help="All trades use the same mark size — useful for stress-testing the timing model alone.",
        )

        def sampler(rng: np.random.Generator) -> float:
            return float(value)

    sampler.__doc__ = f"Sampler: {choice}"
    return sampler


def main() -> None:
    _init_defaults()

    st.set_page_config(page_title="Hawkes Simulator", layout="wide")
    st.title("High-Frequency Order Flow Simulator")
    st.caption("Visualizing clustered order arrivals with Hawkes processes")
    st.markdown(
        "This app simulates how trades in a market can \"snowball\", where one trade nudges the next."
    )

    with st.expander("Learn more"):
        st.write(
            """
            Hawkes processes model event clustering in finance, seismology, and even social media. In
            markets they describe bursts of trading activity: a burst of orders today makes another burst
            more likely a moment later. Experiment with the base activity (μ), how long memory lasts, and the
            distribution of order sizes to see calm periods, frenzies, or near-critical cascades emerge.
            No prior microstructure knowledge is needed: just move the sliders and watch how the timeline reacts.
            """
        )

    st.sidebar.header("Market regimes")

    preset_name = st.sidebar.selectbox(
        "Choose a regime",
        list(PRESET_SCENARIOS.keys()),
        key="preset_choice",
        help="Pick a canned market regime to auto-fill parameters or stay on Custom to explore freely.",
    )
    if preset_name != "Custom" and st.sidebar.button(
        f"Load '{preset_name}'", help="Apply the preset values and rerun the simulation."
    ):
        _apply_preset(PRESET_SCENARIOS[preset_name])
    if preset_name in PRESET_DESCRIPTIONS:
        st.sidebar.caption(PRESET_DESCRIPTIONS[preset_name])

    kernel_choice = st.sidebar.selectbox(
        "Kernel",
        ("Exponential", "Power-law"),
        key="kernel_choice",
        help="Exponential forgets quickly; power-law keeps long memory and heavier clustering.",
    )
    mu = st.sidebar.slider(
        "Base intensity μ",
        0.01,
        2.0,
        value=float(st.session_state.get("mu", DEFAULTS["mu"])),
        step=0.01,
        key="mu",
        help="Background arrival rate when nothing exciting is happening.",
    )
    horizon = st.sidebar.slider(
        "Horizon T",
        10.0,
        1000.0,
        value=float(st.session_state.get("horizon", DEFAULTS["horizon"])),
        step=10.0,
        key="horizon",
        help="How long to simulate the process (in seconds).",
    )
    bins = st.sidebar.slider(
        "Timeline bins",
        20,
        200,
        value=int(st.session_state.get("bins", DEFAULTS["bins"])),
        step=5,
        key="bins",
        help="Controls the resolution of the counts plot (more bins = finer detail).",
    )
    seed = st.sidebar.number_input(
        "Seed",
        value=int(st.session_state.get("seed", DEFAULTS["seed"])),
        step=1,
        key="seed",
        help="Fix the random seed so runs are repeatable. Change it for a fresh sample path.",
    )
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
            "branching_ratio": kernel.branching_ratio(np.mean(timeline.marks) if timeline.marks.size else 0.0),
        }

    comparison_timeline = None
    comparison_label = None
    with st.sidebar.expander("Comparison overlay", expanded=bool(st.session_state.get("compare_enabled", False))):
        compare_enabled = st.checkbox(
            "Overlay a second kernel",
            value=bool(st.session_state.get("compare_enabled", False)),
            key="compare_enabled",
            help="Run a second set of parameters and plot both on the same charts to compare regimes.",
        )
        if compare_enabled:
            compare_kernel_choice = st.selectbox(
                "Comparison kernel",
                ("Exponential", "Power-law"),
                key="compare_kernel",
                help="Choose the structure for the comparison run.",
            )
            if compare_kernel_choice == "Exponential":
                compare_params = _exp_controls("compare_")
                compare_kernel = ExpKernel(**compare_params)
                comparison_timeline = simulate_exp_timeline(
                    mu, compare_kernel, horizon, mark_sampler, seed + 1, bins
                )
                comparison_label = f"{compare_kernel_choice} (overlay)"
            else:
                compare_params = _powerlaw_controls("compare_")
                compare_kernel = PowerLawKernel(**compare_params)
                comparison_timeline = simulate_powerlaw_timeline(
                    mu, compare_kernel, horizon, mark_sampler, seed + 1, bins
                )
                comparison_label = f"{compare_kernel_choice} (overlay)"

    col_plot, col_data = st.columns((2.2, 1))
    with col_plot:
        fig = plot_timeline(
            timeline,
            title=f"{kernel_choice} Hawkes Simulation",
            comparison=comparison_timeline,
            labels=(kernel_choice, comparison_label or "Comparison"),
        )
        st.pyplot(fig)

    with col_data:
        st.subheader("Summary")
        st.json(summary)
        st.metric("Events", int(timeline.times.size))
        st.metric("Average mark", float(np.mean(timeline.marks) if timeline.marks.size else 0.0))
        st.metric("Max intensity", float(timeline.intensity_grid.max() if timeline.intensity_grid.size else mu))

        branching_ratio = summary.get("branching_ratio")
        if branching_ratio is not None and not np.isnan(branching_ratio):
            if np.isinf(branching_ratio) or branching_ratio >= 1.0:
                msg = "Supercritical — cascades will keep growing."
                st.error(f"Branching ratio ≈ {branching_ratio:.2f}. {msg}")
            elif branching_ratio >= 0.85:
                msg = "Near critical — bursts linger."
                st.warning(f"Branching ratio ≈ {branching_ratio:.2f}. {msg}")
            else:
                msg = "Comfortably subcritical — bursts die out."
                st.success(f"Branching ratio ≈ {branching_ratio:.2f}. {msg}")

        if timeline.marks.size:
            hist_fig, hist_ax = plt.subplots(figsize=(4, 3))
            hist_ax.hist(
                timeline.marks,
                bins=min(30, max(5, int(len(timeline.marks) / 5))),
                color="#1f77b4",
                alpha=0.8,
            )
            hist_ax.set_title("Order size distribution")
            hist_ax.set_xlabel("Mark value")
            hist_ax.set_ylabel("Frequency")
            st.pyplot(hist_fig)

        st.download_button(
            "Export simulated order flow (CSV)",
            data="time,mark\n" + "\n".join(f"{t:.6f},{m:.6f}" for t, m in zip(timeline.times, timeline.marks)),
            file_name="hawkes_events.csv",
            mime="text/csv",
        )

    notebooks_dir = Path(__file__).resolve().parent.parent / "docs" / "notebooks"
    if notebooks_dir.exists():
        st.sidebar.markdown("---")
        st.sidebar.subheader("Go deeper")
        st.sidebar.info(
            "See how we calibrate Hawkes models on real Binance order flow and synthetic data."
        )
        for notebook_path in notebooks_dir.glob("*.ipynb"):
            notebook_bytes = notebook_path.read_bytes()
            st.sidebar.download_button(
                label=f"Download {notebook_path.stem.replace('_', ' ').title()} notebook",
                data=notebook_bytes,
                file_name=notebook_path.name,
                mime="application/x-ipynb+json",
            )


if __name__ == "__main__":
    main()
