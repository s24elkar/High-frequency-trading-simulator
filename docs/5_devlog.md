# Week 5 Devlog – Hawkes Process Dynamics

## Implementation Overview
- Added fast exponential-kernel utilities (`decay`, `jump`, `intensity`) and stateful scheduling helpers to the native simulator, enabling O(1) intensity updates per event instead of resumming history (`src/order_flow/OrderFlow.hpp:71-198`, `src/order_flow/OrderFlow.cpp:284-471`).
- The updated `HawkesProcess::simulate` branches between the cached exponential path and the historical fallback while reusing the new `schedule_next_event`, `decay_state`, and `register_event` helpers to keep the acceptance–rejection logic explicit (`src/order_flow/OrderFlow.cpp:316-389`).
- A dedicated Monte Carlo driver now lives in `python/scripts/hawkes_validation.py`, wiring the C++ bridge, diagnostic metrics, plotting, and benchmark instrumentation end to end (`python/scripts/hawkes_validation.py:1-312`).

## Parameter Grid & Assumptions
- Marks are kept constant at unity to isolate excitation dynamics; Poisson baselines use the same marks for apples-to-apples clustering metrics.
- Simulations run on a 600-unit horizon with 128 paths per configuration:
  - `subcritical`: μ = 0.25, α = 0.15, β = 1.40 (ρ ≈ 0.11)
  - `moderate`: μ = 0.25, α = 0.35, β = 1.20 (ρ ≈ 0.29)
  - `near_critical`: μ = 0.30, α = 0.60, β = 1.05 (ρ ≈ 0.57)
- Stability checks are baked into the metrics; all selected configurations satisfy ρ < 1.

## Empirical Validation Highlights
- Empirical vs theoretical mean intensities align within ±1.9 % across the grid, matching the stabilised branching ratios:

| configuration | λ̄ empirical | λ̄ theoretical | relative bias | Hawkes cond. var | Poisson cond. var |
| --- | --- | --- | --- | --- | --- |
| subcritical | 0.2820 | 0.2800 | +0.72 % | 3.6×10⁻³ | 0 |
| moderate | 0.3514 | 0.3529 | −0.43 % | 4.6×10⁻² | 0 |
| near_critical | 0.6871 | 0.7000 | −1.84 % | 5.9×10⁻¹ | 0 |

- Hawkes inter-arrival ACFs show clear positive correlation at short lags, while Poisson baselines fluctuate around zero (see `results/w5/hawkes_validation/figures/interarrival_acf.png`).
- Event-count histograms widen and shift right as excitation increases, diverging sharply from the Poisson reference mean in the near-critical regime (`results/w5/hawkes_validation/figures/event_count_hist.png`).

## Diagnostics & Artefacts
- Generated assets: intensity trajectories, ACF stems, and event-count histograms under `results/w5/hawkes_validation/figures/`.
- Full metric dump (including branching ratios, bias, and histogram bins) stored in `results/w5/hawkes_validation/metrics.json` with a CSV summary alongside.
- The validation entry point `python/scripts/hawkes_validation.py` accepts `--paths`, `--horizon`, `--max-lag`, and `--step` flags to reproduce or extend the study; figures and metrics are regenerated on each run.

## Performance Notes
- Total Hawkes throughput: 101 419 events in ~0.241 s, yielding an average 2.38 µs latency per simulated event (meta block in `results/w5/hawkes_validation/metrics.json`).
- Peak RSS increased by ~5.5 MB during the diagnostic run, primarily from Matplotlib’s transient cache; the script parks cache files under `results/.mpl_cache` so runs can prune them alongside other artefacts.
- The cached exponential state eliminates the quadratic history walk previously required for intensity evaluation, enabling the measured sub-millisecond path simulations even in the near-critical case.
