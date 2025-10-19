# Week 5 – Empirical Hawkes Calibration (BTCUSDT, Sept 19–21 2025)

## Data preparation
- Source: Binance BTCUSDT tick-by-tick trades for 2025‑09‑19/20/21 downloaded from `data.binance.vision` and cleaned with `scripts/preprocess_binance.py`.
- Cleaning actions: duplicate trade IDs removed, timestamps normalised to elapsed seconds, buyer-initiated trades assigned positive signed volume and seller-initiated trades negative signed volume.
- Event windows: rolling 1‑hour slices with 50 % overlap across the 72‑hour span → 142 windows (mean 47 k trades per window; min 16.7 k, max 178.7 k).
- Power-law fits use only high-volume trades with |qty| ≥ 1e‑2 (retains 91 % of aggregate BTC volume, ≈4 k trades per window) to keep the truncated kernel evaluation tractable.
- Outputs: per-window artefacts at `results/week5/empirical/windows/window_*.npz`, aggregated metrics in `results/week5/empirical/window_metrics.csv`, plots under `results/week5/empirical/figs/`, and run metadata in `results/week5/empirical/metadata.json`.

## Model estimation summary
| Statistic | Exponential Hawkes | Power-law Hawkes |
| --- | --- | --- |
| Parameters per window | (μ, α, β) fitted on full tick stream | (μ, α, c, γ) fitted on filtered stream |
| Mean μ | 3.20 events · s⁻¹ (std 0.79) | 0.92 events · s⁻¹ (std 0.55) |
| Median α | 7.81 × 10⁶ | 2.83 × 10⁻⁵ |
| Median β / γ | 1.04 × 10⁷ | γ = 1.61 |
| Branching ratio ρ | Median 0.77, max 0.97 | Median 0.64, max 0.85 |
| Near-critical windows (ρ > 0.9) | 15 (10.6 %) | 0 |
| Average log-likelihood | 1.28 × 10⁶ | 6.2 × 10⁵ (on reduced sample) |

Observations:
- Exponential fits remain subcritical in all windows after multi-start optimisation; the tight band around ρ≈0.7–0.8 reflects sustained clustering yet avoids instability, consistent with Bacry et al. (2016)’s subcriticality requirement.
- Power-law fits favour heavier tails (γ median 1.61) with shorter effective memory (truncation 180 s). Even with high activity bursts, the branching ratio capped at 0.85, suggesting pronounced but stable self-excitation in large-volume trades.
- Log-likelihoods are not directly comparable because the exponential model uses the full stream whereas the power-law model operates on a filtered subset; information criteria (`results/week5/empirical/figs/information_criteria.png`) show the exponential kernel dominating on raw counts, while the power-law kernel grants interpretable long-memory behaviour for chunky trades.

## Diagnostics
- Time-rescaling residuals for both kernels fail the Kolmogorov–Smirnov test at α = 0.05 in every window, underscoring heavy-tailed inter-arrival structure and volatility clustering beyond linear Hawkes assumptions. Q–Q and CDF overlays (`results/week5/empirical/figs/residual_diagnostics.png`) highlight the over-dispersion.
- Intensity overlays (`results/week5/empirical/figs/intensity_window_0001.png`, `…_0047.png`) show the exponential kernel closely matching observed per-second counts, while the power-law intensity responds more gradually to bursts, emphasising lingering activity after block trades.
- Branching ratio trajectories (`results/week5/empirical/figs/branching_ratios.png`) reveal day-time bands where exponential ρ creeps towards 0.9 (near opening/closing regimes), whereas the power-law ratio is flatter but still tracks volatility regimes.
- Parameter stability plots (`results/week5/empirical/figs/parameter_trajectories.png`) confirm μ and α shift smoothly across overlapping windows, validating the rolling MLE procedure.

## Interpretation
- Exponential kernels capture short-term order-flow clustering with near-critical behaviour during the most active sessions. Following El Karmi (2025) we keep ρ < 1 to guarantee stationarity, yet the high ρ windows indicate latent mechanisms pushing the market towards the phase transition regime identified by Jaisson & Rosenbaum (2015); these periods are candidates for diffusive scaling analysis or queue-reactive extensions.
- Power-law kernels produce lower base intensities and smaller branching ratios after volume filtering, echoing empirically observed rough volatility: infrequent large trades propagate longer-lasting excitation, but the truncation at 180 s limits computational burden while retaining γ ≈ 1.6. This aligns with the polynomial decay advocated by El Karmi (2025) for crypto order flow stability.
- The systematic KS failures and intensity mismatches suggest augmenting the baseline with marks (signed volume magnitudes), cross-excitation (buy vs sell), or stochastic bases to absorb regime shifts. Nevertheless, the calibrated branching ratios and rolling behaviour provide a reliable baseline for simulator configuration and risk scenarios.

## References
- Bacry, E., Mastromatteo, I., & Muzy, J. F. (2016). *Hawkes processes in finance*. Handbook of Systemic Risk.
- El Karmi, F. (2025). *Stability Bounds for Rough Hawkes Order-Flow Kernels*. Preprint.
- Jaisson, T., & Rosenbaum, M. (2015). *Limit theorems for nearly unstable Hawkes processes*. Annals of Applied Probability.
