# Hawkes Processes & Market Microstructure

## 1. Point Processes Refresher
A point process models the arrival times of discrete events \(\{T_i\}\). The *conditional intensity* \(\lambda(t)\) is the instantaneous arrival rate given the history \(\mathcal{H}_t\). For a Hawkes process, \(\lambda\) is *self-exciting*:
\[
\lambda(t) = \mu + \sum_{T_i < t} \phi(t - T_i, \, m_i),
\]
where \(\mu\) is the base intensity, \(\phi\) is a kernel, and \(m_i\) are optional marks (e.g., trade volumes).

## 2. Campbell’s Theorem & First-Order Moments
Campbell’s theorem states that for any function \(g\),
\[
\mathbb{E}\left[\sum_i g(T_i)\right] = \mathbb{E}\left[\int g(t) \lambda(t) dt\right].
\]
This connects the *expected* cumulative impact of events to the conditional intensity and underpins the estimator for Hawkes parameters via maximum likelihood. In practice, the likelihood of a Hawkes process is
\[
\mathcal{L}(\theta) = \sum_i \log \lambda(T_i; \theta) - \int_0^T \lambda(t; \theta) \, dt,
\]
which decomposes into a sum over observed events minus the integrated intensity.

## 3. Branching Interpretation
A Hawkes process can be interpreted as a branching (Galton–Watson) process:
- Immigrants arrive according to \(\mu\).
- Each event spawns *offspring* according to the kernel \(\phi\).
The *branching ratio* is the expected number of offspring per event:
\[
\eta = \int_0^\infty \mathbb{E}[\phi(u, m)] du.
\]
For exponential kernels \(\phi(u) = \alpha e^{-\beta u}\), \(\eta = \alpha / \beta\). Subcritical dynamics require \(\eta < 1\). In market microstructure, \(\eta\) quantifies clustering and market reflexivity—values near 1 indicate order flow that *self-perpetuates*.

## 4. Why Hawkes for Order Flow?
1. **Queue dynamics**: Market/limit orders cluster temporally.
2. **Sign persistence**: Buy trades increase the odds of near-term buys (long-memory of order flow).
3. **Venue interaction**: Multivariate Hawkes captures cross-excitation between exchanges and instruments.
4. **Risk metrics**: The branching ratio and intensity shape liquidity/density forecasts for execution algorithms.

## 5. Modelling Variants
| Variant | Kernel | Notes |
| --- | --- | --- |
| Classical MLE | Exponential / Sums of exponentials | Closed-form gradients, efficient EM; implemented in `tick` |
| Neural Hawkes (Du et al. 2016) | RNN with marked intensity | Learns hidden dynamics; requires numerical integration to evaluate log-likelihood |
| Transformer Hawkes (Zuo et al. 2020) | Self-attention + decay | Scales to long-range dependencies; integrates time-modulated attention |
| Continuous-time Normalizing Flows | Flow-based intensity | Flexible but expensive; ideal for calibration tasks |

## 6. Diagnostics
- **Time-rescaling test**: Transform inter-arrival times via \(u_i = 1 - e^{-\Delta \Lambda_i}\); check against \(\mathrm{Uniform}(0, 1)\) via KS/QQ plots.
- **Intensity reconstruction**: Compare fitted vs empirical intensity profiles on hold-out segments.
- **Kernel visualization**: Plot \(\phi(u)\) and its cumulative to interpret memory decay.

## 7. Benchmark Checklist
1. Fit classical Hawkes (MLE) on each symbol/exchange and compute log-likelihood, branching ratio, and diagnostics.
2. Train neural surrogates (GRU/LSTM, Transformer, MLP) and report accuracy/MAE/log-likelihood proxies.
3. Record runtime on CPU/GPU.
4. Run ablations: input marks, backbone type, window size.

## 8. Data Processing & Modelling Workflow
- **Preprocessing**
  - Convert trade timestamps to elapsed seconds from the start of each session so the intensity operates in a common time grid.
  - Define marks as signed trade size (positive for buys, negative for sells) to encode order-flow direction and magnitude.
  - Segment streams into rolling windows (e.g., 10-minute or 1-hour horizons) for stable estimation and for regime comparisons (day vs. night).
- **Estimation**
  - Fit exponential-kernel Hawkes processes by maximum likelihood (e.g., `tick.hawkes.HawkesExpKern`) to obtain baseline intensity `μ`, adjacency matrix `Ω`, and branching ratio `ρ(Ω)`.
  - Compare against rough/power-law kernels (or log-convex sums of exponentials) to gauge long-memory effects.
  - Report parameter tables per symbol/venue, including standard errors when available.
- **Diagnostics**
  - Apply the time-rescaling transform and generate QQ plots against the Exp(1) reference; run KS statistics and log-likelihood comparisons on held-out windows.
  - Plot fitted vs. empirical intensities and cumulative counts to visualise goodness-of-fit.
  - Document runtime (CPU/GPU) for calibration and inference.
- **Interpretation**
  - Discuss the magnitude of `ρ(Ω)`; values near 1 indicate near-critical behaviour and self-exciting cascades—a stylised fact in crypto/equity order flow.
  - Contrast clustering in real BTC data with simulated baselines; relate findings to liquidity resilience and volatility clustering.
  - Highlight venue or asset divergences (e.g., BTC vs. ETH, Binance vs. CME).

## 9. Author Checklist for Publication
- **Data appendix**: Summary statistics of each dataset (trade counts, volatility, tick size).
- **Calibration notebook**: Step-by-step replicable pipeline for each model family.
- **Aggregated benchmarks**: Tables comparing log-likelihood, KS statistics, and runtime.
- **Interpretation section**: Plot branching ratios, cross-excitation matrices, and discuss market implications.

Use this primer as a theoretical backdrop for writing the methodology section of your manuscript.
