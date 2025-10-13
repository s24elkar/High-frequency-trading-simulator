# 5. Poisson to Hawkes Order-Flow Modelling

This note motivates the transition from memoryless Poisson arrivals to
self-exciting Hawkes processes when modelling high-frequency order flow. It
summarises the stability criteria highlighted in *Mathematical Foundations of
Marked Multivariate Hawkes Processes in Financial Markets* (El Karmi, 2025) and
connects them with the branching interpretation in Bacry et al. (2016).

## 5.1 Poisson Baseline

The homogeneous Poisson process assumes independent inter-arrival times
\(\Delta t_i\) with density \(\mu e^{-\mu \Delta t_i}\). Its likelihood over
\([0, T]\) factorises into a product of \(\mu\) terms minus the integrated rate:
\[
\log \mathcal{L}_{\text{Pois}}(\mu) = N_T \log \mu - \mu T,
\]
where \(N_T\) is the number of arrivals observed in the window. Because the
intensity is constant, the model is tractable and serves as a calibration
baseline. However, it cannot reproduce the empirically observed burstiness in
market order flow, nor the long-memory in order-sign sequences.

## 5.2 Hawkes Intensities and Kernels

A (marked) Hawkes process introduces self-excitation via a kernel \(\phi\):
\[
\lambda(t) = \mu + \sum_{t_i < t} \phi(t - t_i, m_i).
\]
The base rate \(\mu\) captures exogenous flow, while the kernel encodes how
recent events increase (or decrease) the probability of imminent arrivals. In
practice we focus on:

- **Exponential decay** \(\phi(u, m) = \alpha m e^{-\beta u}\) — efficient to
  simulate via a single state variable with recursion
  \(S_{t+} = S_{t-} e^{-\beta \Delta t} + \alpha m\).
- **Power-law decay** \(\phi(u, m) = \alpha m (u + c)^{-\gamma}\) — captures
  heavier tails and slow decay; simulation requires recomputing the kernel
  contributions explicitly.
- **Custom kernels** — user-defined functions with integrable tails can reproduce
  empirically motivated shapes (e.g. sums of exponentials or power-law/exponential
  hybrids).

## 5.3 Stability via Reproduction Matrices

El Karmi (2025) emphasises the *reproduction matrix*
\(\Omega = [\omega_{ij}]\), where
\(\omega_{ij} = \mathbb{E}[m_j] \int_0^\infty \phi_{ij}(u)\,\mathrm{d}u\) measures
the expected number of type-\(j\) offspring triggered by a type-\(i\) event.
Stationarity in the multivariate setting is guaranteed if and only if the
spectral radius obeys
\[
\rho(\Omega) < 1.
\]
This prevents explosive cascades and ensures the branching interpretation remains
subcritical. In the univariate case the matrix collapses to the scalar branching
ratio \(\eta = \Omega_{11}\). For exponential kernels this reduces to
\(\eta = (\alpha / \beta) \mathbb{E}[m]\); for power-law kernels with \(\gamma > 1\),
\(\eta = \alpha \mathbb{E}[m] / \bigl((\gamma - 1) c^{\gamma - 1}\bigr)\).

Failing the spectral-radius condition results in explosive behaviour where the
expected event count diverges and the point process ceases to be stationary.
Empirically, calibrated branching ratios often hover just below one, indicating
near-critical order flow (Bacry et al., 2016).

## 5.4 Practical Calibration

Maximum likelihood estimation remains the workhorse for fitting Hawkes
parameters:
\[
\log \mathcal{L}_{\text{Hawkes}}(\theta) =
\sum_{t_i \leq T} \log \lambda(t_i; \theta) -
\int_0^T \lambda(t; \theta)\,\mathrm{d}t.
\]
For exponential kernels the integral has a closed form, which greatly simplifies
numerical optimisation. The new `order_flow.calibration` module implements
`log_likelihood_hawkes_exp` and an L-BFGS-B wrapper for univariate exponential
kernels. Branching ratios are computed directly from the kernel integrals,
allowing quick checks of the stability condition \(\rho(\Omega) < 1\).

## 5.5 References

- Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2016). *Hawkes processes in
  finance*. **Market Microstructure and Liquidity**, 1(1), 1550005.
  https://doi.org/10.1142/S2382626615500057
- El Karmi, S. (2025). *Mathematical Foundations of Marked Multivariate Hawkes
  Processes in Financial Markets*. (Manuscript).

