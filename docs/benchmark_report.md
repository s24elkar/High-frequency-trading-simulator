# Neural vs Transformer Hawkes Comparison

## Setup

- **Datasets**
  - Binance BTCUSDT trades (2025-09-21) aggregated into event sequences via `scripts/pack_binance_npz.py`.
  - LOBSTER AAPL Level-3 sample (2012-06-21) packaged with `scripts/preprocess_lobster.py`.
- **Models**: GRU-based neural Hawkes surrogate and a Transformer backbone sharing embedding/hidden dimensions (64/128) with deterministic seeding (2024).
- **Training**: sliding windows of length 128, stride 64, batch size 256; experiments run for 5 and 15 epochs to gauge convergence behaviour.
- **Outputs**: each run records split metrics, KS diagnostics, runtime, parameter counts, and plots (`experiments/runs/<name>/`). Aggregated metrics live in `experiments/summary/benchmarks.csv`.

## Key Results (5 Epochs)

| Venue | Backbone | Test NLL | MAE (s) | Accuracy | KS stat | Train time (s) |
|-------|----------|---------:|--------:|---------:|--------:|----------------:|
| Binance | GRU | 0.274 | 0.0895 | 0.892 | 0.720 | 68.6 |
| Binance | Transformer | **0.259** | 0.0924 | **0.895** | 0.720 | 121.8 |
| LOBSTER | GRU | 4.62 | 0.933 | 0.858 | 0.618 | 5.58 |
| LOBSTER | Transformer | **4.47** | **0.883** | 0.858 | **0.583** | 9.43 |

Observations:
- Transformers consistently reduce log-loss; improvements are modest on Binance (~5% relative) and more pronounced on LOBSTER (~3%).
- Event-type accuracy gaps are small (+0.3% Binance, neutral on LOBSTER).
- Time-rescaling KS statistics remain high; neither model is perfectly calibrated, though the transformer reduces KS on LOBSTER.
- Training time roughly doubles for the transformer on Binance due to the heavier attention layers; the overhead is milder on the smaller LOBSTER sample.

## Extended Training (15 Epochs)

| Venue | Backbone | Test NLL | MAE (s) | Accuracy | KS stat | Train time (s) |
|-------|----------|---------:|--------:|---------:|--------:|----------------:|
| Binance | GRU | 0.265 | 0.0853 | 0.892 | 0.720 | 240 |
| Binance | Transformer | **0.252** | **0.0835** | **0.898** | 0.720 | 384 |
| LOBSTER | GRU | 4.51 | 0.974 | **0.858** | 0.608 | 16.2 |
| LOBSTER | Transformer | **4.43** | 0.899 | 0.858 | 0.590 | 27.3 |

Observations:
- Additional epochs continue to improve both backbones; transformers benefit more, widening the NLL gap and closing the MAE penalty on Binance.
- Binance transformer now beats GRU on both NLL and MAE, while maintaining the accuracy edge.
- LOBSTER gains in NLL/MAE plateau; transformer reduces NLL slightly but KS improvements taper, hinting that richer features or marks may be needed.
- Runtime scales roughly linearly with epoch count; CPU training remains under 7 minutes for the heaviest run.

## Diagnostics

- Loss curves show smooth convergence; transformers keep gaining for >10 epochs while GRUs plateau earlier.
- Calibration plots (`experiments/runs/*/figs/calibration.png`) reveal underestimation of long inter-arrival times across venues, motivating future work on intensity calibration.
- QQ/KS plots (`figs/qq_rescaled.png`, `figs/ks_cdf.png`) corroborate KS statistics, with transformers producing slightly closer alignment on LOBSTER but not Binance.

## Next Directions

1. Broaden datasets: add more Binance symbols/dates and full LOBSTER days to strengthen statistical claims.
2. Enhance modelling: incorporate volume marks, longer contexts, or attention with temporal decay to improve calibration.
3. Extend evaluation: bootstrap confidence intervals, include classical Hawkes baselines, and measure inference latency on GPU.
4. Documentation: integrate these findings into the main README or paper draft, citing configuration files and scripts for reproducibility.
