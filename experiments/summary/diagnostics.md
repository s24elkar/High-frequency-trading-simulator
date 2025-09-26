# Benchmark Diagnostics Summary

## Binance BTCUSDT (2025-09-21)

- **GRU backbone**
  - Test NLL/MAE/Accuracy: 0.274 / 0.0895 s / 0.892.
  - KS statistic 0.72, indicating noticeable deviation from the ideal exponential rescaling.
  - Loss curve flattens by epoch five; validation closely tracks training suggesting limited overfitting.
- **Transformer backbone**
  - Test NLL/MAE/Accuracy: 0.259 / 0.0924 s / 0.895.
  - Gains ~0.015 in NLL and +0.003 in accuracy over GRU at the cost of slightly higher timing MAE and ~1.8× training time.
  - KS statistic identical (0.72) — transformer does not yet reduce calibration mismatch on the Binance stream.

## LOBSTER AAPL sample (2012-06-21)

- **GRU backbone**
  - Test NLL/MAE/Accuracy: 4.62 / 0.933 s / 0.858.
  - KS statistic 0.62; rescaling plots show heavy tails.
- **Transformer backbone**
  - Test NLL/MAE/Accuracy: 4.47 / 0.883 s / 0.858.
  - Improves NLL by ~0.15 and lowers MAE by ~0.05 s with negligible accuracy change. Runtime is ~1.7× GRU but still sub-10 s on CPU.
  - KS statistic drops to 0.58, reflecting better (though still imperfect) calibration relative to GRU.

## Takeaways

- Transformers consistently deliver lower log-loss on both venues; gains are larger on LOBSTER, hinting that attention helps when trade arrival structure is richer.
- Neither backbone passes strict KS tests yet; future work should explore longer context, mark features, or explicit intensity calibration.
- Runtime overhead of the transformer is modest on LOBSTER but pronounced for high-volume Binance streams, so deployment decisions should weigh latency budgets.

Refer to the figures in `experiments/runs/*/figs/` for visual inspection of loss curves, calibration plots, and QQ/KS diagnostics.
