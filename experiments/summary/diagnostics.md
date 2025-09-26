# Benchmark Diagnostics Summary

## Binance BTCUSDT (context = full book, seed 2024)

- **GRU backbone**
  - Test NLL/MAE/Accuracy: 0.265 / 0.085 s / 0.892 (ECE 0.035).
  - KS statistic remains high (0.72 → p≈0), signalling under-dispersed inter-arrival predictions.
  - Loss curve and calibration plots show consistent bias for upper quantiles despite longer context.
- **Transformer backbone**
  - Test NLL/MAE/Accuracy: 0.252 / 0.083 s / 0.898 (ECE 0.040).
  - Achieves the best log-loss while keeping MAE similar; accuracy improves by ~0.6 pp.
  - QQ/KS diagnostics still deviate from the diagonal — attention helps accuracy but not intensity calibration yet.

## LOBSTER AAPL (level-II sample, seed 2024)

- **GRU backbone**
  - Test NLL/MAE/Accuracy: 4.512 / 0.974 s / 0.858 (ECE 0.591).
  - KS statistic 0.61 with very heavy tails; calibration curve bows above the diagonal (late arrivals underpredicted).
- **Transformer backbone**
  - Test NLL/MAE/Accuracy: 4.425 / 0.899 s / 0.858 (ECE 0.637).
  - Improves NLL by ~0.09 and MAE by ~0.075 s while matching accuracy.
  - KS and ECE remain large, highlighting the difficulty of matching long-memory order flow.

## Ablation — context length (GRU)

| Venue | Context | Test NLL ↓ | MAE ↓ | Acc ↑ |
| --- | --- | ---: | ---: | ---: |
| Binance | Short (256) | 0.274 | 0.090 | 0.892 |
| Binance | Long (full) | 0.265 | 0.085 | 0.892 |
| LOBSTER | Short (256) | 4.620 | 0.933 | 0.858 |
| LOBSTER | Long (full) | 4.512 | 0.974 | 0.858 |

- Extending the context lowers NLL on both venues, with the largest gain on Binance (–0.009) but mixed MAE impact (LOBSTER slightly worse due to very long gaps).
- Calibration remains the dominant error source; even the best models fail KS rescaling tests (p≈0) and exhibit ECE ≥0.03.
- See `experiments/summary/figs` for the paper-ready loss, QQ, KS, and calibration panels included in the draft.
