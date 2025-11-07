# Week 7 – PnL Analytics & Risk Diagnostics

## Methodology

- **Data inputs** — `results/week7/strategy_calibration.csv` (324 simulation buckets) plus reference configs in `results/week7/best_configs.json`.
- **Decomposition engine** — `src/pnl_analysis.cpp` ingests the calibration grid, estimates execution slippage, market impact, and inventory carry penalties per run, and back-solves the implied strategy alpha so that  
  `alpha = realized_pnl + slippage + impact + inventory_carry`. The executable writes the aggregated view to `results/week7/pnl_breakdown.csv`.
- **Risk metrics** — `python/scripts/run_week7_pnl_analytics.py` computes VaR/CVaR, realized volatility, drawdowns, Sharpe/Sortino ratios, and generates plots under `results/week7/plots/pnl_risk/`. The script also validates each “best config” row against the source grid to ensure consistency with historical runs.

## PnL decomposition highlights

| Strategy | Avg realized PnL | Strategy alpha | Slippage cost | Market-impact cost | Inventory/carry cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean reversion | −1,900.32 | −1,829.61 | 49.38 | 12.31 | 9.02 |
| Trend following | 781.57 | 860.22 | 57.24 | 12.79 | 8.61 |

Key takeaways:
- The mean-reversion book’s losses trace back to negative alpha (−1.83k) rather than frictions—the execution stack only erodes ~70 bps of the drawdown, implying the predictive signal is the main culprit.
- Trend-following shows modest positive alpha (~0.86k) and similar execution drag (~70), confirming its gains come from signal quality rather than superior fill mechanics.
- Inventory carry costs stay below 10 per run for both books, so inventory risk is not the dominant drag this week.

Full table: `results/week7/pnl_breakdown.csv`.

## Risk diagnostics

Summary statistics (`results/week7/pnl_risk_metrics.csv`):

| Strategy | Obs. | Mean PnL | VaR₉₅ | CVaR₉₅ | Realized vol | Max DD | Sharpe | Sortino |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean reversion | 162 | −1,900 | −3,749 | −4,256 | 1,144 | 309,459 | −1.66 | −0.84 |
| Trend following | 162 | 782 | −2,032 | −2,740 | 1,183 | 6,527 | 0.66 | 0.45 |

Additional observations:
- Mean reversion’s CDF is heavily left-skewed (see `results/week7/plots/pnl_risk/pnl_histograms.png`), producing extreme drawdowns despite similar volatility to trend-following.
- Trend-following’s positive Sharpe/Sortino comes with fat-tailed downside (VaR still −2k), confirming the strategy depends on infrequent but sizable winners.
- The cumulative curves (`results/week7/plots/pnl_risk/cumulative_pnl.png`) show MR drifting steadily lower across the grid, while TF oscillates but trends up after the first 40 parameter buckets.
- Risk–return scatter (`results/week7/plots/pnl_risk/risk_return_scatter.png`) confirms higher inventory volatility correlates with higher absolute PnL swings; MR points cluster in the south-east quadrant (high risk/negative return).

## Validation & anomalies

- The analytics script cross-checked the calibrated “best configs” against raw grid rows (λ/order-size/aggressiveness tuples) and reproduced the documented PnL within floating-point tolerance, confirming the metrics align with the historical runs described in `docs/week7_strategy_simulation.md`.
- No inconsistencies were found between the aggregated breakdown and the source data; the identity column in `pnl_breakdown.csv` stays within floating noise (≤1e−6).
- Remaining risk gap: the MR desk’s tail losses dominate portfolio VaR despite its lower execution drag—further work should target signal quality (e.g., regime filters) instead of microstructure tweaks.

## Artefacts

- `results/week7/pnl_breakdown.csv`
- `results/week7/pnl_risk_metrics.csv`
- `results/week7/plots/pnl_risk/{pnl_histograms,cumulative_pnl,risk_return_scatter}.png`
