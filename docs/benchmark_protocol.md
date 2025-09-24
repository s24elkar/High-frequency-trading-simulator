# Benchmark Protocol

This document outlines the steps to reproduce the benchmarking suite.

## 1. Data Preparation
1. Run `scripts/preprocess_binance.py` on each day/symbol to produce cleaned trade CSVs and event arrays.
2. Catalogue all datasets in `data/catalog.csv` (symbol, venue, date range, event count, mean gap).
3. For cross-exchange studies, repeat for other venues (e.g., CME, Coinbase, NYSE).

## 2. Model Families
- **Classical Hawkes**: exponential kernel via `tick.hawkes.HawkesExpKern` (MLE/EM), optionally sums of exponentials and power-law.
- **Neural Hawkes** (Du et al. 2016): RNN hidden state with learnable intensity; our surrogate approximates this with the `NeuralHawkesModel` class.
- **Transformer Hawkes** (Zuo et al. 2020): implement as an attention-based backbone using time-aware positional encodings.
- **Ablations**: GRU vs LSTM vs MLP, marked vs unmarked, additional features (volume, price delta, imbalance).

## 3. Experiment Configuration
Use `experiments/run_matrix.py` with a config file that lists each symbol/backbone combination. Example fields:
```json
{
  "name": "binance_btc_transformer",
  "dataset": "data/runs/events/BTCUSDT-2025-09-21-combined-times.npy",
  "num_types": 2,
  "training": {
    "window_size": 64,
    "stride": 32,
    "batch_size": 128,
    "epochs": 10,
    "lr": 0.001,
    "backbone": "transformer",
    "hidden_dim": 128,
    "delta_weight": 1.0
  }
}
```

## 4. Metrics
- **Log-likelihood / surrogate loss** on train/val/test splits.
- **Event-type accuracy & MAE** of inter-arrival predictions.
- **Time-rescaling diagnostics**: KS statistic, QQ curves, empirical CDF.
- **Branching ratio** (classical models).
- **Runtime**: CPU vs GPU inference speed per 10k events.

## 5. Reporting
1. Run `experiments/aggregate_results.py` to build comparison tables.
2. Create figures (Jupyter notebooks) for:
   - Intensity reconstruction (fitted vs observed).
   - QQ plots and uniform CDFs.
   - Kernel visualizations (classical and neural equivalents).
3. Write summaries for each symbol/venue focusing on:
   - Which models capture cross-excitation?
   - How does performance vary across regimes? (day/night, high/low volatility)
   - Runtime trade-offs.

This protocol should form the backbone of the experimental section in a research paper.
