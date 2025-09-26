# Neural/Transformer Hawkes Benchmarks

This note captures the end-to-end workflow for recreating the Binance and LOBSTER experiments and aggregating their metrics.

## Environment

1. Create/activate a virtual environment and install dependencies (PyTorch, NumPy, pandas, matplotlib).
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install torch numpy pandas matplotlib
   ```
2. Add the repository root to `PYTHONPATH` when launching experiment scripts:
   ```bash
   export PYTHONPATH=.
   ```

## Data Preparation

### Binance

The repository already contains detrended buy/sell `.npy` arrays under `data/runs/events/`. To rebuild the consolidated dataset:

```bash
python scripts/pack_binance_npz.py \
  --input-dir data/runs/events \
  --symbol BTCUSDT \
  --days 2025-09-21 \
  --output data/runs/events/binance_btcusdt_2025-09-21.npz
```

This produces `binance_btcusdt_2025-09-21.npz` and `binance_btcusdt_2025-09-21.meta.json`.

### LOBSTER

Place the downloaded LOBSTER sample files in `data/lobster/LOBSTER_SampleFile_AAPL_2012-06-21_10/`. Then run:

```bash
python scripts/preprocess_lobster.py \
  --messages data/lobster/LOBSTER_SampleFile_AAPL_2012-06-21_10/AAPL_2012-06-21_34200000_57600000_message_10.csv \
  --symbol AAPL \
  --date 2012-06-21 \
  --output data/runs/events/lobster_aapl_2012-06-21_sample.npz
```

A metadata file with the same stem is emitted alongside the NPZ.

## Running Benchmarks

Two configs sweep GRU vs Transformer backbones:

```bash
PYTHONPATH=. python experiments/run_matrix.py \
  --config experiments/configs/binance_backbones.json \
  --results-dir experiments/results \
  --run-dir experiments/runs

PYTHONPATH=. python experiments/run_matrix.py \
  --config experiments/configs/lobster_backbones.json \
  --results-dir experiments/results \
  --run-dir experiments/runs
```

For each experiment the runner logs deterministic seed/env info, writes the JSON result (`experiments/results/*.json`), and stores artefacts under `experiments/runs/<experiment>/`:

- `metrics.json` — summary per split, KS stats, runtime, parameter counts.
- `curves/loss_curve.csv` and `curves/calibration_next_time.csv` — epoch traces and calibration bins.
- `figs/` — loss, calibration, QQ, and KS plots.

## Aggregation

Collect all run metrics into a single table:

```bash
python scripts/collect_runs.py \
  --run-dir experiments/runs \
  --output experiments/summary/benchmarks.csv
```

The CSV currently includes four rows (Binance/Lobster × GRU/Transformer). Update after additional experiments to keep the summary synchronized.

## Notes

- All commands assume the repository root as working directory.
- Rerun the packers whenever new raw data arrives (additional Binance symbols or LOBSTER days).
- Plots rely on `matplotlib`; ensure the environment has write access to its cache directory or set `MPLCONFIGDIR`.
