# Binance Event Extraction Workflow

## Raw Inputs
- `data/runs/raw/BTCUSDT-trades-2025-09-19.csv`
- `data/runs/raw/BTCUSDT-trades-2025-09-20.csv`
- `data/runs/raw/BTCUSDT-trades-2025-09-21.csv`

## Processing Commands
1. Clean trades and derive 1s bars (run once per day):
   ```bash
   python scripts/preprocess_binance.py data/runs/raw/BTCUSDT-trades-YYYY-MM-DD.csv --quiet
   ```
2. Extract buy-side event times and marks (saves `.npy` arrays per day):
   ```bash
   python3 - <<'PY'
   import pandas as pd, numpy as np
   from pathlib import Path

   Path('data/runs/events').mkdir(parents=True, exist_ok=True)

   for day in ('2025-09-19', '2025-09-20', '2025-09-21'):
       clean = Path(f'data/runs/processed/BTCUSDT-trades-{day}-clean.csv')
       df = pd.read_csv(clean, usecols=['ts_ms', 'side', 'signed_qty'])
       buys = df[df['side'] == 'buy'].copy()
       times = buys['ts_ms'].to_numpy(dtype='float64') / 1000.0
       rel = times - times[0]
       marks = buys['signed_qty'].to_numpy(dtype='float64')
       np.save(f'data/runs/events/BTCUSDT-{day}-buys-times.npy', rel)
       np.save(f'data/runs/events/BTCUSDT-{day}-buys-marks.npy', marks)
       print(f'{day}: saved {len(rel)} buy events')
   PY
   ```
3. Z-score normalise the marks for numerical stability:
   ```bash
   python3 - <<'PY'
   import numpy as np
   from pathlib import Path

   folder = Path('data/runs/events')
   for day in ('2025-09-19', '2025-09-20', '2025-09-21'):
       marks_path = folder / f'BTCUSDT-{day}-buys-marks.npy'
       marks = np.load(marks_path)
       mean, std = marks.mean(), marks.std()
       nz = marks if std == 0 else (marks - mean) / std
       np.save(folder / f'BTCUSDT-{day}-buys-marks-z.npy', nz)
       print(f'{day}: mean={mean:.6f}, std={std:.6f}')
   PY
   ```
4. Fit an exponential Hawkes model (EM variant shown):
   ```bash
   python3 - <<'PY'
   import numpy as np
   from hawkeslib.model.uv_exp import UnivariateExpHawkesProcess

   times = np.load('data/runs/events/BTCUSDT-2025-09-21-buys-times.npy')
   model = UnivariateExpHawkesProcess()
   loglik = model.fit(times, method='em')
   mu, alpha, beta = model.get_params()
   print({'loglik': loglik, 'mu': mu, 'alpha': alpha, 'beta': beta})
   PY
   ```

## Generated Artefacts
- `data/runs/processed/BTCUSDT-trades-*-clean.csv` — cleaned trade-level data
- `data/runs/processed/BTCUSDT-trades-*-1s-bars.csv` — 1-second OHLCV bars
- `data/runs/events/BTCUSDT-*-buys-times.npy` — detrended buy event timestamps (seconds)
- `data/runs/events/BTCUSDT-*-buys-marks.npy` — raw signed buy volumes
- `data/runs/events/BTCUSDT-*-buys-marks-z.npy` — z-scored marks for Hawkes fitting

## Notes
- Replace `YYYY-MM-DD` when running commands for new days.
- Record fitted parameter triples `(mu, alpha, beta)` per dataset to compare stability over time.
- Use the generated `.npy` arrays as inputs to downstream Hawkes simulations or neural extensions.
