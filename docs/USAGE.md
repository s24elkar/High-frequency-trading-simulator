# High-Frequency Trading Simulator — Usage Guide

This guide expands on the README quick start and walks through building, running, and extending the simulator stack. It covers both the C++ limit-order-book core and the accompanying Hawkes-process analytics toolkit.

## 1. Components at a Glance
- **`src/` C++17 order book** — an in-memory limit-order-book (`OrderBook`) with deterministic event processing and a sample driver (`main.cpp`).
- **`src/hawkes.hpp` kernels** — header-only exponential and power-law Hawkes process primitives plus thinning samplers.
- **`src/hawkes_example.cpp` demo** — illustrates the C++ Hawkes API and prints aggregate counts.
- **`python/` analytics package** — NumPy/Matplotlib-powered Hawkes simulator, plotting helpers, and CSV/JSON exporters.
- **`tests/order_tests.cpp`** — Catch2 regression coverage for price-time priority and cancellations.
- **`data/` & `docs/images/`** — deterministic outputs (event streams, plots) produced by the Python demos.

## 2. Prerequisites
### C++ toolchain
- CMake **3.15+**
- A C++17-capable compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 2019+)

### Python environment
- Python **3.10 or newer**
- `pip`, `venv`, and a working C toolchain if you plan to install binary wheels without prebuilt distributions

## 3. Configure and Build
Choose an out-of-source build directory (e.g., `build/release`). All commands below run from the project root.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target hft_sim
```

- Use `-DCMAKE_BUILD_TYPE=Debug` for instrumentation-friendly builds.
- Pass `-GNinja` if you prefer Ninja over Makefiles.
- The resulting binary lives at `build/release/hft_sim` (or `.exe` on Windows).

### Optional targets
- `hawkes_example` — built alongside `hft_sim` when you run `cmake --build … --target hawkes_example`.
- `order_tests` — built when configuration enables tests (see §6).

## 4. Running the Order-Book Driver
Execute the sample program after building:

```bash
./build/release/hft_sim
```

The driver seeds a handful of limit orders, prints the best bid/ask, and shows a level-by-level snapshot. Use it as a template:

```cpp
OrderBook book;
book.addLimitOrder({42, Side::Buy, 100.15, 25, timestamp_ns});
book.cancel(42);
auto best = book.bestBid();
```

### Key API calls (`OrderBook`)
- `addLimitOrder(const Order&)` — inserts a FIFO-respected order; aggressive liquidity removes resting volume if you extend matching logic.
- `cancel(OrderId)` — removes a live order; returns `false` when the ID is unknown.
- `bestBid()/bestAsk()` — expose the current top-of-book `Order` (`std::optional`).
- `levels(depth)` — returns aggregated depth information (price, total quantity, FIFO queue) for the first *depth* levels on each side.

Adapt `src/main.cpp` or build your own executable that links `OrderBook.cpp` to stream events from files, sockets, or strategy code.

## 5. Hawkes Process Toolkit (C++)
Build and run the demo target:

```bash
cmake --build build/release --target hawkes_example
./build/release/hawkes_example
```

You should see counts similar to:

```
[EXP] generated 237 events
[PL]  generated 184 events
```

### Using the header in your code
- Include `src/hawkes.hpp` and add it to your include path (or install it under `include/` if you package the library).
- Pick a kernel (`ExpKernel` or `PowerLawKernel`), provide a mark sampler (std::function returning double marks), and call `simulate_exp` or `simulate_general`.
- Both simulators accept a time horizon `T`, base intensity `mu`, kernel parameters, and an optional RNG seed for reproducibility.

Example snippet:

```cpp
hawkes::ExpKernel kernel{0.6, 1.4};
auto mark_sampler = [](auto& rng) {
    std::lognormal_distribution<double> d(0.0, 0.4);
    return d(rng);
};
auto result = hawkes::simulate_exp(0.25, kernel, mark_sampler, 3600.0, /*seed=*/2024);
```

`result.t` and `result.v` contain event timestamps and marks respectively.

## 6. C++ Tests
Enable tests at configuration time:

```bash
cmake -S . -B build/tests -DHFT_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/tests --target order_tests
ctest --test-dir build/tests --output-on-failure
```

- Tests use Catch2 v3.5.2 (downloaded automatically via CMake FetchContent).
- The suite validates best bid/ask selection, FIFO ordering at a price level, and cancellation handling. Extend `tests/order_tests.cpp` with additional cases when adding features.

## 7. Python Analytics Toolkit
All Python paths below assume a POSIX shell. On Windows, replace activation with `.venv\Scripts\activate`.

### Set up a virtual environment
```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If Matplotlib needs a writable cache (common in read-only CI environments), run the demos with `MPLCONFIGDIR=.matplotlib` to direct cache files locally.

### Run the bundled demo
```bash
MPLCONFIGDIR=.matplotlib python -m demo
```

The script:
- Runs exponential and power-law Hawkes simulations (`ExpKernel` / `PowerLawKernel`).
- Prints branching ratios to confirm subcritical parameter choices.
- Saves plots under `docs/images/` and event streams under `data/runs/` (CSV + JSON with metadata).
- Calls `plt.show()` for interactive inspection.

### Using the Python API programmatically
```python
from python.simulate import simulate_thinning_exp_fast
from python.kernels import ExpKernel

kernel = ExpKernel(alpha=0.7, beta=1.1)
mu = 0.25
T = 500.0
mark_sampler = lambda rng: float(rng.exponential(1.0))
times, marks = simulate_thinning_exp_fast(mu, kernel, mark_sampler, T, seed=9001)
```

Additional helpers:
- `python.viz.intensity_on_grid` — reconstruct λ(t) over any grid.
- `python.viz.plot_counts_acf` / `plot_mark_acf` — quick-look diagnostics for clustering.
- `python.io.save_csv` / `save_json` — persist runs with metadata for reproducible studies.

## 8. Working with Outputs
- Simulation artefacts (CSV/JSON) land under `data/runs/`; filenames encode the kernel (`exp_events`, `power_events`).
- Generated figures live in `docs/images/`. Consider Git LFS if you expect large plots or many variations.
- Each JSON file embeds kernel parameters, base intensity, horizon, branching ratio, and mark distribution description to aid reproducibility.

## 9. Extending the Project
- Add new C++ strategies by creating another executable in `src/` that links `OrderBook.cpp` and your strategy sources.
- Introduce alternative Hawkes kernels by extending `hawkes::Result` utilities in `src/hawkes.hpp` and mirroring them in `python/kernels.py` / `python/simulate.py` for parity.
- When contributing data-heavy assets, accompany them with scripts in `python/` or C++ utilities so results stay regenerable.
- Update Catch2 tests and Python demos whenever you change order book semantics or Hawkes parameterizations.

## 10. Troubleshooting Checklist
- **Missing Catch2 headers** — ensure you configured with `-DHFT_ENABLE_TESTS=ON`; FetchContent will vendor Catch2 automatically.
- **Matplotlib permission errors** — set `MPLCONFIGDIR` to a writable directory (see §7).
- **Hawkes header include failures** — add `src/` (or your relocated include directory) to your compiler include paths when embedding the header-only library elsewhere.
- **Seeding reproducibility** — both C++ and Python simulators accept explicit seeds; fix them when generating artefacts you intend to compare.

For a concise overview and quick-start commands, see the top-level `README.md`.
