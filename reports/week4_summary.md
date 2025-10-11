# Week 4 Technical Report

## 1. Profiling & Performance Results
- Re-ran deterministic order-book benchmarks (`python/scripts/run_benchmarks.py`) with `seed=42`. Baseline throughput rose to **13.1k msg/s** (wall time 0.153s), up ~6.9% vs. the prior logs snapshot (`logs/benchmarks/order_book_benchmarks.json`). The 10× and 100× scenarios improved by 8.1% and 4.9% respectively, lowering wall clock by 4.6–7.5%.
- Normalised benchmark artefacts now live under `results/week4/perf/` with paired metadata (JSON + CSV + plot). `analysis.profiling` consolidates cProfile/tracemalloc capture while `analysis.plots` standardises chart generation.
- Added `scripts/run_ci_checks.sh` so CI and local runs share the black/flake8/pytest sequence with the same deterministic Matplotlib cache.

## 2. Stress Testing & Validation
- Regenerated the Poisson/burst stress suite via `python/scripts/run_stress_suite.py` (seed 2024). Throughput scaled linearly: ×1 = **21.1k msg/s**, ×10 = **21.7k msg/s**, ×100 = **22.5k msg/s** with avg latency holding at 28–30 µs. Outputs+figures reside in `results/week4/stress/analysis/` with metadata sidecars.
- Aggregated log integrity and throughput distributions (`perf_run_summary.csv`) confirm deterministic digests (`de8c7…`, `bdb68…`) and zero orphan cancels/executes.
- Architecture comparison (`benchmark_architectures.py`) shows single-thread loop sustaining **13.1k msg/s** vs **7.4k msg/s** for the concurrent runner (3 seeds), flagging queue hand-off overhead. All runs share digest `79067c52…`, evidencing reproducibility.
- Added deterministic JSON logging (sorted keys) and Matplotlib cache handling; full Python test suite now passes with the new imports and schema changes.

## 3. Key Lessons & Next Week Plan
- Consolidating profiling utilities into `python/analysis` cut duplication and made cross-language tooling (C++/Python) easier to extend.
- Metadata-wrapped artefacts simplified validation and will streamline tagging/publishing for v0.9-beta.
- Matplotlib/fontconfig friction still exists on shared environments—keeping `MPLCONFIGDIR` explicit prevents flaky CI.

**Next Week Priorities**
1. Integrate C++ order-book benchmarks into the automated results pipeline and surface diff tooling for latency histograms.
2. Extend stress validation to cover realistic replay traces with risk-engine assertions (inventory, throttle breaches).
3. Draft automation for publishing releases (Git tag + GitHub release upload) tied to metadata artefacts.
