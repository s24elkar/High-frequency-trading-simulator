# Week 4 Technical Report

## 1. Profiling & Performance Results
- Introduced the `python/perf` package (shared dataclasses, schema helpers, CSV normalisation) and a header-only `perf::ScopedTimer` for C++; benchmark runners and the Streamlit bridge now consume the shared utilities.
- Re-ran deterministic order-book benchmarks with `seed=42`. Throughput improved to **12.4k msg/s** (+1.9%) for baseline, **12.4k msg/s** (+4.8%) for ×10, and **11.3k msg/s** (+1.0%) for ×100 scenarios compared with `logs/benchmarks/order_book_benchmarks.json`; matching digests confirm reproducibility.
- CI gained a smoke-test stage that executes the throughput and stress scripts in a temporary directory under a fixed Matplotlib cache, ensuring the artefact pipeline stays deterministic.

## 2. Stress Testing & Validation
- `run_stress_suite.py` now fixes RNG seeding via `_configure_rngs` and replays Poisson/burst mixes with zero orphan events; scenario throughput landed at **20.3–21.9k msg/s** with 29–32 µs average latency.
- `aggregate_stress_metrics.py` writes scenario, throughput-series, and PnL summaries using canonical headers (`STRESS_SCENARIO_FIELDS`, `THROUGHPUT_SERIES_FIELDS`, `PERF_RUN_SUMMARY_FIELDS`) and regenerates all figures with metadata sidecars.
- Architecture comparison (seed ladder) shows the single-thread loop averaging **12.6k msg/s** versus **7.0k msg/s** for the concurrent runner; identical digests across six runs validate determinism.
- Full Python test suite (`pytest tests/python`) passes on the refactored codebase, providing regression coverage for the new modules.

## 3. Key Lessons & Plan for Week 5
- Centralising performance schemas removed ad-hoc CSV ordering and will simplify future diff tooling (e.g., verifying latency regressions via schema-aware comparisons).
- Deterministic seed plumbing must exist in both Python and C++; introducing `perf::ScopedTimer` hints at a lightweight microbenchmark harness that can feed into the shared schema tooling.
- CI smoke-tests add confidence, but we still need automated diff reports (plots + tabular deltas) to flag regressions without manual inspection.

**Next Week Priorities**
1. Extend benchmark CLI to emit change reports versus a stored baseline (CSV/JSON diff + alert thresholds).
2. Integrate realistic market replays (LOBSTER/Binance) into stress validation with PnL/risk assertions.
3. Automate release packaging: signed `v0.9-beta` tag, GitHub release draft pre-populated with artefact metadata, and docs cross-links.
