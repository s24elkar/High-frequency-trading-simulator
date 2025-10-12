# Release Notes

## v0.9-beta — Week 4 Baseline

### Core Simulation Stability
- Stress harness rerun with the deterministic seed pipeline (`run_stress_suite.py` + `_configure_rngs`), confirming zero orphan cancels/executes and stable digests (`de8c75…`, `bdb68d…`).
- C++ side now exposes `perf::ScopedTimer`, enabling scoped latency capture without duplicating chrono boilerplate; `src/main.cpp` showcases the helper.

### Profiling & Benchmarking
- New `python/perf` package centralises `BenchmarkResult`, architecture run schemas, and CSV normalisation so every export shares the same header order.
- Global RNG initialisation (`random` + `numpy`) added to throughput and stress scripts, guaranteeing repeatable digests across reruns.
- Baseline throughput improved to **12.4k msg/s** (+1.9% vs. early-week logs), with x10 at **12.4k msg/s** (+4.8%) and x100 at **11.3k msg/s** (+1.0%).

### Stress-Test Outcomes
- Poisson/burst suite sustained **20.3–21.9k msg/s** with 29–32 µs average latency; latency histograms and scenario CSVs now adhere to `STRESS_SCENARIO_FIELDS` ordering.
- Architecture comparison shows single-thread loop averaging **12.6k msg/s** versus **7.0k msg/s** for the concurrent runner (matching digests across all runs).

### Reproducibility & CI
- `.gitignore` updated for virtualenvs and `__pycache__`, keeping artefacts clean.
- CI workflow adds a smoke-test stage that executes deterministic benchmark/stress scripts in a tmp directory under a fixed Matplotlib cache.
- `results/week4/README.md` documents the `/perf`, `/stress`, and `/plots` structure; all charts ship with paired `*.meta.json` envelopes.

### Distribution
- Tag recommendation: `git tag -a v0.9-beta -m "Week 4 validated baseline" && git push origin v0.9-beta` once artefacts are committed.
