# Week 6 Performance Optimization

## Profiling approach

- Built an in-process profiler (`perf/Profiler.*`) guarded by `HFT_ENABLE_PROFILING` to capture per-scope wall time without external tools (local environment lacks `perf`, `gprof`, and `valgrind`).
- Instrumented Hawkes intensity sampling, the event loop, order-book matching, and execution cost calculations via `HFT_PROFILE_SCOPE`/`HFT_PROFILE_STACK`.
- Generated folded stacks (`results/week6/perf_*/profile_*.folded`) and converted them into SVG flame graphs with the bespoke `scripts/generate_flamegraph.py`.
- Captured two views: baseline (`results/week6/perf_baseline/profile_baseline.svg`) and optimized (`results/week6/perf_optimized/profile_optimized.svg`).

## Baseline hotspots

- Hawkes intensity dominated CPU time (≈5% of total in a 60 s session) due to vector-of-vector copies inside `sample_next`.
- Event dispatch (`LatencyQueue` priority queue) consumed ≈11.6% during queue flushes.
- Order-book updates and execution recording together accounted for ≈4%.

## Key optimizations

1. **Contiguous Hawkes state (`src/hawkes_core.cpp`)**
   - Flattened `alpha`, `beta`, and excitation matrices into row-major vectors.
   - Reused pre-sized buffers to avoid per-iteration allocation and reduced acceptance loop copy costs.
   - Result: `Hawkes::sample_next` time dropped from 568 µs to 398 µs (−29.9%).

2. **Lock-free latency queue (`src/latency_model.hpp`)**
   - Replaced `std::priority_queue` with a single-producer single-consumer ring buffer feeding a min-heap for ready events.
   - Provides lock-free ingestion and contiguous storage; legacy mode preserved via `HFT_USE_LEGACY_LATENCY_QUEUE`.

3. **Order book locality (`src/OrderBook.*`)**
   - Swapped the locator hash-map with a contiguous lookup table keyed by order id.
   - Reduced indirection during match/cancel churn.

4. **Compiler flags**
   - Enabled `-O3 -march=native -flto -fno-rtti` across all targets (legacy-compatible kernels now avoid RTTI).

## Benchmarking (1M-event synthetic runs)

Executed `perf_benchmark` with Hawkes parameters scaled by ×16 to reach one million arrivals quickly. Results saved to `results/week6/performance_benchmark.csv`.

| Label | Events | Elapsed (s) | Throughput (events/s) | Mean Latency (µs) | Max RSS (MB) |
|-------|--------|-------------|------------------------|-------------------|--------------|
| baseline | 1,000,000 | 1.107 | **903,195** | 117.1 | 577.7 |
| optimized | 1,000,000 | 1.197 | 835,150 | 117.1 | 579.2 |
| baseline_scale4 | 1,000,000 | 1.079 | **926,716** | 119.5 | 578.7 |
| optimized_scale4 | 1,000,000 | 1.118 | 894,734 | 119.5 | 577.1 |

> **Note:** The lock-free latency queue introduces ~7–8% overhead versus the legacy priority queue in single-threaded mode. The Hawkes and order-book optimizations still lower sampling cost (see profiler_report.txt), but further tuning is planned to re-gain throughput while keeping the lock-free design.

## Profiling summary

`profiler_report.txt` aggregates baseline and optimized scope timings. Highlights:

- Hawkes sampling share decreased from 4.97% to 3.49%.
- Event-loop Hawkes sampling scope dropped from 682 µs to 542 µs.
- Queue flush cost increased (11.6% → 14.4%) because of additional heap maintenance in the lock-free queue.

## Flame graphs

- Baseline: `results/week6/perf_baseline/profile_baseline.svg`
- Optimized: `results/week6/perf_optimized/profile_optimized.svg`

## Limitations & next steps

- Hardware PMU tools (`perf stat`, `valgrind --tool=callgrind`) are unavailable in the sandbox, so branch-mispredict and memory-stall counters could not be sampled. The in-process profiler plus flame graphs provide actionable visibility despite this gap.
- Throughput regression suggests revisiting the lock-free queue implementation—candidates include batching heap rebuilds or using a bucketed calendar queue.
- Add automated benchmark targets to CI so regressions in Hawkes sampling or queue dispatch surface immediately.
