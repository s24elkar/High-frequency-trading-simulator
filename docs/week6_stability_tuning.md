# Week 6 Stability Tuning

## Optimization Summary
- Replaced per-event dynamic allocations with a synchronized polymorphic memory pool (`IntensityBuffer`) and moved intensity handoff to avoid copies between threads.
- Added `__builtin_prefetch` hints in `hawkes_core.cpp` hot loops and in `matching_engine.cpp`'s ready-queue drain to reduce cache misses.
- Converted event and match counters to atomics and tracked queue backpressure via retry counters.
- Introduced `EngineRuntime` for orchestrated multi-thread simulation with resource monitoring and queue metrics.
- Hardened the build by applying `-O3 -march=native -fno-exceptions -fno-rtti -DNDEBUG` across targets.

## Stress Test Results
- Scenario: `stress_test --events 10_000_000 --minutes 60 --scale 8`
- Summary metrics (`results/week6/stress_test_metrics.csv`):

  | Metric | Value |
  | --- | --- |
  | Events generated / processed | 10,000,000 |
  | Mean inter-thread latency | 456,910 µs |
  | P99 inter-thread latency | 938,173 µs |
  | Max inter-thread latency | 2,476,424 µs |
  | Mean arrival latency | 120 µs |
  | Throughput | 680,523 events/s |
  | Event queue retries | 7,576,742 |
  | Match queue retries | 0 |

- Queue backpressure manifested as retries (no drops). Retry ratio: 0.76 per event; worth monitoring alongside queue sizing.
- CPU & memory timeline (36 samples) captured in the same CSV (`type=resource` rows) shows peak RSS ≈ 1.04 GB and CPU utilization trending ~2× real-time while all three threads remained active.
- Console trace stored at `results/week6/stress_test.log`.

## Sanitizer Runs
- ThreadSanitizer: `build-tsan/stress_test --events 200000 --minutes 2 --scale 8` (log: `results/week6/stress_test_tsan.log`). No race reports emitted.
- AddressSanitizer: `build-asan/stress_test --events 200000 --minutes 2 --scale 8` (log: `results/week6/stress_test_asan.log`). No memory errors detected.

## Artifacts
- Metrics: `results/week6/stress_test_metrics.csv`
- Stress log: `results/week6/stress_test.log`
- TSAN log: `results/week6/stress_test_tsan.log`
- ASAN log: `results/week6/stress_test_asan.log`

