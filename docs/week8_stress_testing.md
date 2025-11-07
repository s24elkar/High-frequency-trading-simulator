# Week 8 – Stress & Fault-Tolerance Validation

## Test configurations

| Scenario | Message rate | Injection | Duration | Notes |
| --- | ---: | --- | ---: | --- |
| `network_disconnect` | 120k msg/s | Primary feed unplug for 750 ms | 3 min | Replay backlog from checkpoint, validate state replay. |
| `mq_overflow` | 150k msg/s | Queue depth artificially capped to trigger overflow | 3 min | Measures resilience of back-pressure. |
| `cpu_io_bottleneck` | 110k msg/s | Core pinning + throttled disk checkpoints | 3 min | Exercises hot restart path while CPU saturated. |

Input and results stored in `results/week8/fault_tolerance.csv`, `results/week8/resource_profile.txt`, and `logs/week8/failover_tests.log`.

## Fault tolerance results

| Scenario | Recovery time (ms) | Dropped messages | Order-state accuracy (%) | State drift (bps) |
| --- | ---: | ---: | ---: | ---: |
| network_disconnect | 755.2 | 33 | 99.858 | 0.54 |
| mq_overflow | 1,067.4 | 44 | 99.892 | 0.37 |
| cpu_io_bottleneck | 757.8 | 32 | 99.873 | 0.46 |

Takeaways:
- Network cuts recover in ~0.75 s with minimal drift (<1 bp) but still lose 33 packets before the replay window kicks in. To eliminate gaps, add forward error correction or request/response gap fill after 100 ms outage.
- Queue overflow is the most punishing (1.06 s recovery, 44 drops). Increasing the queue depth plus per-producer flow control should reduce both dropped messages and recovery duration.
- CPU/I/O contention has comparable recovery time to network faults, validating that the checkpoint writer can keep up even when cores are pinned.

## Resource profiling

From `results/week8/resource_profile.txt`:

- **CPU hot spots** – `matching::match_into` consumes 47.3 % of samples due to cache thrash at deep book levels. Action: compact price-level iterators and prefetch depth blocks. Feed decode (`feeds::decode_l2`, 18.9 %) is the next target; using SIMD/AVX2 batch decoders should cut this in half.
- **Memory & leaks** – No lasting leaks, though arena fragmentation appears after 45 min soaks; enable periodic scavenging or switch to `mimalloc`.
- **Locks** – Matching spin-lock shows 9.2 % contention at 150k msg/s; replace with queue-based MCS lock. Checkpoint mutex p99 hold time (3.1 ms) blocks failover threads; move flush operations to an async pool.

## Failover & recovery

Entries in `logs/week8/failover_tests.log` show three hot-restart drills. Failover time stays between 626–667 ms with state reload bounded to <190 ms. PnL continuity deltas remain within ±0.85, confirming checkpoint cadence is tight enough for live restarts. Recommendation: add health probes that trigger state dumps every 30 s to keep continuity error below ±0.25.

## Latency under load

- Stress runs still keep real-time loop averages below 60 µs (see `realtime_loop_metrics.csv`), but p99 grows past the 100 µs SLA once jitter/packet loss is introduced. The latency histogram is strongly lognormal; high tails stem from retransmit queues, so job prioritisation is preferable to simply adding threads.
- OMS validation shows entry/cancel p99s (425 µs / 328 µs baseline vs. 650 µs / 496 µs under stress). Trade confirmation accuracy remains ≥99.9 %, but queue overflow pushes backlog to 259 streaming messages. Auto-scaling stream consumers will keep dashboards real-time.

## Optimization recommendations

1. **Back-pressure aware messaging** – Increase inter-process queue size, add adaptive shedding for low-priority metrics, and monitor occupancy to prevent overflow recovery storms.
2. **Lock refactors** – Replace spin-lock with MCS lock and shard risky data structures (inventory map, checkpoint writer) to reduce contention under 150k msg/s.
3. **Checkpoint tuning** – Move disk flushes off the hot path and lower snapshot interval to 30 s, cutting PnL continuity deltas during failover.
4. **Feed resilience** – Implement gap-fill RPC or FEC to absorb >500 ms network disconnects without any dropped packets.
