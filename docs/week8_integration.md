# Week 8 – System Integration & Performance Benchmarks

## Architecture overview

```
┌────────────────────┐      ticks        ┌───────────────────┐      orders      ┌────────────────────┐
│ MarketDataSimulator├───────▶──────────▶│   StrategyEngine  ├──────▶──────────▶│ ExecutionHandler   │
└─────────┬──────────┘                   └─────────┬─────────┘                 └─────────┬──────────┘
          │                                         │                                       │
          │                                         │fills                                  │fills
          │                                         ▼                                       ▼
          │                             ┌───────────────────┐                    ┌────────────────────┐
          └──────────────ticks─────────▶│   RiskAnalytics   │◀──────────────────▶│   EventBus / PnL   │
                                        └───────────────────┘                    └────────────────────┘
```

- **Event bus & executor** – a lightweight asynchronous messaging fabric built on a work-stealing task executor. Producers (`MarketDataSimulator`) publish events, while subscribers (strategy, execution handler, risk analytics) run on worker threads via non-blocking queues. This mimics ZeroMQ/Boost.Asio style fan-out without introducing external dependencies.
- **Market simulator** – generates synthetic prices at configurable rates using Gaussian shocks. Events carry timestamps and sequence numbers, ensuring deterministic replay.
- **Strategy engine** – single-feed signal processor (EMA vs. spot divergence). Decisions are stateless aside from EMA/position caches, so multiple strategies can subscribe in parallel.
- **Execution handler** – consumes orders, samples latency/slippage, and emits execution reports asynchronously (latency is recorded, not slept). PnL impact is fed back immediately.
- **Risk/PnL analytics** – maintains inventory lots, realised/unrealised PnL, and streams structured updates. Inventory marking relies on the latest market tick.
- **Structured logging** – `spdlog`-style JSON logger (`logs/week8/integration_test.log`) captures every order, execution, PnL update, scenario boundary, and integration summary in a single file suitable for downstream ingestion.

## Inter-module interfaces

| Channel   | Producer                | Consumer(s)           | Payload fields                                             |
| ---       | ---                     | ---                   | ---                                                         |
| `Market`  | MarketDataSimulator     | StrategyEngine, RiskAnalytics | `price`, `timestamp_s`, `sequence`                     |
| `Orders`  | StrategyEngine          | ExecutionHandler      | `order_id`, `side`, `price`, `quantity`, `strategy`         |
| `Executions` | ExecutionHandler     | RiskAnalytics         | `order_id`, `side`, `fill_price`, `quantity`, `latency_us`  |
| `Metrics` | RiskAnalytics + Runner  | CSV/log writers       | aggregated latency vectors, throughput, CPU & memory stats  |

All channels are asynchronous and non-blocking; handlers are scheduled on the executor’s worker pool, giving predictable latency under load while keeping module code largely synchronous.

## Performance benchmarks

Metrics recorded in `results/week8/system_perf.csv`, produced directly by `build/week8_integration`:

| Scenario  | Event rate (Hz) | Events | Avg order latency (µs) | Throughput (events/s) | CPU (%) | Memory (MB) |
| ---       | ---:            | ---:   | ---:                   | ---:                  | ---:    | ---:        |
| baseline  | 1,500           | 1,200  | 249.26                 | 2,235.15              | 34.77   | 4.13        |
| high_load | 3,200           | 2,500  | 320.18                 | 4,695.73              | 67.79   | 4.13        |
| stress    | 5,500           | 4,000  | 375.87                 | 7,256.74              | 71.45   | 4.13        |

Observations:
- Latency scales modestly with throughput: stress tests add ~50% latency over baseline while throughput more than triples.
- CPU utilisation remains under 75% even in the stress scenario thanks to the executor’s bounded worker pool and zero-copy event dispatch.
- Memory footprint is flat (~4 MB RSS), confirming queues remain bounded despite heavy messaging.

## Logging & monitoring

- Real-time JSON logs exist at `logs/week8/integration_test.log`. Each record includes ISO-8601 timestamps, event type (`order`, `execution`, `pnl_update`, etc.), and key metrics (prices, sizes, latencies), satisfying the structured logging requirement.
- The logging surface is intentionally verbose to facilitate live dashboards; downstream tooling can tail the log and update time-series widgets for orders/fills/PnL without touching the CSV outputs.

## Reproduction

1. Build the integration runtime: `g++ -std=c++20 -O3 -Wall -Wextra -pedantic -Iinclude -o build/week8_integration src/week8_system_integration.cpp -lpthread`
2. Run: `./build/week8_integration`
3. Artefacts generated:
   - `/results/week8/system_perf.csv`
   - `/logs/week8/integration_test.log`

These, together with this document, complete the Phase I integration deliverables.

## Real-time validation & OMS checks

- **Real-time loop metrics (`results/week8/realtime_loop_metrics.csv`)** – baseline runs stay within the 100 µs SLA (avg 54.8 µs, p99 98.2 µs), while jittered and lossy testbeds highlight tail inflation (p99 up to 119.8 µs, 276 breaches). These distributions effectively serve as latency histograms: the lognormal core is tight, but packet loss plus jitter adds a high-latency tail that now masks 8–9% of the budget.
- **Network resilience** – injected jitter (±25 µs) and packet-loss up to 0.4% caused only 13 dropped events thanks to queue replays, but tail latency breached the budget. Remediation plan: add adaptive batching for bursts and leverage priority queues for retransmits.
- **OMS validation (`results/week8/oms_validation.csv`)** – entry latency averages 216 µs (p99 425 µs) in baseline, climbing to 341 / 650 µs under stress. Cancel paths are faster (180 / 328 µs baseline). Trade confirmation accuracy remains ≥99.9% and position drift stays under 1.5 bps. Bottleneck: confirmation fan-out during stress saturates the in-memory queue; to remediate, rate-limit cancel bursts and shard confirmation writers.

## PnL streaming telemetry

- `logs/week8/pnl_stream.log` captures six high-frequency batches pumped across the in-memory bus (Redis Streams analogue). Average end-to-end latency stayed between 0.6–3.5 ms with gradual backlog growth (33 → 259 messages) as throughput exceeded consumer capacity. Dashboard widgets kept pace because backlog was bounded and max latency stayed below 6.4 ms; the remediation plan is to auto-scale consumers once backlog exceeds 200 messages for >3 consecutive batches.

## Reproduction (Phase II)

1. Build and run the integration binary as above for core performance numbers.
2. Execute the real-time validation harness: `python3 python/scripts/run_week8_realtime_validation.py`
   - Produces `/results/week8/realtime_loop_metrics.csv`, `/results/week8/oms_validation.csv`, and `/logs/week8/pnl_stream.log`.
