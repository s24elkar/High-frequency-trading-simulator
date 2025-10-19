#pragma once

#include "hawkes_engine.hpp"
#include "latency_model.hpp"
#include "OrderBook.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace simulator {

struct SimulationConfig {
    std::vector<double> mu;
    ExponentialHawkesProcess::Matrix alpha;
    ExponentialHawkesProcess::Matrix beta;
    double session_length{60.0};
    double latency_mean_us{120.0};
    std::uint64_t seed{1337};
    std::filesystem::path event_log_path{};
};

struct IntensitySample {
    double time{};
    std::vector<double> lambda;
};

struct ArrivalRecord {
    double raw_time{};
    double arrival_time{};
    double latency{};
    std::size_t dimension{};
    double intensity_total{};
    double intensity_dimension{};
};

struct SimulationResult {
    double horizon{};
    std::vector<IntensitySample> intensity_trace;
    std::vector<ArrivalRecord> arrivals;
    double mean_interarrival{};
    double variance_interarrival{};
    double mean_intensity{};
};

class SimulatorCore {
public:
    explicit SimulatorCore(SimulationConfig config);

    SimulationResult run();

private:
    struct PendingPayload {
        HawkesEvent event;
    };

    void seed_order_book(OrderBook& book);
    void update_summary_metrics(SimulationResult& result) const;
    void write_event_log(const SimulationResult& result) const;

    SimulationConfig config_;
    ExponentialLatencyModel latency_model_;
    mutable std::uint64_t next_order_id_{1};
};

SimulationConfig default_btcusdt_config();

} // namespace simulator
