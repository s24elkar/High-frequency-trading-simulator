#include "simulator_core.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace simulator {

namespace {

constexpr double kMidPrice = 27000.0;
constexpr double kTickSize = 0.5;
constexpr std::int32_t kBaseQuantity = 5;
constexpr std::size_t kSeedDepth = 8;

SimulationConfig make_default_config() {
    SimulationConfig cfg;
    cfg.mu = {2.7357491833312156, 4.993915774878758};
    cfg.alpha = {
        {14.663804592131706, 0.031195459192378452},
        {0.031083754454383515, 15.999936416220933}
    };
    cfg.beta = {
        {19.99999968685742, 19.99999968685742},
        {19.99999968685742, 19.99999968685742}
    };
    cfg.session_length = 60.0;
    cfg.latency_mean_us = 120.0;
    cfg.seed = 20240607;
    cfg.event_log_path = std::filesystem::path("results/week6/simulation_arrivals.csv");
    return cfg;
}

std::int64_t to_nanoseconds(double seconds) {
    return static_cast<std::int64_t>(seconds * 1'000'000'000.0);
}

Side side_from_dimension(std::size_t dim) noexcept {
    return dim == 0 ? Side::Buy : Side::Sell;
}

} // namespace

SimulationConfig default_btcusdt_config() {
    static const SimulationConfig cfg = make_default_config();
    return cfg;
}

SimulatorCore::SimulatorCore(SimulationConfig config)
    : config_(std::move(config)),
      latency_model_(config_.latency_mean_us) {}

void SimulatorCore::seed_order_book(OrderBook& book) {
    for (std::size_t level = 0; level < kSeedDepth; ++level) {
        const double price_offset = static_cast<double>(level) * kTickSize;
        const std::int32_t qty = kBaseQuantity + static_cast<std::int32_t>(level);
        book.addLimitOrder(Order{
            static_cast<std::int64_t>(next_order_id_++),
            Side::Buy,
            kMidPrice - price_offset,
            qty,
            0
        });
        book.addLimitOrder(Order{
            static_cast<std::int64_t>(next_order_id_++),
            Side::Sell,
            kMidPrice + price_offset,
            qty,
            0
        });
    }
}

SimulationResult SimulatorCore::run() {
    if (config_.mu.empty()) {
        throw std::invalid_argument("SimulatorCore requires calibrated parameters");
    }

    std::mt19937_64 rng(config_.seed);
    ExponentialHawkesProcess process(config_.mu, config_.alpha, config_.beta);
    process.reset(0.0);

    SimulationResult result;
    result.horizon = config_.session_length;

    OrderBook book;
    seed_order_book(book);

    LatencyQueue<PendingPayload> queue;

    auto flush_ready = [&](double time_limit) {
        while (!queue.empty() && queue.top().ready_time <= time_limit) {
            auto entry = queue.pop();
            if (entry.ready_time > config_.session_length) {
                continue;
            }
            ArrivalRecord record;
            record.raw_time = entry.payload.event.time;
            record.arrival_time = entry.ready_time;
            record.latency = entry.delay;
            record.dimension = entry.payload.event.dimension;
            record.intensity_total = entry.payload.event.intensity_total;
            record.intensity_dimension = entry.payload.event.intensity_dimension;

            const Side side = side_from_dimension(record.dimension);
            const std::int32_t quantity = 1;
            const std::int64_t ts_ns = to_nanoseconds(record.arrival_time);

            auto fills = book.match(side, 0.0, quantity);
            if (fills.empty()) {
                seed_order_book(book);
                fills = book.match(side, 0.0, quantity);
            }

            // Maintain simple liquidity replenishment
            if (side == Side::Buy) {
                book.addLimitOrder(Order{
                    static_cast<std::int64_t>(next_order_id_++),
                    Side::Sell,
                    kMidPrice + kTickSize,
                    quantity,
                    ts_ns
                });
            } else {
                book.addLimitOrder(Order{
                    static_cast<std::int64_t>(next_order_id_++),
                    Side::Buy,
                    kMidPrice - kTickSize,
                    quantity,
                    ts_ns
                });
            }

            result.arrivals.push_back(record);
        }
    };

    while (process.current_time() < config_.session_length) {
        const HawkesEvent event = process.sample_next(rng);

        IntensitySample sample;
        sample.time = event.time;
        sample.lambda = process.intensities();
        result.intensity_trace.push_back(std::move(sample));

        const double latency = latency_model_.sample_delay(rng);
        const double ready_time = event.time + latency;
        queue.push(ready_time, latency, PendingPayload{event});

        flush_ready(event.time);
        if (event.time > config_.session_length) {
            break;
        }
    }

    flush_ready(config_.session_length);

    update_summary_metrics(result);
    write_event_log(result);
    return result;
}

void SimulatorCore::update_summary_metrics(SimulationResult& result) const {
    if (result.arrivals.size() < 2) {
        result.mean_interarrival = 0.0;
        result.variance_interarrival = 0.0;
        result.mean_intensity = 0.0;
        return;
    }

    std::vector<double> deltas;
    deltas.reserve(result.arrivals.size() - 1);
    for (std::size_t i = 1; i < result.arrivals.size(); ++i) {
        const double delta = result.arrivals[i].arrival_time - result.arrivals[i - 1].arrival_time;
        if (delta > 0.0) {
            deltas.push_back(delta);
        }
    }
    if (deltas.empty()) {
        result.mean_interarrival = 0.0;
        result.variance_interarrival = 0.0;
    } else {
        const double sum = std::accumulate(deltas.begin(), deltas.end(), 0.0);
        const double sumsq = std::accumulate(deltas.begin(), deltas.end(), 0.0,
                                             [](double acc, double value) {
                                                 return acc + value * value;
                                             });
        const double n = static_cast<double>(deltas.size());
        const double mean = sum / n;
        const double variance = std::max(0.0, (sumsq / n) - (mean * mean));
        result.mean_interarrival = mean;
        result.variance_interarrival = variance;
    }

    const double event_count = static_cast<double>(result.arrivals.size());
    result.mean_intensity = (result.horizon > 0.0) ? event_count / result.horizon : 0.0;
}

void SimulatorCore::write_event_log(const SimulationResult& result) const {
    if (config_.event_log_path.empty()) {
        return;
    }

    const std::filesystem::path arrivals_path = config_.event_log_path;
    std::filesystem::create_directories(arrivals_path.parent_path());

    std::ofstream arrivals(arrivals_path);
    arrivals << "raw_time,arrival_time,latency,dimension,intensity_total,intensity_dimension\n";
    arrivals << std::setprecision(10);
    for (const auto& record : result.arrivals) {
        arrivals << record.raw_time << ','
                 << record.arrival_time << ','
                 << record.latency << ','
                 << record.dimension << ','
                 << record.intensity_total << ','
                 << record.intensity_dimension << '\n';
    }

    std::filesystem::path intensity_path = arrivals_path;
    const std::string stem = intensity_path.stem().string();
    intensity_path.replace_filename(stem + "_intensity.csv");

    std::ofstream intensity(intensity_path);
    intensity << "time";
    if (!result.intensity_trace.empty()) {
        const std::size_t dims = result.intensity_trace.front().lambda.size();
        for (std::size_t i = 0; i < dims; ++i) {
            intensity << ",lambda_" << i;
        }
    }
    intensity << '\n';
    intensity << std::setprecision(10);
    for (const auto& sample : result.intensity_trace) {
        intensity << sample.time;
        for (double value : sample.lambda) {
            intensity << ',' << value;
        }
        intensity << '\n';
    }
}

} // namespace simulator
