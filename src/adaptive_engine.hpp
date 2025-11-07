#pragma once

#include "execution_cost.hpp"

#include <cstddef>
#include <deque>

namespace simulator {

struct AdaptiveExecutionConfig {
    ExecutionCostConfig cost;
    double base_aggressiveness{0.9};
    double min_aggressiveness{0.2};
    double max_aggressiveness{2.5};
    double base_risk{25.0};
    double min_risk{5.0};
    double max_risk{250.0};
    double latency_scale{50.0};
    double pnl_scale{2'000.0};
    double volatility_scale{0.75};
    std::size_t latency_window{50};
    std::size_t pnl_window{60};
    std::size_t volatility_window{40};
};

struct AdaptiveSnapshot {
    double aggressiveness{0.0};
    double risk_limit{0.0};
    double latency_variance{0.0};
    double pnl_drift{0.0};
    double volatility{0.0};
};

class AdaptiveExecutionEngine {
public:
    explicit AdaptiveExecutionEngine(AdaptiveExecutionConfig config = {});

    ExecutionRecord execute(
        const Order& order,
        double reference_price,
        const std::vector<OrderBook::Fill>& fills,
        double observed_latency_ms,
        double pnl_delta,
        double short_term_volatility
    );

    [[nodiscard]] AdaptiveSnapshot snapshot() const noexcept;

private:
    class SlidingWindowStat {
    public:
        explicit SlidingWindowStat(std::size_t window = 1);
        void set_window(std::size_t window);
        void add(double value);
        [[nodiscard]] std::size_t count() const noexcept;
        [[nodiscard]] double mean() const noexcept;
        [[nodiscard]] double variance() const noexcept;
        [[nodiscard]] double drift() const noexcept;

    private:
        void trim();

        std::size_t window_;
        std::deque<double> buffer_;
        double sum_{0.0};
        double sumsq_{0.0};
    };

    void update_statistics(double latency_ms, double pnl_delta, double st_vol);
    void recompute_controls();

    AdaptiveExecutionConfig config_;
    ExecutionEngine engine_;
    SlidingWindowStat latency_stats_;
    SlidingWindowStat pnl_stats_;
    SlidingWindowStat volatility_stats_;
    double aggressiveness_;
    double risk_limit_;
    AdaptiveSnapshot last_snapshot_;
};

} // namespace simulator
