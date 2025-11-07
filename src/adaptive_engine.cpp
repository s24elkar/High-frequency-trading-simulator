#include "adaptive_engine.hpp"

#include <algorithm>
#include <cmath>

namespace simulator {

AdaptiveExecutionEngine::SlidingWindowStat::SlidingWindowStat(std::size_t window)
    : window_(std::max<std::size_t>(1, window)) {}

void AdaptiveExecutionEngine::SlidingWindowStat::set_window(std::size_t window) {
    window_ = std::max<std::size_t>(1, window);
    trim();
}

void AdaptiveExecutionEngine::SlidingWindowStat::add(double value) {
    buffer_.push_back(value);
    sum_ += value;
    sumsq_ += value * value;
    trim();
}

std::size_t AdaptiveExecutionEngine::SlidingWindowStat::count() const noexcept {
    return buffer_.size();
}

double AdaptiveExecutionEngine::SlidingWindowStat::mean() const noexcept {
    if (buffer_.empty()) {
        return 0.0;
    }
    return sum_ / static_cast<double>(buffer_.size());
}

double AdaptiveExecutionEngine::SlidingWindowStat::variance() const noexcept {
    if (buffer_.size() < 2) {
        return 0.0;
    }
    const double n = static_cast<double>(buffer_.size());
    const double mean_val = sum_ / n;
    const double var = (sumsq_ / n) - (mean_val * mean_val);
    return var > 0.0 ? var : 0.0;
}

double AdaptiveExecutionEngine::SlidingWindowStat::drift() const noexcept {
    if (buffer_.size() < 2) {
        return 0.0;
    }
    const double first = buffer_.front();
    const double last = buffer_.back();
    return (last - first) / static_cast<double>(buffer_.size() - 1);
}

void AdaptiveExecutionEngine::SlidingWindowStat::trim() {
    while (buffer_.size() > window_) {
        const double value = buffer_.front();
        buffer_.pop_front();
        sum_ -= value;
        sumsq_ -= value * value;
    }
}

AdaptiveExecutionEngine::AdaptiveExecutionEngine(AdaptiveExecutionConfig config)
    : config_(config),
      engine_(ExecutionEngine(config.cost)),
      latency_stats_(config.latency_window),
      pnl_stats_(config.pnl_window),
      volatility_stats_(config.volatility_window),
      aggressiveness_(config.base_aggressiveness),
      risk_limit_(config.base_risk) {}

ExecutionRecord AdaptiveExecutionEngine::execute(
    const Order& order,
    double reference_price,
    const std::vector<OrderBook::Fill>& fills,
    double observed_latency_ms,
    double pnl_delta,
    double short_term_volatility
) {
    update_statistics(observed_latency_ms, pnl_delta, short_term_volatility);
    recompute_controls();

    ExecutionRecord record = engine_.record_execution(
        order,
        reference_price,
        aggressiveness_,
        fills
    );

    last_snapshot_.aggressiveness = aggressiveness_;
    last_snapshot_.risk_limit = risk_limit_;
    last_snapshot_.latency_variance = latency_stats_.variance();
    last_snapshot_.pnl_drift = pnl_stats_.drift();
    last_snapshot_.volatility = volatility_stats_.mean();

    return record;
}

AdaptiveSnapshot AdaptiveExecutionEngine::snapshot() const noexcept {
    return last_snapshot_;
}

void AdaptiveExecutionEngine::update_statistics(
    double latency_ms,
    double pnl_delta,
    double st_vol
) {
    latency_stats_.add(latency_ms);
    pnl_stats_.add(pnl_delta);
    volatility_stats_.add(st_vol);
}

void AdaptiveExecutionEngine::recompute_controls() {
    constexpr double epsilon = 1e-9;
    const double latency_var = latency_stats_.variance();
    const double pnl_drift = pnl_stats_.drift();
    const double vol_level = volatility_stats_.mean();

    // Reduce aggressiveness when latency variance balloons or volatility spikes,
    // increase when PnL drift is favourable.
    double latency_penalty = latency_var / (config_.latency_scale + latency_var + epsilon);
    double pnl_boost = pnl_drift / (config_.pnl_scale + std::abs(pnl_drift) + epsilon);
    double volatility_penalty = vol_level / (config_.volatility_scale + vol_level + epsilon);

    double adjusted_aggr = config_.base_aggressiveness;
    adjusted_aggr *= (1.0 - latency_penalty);
    adjusted_aggr *= (1.0 - volatility_penalty);
    adjusted_aggr += pnl_boost;

    aggressiveness_ = std::clamp(adjusted_aggr, config_.min_aggressiveness, config_.max_aggressiveness);

    double adjusted_risk = config_.base_risk;
    adjusted_risk += pnl_boost * config_.base_risk;
    adjusted_risk *= (1.0 - volatility_penalty);
    adjusted_risk = std::clamp(adjusted_risk, config_.min_risk, config_.max_risk);
    risk_limit_ = adjusted_risk;
}

} // namespace simulator
