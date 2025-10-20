#include "execution_cost.hpp"

#include "perf/Profiler.hpp"

#include <cmath>

namespace simulator {

ExecutionEngine::ExecutionEngine(ExecutionCostConfig config)
    : config_(config) {}

ExecutionRecord ExecutionEngine::record_execution(
    const Order& requested_order,
    double reference_price,
    double aggressiveness,
    const std::vector<OrderBook::Fill>& fills
) {
    HFT_PROFILE_SCOPE("ExecutionEngine::record_execution");
    Order executed = requested_order;
    double filled_quantity = 0.0;
    double notional = 0.0;

    for (const auto& fill : fills) {
        filled_quantity += static_cast<double>(fill.executedQuantity);
        notional += static_cast<double>(fill.executedQuantity) * fill.fillPrice;
    }

    if (filled_quantity > 0.0) {
        executed.quantity = static_cast<std::int32_t>(filled_quantity);
        executed.execution_price = notional / filled_quantity;
    } else {
        executed.quantity = 0;
        executed.execution_price = reference_price;
    }

    const double direction = (requested_order.side == Side::Buy) ? 1.0 : -1.0;
    const double abs_quantity = std::fabs(filled_quantity);

    executed.slippage = direction * (executed.execution_price - reference_price);

    const double clamped_aggressiveness = aggressiveness > 0.0 ? aggressiveness : 0.0;
    const double temporary_cost = config_.temporary_eta * abs_quantity * clamped_aggressiveness;
    const double permanent_cost = 0.5 * config_.permanent_gamma * abs_quantity * abs_quantity;

    executed.temporary_impact = temporary_cost;
    executed.permanent_impact = permanent_cost;

    const double implementation_shortfall = executed.slippage * abs_quantity;
    const double total_cost = implementation_shortfall + temporary_cost + permanent_cost;

    ExecutionRecord record;
    record.order = executed;
    record.reference_price = reference_price;
    record.aggressiveness = clamped_aggressiveness;
    record.implementation_shortfall = implementation_shortfall;
    record.total_cost = total_cost;

    ledger_.push_back(record);
    total_cost_ += total_cost;
    total_temporary_ += temporary_cost;
    total_permanent_ += permanent_cost;
    total_shortfall_ += implementation_shortfall;

    return record;
}

const std::vector<ExecutionRecord>& ExecutionEngine::history() const noexcept {
    return ledger_;
}

double ExecutionEngine::cumulative_cost() const noexcept {
    return total_cost_;
}

double ExecutionEngine::cumulative_temporary_cost() const noexcept {
    return total_temporary_;
}

double ExecutionEngine::cumulative_permanent_cost() const noexcept {
    return total_permanent_;
}

double ExecutionEngine::cumulative_shortfall() const noexcept {
    return total_shortfall_;
}

} // namespace simulator
