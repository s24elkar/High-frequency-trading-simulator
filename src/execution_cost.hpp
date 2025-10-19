#pragma once

#include "Order.hpp"
#include "OrderBook.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace simulator {

struct ExecutionCostConfig {
    double temporary_eta{0.0};   // temporary impact coefficient (per share per aggressiveness unit)
    double permanent_gamma{0.0}; // permanent impact coefficient (per share^2)
};

struct ExecutionRecord {
    Order order;
    double reference_price{0.0};
    double aggressiveness{1.0};
    double implementation_shortfall{0.0};
    double total_cost{0.0};
};

class ExecutionEngine {
public:
    explicit ExecutionEngine(ExecutionCostConfig config = {});

    ExecutionRecord record_execution(
        const Order& requested_order,
        double reference_price,
        double aggressiveness,
        const std::vector<OrderBook::Fill>& fills
    );

    [[nodiscard]] const std::vector<ExecutionRecord>& history() const noexcept;
    [[nodiscard]] double cumulative_cost() const noexcept;
    [[nodiscard]] double cumulative_temporary_cost() const noexcept;
    [[nodiscard]] double cumulative_permanent_cost() const noexcept;
    [[nodiscard]] double cumulative_shortfall() const noexcept;

private:
    ExecutionCostConfig config_;
    std::vector<ExecutionRecord> ledger_;
    double total_cost_{0.0};
    double total_temporary_{0.0};
    double total_permanent_{0.0};
    double total_shortfall_{0.0};
};

} // namespace simulator

