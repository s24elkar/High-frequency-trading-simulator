#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "execution_cost.hpp"

using simulator::ExecutionCostConfig;
using simulator::ExecutionEngine;
using simulator::ExecutionRecord;
using simulator::Order;
using simulator::OrderBook;
using simulator::Side;

namespace {
OrderBook::Fill make_fill(std::int64_t id, Side side, double price, std::int32_t executed) {
    Order resting{id, side, price, executed, 0};
    OrderBook::Fill fill;
    fill.order = resting;
    fill.executedQuantity = executed;
    fill.fillPrice = price;
    return fill;
}
}

TEST_CASE("ExecutionEngine computes Almgren-Chriss costs for buy order") {
    ExecutionCostConfig cfg;
    cfg.temporary_eta = 0.01;
    cfg.permanent_gamma = 1e-5;

    ExecutionEngine engine(cfg);

    Order request{42, Side::Buy, 100.0, 10, 0};
    std::vector<OrderBook::Fill> fills;
    fills.push_back(make_fill(1, Side::Sell, 100.50, 5));
    fills.push_back(make_fill(2, Side::Sell, 100.60, 5));

    const double reference_price = 100.0;
    const double aggressiveness = 1.2;

    ExecutionRecord record = engine.record_execution(request, reference_price, aggressiveness, fills);

    REQUIRE(record.order.quantity == 10);
    CHECK(record.order.execution_price == Catch::Approx(100.55));
    CHECK(record.order.slippage == Catch::Approx(0.55));

    const double expected_shortfall = 0.55 * 10.0;
    const double expected_temporary = cfg.temporary_eta * 10.0 * aggressiveness;
    const double expected_permanent = 0.5 * cfg.permanent_gamma * 100.0;
    const double expected_total = expected_shortfall + expected_temporary + expected_permanent;

    CHECK(record.implementation_shortfall == Catch::Approx(expected_shortfall));
    CHECK(record.order.temporary_impact == Catch::Approx(expected_temporary));
    CHECK(record.order.permanent_impact == Catch::Approx(expected_permanent));
    CHECK(record.total_cost == Catch::Approx(expected_total));

    CHECK(engine.cumulative_cost() == Catch::Approx(expected_total));
    CHECK(engine.cumulative_temporary_cost() == Catch::Approx(expected_temporary));
    CHECK(engine.cumulative_permanent_cost() == Catch::Approx(expected_permanent));
    CHECK(engine.cumulative_shortfall() == Catch::Approx(expected_shortfall));
}

TEST_CASE("ExecutionEngine tracks costs for sell orders") {
    ExecutionCostConfig cfg;
    cfg.temporary_eta = 0.02;
    cfg.permanent_gamma = 2e-5;

    ExecutionEngine engine(cfg);

    Order request{99, Side::Sell, 101.0, 8, 0};
    std::vector<OrderBook::Fill> fills;
    fills.push_back(make_fill(10, Side::Buy, 100.40, 3));
    fills.push_back(make_fill(11, Side::Buy, 100.30, 5));

    const double reference_price = 101.0;
    const double aggressiveness = 0.6;

    ExecutionRecord record = engine.record_execution(request, reference_price, aggressiveness, fills);

    REQUIRE(record.order.quantity == 8);
    CHECK(record.order.execution_price == Catch::Approx(100.325));

    const double slippage = record.order.slippage;
    CHECK(slippage == Catch::Approx(0.675));

    const double expected_shortfall = slippage * 8.0;
    const double expected_temporary = cfg.temporary_eta * 8.0 * aggressiveness;
    const double expected_permanent = 0.5 * cfg.permanent_gamma * 64.0;
    const double expected_total = expected_shortfall + expected_temporary + expected_permanent;

    CHECK(record.implementation_shortfall == Catch::Approx(expected_shortfall));
    CHECK(record.total_cost == Catch::Approx(expected_total));

    CHECK(engine.cumulative_cost() == Catch::Approx(expected_total));
    CHECK(engine.history().size() == 1);
}

TEST_CASE("ExecutionEngine handles zero-liquidity gracefully") {
    ExecutionCostConfig cfg;
    cfg.temporary_eta = 0.05;
    cfg.permanent_gamma = 1e-4;

    ExecutionEngine engine(cfg);
    Order request{7, Side::Buy, 100.0, 5, 0};
    std::vector<OrderBook::Fill> fills; // empty

    ExecutionRecord record = engine.record_execution(request, 100.0, 1.0, fills);

    CHECK(record.order.quantity == 0);
    CHECK(record.total_cost == 0.0);
    CHECK(engine.cumulative_cost() == 0.0);
}
