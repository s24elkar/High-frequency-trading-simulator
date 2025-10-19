#pragma once
#include <cstdint>
#include <string>

enum class Side : uint8_t { Buy = 0, Sell = 1 };

struct Order {
    std::int64_t id;
    Side side;
    double price;
    std::int32_t quantity;
    std::int64_t ts_ns;
    double execution_price{0.0};
    double slippage{0.0};
    double temporary_impact{0.0};
    double permanent_impact{0.0};
};

inline std::string to_string(const Order& o){
    return "Order{id=" + std::to_string(o.id) +
           ", side=" + std::string(o.side == Side::Buy ? "Buy" : "Sell") +
           ", price=" + std::to_string(o.price) +
           ", quantity=" + std::to_string(o.quantity) +
           ", ts_ns=" + std::to_string(o.ts_ns) +
           ", execution_price=" + std::to_string(o.execution_price) +
           ", slippage=" + std::to_string(o.slippage) +
           ", temporary_impact=" + std::to_string(o.temporary_impact) +
           ", permanent_impact=" + std::to_string(o.permanent_impact) + "}";
}
