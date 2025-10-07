#include "OrderBook.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

extern "C" {

enum ob_side : int { OB_SIDE_BUY = 0, OB_SIDE_SELL = 1 };

struct ob_order {
    std::int64_t id;
    int side;
    double price;
    std::int32_t quantity;
    std::int64_t ts_ns;
};

struct ob_fill {
    ob_order order;
    std::int32_t executed_quantity;
    double fill_price;
};

struct ob_level {
    double price;
    int side;
    std::int64_t total_quantity;
    ob_order* orders;
    std::size_t order_count;
};

struct ob_book {
    OrderBook book;
};

static ob_side to_ob_side(Side side) {
    return side == Side::Buy ? OB_SIDE_BUY : OB_SIDE_SELL;
}

static Side from_ob_side(int side) {
    return side == OB_SIDE_SELL ? Side::Sell : Side::Buy;
}

static ob_order to_ob_order(const Order& order) {
    ob_order result{};
    result.id = order.id;
    result.side = to_ob_side(order.side);
    result.price = order.price;
    result.quantity = order.quantity;
    result.ts_ns = order.ts_ns;
    return result;
}

ob_book* ob_create() {
    try {
        return new ob_book{};
    } catch (...) {
        return nullptr;
    }
}

void ob_destroy(ob_book* handle) {
    delete handle;
}

int ob_add_order(
    ob_book* handle,
    std::int64_t id,
    int side,
    double price,
    std::int32_t quantity,
    std::int64_t ts_ns
) {
    if (!handle) return 0;
    Order order{id, from_ob_side(side), price, quantity, ts_ns};
    handle->book.addLimitOrder(order);
    return 1;
}

int ob_cancel_order(ob_book* handle, std::int64_t id) {
    if (!handle) return 0;
    return handle->book.cancel(id) ? 1 : 0;
}

int ob_best_bid(ob_book* handle, ob_order* out_order) {
    if (!handle || !out_order) return 0;
    auto best = handle->book.bestBid();
    if (!best.has_value()) return 0;
    *out_order = to_ob_order(best.value());
    return 1;
}

int ob_best_ask(ob_book* handle, ob_order* out_order) {
    if (!handle || !out_order) return 0;
    auto best = handle->book.bestAsk();
    if (!best.has_value()) return 0;
    *out_order = to_ob_order(best.value());
    return 1;
}

int ob_execute_order(
    ob_book* handle,
    int side,
    double price,
    std::int32_t quantity,
    ob_fill** out_fills,
    std::size_t* out_count
) {
    if (!handle || !out_fills || !out_count) return 0;
    auto fills = handle->book.match(from_ob_side(side), price, quantity);
    if (fills.empty()) {
        *out_fills = nullptr;
        *out_count = 0;
        return 1;
    }

    ob_fill* buffer = new ob_fill[fills.size()];
    for (std::size_t idx = 0; idx < fills.size(); ++idx) {
        buffer[idx].order = to_ob_order(fills[idx].order);
        buffer[idx].executed_quantity = fills[idx].executedQuantity;
        buffer[idx].fill_price = fills[idx].fillPrice;
    }
    *out_fills = buffer;
    *out_count = fills.size();
    return 1;
}

int ob_snapshot(
    ob_book* handle,
    std::size_t depth,
    ob_level** out_levels,
    std::size_t* out_count
) {
    if (!handle || !out_levels || !out_count) return 0;
    auto levels = handle->book.levels(depth);
    if (levels.empty()) {
        *out_levels = nullptr;
        *out_count = 0;
        return 1;
    }

    ob_level* buffer = new ob_level[levels.size()];
    for (std::size_t idx = 0; idx < levels.size(); ++idx) {
        const auto& lvl = levels[idx];
        buffer[idx].price = lvl.price;
        buffer[idx].side = to_ob_side(lvl.side);
        buffer[idx].total_quantity = lvl.totalQuantity;
        buffer[idx].order_count = lvl.resting.size();
        if (!lvl.resting.empty()) {
            buffer[idx].orders = new ob_order[lvl.resting.size()];
            for (std::size_t j = 0; j < lvl.resting.size(); ++j) {
                buffer[idx].orders[j] = to_ob_order(lvl.resting[j]);
            }
        } else {
            buffer[idx].orders = nullptr;
        }
    }

    *out_levels = buffer;
    *out_count = levels.size();
    return 1;
}

void ob_free_fills(ob_fill* fills, std::size_t /*count*/) {
    delete[] fills;
}

void ob_free_levels(ob_level* levels, std::size_t count) {
    if (!levels) return;
    for (std::size_t idx = 0; idx < count; ++idx) {
        delete[] levels[idx].orders;
    }
    delete[] levels;
}

}  // extern "C"
