#include "OrderBook.hpp"

#include <cmath>      // std::llround
#include <iostream>
#include <numeric>    // std::accumulate
#include <utility>    // std::move


OrderBook::PriceTick OrderBook::toTicks(double price) {
    return static_cast<PriceTick>(std::llround(price * 10000.0));
}

double OrderBook::fromTicks(PriceTick ticks) {
    return static_cast<double>(ticks) / 10000.0;
}

OrderBook::PriceLevel& OrderBook::levelFor(const Order& order) {
    const auto price = toTicks(order.price);
    if (order.side == Side::Buy) {
        return bids_[price];         // creates level if missing
    }
    return asks_[price];
}

const OrderBook::PriceLevel* OrderBook::bestLevel(Side side) const {
    if (side == Side::Buy) {
        if (bids_.empty()) return nullptr;
        return &bids_.begin()->second;  // highest price because comparator is greater<>
    }
    if (asks_.empty()) return nullptr;
    return &asks_.begin()->second;      // lowest ask
}

void OrderBook::addLimitOrder(const Order& order) {
    const auto price = toTicks(order.price);
    auto& level = levelFor(order);
    level.orders.push_back(order);
    locators_[order.id] = Locator{order.side, price, std::prev(level.orders.end())};
}

bool OrderBook::cancel(OrderId id) {
    auto loc = locators_.find(id);
    if (loc == locators_.end()) return false;

    const auto side  = loc->second.side;
    const auto price = loc->second.price;
    auto  orderIt    = loc->second.it;

    if (side == Side::Buy) {
        auto levelIt = bids_.find(price);
        if (levelIt == bids_.end()) return false;
        levelIt->second.orders.erase(orderIt);
        if (levelIt->second.orders.empty()) bids_.erase(levelIt);
    } else {
        auto levelIt = asks_.find(price);
        if (levelIt == asks_.end()) return false;
        levelIt->second.orders.erase(orderIt);
        if (levelIt->second.orders.empty()) asks_.erase(levelIt);
    }

    locators_.erase(loc);
    return true;
}

std::optional<Order> OrderBook::bestBid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->second.orders.front();
}

std::optional<Order> OrderBook::bestAsk() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->second.orders.front();
}

std::vector<OrderBook::LevelSnapshot> OrderBook::levels(std::size_t depth) const {
    std::vector<LevelSnapshot> snapshot;
    snapshot.reserve(depth * 2);

    std::size_t count = 0;
    for (const auto& [price, level] : bids_) {
        if (count++ == depth) break;
        LevelSnapshot snap;
        snap.price = fromTicks(price);
        snap.totalQuantity = std::accumulate(
            level.orders.begin(), level.orders.end(), static_cast<std::int64_t>(0),
            [](std::int64_t acc, const Order& o) { return acc + o.quantity; });
        snap.resting.assign(level.orders.begin(), level.orders.end());
        snapshot.push_back(std::move(snap));
    }

    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count++ == depth) break;
        LevelSnapshot snap;
        snap.price = fromTicks(price);
        snap.totalQuantity = std::accumulate(
            level.orders.begin(), level.orders.end(), static_cast<std::int64_t>(0),
            [](std::int64_t acc, const Order& o) { return acc + o.quantity; });
        snap.resting.assign(level.orders.begin(), level.orders.end());
        snapshot.push_back(std::move(snap));
    }

    return snapshot;
}

void OrderBook::printTopLevels(std::size_t depth) const {
    for (const auto& level : levels(depth)) {
        std::cout << level.price << " qty=" << level.totalQuantity << '\n';
    }
}
