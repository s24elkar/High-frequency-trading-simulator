#pragma once

#include "Order.hpp"

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

class OrderBook {
public:
    using OrderId   = std::int64_t;
    using PriceTick = std::int64_t;  // scaled price (e.g. price * 10000)

    struct LevelSnapshot {
        double price;                     // display price
        std::int64_t totalQuantity;       // aggregate resting size
        std::vector<Order> resting;       // FIFO orders at this price
    };

    void addLimitOrder(const Order& order);
    bool cancel(OrderId id);

    std::optional<Order> bestBid() const;
    std::optional<Order> bestAsk() const;

    std::vector<LevelSnapshot> levels(std::size_t depth) const;
    void printTopLevels(std::size_t depth) const;

private:
    struct PriceLevel {
        std::list<Order> orders;          // preserves price-time priority
    };

    struct Locator {
        Side side;
        PriceTick price;
        std::list<Order>::iterator it;
    };

    using BidBook = std::map<PriceTick, PriceLevel, std::greater<PriceTick>>;
    using AskBook = std::map<PriceTick, PriceLevel, std::less<PriceTick>>;

    static PriceTick toTicks(double price);
    static double fromTicks(PriceTick ticks);

    PriceLevel& levelFor(const Order& order);
    const PriceLevel* bestLevel(Side side) const;

    BidBook bids_;
    AskBook asks_;
    std::unordered_map<OrderId, Locator> locators_;
};
