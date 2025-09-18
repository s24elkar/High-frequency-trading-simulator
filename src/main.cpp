#include <chrono>
#include <iostream>
#include <vector>

#include "Order.hpp"
#include "OrderBook.hpp"

int main() {
    using namespace std::chrono;
    auto now = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();

    std::vector<Order> orders = {
        {1, Side::Buy,  100.00, 10, now},
        {2, Side::Sell, 100.20,  7, now + 1'000},
        {3, Side::Buy,   99.95, 15, now + 2'000},
        {4, Side::Buy,  100.05,  8, now + 3'000},
        {5, Side::Sell, 100.15,  5, now + 4'000},
    };

    OrderBook book;
    for (const auto& order : orders) book.addLimitOrder(order);

    std::cout << "Best bid : "
              << (book.bestBid() ? to_string(*book.bestBid()) : std::string{"<none>"})
              << '\n';
    std::cout << "Best ask : "
              << (book.bestAsk() ? to_string(*book.bestAsk()) : std::string{"<none>"})
              << '\n';

    std::cout << "\nTop-of-book snapshot:\n";
    book.printTopLevels(5);
    return 0;
}
