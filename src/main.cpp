#include <iostream>
#include <vector>
#include <chrono>
#include "Order.hpp"

int main() {
    using namespace std::chrono;
    auto now = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();

    std::vector<Order> orders = {
        {1, Side::Buy,  100.00, 10, now},
        {2, Side::Sell, 100.20,  7, now+1000},
        {3, Side::Buy,   99.95, 15, now+2000},
    };

    for (const auto& o : orders) std::cout << to_string(o) << '\n';
    return 0;
}
