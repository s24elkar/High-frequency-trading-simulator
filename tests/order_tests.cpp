#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "OrderBook.hpp"

TEST_CASE("best bid/ask reflect inserted orders") {
    OrderBook book;
    book.addLimitOrder({1, Side::Buy, 100.10, 5, 10});
    book.addLimitOrder({2, Side::Buy, 100.20, 7, 11});
    book.addLimitOrder({3, Side::Sell, 100.30, 4, 12});
    book.addLimitOrder({4, Side::Sell, 100.40, 6, 13});

    auto bestBid = book.bestBid();
    REQUIRE(bestBid);
    CHECK(bestBid->id == 2);
    CHECK(bestBid->price == Catch::Approx(100.20));
    CHECK(bestBid->quantity == 7);

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 3);
    CHECK(bestAsk->price == Catch::Approx(100.30));
    CHECK(bestAsk->quantity == 4);
}

TEST_CASE("orders at same price honor FIFO") {
    OrderBook book;
    book.addLimitOrder({10, Side::Buy, 100.00, 5, 100});
    book.addLimitOrder({11, Side::Buy, 100.00, 7, 101});

    auto first = book.bestBid();
    REQUIRE(first);
    CHECK(first->id == 10);

    REQUIRE(book.cancel(10));
    auto second = book.bestBid();
    REQUIRE(second);
    CHECK(second->id == 11);
}

TEST_CASE("cancel removes top-of-book order") {
    OrderBook book;
    book.addLimitOrder({20, Side::Sell, 101.00, 3, 200});
    book.addLimitOrder({21, Side::Sell, 101.10, 4, 201});

    REQUIRE(book.cancel(20));
    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 21);
    CHECK_FALSE(book.cancel(42));
}
