cmake_minimum_required(VERSION 3.15)
project(hft_sim LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(hft_sim
    src/main.cpp
    src/OrderBook.cpp
)
target_compile_options(hft_sim PRIVATE -O2 -Wall -Wextra -Wpedantic)
target_include_directories(hft_sim PRIVATE src)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(order_tests
    tests/order_tests.cpp
    src/OrderBook.cpp
)
target_link_libraries(order_tests PRIVATE Catch2::Catch2WithMain)
target_include_directories(order_tests PRIVATE src)
add_test(NAME order_tests COMMAND order_tests)
