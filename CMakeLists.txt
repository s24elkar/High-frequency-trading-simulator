cmake_minimum_required(VERSION 3.15)
project(hft_sim LANGUAGES CXX)
<<<<<<< HEAD
enable_testing()
=======
option(HFT_ENABLE_TESTS "Build unit tests" OFF)
if(HFT_ENABLE_TESTS)
    enable_testing()
endif()

>>>>>>> 5a92aa8 (Initial import of trading simulator)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(hft_sim
    src/main.cpp
    src/OrderBook.cpp
)
target_compile_options(hft_sim PRIVATE -O2 -Wall -Wextra -Wpedantic)
target_include_directories(hft_sim PRIVATE src)

<<<<<<< HEAD
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
=======
if(HFT_ENABLE_TESTS)
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
endif()

# Header-only "library" exposing include/ to dependents
add_library(hawkes INTERFACE)
target_include_directories(hawkes INTERFACE ${CMAKE_SOURCE_DIR}/include)  # include/hawkes/hawkes.hpp

# Example app that uses the header
add_executable(hawkes_example src/hawkes_example.cpp)
target_link_libraries(hawkes_example PRIVATE hawkes)
>>>>>>> 5a92aa8 (Initial import of trading simulator)
