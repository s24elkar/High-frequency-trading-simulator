cmake_minimum_required(VERSION 3.15)
project(hft_sim LANGUAGES CXX)

option(HFT_ENABLE_TESTS "Build unit tests" OFF)
option(ENABLE_DOCS "Generate documentation with Doxygen" OFF)
if(HFT_ENABLE_TESTS)
    enable_testing()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(hft_core STATIC
    src/OrderBook.cpp
    src/order_flow/OrderFlow.cpp
    src/order_flow/hawkes_mle.cpp
    src/hawkes_engine.cpp
    src/simulator_core.cpp
)
target_include_directories(hft_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(hft_core PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_library(order_book_bridge SHARED
    src/order_book_c_api.cpp
)
target_link_libraries(order_book_bridge PRIVATE hft_core)
target_include_directories(order_book_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(order_book_bridge PROPERTIES
    OUTPUT_NAME order_book_bridge
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_library(hawkes_bridge SHARED
    src/hawkes_c_api.cpp
)
target_link_libraries(hawkes_bridge PRIVATE hft_core)
target_include_directories(hawkes_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(hawkes_bridge PRIVATE -O2 -Wall -Wextra -Wpedantic)
set_target_properties(hawkes_bridge PROPERTIES
    OUTPUT_NAME hawkes_bridge
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_executable(hft_sim src/main.cpp)
target_link_libraries(hft_sim PRIVATE hft_core)
target_include_directories(hft_sim PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(hawkes_example src/hawkes_example.cpp)
target_link_libraries(hawkes_example PRIVATE hft_core)
target_include_directories(hawkes_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(hawkes_validation src/hawkes_validation.cpp)
target_link_libraries(hawkes_validation PRIVATE hft_core)
target_include_directories(hawkes_validation PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(hawkes_validation PRIVATE -O2 -Wall -Wextra -Wpedantic)

if(ENABLE_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

if(HFT_ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(order_tests tests/order_tests.cpp)
    target_link_libraries(order_tests PRIVATE Catch2::Catch2WithMain hft_core)
    target_include_directories(order_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME order_tests COMMAND order_tests)
endif()
