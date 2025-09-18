cmake_minimum_required(VERSION 3.15)
project(hft_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(hft_sim
    src/main.cpp
)
target_compile_options(hft_sim PRIVATE -O2 -Wall -Wextra -Wpedantic)
