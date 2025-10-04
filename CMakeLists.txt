cmake_minimum_required(VERSION 3.14)
project(HPML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
