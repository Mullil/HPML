find_package(benchmark REQUIRED)

add_executable(matmul_bench matmul/matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE matmul benchmark::benchmark pthread)
