find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)


add_executable(matmul_bench matmul/matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE matmul benchmark::benchmark pthread)
target_link_libraries(matmul_bench PRIVATE Eigen3::Eigen)

add_custom_target(perf_benchmark
    COMMAND perf stat -e cycles,instructions,cache-references,cache-misses ./matmul_bench
    DEPENDS matmul_bench
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmark with perf"
)
