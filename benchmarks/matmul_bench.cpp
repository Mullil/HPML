#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "../src/matmul.hpp"


void fill_matrix_random(Matrix& M, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    for (size_t i = 0; i < M.rows * M.cols; ++i) {
        M.data[i] = dist(gen);
    }
}

Matrix create_matrix(size_t N, std::vector<double>& storage) {
    storage.resize(N * N);
    return Matrix{N, N, storage.data()};
}

void BM_MatMul(benchmark::State& state, size_t N) {
    // Random generator with seed 22
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> A_data, B_data, C_data;
    Matrix A = create_matrix(N, A_data);
    Matrix B = create_matrix(N, B_data);
    Matrix C = create_matrix(N, C_data);

    fill_matrix_random(A, gen, dist);
    fill_matrix_random(B, gen, dist);

    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data);
    }
}

BENCHMARK_CAPTURE(BM_MatMul, Small, 100)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, Medium, 1000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, MediumLarge, 3000)->Unit(benchmark::kSecond);

BENCHMARK_MAIN();

