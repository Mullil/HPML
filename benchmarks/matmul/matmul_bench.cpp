#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "../../src/linalg/matmul.hpp"


void fill_matrix_random(Matrix& M, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    for (size_t i = 0; i < M.rows * M.cols; ++i) {
        M.data[i] = dist(gen);
    }
}

Matrix create_matrix(size_t M, size_t N, std::vector<double>& X) {
    X.resize(M * N);
    return Matrix{M, N, X.data()};
}

/** Creates two square matrices with given dimensions NxN
 and random double-precision floating point numbers **/
void BM_MatMul(benchmark::State& state, size_t N) {
    // Random generator with seed 22
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> A_data, B_data, C_data;
    Matrix A = create_matrix(N, N, A_data);
    Matrix B = create_matrix(N, N, B_data);
    Matrix C = create_matrix(N, N, C_data);

    fill_matrix_random(A, gen, dist);
    fill_matrix_random(B, gen, dist);

    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data);
    }
}


// Eigen library used to compare speed
void BM_Reference(benchmark::State& state, size_t N) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    for (auto _ : state) {
        Eigen::MatrixXd C = A * B;
        benchmark::DoNotOptimize(C.data());
    }
}

BENCHMARK_CAPTURE(BM_MatMul, 100, 100)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, 1000, 1000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, 3000, 3000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, 5000, 5000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_Reference, 5000, 5000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_MatMul, 8000, 8000)->Unit(benchmark::kSecond);
BENCHMARK_CAPTURE(BM_Reference, 8000, 8000)->Unit(benchmark::kSecond);

BENCHMARK_MAIN();

