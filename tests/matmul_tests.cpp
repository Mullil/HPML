#include <gtest/gtest.h>
#include <numeric>
#include <algorithm>
#include <vector>

#include "../src/linalg/matmul.hpp"


inline void EXPECT_MATRIX_NEAR(const Matrix& M, const std::vector<double>& expected, double tolerance = 1e-9) {
    ASSERT_EQ(M.rows * M.cols, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(M.data[i], expected[i], tolerance) << "Error at index " << i;
    }
}

TEST(matmulSmall, 2x2) {
    double A_data[4] = {1,2,
                        2,4};
    double B_data[4] = {0,3,
                        1,2};
    double C_data[4] = {0,0,0,0};

    Matrix A{2,2,A_data};
    Matrix B{2,2,B_data};
    Matrix C{2,2,C_data};

    matmul(A,B,C);

    EXPECT_MATRIX_NEAR(C, {2, 7,
                           4, 14});
}

TEST(matmulSmall, 3x3) {
    double A_data[9] = {1,2, 0.5,
                        2,4, 0.5,
                        1, 3, 1};

    double B_data[9] = {0.5, 2, 3,
                        1, 2, 0.5,
                        0.5, 0.5, 2};

    double C_data[9] = {0};

    Matrix A{3,3,A_data};
    Matrix B{3,3,B_data};
    Matrix C{3,3,C_data};

    matmul(A,B,C);

    EXPECT_MATRIX_NEAR(C, {
        2.75, 6.25, 5.0,
        5.25, 12.25, 9.0,
        4.0,  8.5,  6.5
    });
}


TEST(MatmulMedium, Random10x10) {
    const int N = 10;
    std::vector<double> A_data(N * N);
    std::vector<double> B_data(N * N);
    std::vector<double> C_data(N * N, 0);

    for (int i = 0; i < N*N; i++) {
        A_data[i] = rand() % 10;
        B_data[i] = rand() % 10;
    }

    Matrix A{N, N, A_data.data()};
    Matrix B{N, N, B_data.data()};
    Matrix C{N, N, C_data.data()};

    matmul(A, B, C);

    std::vector<double> expected(N * N, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                expected[i*N + j] += A.data[i*N + k] * B.data[k*N + j];
            }
        }
    }

    EXPECT_MATRIX_NEAR(C, expected);
}

TEST(MatmulLarge, Random1000x500_times_500x200) {
    const int M = 1000;
    const int K = 500;
    const int N = 200;
    std::vector<double> A_data(M * K);
    std::vector<double> B_data(K * N);
    std::vector<double> C_data(M * N, 0);

    for (int i = 0; i < 1000 * 500; i++) {
        A_data[i] = rand() % 10;
    }

    for (int i = 0; i < 500 * 200; i++) {
        B_data[i] = rand() % 10;
    }

    Matrix A{M, K, A_data.data()};
    Matrix B{K, N, B_data.data()};
    Matrix C{M, N, C_data.data()};

    matmul(A, B, C);

    std::vector<double> expected(M * N, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                expected[i*N + j] += A.data[i*K + k] * B.data[k*N + j];
            }
        }
    }

    EXPECT_MATRIX_NEAR(C, expected);
}

TEST(MatmulLarge, Random1000x1000) {
    const int N = 1000;
    std::vector<double> A_data(N * N);
    std::vector<double> B_data(N * N);
    std::vector<double> C_data(N * N, 0);

    for (int i = 0; i < N*N; i++) {
        A_data[i] = rand() % 10;
        B_data[i] = rand() % 10;
    }

    Matrix A{N, N, A_data.data()};
    Matrix B{N, N, B_data.data()};
    Matrix C{N, N, C_data.data()};

    matmul(A, B, C);

    std::vector<double> expected(N * N, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                expected[i*N + j] += A.data[i*N + k] * B.data[k*N + j];
            }
        }
    }

    EXPECT_MATRIX_NEAR(C, expected);
}


TEST(MatmulSpecial, Identity) {
    double A_data[4] = {5, -1,
                        2,  3};
    double B_data[4] = {1, 0,
                        0, 1};
    double C_data[4] = {0, 0, 0, 0};

    Matrix A{2, 2, A_data};
    Matrix B{2, 2, B_data};
    Matrix C{2, 2, C_data};

    matmul(A, B, C);

    EXPECT_MATRIX_NEAR(C, {5, -1,
                           2, 3});
}

TEST(MatmulSpecial, ZeroMatrix) {
    double A_data[6] = {1, 2,
                        3, 4,
                        5, 6};
    double B_data[6] = {0, 0, 0,
                        0, 0, 0};
    double C_data[6] = {0};

    Matrix A{3, 2, A_data};
    Matrix B{2, 3, B_data};
    Matrix C{3, 3, C_data};

    matmul(A, B, C);

    EXPECT_MATRIX_NEAR(C, {0, 0, 0,
                           0, 0, 0,
                           0, 0, 0});
}

TEST(MatmulRectangular, 2x3_times_3x2) {
    double A_data[6] = {1, 2, 3,
                        4, 5, 6};
    double B_data[6] = {7, 8,
                        9, 10,
                        11, 12};
    double C_data[4] = {0};

    Matrix A{2, 3, A_data};
    Matrix B{3, 2, B_data};
    Matrix C{2, 2, C_data};

    matmul(A, B, C);

    EXPECT_MATRIX_NEAR(C, {58, 64,
                           139, 154});
}


TEST(MatmulNegative, HandlesNegative) {
    double A_data[4] = {-1, 2,
                        3, -4};
    double B_data[4] = {2, -3,
                        -1, 5};
    double C_data[4] = {0};

    Matrix A{2, 2, A_data};
    Matrix B{2, 2, B_data};
    Matrix C{2, 2, C_data};

    matmul(A, B, C);

    EXPECT_MATRIX_NEAR(C, {-4, 13,
                           10, -29});
}

TEST(MatmulError, DimensionMismatch) {
    double A_data[6] = {1, 2, 3,
                        4, 5, 6};
    double B_data[4] = {1, 2,
                        3, 4};
    double C_data[4] = {0};

    Matrix A{2, 3, A_data};
    Matrix B{2, 2, B_data};
    Matrix C{2, 2, C_data};

    EXPECT_THROW(matmul(A, B, C), std::invalid_argument);
}

TEST(MatmulError, OutputDimensionMismatch) {
    double A_data[6] = {1, 2, 3,
                        4, 5, 6};
    double B_data[4] = {1, 2,
                        3, 4};
    double C_data[5] = {0};

    Matrix A{2, 3, A_data};
    Matrix B{2, 2, B_data};
    Matrix C{2, 2, C_data};

    EXPECT_THROW(matmul(A, B, C), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}