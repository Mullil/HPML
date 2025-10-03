#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>

struct Matrix {
    size_t rows, cols;
    double* data;
};

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    const size_t TILE = 64;
    // Check matrix dimensions
    if (A.cols != B.rows) {
        throw std::invalid_argument("Faulty input matrix dimensions");
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        throw std::invalid_argument("Faulty output matrix dimensions");
    }
    // Matmul with added tiling and contiguous memory access
    for (size_t ii = 0; ii < A.rows; ii+=TILE) {
        for (size_t jj = 0; jj < B.cols; jj+=TILE) {
            for (size_t kk = 0; kk < A.cols; kk+=TILE) {
                size_t i_max = std::min(ii + TILE, A.rows);
                size_t j_max = std::min(jj + TILE, B.cols);
                size_t k_max = std::min(kk + TILE, A.cols);
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        for (size_t j = jj; j < j_max; ++j) {
                            C.data[i * C.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                        }
                    }
                }
            }
            
        }
    }
}