#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>

struct Matrix {
    size_t rows, cols;
    double* data;
};

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    // Check matrix dimensions
    if (A.cols != B.rows) {
        throw std::invalid_argument("Faulty input matrix dimensions");
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        throw std::invalid_argument("Faulty output matrix dimensions");
    }
    // Unoptimized matrix multiplication with a flattened matrix    
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            for (size_t k = 0; k < A.cols; ++k) {
                C.data[i * C.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            
        }
    }
}