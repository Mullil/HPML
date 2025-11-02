#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <stdlib.h>

struct Matrix {
    size_t rows, cols;
    double* data;
};

void matmul(const Matrix& A, const Matrix& B, Matrix& C);

#endif