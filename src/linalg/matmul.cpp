#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>

// row-major matrix
struct Matrix {
    size_t rows, cols;
    double* data;
};


// Tiling parameters
constexpr size_t MACRO_TILE = 128;
constexpr size_t MICRO_TILE = 4;


/*
-----------------------------------------------------------------------
 pack_A and pack_B copy a block of matrices A and B into a contiguous
 buffers A_block and B_block to make reading more cache-friendly
-----------------------------------------------------------------------
*/

void pack_A(const double* A, double* A_block, size_t lda, size_t i_max, size_t k_max) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < i_max; ++i)
        for (size_t k = 0; k < k_max; ++k)
            A_block[i * k_max + k] = A[i * lda + k];
}

void pack_B(const double* B, double* B_block, size_t ldb, size_t k_max, size_t j_max) {
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < k_max; ++k)
        for (size_t j = 0; j < j_max; ++j)
            B_block[k * j_max + j] = B[k * ldb + j];
}

/*
-------------------------------------------------------------------------------
 micro_matmul computes a small 4x4 tile ideally fitting into registers

 Input arguments:
    k_max: length of the inner dimension
    A_block: pointer to the packed A block
    lda: stride between A_block rows
    B_block: pointer to the packed B block
    ldb: stride between B_block rows
    C_block: pointer to the C block to update
    ldc: stride between the rows of C
    mr, nr: micro-tile sizes that can be smaller than MICRO_TILE to handle edges
---------------------------------------------------------------------------------
*/

inline void micro_matmul(size_t max_k,
                         const double* A_block, size_t lda,
                         const double* B_block, size_t ldb,
                         double* C_block, size_t ldc,
                         size_t mr, size_t nr)
{

    double C_tile[MICRO_TILE][MICRO_TILE] = {0.0};
    for (size_t k = 0; k < max_k; ++k) {
        for (size_t i = 0; i < mr; ++i) {
            double a = A_block[i * lda + k];
            const double* B_row = B_block + k * ldb;
            for (size_t j = 0; j < nr; ++j) {
                C_tile[i][j] += a * B_row[j];
            }
        }
    }
    // Write the tile into a block of C
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            C_block[i * ldc + j] += C_tile[i][j];
        } 
    }
}

/*
-----------------------------------------------------------------------------

matmul uses two-level tiling for cache-friendly matrix multiplication

Outer loops with ii, kk and jj are macro-tiles that should fit the L2 cache
and the inner loops with i and j are micro-tiles (inside the macro-tiles)
that should fit registers.

Explanation of variables:
    i_max is the number of rows processed in A and C in the specific tile
    j_max is the number of columns processed in B and C in the specific tile
    k_max is the number of A columns and B rows to multiply

-----------------------------------------------------------------------------
*/

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {

    // Check matrix dimensions
    if (A.cols != B.rows) {
        throw std::invalid_argument("Faulty input matrix dimensions");
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        throw std::invalid_argument("Faulty output matrix dimensions");
    }
    std::vector<double> A_block;
    std::vector<double> B_block;
    A_block.reserve(MACRO_TILE * MACRO_TILE);
    B_block.reserve(MACRO_TILE * MACRO_TILE);

    #pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < A.rows; ii+=MACRO_TILE) {
        // prevents race conditions
        std::vector<double> A_block;
        std::vector<double> B_block;
        for (size_t kk = 0; kk < A.cols; kk+=MACRO_TILE) {
            // Pack a block of matrix A into a smaller buffer
            size_t i_max = std::min(MACRO_TILE, A.rows - ii);
            size_t k_max = std::min(MACRO_TILE, A.cols - kk);
            A_block.resize(i_max * k_max);
            pack_A(A.data + ii * A.cols + kk, A_block.data(), A.cols, i_max, k_max);

            for (size_t jj = 0; jj < B.cols; jj+=MACRO_TILE) {
                // Pack a block of matrix B into a smaller buffer
                size_t j_max = std::min(MACRO_TILE, B.cols - jj);
                B_block.resize(k_max * j_max);
                pack_B(B.data + kk * B.cols + jj, B_block.data(), B.cols, k_max, j_max);

                // Process micro tiles
                for (size_t i = 0; i < i_max; i += MICRO_TILE) {
                    size_t mr = std::min(MICRO_TILE, i_max - i);
                    for (size_t j = 0; j < j_max; j += MICRO_TILE) {
                        size_t nr = std::min(MICRO_TILE, j_max - j);
                        micro_matmul(k_max, &A_block[i * k_max], k_max, &B_block[j], j_max, &C.data[(ii + i) * C.cols + jj + j], C.cols, mr, nr);
                    }
                }
            }
            
        }
    }
}