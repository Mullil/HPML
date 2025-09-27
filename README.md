# High-Performance Machine Learning

A repository for machine learning and related algorithms in C++

## Requirements

- CMake ≥ 3.14

- C++17

- Google Test

- Google Benchmark

## Build

To build run:

```Bash
mkdir build && cd build

cmake ..

make -j$(nproc)
```

To run tests run:

```Bash
./tests/matmul_test
```

To run benchmarks run:

```Bash
./benchmarks/matmul_bench
```

## Todos

Matrix multiplication:

- Thorough tests
- Optimizing for parallelization
- Benchmarks for non-square matrices

Future algorithms:

- SVD
- PCA
- Linear regression
- Logistic regression
- K-means
- Feedforward neural network
