# High-Performance Machine Learning

A repository for machine learning and related algorithms in C++

## Requirements

- CMake ≥ 3.14

- C++17

- Google Test

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

## Todos

Matrix multiplication:

- Thorough tests
- Benchmarks for baseline solution
- Optimizing for parallelization
