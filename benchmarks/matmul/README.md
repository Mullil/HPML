## Baseline algorithm without parallelization or optimizations

[source code](https://github.com/Mullil/HPML/blob/v1.0.0/src/matmul.cpp)

Intel i5-8250U

Without optimization flags / with -O3 -march=native -fopenmp

- 100x100: 0.005 s / 0.001 s

- 1000x1000: 10.8 s / 2.10 s 

- 3000x3000: 434 s / 177 s


## Added tiling (tile size = 64x64) and contiguous memory access

[source code](https://github.com/Mullil/HPML/blob/v1.0.1/src/matmul.cpp)

Intel i5-8250U

Without optimization flags / with -O3 -march=native -fopenmp

- 100x100: 0.005 s / 0.000 s

- 1000x1000: 5.24 s / 0.221 s

- 3000x3000: 185 s / 10.2 s

- 5000x5000: - / 45.4 s
