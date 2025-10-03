## Baseline algorithm without parallelization or optimizations

[source code](https://github.com/Mullil/HPML/blob/v1.0.0/src/matmul.cpp)

Intel i5-8250U

- 100x100: 0.005 s

- 1000x1000: 10.8 s

- 3000x3000: 434 s


## Added tiling (tile size=64) and contiguous memory access

Intel i5-8250U

- 100x100: 0.005 s

- 1000x1000: 5.24 s

- 3000x3000: 185 s
