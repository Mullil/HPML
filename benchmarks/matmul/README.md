## Baseline algorithm without parallelization or optimizations

[source code](https://github.com/Mullil/HPML/blob/v1.0.0/src/matmul.cpp)

Intel i5-8250U

- 100x100: 0.005 s

- 1000x1000: 10.8 s

- 3000x3000: 434 s


## Added tiling (tile size = 64x64) and contiguous memory access

Intel i5-8250U

- 100x100: 0.005 s

- 1000x1000: 5.24 s

- 3000x3000: 185 s

## Move from single-level tiling to two-level tiling (macro-tile size = 128x128, micro-tile size = 4x4)

Intel i5-8250U

- 100x100: 0.003 s

- 1000x1000: 3.48 s

- 3000x3000: 138 s
