| Implementation (N = 1,048,576) | GPU kernel (ms) | End-to-end (ms) | Throughput (G iter/s) | Speedup vs. C++ OpenMP, same precision |
|---|---:|---:|---:|---:|
| NumPy, vectorised over B | - | 72689.7 | 0.014 | 0.09x |
| C++ OpenMP, FP64 | - | 6651.4 | 0.158 | 1.00x |
| C++ OpenMP, FP32 | - | 5009.8 | 0.209 | 1.00x |
| CUDA, FP64 | 568.0 | 572.5 | 1.831 | **11.6x** |
| CUDA, FP32 | 21.0 | 23.3 | 44.984 | **214.9x** |

- Python loop (coursework style), N = 4,096: 1,796.9 ms; CUDA FP64 is 90x faster; CUDA FP32 is 1,563x faster
- C++ 1 thread, FP64, N = 262,144: 9,428.3 ms; CUDA FP64 is 64x faster; CUDA FP32 is 1,658x faster

### Workflow: README bifurcation diagram (4000 B values, 2000 transient + 300 stored, output included)

| Implementation | Kernel (ms) | D2H copy (ms) | End-to-end (ms) |
|---|---:|---:|---:|
| C++ 1 thread, FP64 | - | - | 315.8 |
| C++ OpenMP, FP64 | - | - | 54.2 |
| CUDA, FP64 | 23.1 | 4.4 | 43.3 |
| CUDA, FP32 | 0.7 | 1.4 | 5.3 |

### Block size (N = 1,048,576, kernel time in ms)

| Block | FP64 | FP32 |
|---:|---:|---:|
| 32 | 573.1 | 22.6 |
| 64 | 576.5 | 20.7 |
| 128 | 578.7 | 21.5 |
| 256 | 582.8 | 21.9 |
| 512 | 588.9 | 21.6 |
| 1024 | 602.5 | 22.1 |
