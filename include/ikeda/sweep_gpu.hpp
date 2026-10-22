// SPDX-License-Identifier: MIT
//
// Public interface to the CUDA sweep. This header is plain C++ (no CUDA
// types), so host code built by an ordinary C++ compiler can include it.
#pragma once

#include <string>

#include "ikeda/sweep.hpp"

namespace ikeda {

// Threads per block. Each thread runs one long, register-resident orbit with
// no shared memory and no inter-thread communication, so the block size only
// affects occupancy, and occupancy isn't the bottleneck: throughput is limited
// by the special-function and FP64 units. With the GPU at its boost clock,
// every block size from 32 to 1024 gives the same kernel time to within ~8%
// run-to-run noise (benchmarks/, "blocksize" section). 256 is a conventional
// choice in that flat region.
inline constexpr int kDefaultBlockSize = 256;

struct GpuTiming {
    float kernel_ms = 0;       // cudaEvent time of the kernel alone
    float transfer_ms = 0;     // cudaEvent time of the device-to-host copies
    double end_to_end_ms = 0;  // host wall clock: allocation + kernel + copies + free
};

struct GpuInfo {
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    double memory_gib = 0;
    int runtime_version = 0;  // e.g. 13030 for CUDA 13.3
    int driver_version = 0;
};

// True if at least one CUDA device can be used.
bool gpu_available();

GpuInfo gpu_info(int device = 0);

// Runs the sweep on the current CUDA device. If timing is non-null, it is
// filled with kernel, transfer, and end-to-end times. Throws
// ikeda::cuda::CudaError on any CUDA failure.
template <typename Real>
SweepResult<Real> sweep_gpu(const SweepConfig& cfg, GpuTiming* timing = nullptr, int block_size = kDefaultBlockSize);

}  // namespace ikeda
