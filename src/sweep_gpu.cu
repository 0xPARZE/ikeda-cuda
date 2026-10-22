// SPDX-License-Identifier: MIT
#include <chrono>
#include <stdexcept>

#include "ikeda/cuda_utils.cuh"
#include "ikeda/sweep_gpu.hpp"

namespace ikeda {
namespace {

// One thread integrates one orbit. All per-orbit state (z, the tangent vector,
// the running log-growth) stays in registers for the whole run, and the only
// global-memory traffic is the coalesced, write-only orbit output. The kernel
// is therefore bound by the throughput of sincos, divide, and log, not by
// memory bandwidth. That matters for precision: consumer and mobile GPUs run
// FP64 at 1/32 of the FP32 rate.
template <typename Real>
__global__ void sweep_kernel(SweepArgs<Real> args, Real* __restrict__ out_x, Real* __restrict__ out_y,
                             Real* __restrict__ out_lyap) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // The last block is partially empty unless n_params is a multiple of the
    // block size.
    if (i >= args.n_params) return;
    out_lyap[i] = integrate_orbit(i, args, out_x, out_y);
}

}  // namespace

bool gpu_available() {
    int n = 0;
    // A missing driver or device is reported as an error code here, not an
    // exception, so the CPU-only code paths still work.
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

GpuInfo gpu_info(int device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    GpuInfo info;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.memory_gib = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

template <typename Real>
SweepResult<Real> sweep_gpu(const SweepConfig& cfg, GpuTiming* timing, int block_size) {
    cfg.validate();
    if (block_size <= 0 || block_size % 32 != 0 || block_size > 1024) {
        throw std::invalid_argument("block_size must be a positive multiple of 32, at most 1024");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    SweepResult<Real> result = make_result<Real>(cfg);
    const std::size_t n_orbit = result.x.size();
    {
        cuda::DeviceBuffer<Real> d_x(n_orbit), d_y(n_orbit), d_lyap(cfg.n_params);
        cuda::Event start, after_kernel, after_copy;

        const auto args = SweepArgs<Real>::from(cfg);
        const std::size_t blocks = (cfg.n_params + block_size - 1) / block_size;

        start.record();
        sweep_kernel<Real><<<static_cast<unsigned>(blocks), block_size>>>(
            args, cfg.store_orbit ? d_x.data() : nullptr, cfg.store_orbit ? d_y.data() : nullptr, d_lyap.data());
        CUDA_CHECK_LAUNCH();
        after_kernel.record();

        d_lyap.copy_to_host(result.lyapunov.data());
        if (cfg.store_orbit) {
            d_x.copy_to_host(result.x.data());
            d_y.copy_to_host(result.y.data());
        }
        after_copy.record();
        after_copy.synchronize();

        if (timing) {
            timing->kernel_ms = cuda::Event::elapsed_ms(start, after_kernel);
            timing->transfer_ms = cuda::Event::elapsed_ms(after_kernel, after_copy);
        }
    }  // device buffers freed here
    if (timing) {
        timing->end_to_end_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    }
    return result;
}

template SweepResult<float> sweep_gpu<float>(const SweepConfig&, GpuTiming*, int);
template SweepResult<double> sweep_gpu<double>(const SweepConfig&, GpuTiming*, int);

}  // namespace ikeda
