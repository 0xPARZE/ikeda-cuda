// SPDX-License-Identifier: MIT
//
// Benchmark driver: CPU (1 thread and all threads, OpenMP) vs GPU (float and
// double) for Lyapunov sweeps of increasing size.
//
// Method
//   * Every configuration is run kWarmup times untimed (CUDA context creation,
//     module load, first-touch page faults, CPU frequency ramp-up) and then
//     kReps times. The repetition with the median end-to-end time is
//     reported.
//   * GPU kernel time comes from cudaEvents bracketing the launch. GPU
//     end-to-end time is host wall clock around sweep_gpu(): device
//     allocation, kernel, device-to-host copy, and free. CPU time is host wall
//     clock around sweep_cpu().
//   * The scaling runs use store_orbit = false, so they measure compute. A
//     separate "workflow" section times the real bifurcation-diagram run
//     (4000 B values x 300 stored samples), where the transfer is part of
//     the cost.
//
// Writes CSV to the path given as argv[1] (default benchmarks/results/latest.csv).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ikeda/sweep.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef IKEDA_HAVE_CUDA
#include "ikeda/sweep_gpu.hpp"
#endif

namespace {

constexpr int kWarmup = 3;
constexpr int kReps = 7;
#ifdef IKEDA_HAVE_CUDA
// Laptop GPUs idle at a few hundred MHz and take a moment to reach boost
// clocks, so the GPU is kept busy for this long before anything is timed.
constexpr double kGpuSpinUpSeconds = 3.0;
#endif
constexpr int kTransient = 500;
constexpr int kRecord = 500;  // 1000 map iterations per parameter in total
// A single-threaded CPU run at 2^20 parameters takes ~30 s per repetition, so
// the serial baseline stops here. Its throughput is flat by then anyway.
constexpr std::size_t kMaxSerialParams = std::size_t{1} << 18;

struct Timing {
    double kernel_ms = 0;
    double transfer_ms = 0;
    double total_ms = 0;
};

// Runs `run` kWarmup times untimed, then `reps` times, and returns the
// repetition with the median end-to-end time. Its kernel and transfer times
// are reported as they were measured together, so they stay consistent.
Timing measure(const std::function<Timing()>& run, int reps) {
    for (int i = 0; i < kWarmup; ++i) run();
    std::vector<Timing> runs;
    for (int i = 0; i < reps; ++i) runs.push_back(run());
    std::sort(runs.begin(), runs.end(), [](const Timing& a, const Timing& b) { return a.total_ms < b.total_ms; });
    return runs[runs.size() / 2];
}

template <typename Real>
Timing time_cpu(const ikeda::SweepConfig& c, int threads) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = ikeda::sweep_cpu<Real>(c, threads);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r.lyapunov.empty()) std::abort();  // keep the result observable
    return {ms, 0.0, ms};
}

#ifdef IKEDA_HAVE_CUDA
template <typename Real>
Timing time_gpu(const ikeda::SweepConfig& c, int block) {
    ikeda::GpuTiming g;
    ikeda::sweep_gpu<Real>(c, &g, block);
    return {g.kernel_ms, g.transfer_ms, g.end_to_end_ms};
}
#endif

class CsvWriter {
 public:
    explicit CsvWriter(const std::string& path) : f_(path) {
        f_ << "section,impl,precision,n_params,iterations,store_orbit,block_size,threads,"
              "kernel_ms,transfer_ms,total_ms,giga_iter_per_s\n";
    }
    void row(const std::string& section, const std::string& impl, const std::string& precision,
             const ikeda::SweepConfig& c, int block, int threads, const Timing& t) {
        const double iters = static_cast<double>(c.n_params) * (c.n_transient + c.n_record);
        const double gips = iters / (t.kernel_ms * 1e6);
        f_ << section << ',' << impl << ',' << precision << ',' << c.n_params << ',' << (c.n_transient + c.n_record)
           << ',' << (c.store_orbit ? 1 : 0) << ',' << block << ',' << threads << ',' << t.kernel_ms << ','
           << t.transfer_ms << ',' << t.total_ms << ',' << gips << '\n';
        f_.flush();
        std::printf("  %-9s %-12s %-6s N=%-8zu kernel %10.3f ms  total %10.3f ms  %7.3f Giter/s\n", section.c_str(),
                    impl.c_str(), precision.c_str(), c.n_params, t.kernel_ms, t.total_ms, gips);
    }

 private:
    std::ofstream f_;
};

ikeda::SweepConfig lyapunov_config(std::size_t n) {
    ikeda::SweepConfig c;
    c.b_min = 0.0;
    c.b_max = 1.0;
    c.n_params = n;
    c.n_transient = kTransient;
    c.n_record = kRecord;
    c.store_orbit = false;
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "benchmarks/results/latest.csv";
    std::filesystem::create_directories(std::filesystem::path(out).parent_path());
    CsvWriter csv(out);

#ifdef _OPENMP
    const int all_threads = omp_get_max_threads();
#else
    const int all_threads = 1;
#endif
    std::printf("CPU threads: %d (hardware_concurrency = %u)\n", all_threads, std::thread::hardware_concurrency());
#ifdef IKEDA_HAVE_CUDA
    const bool gpu = ikeda::gpu_available();
    if (gpu) {
        const auto info = ikeda::gpu_info();
        const auto t0 = std::chrono::steady_clock::now();
        const auto spin = lyapunov_config(std::size_t{1} << 20);
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < kGpuSpinUpSeconds) {
            ikeda::sweep_gpu<float>(spin);
        }
        std::printf("GPU: %s, sm_%d%d, %d SMs, %.1f GiB, CUDA runtime %d, driver %d\n", info.name.c_str(),
                    info.cc_major, info.cc_minor, info.sm_count, info.memory_gib, info.runtime_version,
                    info.driver_version);
    }
#else
    [[maybe_unused]] const bool gpu = false;
#endif

    // 1. Scaling with the number of parameters (compute only).
    std::printf("\n[scaling] %d iterations per parameter, Lyapunov only\n", kTransient + kRecord);
    for (std::size_t n = 256; n <= (std::size_t{1} << 20); n *= 4) {
        const auto c = lyapunov_config(n);
        if (n <= kMaxSerialParams) {
            csv.row("scaling", "cpu_serial", "double", c, 0, 1,
                    measure([&] { return time_cpu<double>(c, 1); }, n >= 65536 ? 1 : kReps));
        }
        csv.row("scaling", "cpu_openmp", "double", c, 0, all_threads,
                measure([&] { return time_cpu<double>(c, all_threads); }, kReps));
        csv.row("scaling", "cpu_openmp", "float", c, 0, all_threads,
                measure([&] { return time_cpu<float>(c, all_threads); }, kReps));
#ifdef IKEDA_HAVE_CUDA
        if (gpu) {
            csv.row("scaling", "gpu", "double", c, ikeda::kDefaultBlockSize, 0,
                    measure([&] { return time_gpu<double>(c, ikeda::kDefaultBlockSize); }, kReps));
            csv.row("scaling", "gpu", "float", c, ikeda::kDefaultBlockSize, 0,
                    measure([&] { return time_gpu<float>(c, ikeda::kDefaultBlockSize); }, kReps));
        }
#endif
    }

#ifdef IKEDA_HAVE_CUDA
    if (gpu) {
        // 2. Block-size sweep at the largest size.
        std::printf("\n[blocksize] N = 2^20\n");
        const auto c = lyapunov_config(std::size_t{1} << 20);
        for (int block : {32, 64, 128, 256, 512, 1024}) {
            csv.row("blocksize", "gpu", "double", c, block, 0,
                    measure([&] { return time_gpu<double>(c, block); }, kReps));
            csv.row("blocksize", "gpu", "float", c, block, 0,
                    measure([&] { return time_gpu<float>(c, block); }, kReps));
        }
    }
#endif

    // 3. The real workflow: the README's bifurcation diagram, with output.
    std::printf("\n[workflow] bifurcation diagram: 4000 x (2000 transient + 300 stored)\n");
    const ikeda::SweepConfig w;  // defaults are the figure configuration
    csv.row("workflow", "cpu_serial", "double", w, 0, 1, measure([&] { return time_cpu<double>(w, 1); }, kReps));
    csv.row("workflow", "cpu_openmp", "double", w, 0, all_threads,
            measure([&] { return time_cpu<double>(w, all_threads); }, kReps));
#ifdef IKEDA_HAVE_CUDA
    if (gpu) {
        csv.row("workflow", "gpu", "double", w, ikeda::kDefaultBlockSize, 0,
                measure([&] { return time_gpu<double>(w, ikeda::kDefaultBlockSize); }, kReps));
        csv.row("workflow", "gpu", "float", w, ikeda::kDefaultBlockSize, 0,
                measure([&] { return time_gpu<float>(w, ikeda::kDefaultBlockSize); }, kReps));
    }
#endif

    std::printf("\nwrote %s\n", out.c_str());
    return 0;
}
