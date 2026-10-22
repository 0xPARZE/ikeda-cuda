// SPDX-License-Identifier: MIT
//
// ikeda_sweep: run a parameter sweep on the CPU or GPU and write .npy output.
//
//   ikeda_sweep --config configs/bifurcation.cfg
//   ikeda_sweep --device cpu --n-params 1000 --precision float --out out/test
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>

#include "ikeda/io.hpp"
#include "ikeda/options.hpp"
#include "ikeda/sweep.hpp"

#ifdef IKEDA_HAVE_CUDA
#include "ikeda/sweep_gpu.hpp"
#endif

namespace {

template <typename Real>
int run(const ikeda::RunOptions& opt) {
    std::ostringstream extra;
    ikeda::SweepResult<Real> result;
    const char* precision = sizeof(Real) == 4 ? "float" : "double";

    if (opt.device == ikeda::Device::Gpu) {
#ifdef IKEDA_HAVE_CUDA
        if (!ikeda::gpu_available()) {
            std::cerr << "error: no CUDA device found; rerun with --device cpu\n";
            return 1;
        }
        const auto info = ikeda::gpu_info();
        ikeda::GpuTiming t;
        result = ikeda::sweep_gpu<Real>(opt.sweep, &t, opt.block_size > 0 ? opt.block_size : ikeda::kDefaultBlockSize);
        std::printf("GPU %s (sm_%d%d), %s: kernel %.2f ms, D2H %.2f ms, end-to-end %.2f ms\n", info.name.c_str(),
                    info.cc_major, info.cc_minor, precision, t.kernel_ms, t.transfer_ms, t.end_to_end_ms);
        extra << "\"device\": \"gpu\", \"gpu\": \"" << info.name << "\", \"kernel_ms\": " << t.kernel_ms
              << ", \"end_to_end_ms\": " << t.end_to_end_ms;
#else
        std::cerr << "error: built without CUDA support; rerun with --device cpu\n";
        return 1;
#endif
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        result = ikeda::sweep_cpu<Real>(opt.sweep, opt.cpu_threads);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("CPU, %s: %.2f ms\n", precision, ms);
        extra << "\"device\": \"cpu\", \"cpu_ms\": " << ms;
    }

    ikeda::io::write_sweep(opt.out_dir, opt.sweep, result, extra.str());
    std::printf("wrote %zu parameters x %d samples to %s/\n", opt.sweep.n_params,
                opt.sweep.store_orbit ? opt.sweep.n_record : 0, opt.out_dir.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto opt = ikeda::parse_options(argc, argv);
        if (opt.help) {
            std::cout << ikeda::usage(argv[0]);
            return 0;
        }
        return opt.precision == ikeda::Precision::Float ? run<float>(opt) : run<double>(opt);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
