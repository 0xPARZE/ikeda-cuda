// SPDX-License-Identifier: MIT
//
// GPU results against the CPU reference.
//
// What "agreement" can mean. CUDA's sin/cos/log are correctly rounded to
// within 1-2 ulp but are not bit-identical to glibc's, and nvcc and g++
// contract a*b+c into FMAs in different places. The two backends therefore
// start to differ at the 1e-16 level on the very first step. What happens to
// that difference depends on the dynamics:
//
//   * on a stable periodic orbit (lambda < 0) it is damped, so the recorded
//     orbits must agree pointwise to near machine precision;
//   * on the chaotic attractor (lambda > 0) it grows like e^(lambda n) and
//     saturates at the size of the attractor after ~70 iterations. Pointwise
//     comparison is then meaningless, and the right check is that invariant
//     statistics, here the Lyapunov exponent over a long window, agree within
//     their sampling error.
#include <gtest/gtest.h>

#include <cmath>

#include "ikeda/analysis.hpp"
#include "ikeda/sweep.hpp"
#include "ikeda/sweep_gpu.hpp"

namespace {

class SweepGpu : public ::testing::Test {
 protected:
    void SetUp() override {
        if (!ikeda::gpu_available()) GTEST_SKIP() << "no CUDA device available";
    }
};

// Periodic window B in [0, 0.64]: below the accumulation point of the cascade
// (B_inf ~ 0.6463), so every orbit is a stable cycle.
ikeda::SweepConfig periodic_window(std::size_t n_params) {
    ikeda::SweepConfig c;
    c.b_min = 0.0;
    c.b_max = 0.64;
    c.n_params = n_params;
    c.n_transient = 3000;
    c.n_record = 64;
    return c;
}

template <typename Real>
void expect_close(const ikeda::SweepResult<Real>& gpu, const ikeda::SweepResult<Real>& cpu, double orbit_tol,
                  double lyap_tol) {
    ASSERT_EQ(gpu.x.size(), cpu.x.size());
    ASSERT_EQ(gpu.lyapunov.size(), cpu.lyapunov.size());
    for (std::size_t k = 0; k < gpu.x.size(); ++k) {
        ASSERT_NEAR(gpu.x[k], cpu.x[k], orbit_tol) << "flat index " << k;
        ASSERT_NEAR(gpu.y[k], cpu.y[k], orbit_tol) << "flat index " << k;
    }
    for (std::size_t i = 0; i < gpu.lyapunov.size(); ++i) {
        if (std::isinf(cpu.lyapunov[i])) {
            EXPECT_EQ(gpu.lyapunov[i], cpu.lyapunov[i]);  // B = 0
        } else {
            EXPECT_NEAR(gpu.lyapunov[i], cpu.lyapunov[i], lyap_tol) << "param " << i;
        }
    }
}

TEST_F(SweepGpu, DoubleMatchesCpuOnPeriodicOrbits) {
    const auto c = periodic_window(2000);
    expect_close(ikeda::sweep_gpu<double>(c), ikeda::sweep_cpu<double>(c), 1e-10, 1e-9);
}

// Float near a bifurcation converges slowly (|mu| -> 1), so float round-off
// differences are damped less there; 1e-3 is still 1000x smaller than the
// spacing between the branches the diagram resolves.
TEST_F(SweepGpu, FloatMatchesCpuOnPeriodicOrbits) {
    const auto c = periodic_window(2000);
    expect_close(ikeda::sweep_gpu<float>(c), ikeda::sweep_cpu<float>(c), 1e-3, 1e-2);
}

// Parameter counts that are not multiples of the block size (including a
// single parameter and fewer parameters than one warp) exercise the bounds
// check in the kernel. Every block size must give bitwise identical results,
// since each orbit is computed independently.
TEST_F(SweepGpu, ArbitrarySizesAndBlockSizes) {
    for (std::size_t n : {1u, 7u, 31u, 255u, 257u, 1000u}) {
        const auto c = periodic_window(n);
        const auto cpu = ikeda::sweep_cpu<double>(c);
        const auto reference = ikeda::sweep_gpu<double>(c, nullptr, 256);
        expect_close(reference, cpu, 1e-10, 1e-9);
        for (int block : {32, 128, 1024}) {
            const auto r = ikeda::sweep_gpu<double>(c, nullptr, block);
            EXPECT_EQ(r.x, reference.x) << "n = " << n << ", block = " << block;
            EXPECT_EQ(r.lyapunov, reference.lyapunov) << "n = " << n << ", block = " << block;
        }
    }
}

TEST_F(SweepGpu, SingleParameterAtFixedPoint) {
    ikeda::SweepConfig c;
    c.b_min = c.b_max = 0.3;
    c.n_params = 1;
    c.n_transient = 5000;
    c.n_record = 10;
    const auto r = ikeda::sweep_gpu<double>(c);
    const auto z = ikeda::find_cycle(0.3, 1, {0.7, 0.0});
    ASSERT_TRUE(z.has_value());
    EXPECT_NEAR(r.x_at(0, 9), z->x, 1e-12);
    EXPECT_NEAR(r.y_at(0, 9), z->y, 1e-12);
}

// In the chaotic regime, compare the invariant (lambda), not the trajectory.
// With N = 4e5 samples the standard error of a finite-time Lyapunov estimate
// on this attractor is ~3e-3, so 0.015 is a 5-sigma bound.
// B = 0.82, 0.86, 0.90 are chosen to be on the chaotic attractor. Nearby
// values such as 0.85 and 0.87 sit in periodic windows, and from z_0 = 0
// every B > 0.9 falls onto a coexisting stable focus.
TEST_F(SweepGpu, ChaoticLyapunovExponentsAgreeStatistically) {
    ikeda::SweepConfig c;
    c.b_min = 0.82;
    c.b_max = 0.90;
    c.n_params = 3;
    c.n_transient = 1000;
    c.n_record = 400000;
    c.store_orbit = false;
    const auto gpu = ikeda::sweep_gpu<double>(c);
    const auto cpu = ikeda::sweep_cpu<double>(c);
    for (std::size_t i = 0; i < c.n_params; ++i) {
        EXPECT_GT(gpu.lyapunov[i], 0.0);
        EXPECT_NEAR(gpu.lyapunov[i], cpu.lyapunov[i], 0.015) << "B = " << c.b_value(i);
    }
}

// The noise stream is a pure function of (seed, parameter, iteration), so on
// a contracting orbit the noisy GPU and CPU runs agree pointwise.
TEST_F(SweepGpu, NoiseMatchesCpuOnContractingOrbits) {
    auto c = periodic_window(500);
    c.b_max = 0.35;
    c.noise_sigma = 1e-3;
    expect_close(ikeda::sweep_gpu<double>(c), ikeda::sweep_cpu<double>(c), 1e-10, 1e-9);
}

TEST_F(SweepGpu, TimingIsReported) {
    ikeda::GpuTiming t;
    ikeda::sweep_gpu<float>(periodic_window(4096), &t);
    EXPECT_GT(t.kernel_ms, 0.0f);
    EXPECT_GE(t.end_to_end_ms, static_cast<double>(t.kernel_ms));
}

TEST_F(SweepGpu, RejectsInvalidBlockSize) {
    const auto c = periodic_window(10);
    EXPECT_THROW(ikeda::sweep_gpu<double>(c, nullptr, 100), std::invalid_argument);
    EXPECT_THROW(ikeda::sweep_gpu<double>(c, nullptr, 2048), std::invalid_argument);
}

}  // namespace
