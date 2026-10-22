// SPDX-License-Identifier: MIT
//
// Parameter sweeps over the mirror reflectivity B.
//
// A bifurcation diagram needs thousands to millions of independent orbits, one
// per value of B. Each orbit is a strictly sequential recurrence: iteration n+1
// can't start until iteration n is done. The parallelism is therefore *across*
// parameters, never within an orbit. The design follows from that:
//
//   * one GPU thread (or one OpenMP loop iteration) owns one value of B;
//   * the whole per-orbit computation lives in integrate_orbit() below, which
//     the CPU reference and the CUDA kernel both call, so the two can't
//     disagree about the physics;
//   * orbit samples are stored iteration-major, out[k * n_params + i]. At a
//     fixed iteration k, consecutive threads i write consecutive addresses, so
//     every store from a warp is coalesced into a few wide transactions. The
//     original prototype used parameter-major storage (out[i * n_record + k]),
//     which made a warp's 32 stores land n_record elements apart.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ikeda/counter_rng.hpp"
#include "ikeda/ikeda_map.hpp"

namespace ikeda {

// User-facing description of a sweep. Defaults reproduce the figures in the
// README.
struct SweepConfig {
    double b_min = 0.0;           // first reflectivity in the sweep
    double b_max = 1.0;           // last reflectivity in the sweep (inclusive)
    std::size_t n_params = 4000;  // number of B values

    // Iterations discarded before recording, so the orbit can settle onto its
    // attractor. Near a bifurcation, convergence slows like |mu|^n with |mu| -> 1,
    // so a few thousand round trips are needed for a clean diagram.
    int n_transient = 2000;

    // Iterations recorded per B, and also the averaging window for the
    // Lyapunov exponent.
    int n_record = 300;

    // If false, only the Lyapunov exponents are returned. This is the
    // compute-bound mode used by the benchmarks, since it has almost no output
    // traffic.
    bool store_orbit = true;

    double x0 = 0.0;  // initial field, Re z_0
    double y0 = 0.0;  // initial field, Im z_0

    // Standard deviation of the complex Gaussian noise added each round trip
    // (0 means deterministic).
    double noise_sigma = 0.0;
    std::uint64_t seed = 20260620;  // only used when noise_sigma > 0

    CavityParams<double> cavity = default_cavity<double>();

    // Spacing of the B grid. A single-point sweep has step 0 and evaluates b_min.
    double b_step() const { return n_params > 1 ? (b_max - b_min) / static_cast<double>(n_params - 1) : 0.0; }

    // Value of B for parameter index i, using the same formula as the kernel.
    double b_value(std::size_t i) const { return b_min + b_step() * static_cast<double>(i); }

    void validate() const {
        if (n_params == 0) throw std::invalid_argument("n_params must be >= 1");
        if (n_transient < 0) throw std::invalid_argument("n_transient must be >= 0");
        if (n_record < 1) throw std::invalid_argument("n_record must be >= 1");
        if (!(b_max >= b_min)) throw std::invalid_argument("require b_max >= b_min");
        if (noise_sigma < 0) throw std::invalid_argument("noise_sigma must be >= 0");
    }
};

// Results of a sweep, in host memory.
template <typename Real>
struct SweepResult {
    std::size_t n_params = 0;
    int n_record = 0;
    std::vector<Real> x;         // [n_record * n_params], iteration-major; empty if !store_orbit
    std::vector<Real> y;         // same layout as x
    std::vector<Real> lyapunov;  // [n_params], largest Lyapunov exponent per B

    Real x_at(std::size_t param, int k) const { return x[static_cast<std::size_t>(k) * n_params + param]; }
    Real y_at(std::size_t param, int k) const { return y[static_cast<std::size_t>(k) * n_params + param]; }
};

// Flattened, trivially copyable arguments passed by value to the kernel.
template <typename Real>
struct SweepArgs {
    CavityParams<Real> cavity;
    Real b_min;
    Real b_step;
    std::size_t n_params;
    int n_transient;
    int n_record;
    Real x0, y0;
    Real noise_sigma;
    std::uint64_t seed;

    static SweepArgs from(const SweepConfig& c) {
        return {{Real(c.cavity.A), Real(c.cavity.p), Real(c.cavity.phi)},
                Real(c.b_min),
                Real(c.b_step()),
                c.n_params,
                c.n_transient,
                c.n_record,
                Real(c.x0),
                Real(c.y0),
                Real(c.noise_sigma),
                c.seed};
    }
};

// Integrates one orbit for parameter index i, writes its recorded samples
// (if out_x != nullptr), and returns the largest Lyapunov exponent.
//
// The Lyapunov exponent uses the standard tangent-vector method (Benettin et
// al., 1980): push a unit vector v through the analytic Jacobian, accumulate
// ln|J v|, and renormalise each step so it doesn't overflow or underflow. A
// finite-difference "two nearby trajectories" estimate would need a second
// orbit and a tuned separation. The analytic Jacobian is exact and nearly
// free, since it reuses the sin/cos from the step.
template <typename Real>
IKEDA_HD Real integrate_orbit(std::size_t i, const SweepArgs<Real>& a, Real* out_x, Real* out_y) {
    const Real B = a.b_min + a.b_step * static_cast<Real>(i);
    const bool noisy = a.noise_sigma > Real(0);
    State<Real> z{a.x0, a.y0};

    std::uint64_t n = 0;  // global iteration counter, used as the RNG counter
    for (int k = 0; k < a.n_transient; ++k, ++n) {
        z = step(z, B, a.cavity);
        if (noisy) {
            Real gx, gy;
            gaussian_pair(a.seed, i, n, &gx, &gy);
            z.x += a.noise_sigma * gx;
            z.y += a.noise_sigma * gy;
        }
    }

    Real vx = 1, vy = 0;  // tangent vector
    Real log_growth = 0;
    for (int k = 0; k < a.n_record; ++k, ++n) {
        Jacobian<Real> J;
        z = step_with_jacobian(z, B, a.cavity, &J);
        if (noisy) {
            Real gx, gy;
            gaussian_pair(a.seed, i, n, &gx, &gy);
            z.x += a.noise_sigma * gx;
            z.y += a.noise_sigma * gy;
        }

        const Real wx = J.j00 * vx + J.j01 * vy;
        const Real wy = J.j10 * vx + J.j11 * vy;
        const Real norm = std::sqrt(wx * wx + wy * wy);
        // At B = 0 the Jacobian is identically zero and lambda = -inf. Adding
        // log(0) = -inf is correct, and skipping the renormalisation keeps v
        // finite, so no NaN can appear.
        log_growth += std::log(norm);
        if (norm > Real(0)) {
            vx = wx / norm;
            vy = wy / norm;
        }

        if (out_x) {
            const std::size_t w = static_cast<std::size_t>(k) * a.n_params + i;
            out_x[w] = z.x;
            out_y[w] = z.y;
        }
    }
    return log_growth / static_cast<Real>(a.n_record);
}

// CPU reference implementation (src/sweep_cpu.cpp). n_threads <= 0 uses
// every OpenMP thread; 1 gives a strictly serial run.
template <typename Real>
SweepResult<Real> sweep_cpu(const SweepConfig& cfg, int n_threads = 0);

// Shared helper that allocates a host result with the right shape.
template <typename Real>
SweepResult<Real> make_result(const SweepConfig& cfg) {
    SweepResult<Real> r;
    r.n_params = cfg.n_params;
    r.n_record = cfg.n_record;
    r.lyapunov.resize(cfg.n_params);
    if (cfg.store_orbit) {
        const std::size_t n = cfg.n_params * static_cast<std::size_t>(cfg.n_record);
        r.x.resize(n);
        r.y.resize(n);
    }
    return r;
}

}  // namespace ikeda
