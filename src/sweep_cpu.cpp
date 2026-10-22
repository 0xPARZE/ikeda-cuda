// SPDX-License-Identifier: MIT
#include "ikeda/sweep.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ikeda {

template <typename Real>
SweepResult<Real> sweep_cpu(const SweepConfig& cfg, int n_threads) {
    cfg.validate();
    SweepResult<Real> result = make_result<Real>(cfg);
    const auto args = SweepArgs<Real>::from(cfg);
    Real* out_x = cfg.store_orbit ? result.x.data() : nullptr;
    Real* out_y = cfg.store_orbit ? result.y.data() : nullptr;
    const auto n = static_cast<std::ptrdiff_t>(cfg.n_params);

#ifdef _OPENMP
    const int threads = n_threads > 0 ? n_threads : omp_get_max_threads();
    // Every orbit runs the same number of iterations, so a static schedule
    // already balances the load and avoids dynamic-scheduling overhead.
#pragma omp parallel for schedule(static) num_threads(threads)
#else
    static_cast<void>(n_threads);
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        result.lyapunov[idx] = integrate_orbit(idx, args, out_x, out_y);
    }
    return result;
}

template SweepResult<float> sweep_cpu<float>(const SweepConfig&, int);
template SweepResult<double> sweep_cpu<double>(const SweepConfig&, int);

}  // namespace ikeda
