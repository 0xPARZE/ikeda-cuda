// SPDX-License-Identifier: MIT
//
// A stateless, counter-based Gaussian noise source shared by the CPU and GPU.
//
// Stochastic runs add a small complex Gaussian kick to each round trip. That
// models spontaneous emission and thermal fluctuations in the cavity. For the
// results to be reproducible, and for the GPU to be testable against the CPU,
// both implementations must draw *exactly the same* noise samples.
//
// A stateful generator (std::mt19937, or curand's XORWOW) doesn't fit that
// requirement well. Each GPU thread would need its own seeded state, and the
// CPU would have to reproduce the GPU's per-thread streams. Instead, each
// sample is a pure function of (seed, parameter index, iteration). We hash
// those three values with the SplitMix64 finaliser (Steele et al., 2014) and
// turn the hash into a pair of normals via Box-Muller. The result doesn't
// depend on the thread count, the launch configuration, or the order in which
// the work runs.
#pragma once

#include <cmath>
#include <cstdint>

#include "ikeda/ikeda_map.hpp"

namespace ikeda {

IKEDA_HD std::uint64_t splitmix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Two independent N(0, 1) samples for (seed, stream, counter).
template <typename Real>
IKEDA_HD void gaussian_pair(std::uint64_t seed, std::uint64_t stream, std::uint64_t counter, Real* g0, Real* g1) {
    const std::uint64_t h = splitmix64(splitmix64(seed ^ splitmix64(stream)) + counter);

    // Split the 64-bit hash into two 32-bit uniforms. u1 lies in (0, 1], so
    // log(u1) is always finite. Box-Muller is evaluated in double even for
    // float runs, so the float and double sweeps see the same noise sequence
    // (it only runs when noise is enabled, so the cost doesn't matter).
    constexpr double kInv2Pow32 = 1.0 / 4294967296.0;
    const double u1 = (static_cast<double>(h >> 32) + 1.0) * kInv2Pow32;
    const double u2 = static_cast<double>(h & 0xFFFFFFFFull) * kInv2Pow32;

    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const double r = std::sqrt(-2.0 * std::log(u1));
    double s, c;
    detail::sincos_hd(kTwoPi * u2, &s, &c);
    *g0 = static_cast<Real>(r * c);
    *g1 = static_cast<Real>(r * s);
}

}  // namespace ikeda
