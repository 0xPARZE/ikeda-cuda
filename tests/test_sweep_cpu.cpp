// SPDX-License-Identifier: MIT
//
// Tests of the CPU reference sweep against analytic and semi-analytic results.
#include <gtest/gtest.h>

#include <cmath>

#include "ikeda/analysis.hpp"
#include "ikeda/sweep.hpp"

namespace {

ikeda::SweepConfig single(double B, int transient, int record) {
    ikeda::SweepConfig c;
    c.b_min = c.b_max = B;
    c.n_params = 1;
    c.n_transient = transient;
    c.n_record = record;
    return c;
}

// For B below ~0.2 the fixed point is a stable focus: its two multipliers are a
// complex-conjugate pair, so |mu| = sqrt(det J) = B and the Lyapunov exponent
// is exactly ln B. The finite-time estimate converges as O(1/N).
TEST(SweepCpu, LyapunovEqualsLnBInTheFocusRegime) {
    ikeda::SweepConfig c;
    c.b_min = 0.05;
    c.b_max = 0.18;
    c.n_params = 14;
    c.n_transient = 1000;
    c.n_record = 20000;
    c.store_orbit = false;
    const auto r = ikeda::sweep_cpu<double>(c);
    for (std::size_t i = 0; i < c.n_params; ++i) {
        EXPECT_NEAR(r.lyapunov[i], std::log(c.b_value(i)), 1e-3) << "B = " << c.b_value(i);
    }
}

// Between the focus regime and B_1 the multipliers are real. The Lyapunov
// exponent is then ln |mu_max| of the Newton-computed fixed point, and the
// settled orbit sits on that fixed point.
TEST(SweepCpu, MatchesNewtonFixedPointBelowFirstBifurcation) {
    for (double B : {0.25, 0.3, 0.35, 0.39}) {
        const auto r = ikeda::sweep_cpu<double>(single(B, 5000, 20000));
        const auto z = ikeda::find_cycle(B, 1, {0.7, 0.0});
        ASSERT_TRUE(z.has_value());
        EXPECT_NEAR(r.x_at(0, 0), z->x, 1e-10);
        EXPECT_NEAR(r.y_at(0, 0), z->y, 1e-10);
        const auto mu = ikeda::eigenvalues(ikeda::monodromy(*z, B, 1));
        EXPECT_NEAR(r.lyapunov[0], std::log(std::abs(mu[0])), 1e-3) << "B = " << B;
    }
}

// Past each threshold B_k, the orbit from z_0 = 0 settles onto the 2^k-cycle of
// the cascade. The exception is B = 0.64, which is between B_3 and B_4, so a
// period-8 cycle exists there. But z_0 = 0 lies in the basin of a *coexisting*
// period-6 orbit instead. Multistability like this is why the sweep records the
// initial condition in meta.json.
TEST(SweepCpu, OrbitPeriodsFollowTheCascade) {
    struct Case {
        double B;
        int period;
    };
    for (const auto [B, period] : {Case{0.30, 1}, Case{0.50, 2}, Case{0.62, 4}, Case{0.64, 6}}) {
        const auto r = ikeda::sweep_cpu<double>(single(B, 20000, 64));
        for (int k = 0; k + period < 64; ++k) {
            EXPECT_NEAR(r.x_at(0, k + period), r.x_at(0, k), 1e-8) << "B = " << B;
        }
        for (int q = 1; q < period; ++q) {
            // ...and no shorter period.
            EXPECT_GT(std::abs(r.x_at(0, q) - r.x_at(0, 0)), 1e-4) << "B = " << B << ", q = " << q;
        }
        EXPECT_LT(r.lyapunov[0], 0.0);
    }
}

// The classic Ikeda attractor (B = 0.9) is chaotic with Lyapunov exponents
// (0.5076, -0.7183): J. C. Sprott and A. Xiong, Chaos 25, 083101 (2015).
TEST(SweepCpu, ClassicAttractorIsChaotic) {
    const auto r = ikeda::sweep_cpu<double>(single(0.9, 1000, 2000000));
    EXPECT_NEAR(r.lyapunov[0], 0.5076, 0.015);
}

TEST(SweepCpu, ZeroReflectivityGivesMinusInfinityNotNaN) {
    const auto r = ikeda::sweep_cpu<double>(single(0.0, 10, 10));
    EXPECT_TRUE(std::isinf(r.lyapunov[0]) && r.lyapunov[0] < 0);
    EXPECT_DOUBLE_EQ(r.x_at(0, 0), 1.0);  // z = A once the cavity has no feedback
}

TEST(SweepCpu, SingleParameterMatchesTheSameColumnOfALargerSweep) {
    ikeda::SweepConfig big;
    big.n_params = 101;
    big.b_min = 0.0;
    big.b_max = 1.0;
    big.n_transient = 200;
    big.n_record = 50;
    const auto r = ikeda::sweep_cpu<double>(big);
    const std::size_t col = 90;  // B = 0.9
    const auto s = ikeda::sweep_cpu<double>(single(big.b_value(col), 200, 50));
    for (int k = 0; k < 50; ++k) EXPECT_EQ(s.x_at(0, k), r.x_at(col, k));
    EXPECT_EQ(s.lyapunov[0], r.lyapunov[col]);
}

// Results are bitwise independent of the thread count (with and without
// noise). Each orbit is computed by one thread, with no shared reductions.
TEST(SweepCpu, DeterministicAcrossThreadCounts) {
    for (double sigma : {0.0, 0.01}) {
        ikeda::SweepConfig c;
        c.n_params = 503;  // prime: uneven split across threads
        c.n_transient = 300;
        c.n_record = 40;
        c.noise_sigma = sigma;
        const auto a = ikeda::sweep_cpu<double>(c, 1);
        const auto b = ikeda::sweep_cpu<double>(c, 4);
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.lyapunov, b.lyapunov);
    }
}

TEST(SweepCpu, StoreOrbitDoesNotChangeLyapunov) {
    ikeda::SweepConfig c;
    c.n_params = 200;
    c.n_transient = 300;
    c.n_record = 100;
    const auto with = ikeda::sweep_cpu<double>(c);
    c.store_orbit = false;
    const auto without = ikeda::sweep_cpu<double>(c);
    EXPECT_TRUE(without.x.empty());
    EXPECT_EQ(with.lyapunov, without.lyapunov);
}

TEST(SweepCpu, NoiseIsSeededAndReproducible) {
    auto c = single(0.9, 100, 200);
    c.noise_sigma = 0.01;
    const auto a = ikeda::sweep_cpu<double>(c);
    const auto b = ikeda::sweep_cpu<double>(c);
    EXPECT_EQ(a.x, b.x);
    c.seed += 1;
    const auto d = ikeda::sweep_cpu<double>(c);
    EXPECT_NE(a.x, d.x);
}

// In the strongly contracting regime, noise of size sigma displaces the orbit
// from the fixed point by O(sigma), not more.
TEST(SweepCpu, WeakNoiseStaysNearTheFixedPoint) {
    auto c = single(0.3, 2000, 5000);
    c.noise_sigma = 1e-4;
    const auto r = ikeda::sweep_cpu<double>(c);
    const auto z = ikeda::find_cycle(0.3, 1, {0.7, 0.0});
    ASSERT_TRUE(z.has_value());
    for (int k = 0; k < c.n_record; ++k) {
        EXPECT_LT(std::hypot(r.x_at(0, k) - z->x, r.y_at(0, k) - z->y), 10 * c.noise_sigma);
    }
}

TEST(SweepConfig, ValidateRejectsBadInput) {
    ikeda::SweepConfig c;
    c.n_params = 0;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c = {};
    c.n_record = 0;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c = {};
    c.b_min = 1.0;
    c.b_max = 0.5;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c = {};
    c.noise_sigma = -1;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(SweepConfig, GridEndpointsAreExact) {
    ikeda::SweepConfig c;
    c.b_min = 0.2;
    c.b_max = 0.8;
    c.n_params = 7;
    EXPECT_DOUBLE_EQ(c.b_value(0), 0.2);
    EXPECT_NEAR(c.b_value(6), 0.8, 1e-15);
    c.n_params = 1;
    EXPECT_DOUBLE_EQ(c.b_value(0), 0.2);
}

}  // namespace
