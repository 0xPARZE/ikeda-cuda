// SPDX-License-Identifier: MIT
//
// Tests of the map and its analytic Jacobian, against identities that hold
// exactly.
#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "ikeda/counter_rng.hpp"
#include "ikeda/ikeda_map.hpp"

namespace {

using ikeda::Jacobian;
using ikeda::State;

const auto kCav = ikeda::default_cavity<double>();

// A spread of points covering the attractor and the far field.
const State<double> kPoints[] = {{0.0, 0.0},  {0.5, -0.3}, {1.2, 0.7}, {-0.8, 1.5},
                                 {2.0, -2.0}, {0.1, 0.05}, {-3.0, 0.4}};
const double kBs[] = {0.1, 0.4, 0.6, 0.9, 0.99};

// The real-valued map must agree with the original complex form
// z' = A + B z exp(i (phi - p / (1 + |z|^2))).
TEST(IkedaMap, MatchesComplexForm) {
    for (const auto& z : kPoints) {
        for (double B : kBs) {
            const std::complex<double> zc(z.x, z.y);
            const std::complex<double> expected =
                kCav.A + B * zc * std::exp(std::complex<double>(0, kCav.phi - kCav.p / (1.0 + std::norm(zc))));
            const auto got = ikeda::step(z, B, kCav);
            EXPECT_NEAR(got.x, expected.real(), 1e-14);
            EXPECT_NEAR(got.y, expected.imag(), 1e-14);
        }
    }
}

TEST(IkedaMap, StepWithJacobianMatchesStep) {
    for (const auto& z : kPoints) {
        Jacobian<double> J;
        const auto a = ikeda::step(z, 0.9, kCav);
        const auto b = ikeda::step_with_jacobian(z, 0.9, kCav, &J);
        EXPECT_DOUBLE_EQ(a.x, b.x);
        EXPECT_DOUBLE_EQ(a.y, b.y);
    }
}

// The analytic Jacobian against a central finite difference, O(h^2) accurate.
TEST(IkedaMap, JacobianMatchesFiniteDifference) {
    constexpr double h = 1e-6;
    for (const auto& z : kPoints) {
        for (double B : kBs) {
            Jacobian<double> J;
            ikeda::step_with_jacobian(z, B, kCav, &J);
            const auto px = ikeda::step({z.x + h, z.y}, B, kCav);
            const auto mx = ikeda::step({z.x - h, z.y}, B, kCav);
            const auto py = ikeda::step({z.x, z.y + h}, B, kCav);
            const auto my = ikeda::step({z.x, z.y - h}, B, kCav);
            EXPECT_NEAR(J.j00, (px.x - mx.x) / (2 * h), 1e-7);
            EXPECT_NEAR(J.j01, (py.x - my.x) / (2 * h), 1e-7);
            EXPECT_NEAR(J.j10, (px.y - mx.y) / (2 * h), 1e-7);
            EXPECT_NEAR(J.j11, (py.y - my.y) / (2 * h), 1e-7);
        }
    }
}

// det J = B^2 everywhere: the map contracts phase-space area uniformly.
TEST(IkedaMap, JacobianDeterminantIsBSquared) {
    for (const auto& z : kPoints) {
        for (double B : kBs) {
            Jacobian<double> J;
            ikeda::step_with_jacobian(z, B, kCav, &J);
            EXPECT_NEAR(J.j00 * J.j11 - J.j01 * J.j10, B * B, 1e-12);
        }
    }
}

TEST(IkedaMap, FloatAndDoubleAgreeForOneStep) {
    const auto cavf = ikeda::default_cavity<float>();
    for (const auto& z : kPoints) {
        const auto d = ikeda::step(z, 0.9, kCav);
        const auto f = ikeda::step(State<float>{float(z.x), float(z.y)}, 0.9f, cavf);
        EXPECT_NEAR(f.x, d.x, 1e-5);
        EXPECT_NEAR(f.y, d.y, 1e-5);
    }
}

TEST(CounterRng, IsAPureFunctionOfItsInputs) {
    double a0, a1, b0, b1;
    ikeda::gaussian_pair<double>(42, 7, 1000, &a0, &a1);
    ikeda::gaussian_pair<double>(42, 7, 1000, &b0, &b1);
    EXPECT_EQ(a0, b0);
    EXPECT_EQ(a1, b1);
    ikeda::gaussian_pair<double>(43, 7, 1000, &b0, &b1);
    EXPECT_NE(a0, b0);
}

TEST(CounterRng, HasStandardNormalMoments) {
    constexpr int n = 200000;
    double sum = 0, sum2 = 0, sum4 = 0;
    for (int k = 0; k < n; ++k) {
        double g0, g1;
        ikeda::gaussian_pair<double>(1, 0, static_cast<std::uint64_t>(k), &g0, &g1);
        for (double g : {g0, g1}) {
            sum += g;
            sum2 += g * g;
            sum4 += g * g * g * g;
        }
    }
    const double m = 2.0 * n;
    // Standard errors with 4e5 samples: mean ~0.0016, variance ~0.0022, kurtosis ~0.008.
    EXPECT_NEAR(sum / m, 0.0, 0.01);
    EXPECT_NEAR(sum2 / m, 1.0, 0.01);
    EXPECT_NEAR(sum4 / m, 3.0, 0.05);
}

}  // namespace
