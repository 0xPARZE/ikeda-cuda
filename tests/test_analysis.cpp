// SPDX-License-Identifier: MIT
//
// Tests of the Newton/continuation tools and the CW steady-state analysis.
#include <gtest/gtest.h>

#include <cmath>

#include "ikeda/analysis.hpp"

namespace {

constexpr double kFeigenbaumDelta = 4.669201609102990;

TEST(Analysis, FixedPointSatisfiesMapEquation) {
    for (double B : {0.1, 0.3, 0.5, 0.9}) {
        const auto z = ikeda::find_cycle(B, 1, {0.5, 0.0});
        ASSERT_TRUE(z.has_value()) << "B = " << B;
        const auto fz = ikeda::step(*z, B, ikeda::default_cavity<double>());
        EXPECT_NEAR(fz.x, z->x, 1e-12);
        EXPECT_NEAR(fz.y, z->y, 1e-12);
    }
}

// The fixed point of the map is a CW steady state, so its intensity must
// solve the CW equation u (1 + B^2 - 2 B cos t(u)) = A^2.
TEST(Analysis, FixedPointIsACwSteadyState) {
    const auto cav = ikeda::default_cavity<double>();
    for (double B : {0.2, 0.4, 0.6}) {
        const auto z = ikeda::find_cycle(B, 1, {0.5, 0.0});
        ASSERT_TRUE(z.has_value());
        const double u = z->x * z->x + z->y * z->y;
        const double lhs = u * (1 + B * B - 2 * B * std::cos(cav.phi - cav.p / (1 + u)));
        EXPECT_NEAR(lhs, cav.A * cav.A, 1e-11);
    }
}

TEST(Analysis, MonodromyDeterminantIsBToThe2n) {
    for (int period : {1, 2, 4, 8}) {
        const double B = 0.63;
        const auto M = ikeda::monodromy({0.4, -0.2}, B, period);
        EXPECT_NEAR(M.j00 * M.j11 - M.j01 * M.j10, std::pow(B, 2 * period), 1e-12);
    }
}

TEST(Analysis, EigenvaluesOfKnownMatrices) {
    const auto real = ikeda::eigenvalues({2, 0, 0, -3});
    EXPECT_DOUBLE_EQ(real[0].real(), -3);
    EXPECT_DOUBLE_EQ(real[1].real(), 2);
    const auto rot = ikeda::eigenvalues({0, -1, 1, 0});  // rotation by 90 degrees
    EXPECT_NEAR(std::abs(rot[0].imag()), 1.0, 1e-15);
    EXPECT_NEAR(rot[0].real(), 0.0, 1e-15);
}

class Cascade : public ::testing::Test {
 protected:
    static void SetUpTestSuite() { result_ = ikeda::period_doubling_cascade(6); }
    static ikeda::CascadeResult result_;
};
ikeda::CascadeResult Cascade::result_;

// At B_1 the fixed point must have a multiplier of exactly -1. Since the
// multipliers then satisfy mu_1 mu_2 = B^2, the other one is -B^2.
TEST_F(Cascade, FirstThresholdIsAFlipBifurcationOfTheFixedPoint) {
    const double B1 = result_.thresholds[0];
    const auto z = ikeda::find_cycle(B1, 1, {0.7, 0.1});
    ASSERT_TRUE(z.has_value());
    const auto mu = ikeda::eigenvalues(ikeda::monodromy(*z, B1, 1));
    EXPECT_NEAR(mu[0].real(), -1.0, 1e-8);
    EXPECT_NEAR(mu[1].real(), -B1 * B1, 1e-8);
    EXPECT_NEAR(B1, 0.4027006301, 1e-9);  // regression value
}

TEST_F(Cascade, ThresholdsAccumulateAtTheFeigenbaumRate) {
    const auto& t = result_.thresholds;
    ASSERT_EQ(t.size(), 6u);
    for (std::size_t k = 1; k < t.size(); ++k) EXPECT_GT(t[k], t[k - 1]);
    // Successive ratios converge to Feigenbaum's delta. The deepest estimate
    // should be within 1%.
    ASSERT_EQ(result_.feigenbaum.size(), 4u);
    EXPECT_NEAR(result_.feigenbaum.back() / kFeigenbaumDelta, 1.0, 0.01);
}

TEST(Analysis, CwSteadyStateCount) {
    EXPECT_EQ(ikeda::count_steady_states(0.0), 1);
    EXPECT_EQ(ikeda::count_steady_states(0.5), 1);
    EXPECT_EQ(ikeda::count_steady_states(0.8), 3);
}

// Ikeda's central result: the round-trip delay destabilises the cavity (B_1)
// well before CW theory predicts any instability (the bistable fold, B_cw).
TEST(Analysis, DelayInstabilityPrecedesCwBistability) {
    const double b_cw = ikeda::bistability_threshold();
    EXPECT_NEAR(b_cw, 0.7083763219, 1e-8);  // regression value
    EXPECT_EQ(ikeda::count_steady_states(b_cw - 1e-6), 1);
    EXPECT_EQ(ikeda::count_steady_states(b_cw + 1e-6), 3);
    EXPECT_LT(ikeda::period_doubling_cascade(1).thresholds[0], b_cw);
}

}  // namespace
