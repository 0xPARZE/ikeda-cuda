// SPDX-License-Identifier: MIT
#include "ikeda/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ikeda {
namespace {

// Newton's method settings for periodic orbits.
constexpr int kNewtonMaxIter = 60;
constexpr double kNewtonTol = 1e-12;

// Continuation settings for the cascade.
constexpr double kCascadeStartB = 0.05;  // fixed point is strongly attracting here
constexpr double kFirstStepB = 1e-3;     // march step while tracking the fixed point
constexpr double kStepsPerGap = 200.0;   // later steps: (B_k - B_{k-1}) / kStepsPerGap
constexpr double kNewbornOffset = 0.05;  // start the new cycle this fraction of a gap past B_k
constexpr int kSettleIters = 200000;     // iterations to fall onto the newborn cycle
constexpr double kMinCycleSeparation = 1e-7;

// Grid resolution for locating extrema of the CW steady-state function.
constexpr int kSteadyStateGrid = 20000;
constexpr int kGoldenIters = 100;

Jacobian<double> matmul(const Jacobian<double>& a, const Jacobian<double>& b) {
    return {a.j00 * b.j00 + a.j01 * b.j10, a.j00 * b.j01 + a.j01 * b.j11, a.j10 * b.j00 + a.j11 * b.j10,
            a.j10 * b.j01 + a.j11 * b.j11};
}

double distance(Point a, Point b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

}  // namespace

Point iterate(Point z, double B, int n, const Cavity& cav) {
    for (int k = 0; k < n; ++k) z = step(z, B, cav);
    return z;
}

Jacobian<double> monodromy(Point z, double B, int period, const Cavity& cav) {
    Jacobian<double> M{1, 0, 0, 1};
    for (int k = 0; k < period; ++k) {
        Jacobian<double> J;
        z = step_with_jacobian(z, B, cav, &J);
        M = matmul(J, M);
    }
    return M;
}

std::array<std::complex<double>, 2> eigenvalues(const Jacobian<double>& M) {
    const double tr = M.j00 + M.j11;
    const double det = M.j00 * M.j11 - M.j01 * M.j10;
    const std::complex<double> disc = std::sqrt(std::complex<double>(tr * tr - 4.0 * det));
    std::complex<double> a = 0.5 * (tr + disc);
    std::complex<double> b = 0.5 * (tr - disc);
    if (std::abs(b) > std::abs(a)) std::swap(a, b);
    return {a, b};
}

std::optional<Point> find_cycle(double B, int period, Point guess, const Cavity& cav) {
    Point z = guess;
    for (int it = 0; it < kNewtonMaxIter; ++it) {
        const Point fz = iterate(z, B, period, cav);
        const double fx = fz.x - z.x;
        const double fy = fz.y - z.y;
        if (std::hypot(fx, fy) < kNewtonTol) return z;

        // Solve (M - I) dz = -F with Cramer's rule.
        Jacobian<double> M = monodromy(z, B, period, cav);
        M.j00 -= 1.0;
        M.j11 -= 1.0;
        const double det = M.j00 * M.j11 - M.j01 * M.j10;
        if (det == 0.0 || !std::isfinite(det)) return std::nullopt;
        z.x += (-fx * M.j11 + fy * M.j01) / det;
        z.y += (-fy * M.j00 + fx * M.j10) / det;
    }
    return std::nullopt;
}

double leftmost_multiplier(Point cycle_point, double B, int period, const Cavity& cav) {
    const auto mu = eigenvalues(monodromy(cycle_point, B, period, cav));
    return std::min(mu[0].real(), mu[1].real());
}

CascadeResult period_doubling_cascade(int levels, double tol, const Cavity& cav) {
    auto require = [](std::optional<Point> p, double B, int period) {
        if (!p) {
            throw std::runtime_error("Newton failed to find the period-" + std::to_string(period) +
                                     " orbit at B = " + std::to_string(B));
        }
        return *p;
    };

    CascadeResult out;
    int period = 1;
    double b = kCascadeStartB;
    Point z = require(find_cycle(b, period, iterate({0, 0}, b, kSettleIters, cav), cav), b, period);
    double db = kFirstStepB;

    for (int level = 0; level < levels; ++level) {
        // March in B, tracking the cycle, until it loses stability.
        double b_lo = b;
        Point z_lo = z;
        double b_hi = b;
        for (;;) {
            b_hi = b_lo + db;
            if (b_hi >= 1.0) throw std::runtime_error("cascade search left the range B < 1");
            const Point z_hi = require(find_cycle(b_hi, period, z_lo, cav), b_hi, period);
            if (leftmost_multiplier(z_hi, b_hi, period, cav) < -1.0) break;
            b_lo = b_hi;
            z_lo = z_hi;
        }

        // Bisect for the flip bifurcation, keeping the guess on the stable side.
        while (b_hi - b_lo > tol) {
            const double mid = 0.5 * (b_lo + b_hi);
            const Point zm = require(find_cycle(mid, period, z_lo, cav), mid, period);
            if (leftmost_multiplier(zm, mid, period, cav) < -1.0) {
                b_hi = mid;
            } else {
                b_lo = mid;
                z_lo = zm;
            }
        }
        const double b_k = 0.5 * (b_lo + b_hi);
        out.thresholds.push_back(b_k);
        if (level + 1 == levels) break;

        // Move onto the newborn 2*period cycle. Just past B_k it is attracting,
        // so we iterate onto it from a point next to the now-unstable cycle and
        // then polish it with Newton.
        const double gap = level == 0 ? 0.2 : b_k - out.thresholds[out.thresholds.size() - 2];
        b = b_k + kNewbornOffset * gap;
        const Point kicked{z_lo.x + 1e-4, z_lo.y + 1e-4};
        period *= 2;
        z = require(find_cycle(b, period, iterate(kicked, b, kSettleIters, cav), cav), b, period);
        if (distance(iterate(z, b, period / 2, cav), z) < kMinCycleSeparation) {
            throw std::runtime_error("continuation fell back onto the period-" + std::to_string(period / 2) + " orbit");
        }
        db = level == 0 ? kFirstStepB : gap / kStepsPerGap;
    }

    for (std::size_t k = 1; k + 1 < out.thresholds.size(); ++k) {
        const auto& t = out.thresholds;
        out.feigenbaum.push_back((t[k] - t[k - 1]) / (t[k + 1] - t[k]));
    }
    return out;
}

int count_steady_states(double B, const Cavity& cav) {
    if (!(B >= 0.0 && B < 1.0)) throw std::invalid_argument("count_steady_states requires 0 <= B < 1");
    const double A2 = cav.A * cav.A;
    auto g = [&](double u) { return u * (1.0 + B * B - 2.0 * B * std::cos(cav.phi - cav.p / (1.0 + u))) - A2; };

    // Every steady state satisfies u <= A^2 / (1 - B)^2, since |1 - B e^{it}| >= 1 - B.
    // Between consecutive extrema g is monotonic, so the roots of g equal the
    // sign changes along [g(0), g(extremum_1), ..., g(u_max)]. Extrema are
    // bracketed on a grid and refined by golden-section search. That makes the
    // count exact even when two roots are about to merge at a fold, where a
    // plain sign-change scan would miss them.
    const double u_max = A2 / ((1.0 - B) * (1.0 - B)) * (1.0 + 1e-9);
    const double h = u_max / kSteadyStateGrid;
    std::vector<double> samples{g(0.0)};
    double prev = g(0.0), cur = g(h);
    for (int j = 1; j < kSteadyStateGrid; ++j) {
        const double next = g((j + 1) * h);
        if ((cur - prev) * (next - cur) < 0.0) {
            const bool is_max = cur > prev;
            double lo = (j - 1) * h, hi = (j + 1) * h;
            constexpr double kInvPhi = 0.6180339887498949;
            for (int it = 0; it < kGoldenIters; ++it) {
                const double m1 = hi - kInvPhi * (hi - lo);
                const double m2 = lo + kInvPhi * (hi - lo);
                if ((g(m1) < g(m2)) == is_max) {
                    lo = m1;
                } else {
                    hi = m2;
                }
            }
            samples.push_back(g(0.5 * (lo + hi)));
        }
        prev = cur;
        cur = next;
    }
    samples.push_back(g(u_max));

    int roots = 0;
    for (std::size_t k = 1; k < samples.size(); ++k) {
        if ((samples[k - 1] < 0.0) != (samples[k] < 0.0)) ++roots;
    }
    return roots;
}

double bistability_threshold(double b_lo, double b_hi, double tol, const Cavity& cav) {
    constexpr int kScan = 1000;
    double prev = b_lo;
    for (int j = 1; j <= kScan; ++j) {
        const double b = b_lo + (b_hi - b_lo) * j / kScan;
        if (count_steady_states(b, cav) > 1) {
            double lo = prev, hi = b;
            while (hi - lo > tol) {
                const double mid = 0.5 * (lo + hi);
                (count_steady_states(mid, cav) > 1 ? hi : lo) = mid;
            }
            return 0.5 * (lo + hi);
        }
        prev = b;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace ikeda
