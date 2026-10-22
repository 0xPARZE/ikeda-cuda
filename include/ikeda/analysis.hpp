// SPDX-License-Identifier: MIT
//
// Semi-analytic tools for the Ikeda map: periodic orbits, their stability,
// the period-doubling cascade, and the continuous-wave (CW) steady states.
//
// These run on the CPU in double precision. They serve two purposes:
//   1. physics results: bifurcation thresholds B_k, Feigenbaum ratios, and
//      the CW bistability threshold;
//   2. independent ground truth for the sweep tests. They locate orbits with
//      Newton's method, not brute-force iteration, so they can check the sweep
//      without sharing its failure modes.
#pragma once

#include <array>
#include <complex>
#include <optional>
#include <vector>

#include "ikeda/ikeda_map.hpp"

namespace ikeda {

using Cavity = CavityParams<double>;
using Point = State<double>;

// Iterates the map n times from z.
Point iterate(Point z, double B, int n, const Cavity& cav = default_cavity<double>());

// Monodromy matrix of an n-cycle through z: the product J(z_{n-1}) ... J(z_0).
// Its eigenvalues are the cycle's Floquet multipliers. Because det J = B^2 at
// every point, det M = B^(2n).
Jacobian<double> monodromy(Point z, double B, int period, const Cavity& cav = default_cavity<double>());

// Eigenvalues of a 2x2 matrix, sorted so that |mu[0]| >= |mu[1]|.
std::array<std::complex<double>, 2> eigenvalues(const Jacobian<double>& M);

// Newton's method on F(z) = f^period(z) - z, seeded at `guess`. Its Jacobian
// is M - I, with M the monodromy matrix above. Returns nullopt if Newton
// fails to converge to |F| < 1e-12.
std::optional<Point> find_cycle(double B, int period, Point guess, const Cavity& cav = default_cavity<double>());

// Real part of the multiplier furthest to the left. A period-doubling
// (flip) bifurcation is where this crosses -1.
double leftmost_multiplier(Point cycle_point, double B, int period, const Cavity& cav = default_cavity<double>());

struct CascadeResult {
    std::vector<double> thresholds;  // B_k: the 2^(k-1)-cycle loses stability to a 2^k-cycle
    std::vector<double> feigenbaum;  // delta_k = (B_k - B_{k-1}) / (B_{k+1} - B_k)
};

// Locates the first `levels` period-doubling thresholds by numerical
// continuation. Starting from the stable fixed point at small B, each cycle is
// tracked in B with Newton's method until its leftmost multiplier crosses -1.
// The crossing is then refined by bisection to `tol`, and the search moves on
// to the newly born cycle of twice the period.
CascadeResult period_doubling_cascade(int levels, double tol = 1e-11, const Cavity& cav = default_cavity<double>());

// CW steady states
// ----------------
// Steady-state (CW) theory looks for time-independent fields z = f(z). Put
// u = |z|^2. Then z = A / (1 - B e^{i t(u)}), so u solves the scalar equation
//
//     u (1 + B^2 - 2 B cos(phi - p / (1 + u))) = A^2.
//
// Several roots mean optical bistability, which is the instability CW theory
// predicts. Ikeda's point (1979) was that the round-trip delay makes the
// steady state unstable much earlier, through period doubling into chaos.
// The same cavity can't be described by this analysis alone.

// Number of distinct steady-state intensities u at reflectivity B (< 1).
int count_steady_states(double B, const Cavity& cav = default_cavity<double>());

// Smallest B in [b_lo, b_hi] at which more than one steady state exists, to
// within `tol`, or NaN if there is none.
double bistability_threshold(double b_lo = 0.0, double b_hi = 0.99, double tol = 1e-10,
                             const Cavity& cav = default_cavity<double>());

}  // namespace ikeda
