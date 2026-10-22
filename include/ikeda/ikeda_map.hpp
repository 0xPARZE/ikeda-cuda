// SPDX-License-Identifier: MIT
//
// The Ikeda map, written once and shared by the CPU reference and the CUDA
// kernels.
//
// The Ikeda map models the electric field z_n circulating in a nonlinear
// optical ring cavity, sampled once per round trip:
//
//     z_{n+1} = A + B z_n exp[ i (phi - p / (1 + |z_n|^2)) ]
//
//   A    amplitude of the injected (input) field
//   B    mirror reflection coefficient (fraction of field kept per round trip)
//   phi  linear (empty-cavity) phase shift
//   p    nonlinear (Kerr) phase shift; intensity-dependent detuning
//
// Why a 2D real system instead of std::complex / cuComplex
// ----------------------------------------------------------
// Writing z = x + i y and t = phi - p / (1 + x^2 + y^2) gives
//
//     x_{n+1} = A + B (x cos t - y sin t)
//     y_{n+1} =     B (x sin t + y cos t)
//
// which is just a rotation by t, a contraction by B, and a shift by A. In this
// form one iteration costs a single sincos() plus a handful of FMAs: there is
// no complex exp(), no complex multiply, and no dependency on a complex type
// that behaves differently on host and device. It also gives us a closed-form
// 2x2 Jacobian (below), which is what the Lyapunov exponent needs.
//
// Everything in this header is a plain inline function on POD types so that
// the *same* source is compiled by g++ for the CPU reference and by nvcc for
// the GPU. Any difference between the two outputs therefore comes from the
// floating-point math libraries, not from two hand-maintained copies of the
// equations drifting apart.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define IKEDA_HD __host__ __device__ __forceinline__
#else
#define IKEDA_HD inline
#endif

namespace ikeda {

// Physical parameters of the cavity that stay fixed during a sweep over B.
template <typename Real>
struct CavityParams {
    Real A;    // input field amplitude
    Real p;    // nonlinear phase shift
    Real phi;  // linear phase shift
};

// The classic parameter set from Ikeda (1979) and Hammel, Jones & Moloney
// (1985), which is also the set used in the PHYS 4250 project.
inline constexpr double kDefaultA = 1.0;
inline constexpr double kDefaultP = 6.0;
inline constexpr double kDefaultPhi = 0.4;

template <typename Real>
constexpr CavityParams<Real> default_cavity() {
    return {Real(kDefaultA), Real(kDefaultP), Real(kDefaultPhi)};
}

// The cavity field in real coordinates: z = x + i y.
template <typename Real>
struct State {
    Real x;
    Real y;
};

// Row-major 2x2 Jacobian  d(x', y') / d(x, y).
template <typename Real>
struct Jacobian {
    Real j00, j01;
    Real j10, j11;
};

namespace detail {

// sincos dispatch. On the device we call the CUDA intrinsic, which computes
// both values in one pass. On the host we call std::sin and std::cos; g++
// merges these into a single sincos call at -O2 and above.
IKEDA_HD void sincos_hd(double t, double* s, double* c) {
#if defined(__CUDA_ARCH__)
    ::sincos(t, s, c);
#else
    *s = std::sin(t);
    *c = std::cos(t);
#endif
}

IKEDA_HD void sincos_hd(float t, float* s, float* c) {
#if defined(__CUDA_ARCH__)
    ::sincosf(t, s, c);
#else
    *s = std::sin(t);
    *c = std::cos(t);
#endif
}

}  // namespace detail

// One round trip of the cavity: z_n -> z_{n+1}.
template <typename Real>
IKEDA_HD State<Real> step(State<Real> z, Real B, const CavityParams<Real>& cav) {
    const Real t = cav.phi - cav.p / (Real(1) + z.x * z.x + z.y * z.y);
    Real s, c;
    detail::sincos_hd(t, &s, &c);
    return {cav.A + B * (z.x * c - z.y * s), B * (z.x * s + z.y * c)};
}

// One round trip that also returns the Jacobian evaluated at z_n.
//
// Derivation. Let r2 = x^2 + y^2, d = 1 + r2, and t = phi - p / d. Then
//
//     dt/dx = 2 p x / d^2,      dt/dy = 2 p y / d^2.
//
// Differentiating x' = A + B (x cos t - y sin t) by the chain rule:
//
//     dx'/dx =  B cos t + B (-x sin t - y cos t) dt/dx =  B cos t - y' dt/dx
//     dx'/dy = -B sin t + B (-x sin t - y cos t) dt/dy = -B sin t - y' dt/dy
//
// and y' = B (x sin t + y cos t):
//
//     dy'/dx =  B sin t + B ( x cos t - y sin t) dt/dx =  B sin t + (x' - A) dt/dx
//     dy'/dy =  B cos t + B ( x cos t - y sin t) dt/dy =  B cos t + (x' - A) dt/dy
//
// We use the identities B (x sin t + y cos t) = y' and B (x cos t - y sin t) = x' - A
// so the Jacobian reuses the values already computed for the step.
//
// A useful check: det J = B^2 for every z. The map is a rotation and scaling
// by B composed with a shear, so it contracts phase-space area by exactly B^2
// per round trip. The tests verify this, and it implies lambda_1 + lambda_2 = 2 ln B.
template <typename Real>
IKEDA_HD State<Real> step_with_jacobian(State<Real> z, Real B, const CavityParams<Real>& cav, Jacobian<Real>* J) {
    const Real d = Real(1) + z.x * z.x + z.y * z.y;
    const Real t = cav.phi - cav.p / d;
    Real s, c;
    detail::sincos_hd(t, &s, &c);

    const State<Real> next{cav.A + B * (z.x * c - z.y * s), B * (z.x * s + z.y * c)};

    const Real k = Real(2) * cav.p / (d * d);
    const Real dt_dx = k * z.x;
    const Real dt_dy = k * z.y;
    const Real u = next.x - cav.A;  // = B (x cos t - y sin t)

    J->j00 = B * c - next.y * dt_dx;
    J->j01 = -B * s - next.y * dt_dy;
    J->j10 = B * s + u * dt_dx;
    J->j11 = B * c + u * dt_dy;
    return next;
}

}  // namespace ikeda
