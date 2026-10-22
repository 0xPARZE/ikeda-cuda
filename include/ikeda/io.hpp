// SPDX-License-Identifier: MIT
//
// Output in NumPy's .npy format.
//
// The prototype wrote ~50 MB CSV files, which took longer to format and parse
// than the GPU took to compute them, and lost precision in the default
// 6-significant-digit stream formatting. A .npy file is a short ASCII header
// followed by the raw little-endian array, so writing it is a single
// fwrite, and np.load() reads it back bit-exactly with no parsing.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ikeda/sweep.hpp"

namespace ikeda::io {

// Writes a C-ordered float32/float64 array with the given shape.
template <typename T>
void write_npy(const std::string& path, const T* data, const std::vector<std::size_t>& shape);

// Writes a sweep into `dir` (created if needed):
//   B.npy         [n_params]            reflectivity of each column
//   lyapunov.npy  [n_params]            largest Lyapunov exponent
//   x.npy, y.npy  [n_record, n_params]  recorded orbit (only if store_orbit)
//   meta.json                           run configuration plus `extra_json` fields
// `extra_json` must be a comma-separated list of JSON members (or empty).
template <typename Real>
void write_sweep(const std::string& dir, const SweepConfig& cfg, const SweepResult<Real>& r,
                 const std::string& extra_json = "");

}  // namespace ikeda::io
