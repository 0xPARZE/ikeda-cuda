// SPDX-License-Identifier: MIT
//
// Run settings, read from an optional config file and command-line flags.
//
// Settings come from a `key = value` config file (see configs/), then any
// `--key value` flags on the command line, which override the file. Every key
// works in both places, so `ikeda_sweep --config configs/bifurcation.cfg
// --n-params 10000` reruns a saved configuration at a higher resolution.
#pragma once

#include <string>

#include "ikeda/sweep.hpp"

namespace ikeda {

enum class Device { Cpu, Gpu };
enum class Precision { Float, Double };

struct RunOptions {
    SweepConfig sweep;
    Device device = Device::Gpu;
    Precision precision = Precision::Double;
    int cpu_threads = 0;  // 0 means all hardware threads
    int block_size = 0;   // 0 means kDefaultBlockSize
    std::string out_dir = "out/sweep";
    bool help = false;
};

// Parses argv. Throws std::invalid_argument with a readable message on an
// unknown key or a malformed value.
RunOptions parse_options(int argc, const char* const* argv);

// Usage text listing every key, its meaning, and its default.
std::string usage(const char* program);

}  // namespace ikeda
