// SPDX-License-Identifier: MIT
//
// ikeda_thresholds: compute the period-doubling cascade, the Feigenbaum ratio
// estimates, and the CW bistability threshold for the default cavity
// (A = 1, p = 6, phi = 0.4). Writes a JSON summary that the plotting scripts
// read to annotate the bifurcation diagram.
//
//   ikeda_thresholds [output.json]
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "ikeda/analysis.hpp"

namespace {

constexpr int kCascadeLevels = 6;  // B_1 ... B_6: up to the 32 -> 64 doubling
constexpr double kFeigenbaumDelta = 4.669201609102990;

}  // namespace

int main(int argc, char** argv) {
    try {
        const char* out_path = argc > 1 ? argv[1] : "out/thresholds.json";

        const auto cascade = ikeda::period_doubling_cascade(kCascadeLevels);
        const double b_cw = ikeda::bistability_threshold();

        std::printf("Period-doubling cascade (A = 1, p = 6, phi = 0.4)\n");
        for (std::size_t k = 0; k < cascade.thresholds.size(); ++k) {
            std::printf("  B_%zu = %.10f   (period %d -> %d)\n", k + 1, cascade.thresholds[k], 1 << k, 2 << k);
        }
        std::printf("Feigenbaum ratio estimates (delta = %.6f)\n", kFeigenbaumDelta);
        for (std::size_t k = 0; k < cascade.feigenbaum.size(); ++k) {
            std::printf("  delta_%zu = %.4f   (%.2f%% of delta)\n", k + 1, cascade.feigenbaum[k],
                        100.0 * cascade.feigenbaum[k] / kFeigenbaumDelta);
        }
        std::printf("CW bistability threshold (3 steady states appear)\n  B_cw = %.10f\n", b_cw);

        const auto parent = std::filesystem::path(out_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream f(out_path);
        if (!f) throw std::runtime_error(std::string("cannot open ") + out_path);
        f << std::setprecision(15) << "{\n  \"period_doubling\": [";
        for (std::size_t k = 0; k < cascade.thresholds.size(); ++k) f << (k ? ", " : "") << cascade.thresholds[k];
        f << "],\n  \"feigenbaum\": [";
        for (std::size_t k = 0; k < cascade.feigenbaum.size(); ++k) f << (k ? ", " : "") << cascade.feigenbaum[k];
        f << "],\n  \"cw_bistability\": " << b_cw << "\n}\n";
        std::printf("wrote %s\n", out_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
