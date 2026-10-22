// SPDX-License-Identifier: MIT
#include "ikeda/options.hpp"

#include <charconv>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ikeda {
namespace {

std::string trim(std::string_view s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    const auto e = s.find_last_not_of(" \t\r");
    return std::string(s.substr(b, e - b + 1));
}

template <typename T>
T parse_number(const std::string& key, const std::string& v) {
    T out{};
    const auto* end = v.data() + v.size();
    const auto [ptr, ec] = std::from_chars(v.data(), end, out);
    if (ec != std::errc() || ptr != end) {
        throw std::invalid_argument("invalid value '" + v + "' for '" + key + "'");
    }
    return out;
}

bool parse_bool(const std::string& key, const std::string& v) {
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    throw std::invalid_argument("invalid boolean '" + v + "' for '" + key + "'");
}

using Setter = std::function<void(RunOptions&, const std::string& key, const std::string& value)>;

struct Key {
    Setter set;
    const char* help;
};

const std::map<std::string, Key>& keys() {
    static const std::map<std::string, Key> table = {
        {"b-min",
         {[](auto& o, auto& k, auto& v) { o.sweep.b_min = parse_number<double>(k, v); },
          "first reflectivity B in the sweep [0]"}},
        {"b-max",
         {[](auto& o, auto& k, auto& v) { o.sweep.b_max = parse_number<double>(k, v); },
          "last reflectivity B in the sweep, inclusive [1]"}},
        {"n-params",
         {[](auto& o, auto& k, auto& v) { o.sweep.n_params = parse_number<std::size_t>(k, v); },
          "number of B values [4000]"}},
        {"transient",
         {[](auto& o, auto& k, auto& v) { o.sweep.n_transient = parse_number<int>(k, v); },
          "iterations discarded before recording [2000]"}},
        {"record",
         {[](auto& o, auto& k, auto& v) { o.sweep.n_record = parse_number<int>(k, v); },
          "iterations recorded / averaged for Lyapunov [300]"}},
        {"store-orbit",
         {[](auto& o, auto& k, auto& v) { o.sweep.store_orbit = parse_bool(k, v); },
          "write x.npy/y.npy, not just lyapunov.npy [true]"}},
        {"x0", {[](auto& o, auto& k, auto& v) { o.sweep.x0 = parse_number<double>(k, v); }, "initial Re z_0 [0]"}},
        {"y0", {[](auto& o, auto& k, auto& v) { o.sweep.y0 = parse_number<double>(k, v); }, "initial Im z_0 [0]"}},
        {"noise",
         {[](auto& o, auto& k, auto& v) { o.sweep.noise_sigma = parse_number<double>(k, v); },
          "std. dev. of complex Gaussian noise per round trip [0]"}},
        {"seed",
         {[](auto& o, auto& k, auto& v) { o.sweep.seed = parse_number<std::uint64_t>(k, v); },
          "noise seed [20260620]"}},
        {"A",
         {[](auto& o, auto& k, auto& v) { o.sweep.cavity.A = parse_number<double>(k, v); },
          "input field amplitude [1]"}},
        {"p",
         {[](auto& o, auto& k, auto& v) { o.sweep.cavity.p = parse_number<double>(k, v); },
          "nonlinear phase shift [6]"}},
        {"phi",
         {[](auto& o, auto& k, auto& v) { o.sweep.cavity.phi = parse_number<double>(k, v); },
          "linear phase shift [0.4]"}},
        {"device",
         {[](auto& o, auto& k, auto& v) {
              if (v == "cpu")
                  o.device = Device::Cpu;
              else if (v == "gpu")
                  o.device = Device::Gpu;
              else
                  throw std::invalid_argument("'" + k + "' must be cpu or gpu");
          },
          "cpu or gpu [gpu]"}},
        {"precision",
         {[](auto& o, auto& k, auto& v) {
              if (v == "float")
                  o.precision = Precision::Float;
              else if (v == "double")
                  o.precision = Precision::Double;
              else
                  throw std::invalid_argument("'" + k + "' must be float or double");
          },
          "float or double [double]"}},
        {"threads",
         {[](auto& o, auto& k, auto& v) { o.cpu_threads = parse_number<int>(k, v); }, "CPU threads, 0 = all [0]"}},
        {"block-size",
         {[](auto& o, auto& k, auto& v) { o.block_size = parse_number<int>(k, v); },
          "CUDA threads per block, 0 = default [0]"}},
        {"out", {[](auto& o, auto&, auto& v) { o.out_dir = v; }, "output directory [out/sweep]"}},
    };
    return table;
}

void apply(RunOptions& o, const std::string& key, const std::string& value) {
    const auto it = keys().find(key);
    if (it == keys().end()) throw std::invalid_argument("unknown option '" + key + "'");
    it->second.set(o, key, value);
}

void apply_config_file(RunOptions& o, const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("cannot open config file '" + path + "'");
    std::string line;
    for (int lineno = 1; std::getline(f, line); ++lineno) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        if (trim(line).empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::invalid_argument(path + ":" + std::to_string(lineno) + ": expected 'key = value'");
        }
        apply(o, trim(std::string_view(line).substr(0, eq)), trim(std::string_view(line).substr(eq + 1)));
    }
}

}  // namespace

RunOptions parse_options(int argc, const char* const* argv) {
    RunOptions o;
    // Apply the config file first, wherever it appears, so that command-line
    // flags always take precedence over it.
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--config") apply_config_file(o, argv[i + 1]);
    }
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            o.help = true;
            continue;
        }
        if (arg.substr(0, 2) != "--") throw std::invalid_argument("unexpected argument '" + std::string(arg) + "'");
        if (i + 1 >= argc) throw std::invalid_argument("missing value for '" + std::string(arg) + "'");
        const std::string key(arg.substr(2));
        const std::string value = argv[++i];
        if (key != "config") apply(o, key, value);
    }
    o.sweep.validate();
    return o;
}

std::string usage(const char* program) {
    std::ostringstream s;
    s << "Usage: " << program << " [--config FILE] [--KEY VALUE ...]\n\n"
      << "Every KEY may also be given as 'KEY = VALUE' in the config file;\n"
      << "command-line values override the file.\n\n";
    for (const auto& [name, key] : keys()) {
        s << "  --" << name << std::string(name.size() < 14 ? 14 - name.size() : 1, ' ') << key.help << '\n';
    }
    return s.str();
}

}  // namespace ikeda
