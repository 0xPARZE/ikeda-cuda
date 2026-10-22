// SPDX-License-Identifier: MIT
#include "ikeda/io.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace ikeda::io {
namespace {

static_assert(std::endian::native == std::endian::little, "the .npy writer assumes a little-endian host");

template <typename T>
constexpr const char* npy_descr() {
    if constexpr (std::is_same_v<T, float>) return "<f4";
    if constexpr (std::is_same_v<T, double>) return "<f8";
}

std::string shape_tuple(const std::vector<std::size_t>& shape) {
    std::ostringstream s;
    s << '(';
    for (std::size_t d : shape) s << d << ", ";
    s << ')';
    return s.str();
}

}  // namespace

template <typename T>
void write_npy(const std::string& path, const T* data, const std::vector<std::size_t>& shape) {
    static_assert(std::is_floating_point_v<T>);
    std::string header = std::string("{'descr': '") + npy_descr<T>() +
                         "', 'fortran_order': False, 'shape': " + shape_tuple(shape) + ", }";
    // NPY v1.0: 6-byte magic, 2-byte version, 2-byte little-endian header
    // length, then the header padded with spaces and a final '\n' so the data
    // starts on a 64-byte boundary.
    constexpr std::size_t kPreamble = 10;
    constexpr std::size_t kAlign = 64;
    const std::size_t pad = kAlign - (kPreamble + header.size() + 1) % kAlign;
    header.append(pad % kAlign, ' ');
    header.push_back('\n');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " for writing");
    const auto hlen = static_cast<std::uint16_t>(header.size());
    f.write("\x93NUMPY\x01\x00", 8);
    f.put(static_cast<char>(hlen & 0xFF));
    f.put(static_cast<char>(hlen >> 8));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!f) throw std::runtime_error("failed while writing " + path);
}

template <typename Real>
void write_sweep(const std::string& dir, const SweepConfig& cfg, const SweepResult<Real>& r,
                 const std::string& extra_json) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    const fs::path d(dir);

    std::vector<double> b(cfg.n_params);
    for (std::size_t i = 0; i < cfg.n_params; ++i) b[i] = cfg.b_value(i);
    write_npy((d / "B.npy").string(), b.data(), {cfg.n_params});
    write_npy((d / "lyapunov.npy").string(), r.lyapunov.data(), {cfg.n_params});
    if (cfg.store_orbit) {
        const std::vector<std::size_t> shape{static_cast<std::size_t>(cfg.n_record), cfg.n_params};
        write_npy((d / "x.npy").string(), r.x.data(), shape);
        write_npy((d / "y.npy").string(), r.y.data(), shape);
    }

    std::ofstream meta(d / "meta.json");
    meta << std::setprecision(17) << "{\n"
         << "  \"A\": " << cfg.cavity.A << ",\n"
         << "  \"p\": " << cfg.cavity.p << ",\n"
         << "  \"phi\": " << cfg.cavity.phi << ",\n"
         << "  \"b_min\": " << cfg.b_min << ",\n"
         << "  \"b_max\": " << cfg.b_max << ",\n"
         << "  \"n_params\": " << cfg.n_params << ",\n"
         << "  \"n_transient\": " << cfg.n_transient << ",\n"
         << "  \"n_record\": " << cfg.n_record << ",\n"
         << "  \"store_orbit\": " << (cfg.store_orbit ? "true" : "false") << ",\n"
         << "  \"x0\": " << cfg.x0 << ",\n"
         << "  \"y0\": " << cfg.y0 << ",\n"
         << "  \"noise_sigma\": " << cfg.noise_sigma << ",\n"
         << "  \"seed\": " << cfg.seed << ",\n"
         << "  \"precision\": \"" << (sizeof(Real) == 4 ? "float" : "double") << "\"";
    if (!extra_json.empty()) meta << ",\n  " << extra_json;
    meta << "\n}\n";
    if (!meta) throw std::runtime_error("failed while writing meta.json in " + dir);
}

template void write_npy<float>(const std::string&, const float*, const std::vector<std::size_t>&);
template void write_npy<double>(const std::string&, const double*, const std::vector<std::size_t>&);
template void write_sweep<float>(const std::string&, const SweepConfig&, const SweepResult<float>&, const std::string&);
template void write_sweep<double>(const std::string&, const SweepConfig&, const SweepResult<double>&,
                                  const std::string&);

}  // namespace ikeda::io
