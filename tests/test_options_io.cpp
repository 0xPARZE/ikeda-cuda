// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ikeda/io.hpp"
#include "ikeda/options.hpp"

namespace {

namespace fs = std::filesystem;

fs::path temp_dir(const std::string& name) {
    const auto d = fs::temp_directory_path() / ("ikeda_test_" + name);
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

TEST(Options, DefaultsAndOverrides) {
    const char* argv[] = {"prog", "--n-params", "123", "--device", "cpu", "--precision", "float", "--noise", "0.5"};
    const auto o = ikeda::parse_options(9, argv);
    EXPECT_EQ(o.sweep.n_params, 123u);
    EXPECT_EQ(o.device, ikeda::Device::Cpu);
    EXPECT_EQ(o.precision, ikeda::Precision::Float);
    EXPECT_DOUBLE_EQ(o.sweep.noise_sigma, 0.5);
    EXPECT_EQ(o.sweep.n_transient, ikeda::SweepConfig{}.n_transient);
}

TEST(Options, CommandLineOverridesConfigFile) {
    const auto dir = temp_dir("options");
    const auto cfg = (dir / "run.cfg").string();
    std::ofstream(cfg) << "# comment line\n"
                          "n-params = 50   # trailing comment\n"
                          "record = 7\n"
                          "b-max = 0.9\n";
    const char* argv[] = {"prog", "--record", "9", "--config", cfg.c_str()};
    const auto o = ikeda::parse_options(5, argv);
    EXPECT_EQ(o.sweep.n_params, 50u);
    EXPECT_EQ(o.sweep.n_record, 9);  // flag beats file, even though the file is listed later
    EXPECT_DOUBLE_EQ(o.sweep.b_max, 0.9);
}

TEST(Options, RejectsUnknownAndMalformed) {
    const char* unknown[] = {"prog", "--bogus", "1"};
    EXPECT_THROW(ikeda::parse_options(3, unknown), std::invalid_argument);
    const char* malformed[] = {"prog", "--n-params", "12abc"};
    EXPECT_THROW(ikeda::parse_options(3, malformed), std::invalid_argument);
    const char* missing[] = {"prog", "--n-params"};
    EXPECT_THROW(ikeda::parse_options(2, missing), std::invalid_argument);
}

// Checks the NPY header byte-for-byte against the format spec and that the
// payload round-trips.
TEST(Io, NpyHeaderAndPayload) {
    const auto path = (temp_dir("npy") / "a.npy").string();
    const std::vector<double> data{1.5, -2.25, 3.0, 4.0, 5.0, 6.0};
    ikeda::io::write_npy(path, data.data(), {2, 3});

    std::ifstream f(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_GE(bytes.size(), 10u);
    EXPECT_EQ(bytes.substr(0, 8), std::string("\x93NUMPY\x01\x00", 8));
    const std::size_t hlen =
        std::size_t{static_cast<unsigned char>(bytes[8])} | (std::size_t{static_cast<unsigned char>(bytes[9])} << 8);
    EXPECT_EQ((10 + hlen) % 64, 0u);
    const std::string header = bytes.substr(10, hlen);
    EXPECT_NE(header.find("'descr': '<f8'"), std::string::npos);
    EXPECT_NE(header.find("'shape': (2, 3, )"), std::string::npos);
    EXPECT_EQ(header.back(), '\n');
    ASSERT_EQ(bytes.size(), 10 + hlen + data.size() * sizeof(double));
    std::vector<double> back(data.size());
    std::memcpy(back.data(), bytes.data() + 10 + hlen, data.size() * sizeof(double));
    EXPECT_EQ(back, data);
}

TEST(Io, WriteSweepCreatesAllFiles) {
    const auto dir = temp_dir("sweep");
    ikeda::SweepConfig c;
    c.n_params = 10;
    c.n_transient = 5;
    c.n_record = 4;
    const auto r = ikeda::sweep_cpu<float>(c);
    ikeda::io::write_sweep((dir / "out").string(), c, r);
    for (const char* f : {"B.npy", "lyapunov.npy", "x.npy", "y.npy", "meta.json"}) {
        EXPECT_TRUE(fs::exists(dir / "out" / f)) << f;
    }
}

}  // namespace
