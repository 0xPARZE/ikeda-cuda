include(FetchContent)
find_package(GTest 1.12 QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

add_executable(ikeda_tests
    test_map.cpp
    test_analysis.cpp
    test_sweep_cpu.cpp
    test_options_io.cpp)
target_link_libraries(ikeda_tests PRIVATE ikeda_core ikeda_warnings GTest::gtest_main)
gtest_discover_tests(ikeda_tests DISCOVERY_TIMEOUT 60)

if(IKEDA_HAVE_CUDA)
    # GPU-vs-CPU tests. They are skipped (not failed) on machines without a
    # CUDA device, so a CUDA build can still run ctest on a GPU-less host.
    add_executable(ikeda_gpu_tests test_sweep_gpu.cpp)
    target_link_libraries(ikeda_gpu_tests PRIVATE ikeda_gpu ikeda_warnings GTest::gtest_main)
    gtest_discover_tests(ikeda_gpu_tests DISCOVERY_TIMEOUT 60)
endif()
