// SPDX-License-Identifier: MIT
//
// Small RAII and error-checking helpers for the CUDA runtime API.
//
// Every runtime call goes through CUDA_CHECK, which throws on failure with the
// failing expression and its source location. Kernel launches are followed by
// CUDA_CHECK_LAUNCH(). That catches configuration errors immediately, and in
// debug builds it also synchronises, so asynchronous faults (such as an
// out-of-bounds access) are reported at the launch that caused them instead
// of at some later, unrelated API call.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ikeda::cuda {

class CudaError : public std::runtime_error {
 public:
    using std::runtime_error::runtime_error;
};

inline void check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                        " failed: " + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
    }
}

}  // namespace ikeda::cuda

#define CUDA_CHECK(expr) ::ikeda::cuda::check((expr), #expr, __FILE__, __LINE__)

#ifdef NDEBUG
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#else
#define CUDA_CHECK_LAUNCH()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#endif

namespace ikeda::cuda {

// Owning, move-only handle to a device allocation of n elements of T.
template <typename T>
class DeviceBuffer {
 public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : size_(n) {
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }

    void copy_to_host(T* dst) const {
        if (size_ > 0) CUDA_CHECK(cudaMemcpy(dst, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

 private:
    void reset() noexcept {
        // A destructor can't throw, and a failed cudaFree only happens once the
        // context is already broken, so the error code is deliberately ignored.
        if (ptr_) static_cast<void>(cudaFree(ptr_));
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// Owning handle to a cudaEvent_t, used for GPU-side timing.
class Event {
 public:
    Event() { CUDA_CHECK(cudaEventCreate(&ev_)); }
    ~Event() { static_cast<void>(cudaEventDestroy(ev_)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t s = nullptr) { CUDA_CHECK(cudaEventRecord(ev_, s)); }
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(ev_)); }
    cudaEvent_t get() const noexcept { return ev_; }

    // Milliseconds between two recorded events (resolution ~0.5 us).
    static float elapsed_ms(const Event& start, const Event& stop) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev_, stop.ev_));
        return ms;
    }

 private:
    cudaEvent_t ev_ = nullptr;
};

}  // namespace ikeda::cuda
