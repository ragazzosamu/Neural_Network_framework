#include "cuda/memory.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// memory managment of the gpu. additional layer

// Throws if a CUDA call returned an error. `what` names the failed call,
// so the message says where the error came from.
static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

// Asks the driver how many GPUs there are. A failure here (e.g. no driver)
// is not an error for the program: it just means CUDA is not available.
static bool query_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // Clear the error, otherwise a later cudaGetLastError() would return it.
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

bool cuda_is_available() {
    // static: initialized only on the first call, then the cached value is returned.
    static const bool available = query_cuda_available();
    return available;
}

float *cuda_alloc(size_t n) {
    float *p = nullptr;
    // &p: cudaMalloc writes the new device address into p.
    check(cudaMalloc(&p, sizeof(float) * n), "cudaMalloc");
    return p;
}

// No check(): this function is noexcept (it runs inside the shared_ptr deleter),
// so the error is ignored instead of thrown.
void cuda_free(float *p) noexcept { cudaFree(p); }

void cuda_memset_zero(float *p, size_t n) { check(cudaMemset(p, 0, sizeof(float) * n), "cudaMemset"); }

void cuda_copy_cpu_to_gpu(float *dst, const float *src, size_t n) {
    check(cudaMemcpy(dst, src, sizeof(float) * n, cudaMemcpyHostToDevice), "cudaMemcpy cpu->gpu");
}

void cuda_copy_gpu_to_cpu(float *dst, const float *src, size_t n) {
    check(cudaMemcpy(dst, src, sizeof(float) * n, cudaMemcpyDeviceToHost), "cudaMemcpy gpu->cpu");
}

void cuda_copy_gpu_to_gpu(float *dst, const float *src, size_t n) {
    check(cudaMemcpy(dst, src, sizeof(float) * n, cudaMemcpyDeviceToDevice), "cudaMemcpy gpu->gpu");
}
