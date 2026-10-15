#pragma once
#include <cstddef>

// GPU memory management: a thin layer over the CUDA runtime, so that the rest
// of the framework (Tensor) can allocate, free and copy GPU buffers without
// including any CUDA header. Implemented in src/cuda/memory.cu.
//
// Every size parameter `n` is a number of floats, not of bytes.
// Every function except cuda_free throws std::runtime_error if the CUDA call fails.

/**
 * @brief Returns true if at least one CUDA-capable GPU (with a working driver)
 *        is present. The check is done once, on the first call; later calls
 *        return the cached result.
 */
bool cuda_is_available();

/**
 * @brief Allocates a buffer of @p n floats on the GPU.
 *        The memory is NOT initialized (see cuda_memset_zero()).
 * @return A device pointer: it can be passed to kernels and to the other
 *         cuda_* functions, but must never be dereferenced on the CPU.
 */
float *cuda_alloc(size_t n);

/**
 * @brief Frees a buffer allocated with cuda_alloc(). Passing nullptr is allowed.
 *
 * noexcept: it is called by the shared_ptr deleter, i.e. inside a destructor,
 * which must never throw. Errors are ignored (e.g. at program exit, when the
 * CUDA runtime may already be unloaded).
 */
void cuda_free(float *p) noexcept;

/// @brief Sets all @p n floats of the GPU buffer @p p to 0.0f.
void cuda_memset_zero(float *p, size_t n);

/// @brief Copies @p n floats from CPU memory (@p src) to GPU memory (@p dst).
void cuda_copy_cpu_to_gpu(float *dst, const float *src, size_t n);

/// @brief Copies @p n floats from GPU memory (@p src) to CPU memory (@p dst).
void cuda_copy_gpu_to_cpu(float *dst, const float *src, size_t n);

/// @brief Copies @p n floats between two GPU buffers.
void cuda_copy_gpu_to_gpu(float *dst, const float *src, size_t n);
