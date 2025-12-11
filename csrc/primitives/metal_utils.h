#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/stream.h>

// Metal backend support - conditionally included
#ifdef MLX_BUILD_METAL
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#endif

namespace mlx_audio {

// Library name for our Metal kernels
constexpr const char* METAL_LIB_NAME = "mlx_audio_ext";

/**
 * Check if Metal GPU dispatch is available and should be used.
 *
 * Returns true if:
 * - MLX was built with Metal support
 * - The device is GPU
 * - Metal backend is available
 */
inline bool should_use_metal(mlx::core::StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    auto device = mlx::core::to_device(s);
    return device.type == mlx::core::Device::gpu &&
           mlx::core::is_available(mlx::core::Device::gpu);
#else
    return false;
#endif
}

/**
 * Get the Metal device reference.
 * Only call this if should_use_metal() returns true.
 */
#ifdef MLX_BUILD_METAL
inline mlx::core::metal::Device& get_metal_device(mlx::core::StreamOrDevice s) {
    return mlx::core::metal::device(mlx::core::to_device(s));
}
#endif

/**
 * Helper to compute optimal threadgroup size for 1D dispatch.
 */
inline int get_threadgroup_size_1d(int total_threads, int max_threads = 256) {
    return std::min(total_threads, max_threads);
}

/**
 * Helper to compute optimal threadgroup dimensions for 2D dispatch.
 *
 * Optimized for Apple Silicon GPUs which benefit from wider threadgroups
 * for better occupancy and memory coalescing.
 */
inline std::pair<int, int> get_threadgroup_size_2d(
    int dim0, int dim1, int max_total = 256) {
    // Wider threadgroups (64 vs 32) improve occupancy on Apple Silicon
    int tg0 = std::min(dim0, 64);
    int tg1 = std::min(dim1, max_total / tg0);
    return {tg0, tg1};
}

/**
 * Helper to compute optimal threadgroup dimensions for 3D dispatch.
 */
inline std::tuple<int, int, int> get_threadgroup_size_3d(
    int dim0, int dim1, int dim2, int max_total = 256) {
    // Prioritize first two dimensions
    int tg0 = std::min(dim0, 32);
    int tg1 = std::min(dim1, 8);
    int tg2 = std::min(dim2, max_total / (tg0 * tg1));
    return {tg0, tg1, std::max(1, tg2)};
}

}  // namespace mlx_audio
