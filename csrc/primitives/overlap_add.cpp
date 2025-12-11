#include "primitives/overlap_add.h"
#include "primitives/metal_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

#ifdef MLX_BUILD_METAL
mlx::core::array overlap_add_metal(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int batch_size = frames.shape(0);
    int n_frames_count = frames.shape(1);
    int n_fft = frames.shape(2);

    // Ensure contiguous float32 inputs
    auto frames_f = astype(frames, float32, s);
    auto window_f = astype(window, float32, s);

    // Make sure inputs are contiguous for Metal
    frames_f = flatten(frames_f, s);
    frames_f = reshape(frames_f, {batch_size, n_frames_count, n_fft}, s);
    window_f = flatten(window_f, s);
    window_f = reshape(window_f, {n_fft}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);

    auto lib = d.get_library(METAL_LIB_NAME);
    auto overlap_kernel = d.get_kernel("overlap_add_float", lib);
    auto normalize_kernel = d.get_kernel("normalize_float", lib);

    // Allocate output arrays
    auto output = zeros({batch_size, output_length}, float32, s);
    auto window_sum = zeros({output_length}, float32, s);

    // Evaluate inputs to ensure they're ready
    eval({frames_f, window_f, output, window_sum});

    // Get command encoder and set up overlap_add kernel
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(overlap_kernel);

    // Set buffers - must match kernel signature order
    enc.set_input_array(frames_f, 0);      // frames
    enc.set_input_array(window_f, 1);      // window
    enc.set_output_array(output, 2);       // output
    enc.set_output_array(window_sum, 3);   // window_sum
    enc.set_bytes(batch_size, 4);          // batch_size
    enc.set_bytes(n_frames_count, 5);      // n_frames
    enc.set_bytes(n_fft, 6);               // n_fft
    enc.set_bytes(hop_length, 7);          // hop_length
    enc.set_bytes(output_length, 8);       // output_length

    // Dispatch threads: 2D grid (output_length, batch_size)
    auto [tg0, tg1] = get_threadgroup_size_2d(output_length, batch_size);
    MTL::Size grid_dims = MTL::Size(output_length, batch_size, 1);
    MTL::Size group_dims = MTL::Size(tg0, tg1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    // Now dispatch normalization kernel
    enc.set_compute_pipeline_state(normalize_kernel);
    enc.set_output_array(output, 0);       // output (in-place)
    enc.set_input_array(window_sum, 1);    // window_sum
    enc.set_bytes(batch_size, 2);          // batch_size
    enc.set_bytes(output_length, 3);       // output_length

    enc.dispatch_threads(grid_dims, group_dims);

    return output;
}
#endif  // MLX_BUILD_METAL

mlx::core::array overlap_add_cpu(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int batch_size = frames.shape(0);
    int n_frames_count = frames.shape(1);
    int n_fft = frames.shape(2);

    // Ensure inputs are float32
    auto frames_f = astype(frames, float32, s);
    auto window_f = astype(window, float32, s);

    // GPU-friendly gather-based overlap-add using pure MLX ops
    // Each output position gathers contributions from overlapping frames

    // Apply window to frames: (batch, n_frames, n_fft)
    auto windowed_frames = frames_f * window_f;

    // Window squared for normalization: (n_fft,)
    auto window_sq = window_f * window_f;

    // Create position indices for all (frame, sample) pairs
    auto frame_idx = arange(n_frames_count, int32, s);  // (n_frames,)
    auto sample_idx = arange(n_fft, int32, s);  // (n_fft,)

    // Output positions: out_pos[f,s] = f * hop_length + s
    auto frame_starts = frame_idx * hop_length;  // (n_frames,)
    auto out_positions = reshape(frame_starts, {n_frames_count, 1}, s) +
                         reshape(sample_idx, {1, n_fft}, s);  // (n_frames, n_fft)

    // Create gather indices for each output position
    // Max frames that can contribute to any position
    int max_contributing_frames = (n_fft + hop_length - 1) / hop_length + 1;

    // For each output position, store indices of contributing (frame, sample) pairs
    std::vector<int32_t> gather_frame_idx(output_length * max_contributing_frames, 0);
    std::vector<int32_t> gather_sample_idx(output_length * max_contributing_frames, 0);
    std::vector<float> gather_mask(output_length * max_contributing_frames, 0.0f);

    for (int i = 0; i < output_length; ++i) {
        int first_frame = std::max(0, (i - n_fft + 1 + hop_length - 1) / hop_length);
        int last_frame = std::min(n_frames_count - 1, i / hop_length);

        int contrib_idx = 0;
        for (int f = first_frame; f <= last_frame && contrib_idx < max_contributing_frames; ++f) {
            int sample_in_frame = i - f * hop_length;
            if (sample_in_frame >= 0 && sample_in_frame < n_fft) {
                int flat_idx = i * max_contributing_frames + contrib_idx;
                gather_frame_idx[flat_idx] = f;
                gather_sample_idx[flat_idx] = sample_in_frame;
                gather_mask[flat_idx] = 1.0f;
                contrib_idx++;
            }
        }
    }

    // Convert to arrays
    auto frame_gather_idx = array(gather_frame_idx.data(), {output_length, max_contributing_frames}, int32);
    auto sample_gather_idx = array(gather_sample_idx.data(), {output_length, max_contributing_frames}, int32);
    auto contribution_mask = array(gather_mask.data(), {output_length, max_contributing_frames}, float32);

    // Gather window values and compute window_sum
    auto gathered_window = take(window_f, sample_gather_idx, s);  // (output_length, max_contrib)
    auto gathered_window_sq = gathered_window * gathered_window * contribution_mask;
    auto window_sum = sum(gathered_window_sq, 1, false, s);  // (output_length,)
    window_sum = maximum(window_sum, array(1e-8f), s);

    // For signal: gather from windowed_frames using 2D indexing
    // Compute flat indices into (n_frames, n_fft) slice
    auto flat_gather_idx = frame_gather_idx * n_fft + sample_gather_idx;  // (output_length, max_contrib)

    // For each batch, gather and sum
    std::vector<array> batch_results;
    batch_results.reserve(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        // Get this batch's windowed frames: (n_frames, n_fft)
        auto batch_frames = slice(windowed_frames, {b, 0, 0}, {b + 1, n_frames_count, n_fft}, s);
        batch_frames = reshape(batch_frames, {n_frames_count * n_fft}, s);  // flatten

        // Gather values
        auto gathered_vals = take(batch_frames, flat_gather_idx, s);  // (output_length, max_contrib)
        gathered_vals = gathered_vals * contribution_mask;

        // Sum contributions and normalize
        auto batch_output = sum(gathered_vals, 1, false, s);  // (output_length,)
        batch_output = batch_output / window_sum;
        batch_results.emplace_back(reshape(batch_output, {1, output_length}, s));
    }

    // Stack batch results
    auto output = concatenate(batch_results, 0, s);  // (batch, output_length)

    return output;
}

mlx::core::array overlap_add(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (frames.ndim() != 3) {
        throw std::invalid_argument(
            "frames must be 3-dimensional (batch, n_frames, n_fft)");
    }
    if (window.ndim() != 1) {
        throw std::invalid_argument(
            "window must be 1-dimensional (n_fft,)");
    }
    if (frames.shape(2) != window.shape(0)) {
        throw std::invalid_argument(
            "frames n_fft dimension must match window length");
    }
    if (hop_length <= 0) {
        throw std::invalid_argument("hop_length must be positive");
    }
    if (output_length <= 0) {
        throw std::invalid_argument("output_length must be positive");
    }

#ifdef MLX_BUILD_METAL
    // Try Metal GPU path
    if (should_use_metal(s)) {
        return overlap_add_metal(frames, window, hop_length, output_length, s);
    }
#endif

    // CPU fallback
    return overlap_add_cpu(frames, window, hop_length, output_length, s);
}

}  // namespace mlx_audio
