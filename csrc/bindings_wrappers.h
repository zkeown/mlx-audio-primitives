#pragma once

#include <optional>
#include <mlx/mlx.h>
#include "primitives/windows.h"
#include "primitives/frame_signal.h"
#include "primitives/overlap_add.h"
#include "primitives/pad_signal.h"
#include "primitives/mel_filterbank.h"

namespace mlx_audio {

// Wrapper functions that take optional stream for proper nanobind binding

inline mlx::core::array generate_window_wrapper(
    const std::string& window_type,
    int length,
    bool periodic,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return generate_window(window_type, length, periodic, stream.value());
    }
    return generate_window(window_type, length, periodic);
}

inline mlx::core::array frame_signal_wrapper(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return frame_signal(signal, frame_length, hop_length, stream.value());
    }
    return frame_signal(signal, frame_length, hop_length);
}

inline mlx::core::array overlap_add_wrapper(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return overlap_add(frames, window, hop_length, output_length, stream.value());
    }
    return overlap_add(frames, window, hop_length, output_length);
}

inline mlx::core::array pad_signal_wrapper(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return pad_signal(signal, pad_length, mode, stream.value());
    }
    return pad_signal(signal, pad_length, mode);
}

inline mlx::core::array hz_to_mel_wrapper(
    const mlx::core::array& frequencies,
    bool htk,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return hz_to_mel(frequencies, htk, stream.value());
    }
    return hz_to_mel(frequencies, htk);
}

inline mlx::core::array mel_to_hz_wrapper(
    const mlx::core::array& mels,
    bool htk,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return mel_to_hz(mels, htk, stream.value());
    }
    return mel_to_hz(mels, htk);
}

inline mlx::core::array mel_filterbank_wrapper(
    int sr,
    int n_fft,
    int n_mels,
    float fmin,
    std::optional<float> fmax,
    bool htk,
    const std::string& norm,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk, norm, stream.value());
    }
    return mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk, norm);
}

}  // namespace mlx_audio
