#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "bindings_wrappers.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "MLX Audio Primitives C++ Extension";

    // Core STFT operations
    m.def(
        "overlap_add",
        &mlx_audio::overlap_add_wrapper,
        "frames"_a,
        "window"_a,
        "hop_length"_a,
        "output_length"_a,
        "stream"_a = nb::none(),
        R"(
        Perform overlap-add reconstruction with window normalization.

        Reconstructs a time-domain signal from overlapping windowed frames
        using the overlap-add method with squared window normalization.

        Parameters
        ----------
        frames : mlx.core.array
            Input frames of shape (batch, n_frames, n_fft).
        window : mlx.core.array
            Window function of shape (n_fft,).
        hop_length : int
            Number of samples between consecutive frames.
        output_length : int
            Desired length of output signal.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Reconstructed signal of shape (batch, output_length).
        )");

    m.def(
        "frame_signal",
        &mlx_audio::frame_signal_wrapper,
        "signal"_a,
        "frame_length"_a,
        "hop_length"_a,
        "stream"_a = nb::none(),
        R"(
        Frame a signal into overlapping windows.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (batch, samples) or (samples,).
        frame_length : int
            Length of each frame (typically n_fft).
        hop_length : int
            Number of samples between consecutive frames.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Framed signal of shape (batch, n_frames, frame_length) or
            (n_frames, frame_length) if input was 1D.
        )");

    m.def(
        "pad_signal",
        &mlx_audio::pad_signal_wrapper,
        "signal"_a,
        "pad_length"_a,
        "mode"_a = "constant",
        "stream"_a = nb::none(),
        R"(
        Pad signal on both sides.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (batch, samples).
        pad_length : int
            Number of samples to pad on each side.
        mode : str, optional
            Padding mode: 'constant', 'reflect', or 'edge'.
            Default: 'constant'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Padded signal of shape (batch, samples + 2 * pad_length).
        )");

    // Window functions
    m.def(
        "generate_window",
        &mlx_audio::generate_window_wrapper,
        "window_type"_a,
        "length"_a,
        "periodic"_a = true,
        "stream"_a = nb::none(),
        R"(
        Generate a window function.

        Parameters
        ----------
        window_type : str
            Window type: 'hann', 'hamming', 'blackman', 'bartlett', 'rectangular'.
        length : int
            Window length.
        periodic : bool, optional
            If True, create periodic (DFT-even) window for FFT. Default: True.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Window array of shape (length,) with dtype float32.
        )");

    // Mel-scale operations
    m.def(
        "hz_to_mel",
        &mlx_audio::hz_to_mel_wrapper,
        "frequencies"_a,
        "htk"_a = false,
        "stream"_a = nb::none(),
        R"(
        Convert frequency in Hz to mel scale.

        Parameters
        ----------
        frequencies : mlx.core.array
            Frequencies in Hz.
        htk : bool, optional
            If True, use HTK formula. If False, use Slaney (librosa default).
            Default: False.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Frequencies in mel scale.
        )");

    m.def(
        "mel_to_hz",
        &mlx_audio::mel_to_hz_wrapper,
        "mels"_a,
        "htk"_a = false,
        "stream"_a = nb::none(),
        R"(
        Convert mel scale to frequency in Hz.

        Parameters
        ----------
        mels : mlx.core.array
            Frequencies in mel scale.
        htk : bool, optional
            If True, use HTK formula. If False, use Slaney (librosa default).
            Default: False.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Frequencies in Hz.
        )");

    m.def(
        "mel_filterbank",
        &mlx_audio::mel_filterbank_wrapper,
        "sr"_a,
        "n_fft"_a,
        "n_mels"_a = 128,
        "fmin"_a = 0.0f,
        "fmax"_a = nb::none(),
        "htk"_a = false,
        "norm"_a = "slaney",
        "stream"_a = nb::none(),
        R"(
        Create a mel-scale filterbank matrix.

        Parameters
        ----------
        sr : int
            Sample rate of the audio.
        n_fft : int
            FFT size.
        n_mels : int, optional
            Number of mel bands. Default: 128.
        fmin : float, optional
            Minimum frequency in Hz. Default: 0.0.
        fmax : float, optional
            Maximum frequency in Hz. Default: sr / 2.
        htk : bool, optional
            If True, use HTK formula for mel scale. Default: False.
        norm : str, optional
            Normalization mode: 'slaney' or empty string for none.
            Default: 'slaney'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        )");
}
