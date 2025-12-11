"""
Benchmarks comparing C++ extension vs pure Python/MLX implementations.

This module benchmarks the C++ extension primitives against their Python
counterparts to measure the performance benefit of the native implementations.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio_primitives._extension import HAS_CPP_EXT
from .utils import BenchmarkResult, compute_accuracy, generate_test_signal, time_function


def _force_python_autocorrelation(signal, max_lag, normalize, center):
    """Pure Python/MLX autocorrelation implementation."""
    y = signal
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    n = y.shape[-1]
    if max_lag is None or max_lag <= 0:
        max_lag = n
    max_lag = min(max_lag, n)

    y = y.astype(mx.float32)

    if center:
        mean_val = mx.mean(y, axis=-1, keepdims=True)
        y = y - mean_val

    # FFT-based autocorrelation
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2

    Y = mx.fft.rfft(y, n=n_fft, axis=-1)
    power = Y * mx.conj(Y)
    r = mx.fft.irfft(power, n=n_fft, axis=-1)
    r = r[..., :max_lag]

    if normalize:
        r0 = mx.maximum(r[..., :1], mx.array(1e-10))
        r = r / r0

    if input_is_1d:
        r = r[0]

    return r


def _force_python_resample_fft(signal, num_samples):
    """Pure Python/MLX resample implementation."""
    y = signal
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    orig_length = y.shape[-1]
    if num_samples == orig_length:
        return y[0] if input_is_1d else y

    y = y.astype(mx.float32)

    # Full FFT for proper handling
    Y = mx.fft.fft(y, axis=-1)
    scale_factor = num_samples / orig_length

    batch_size = y.shape[0]

    if num_samples > orig_length:
        n_pos = (orig_length + 1) // 2
        Y_pos = Y[:, :n_pos]
        Y_neg = Y[:, n_pos:]
        n_pad = num_samples - orig_length
        zeros_pad = mx.zeros((batch_size, n_pad), dtype=mx.complex64)
        Y_resampled = mx.concatenate([Y_pos, zeros_pad, Y_neg], axis=-1)
    else:
        n_pos_new = (num_samples + 1) // 2
        n_neg_new = num_samples // 2
        Y_pos = Y[:, :n_pos_new]
        Y_neg = Y[:, -n_neg_new:] if n_neg_new > 0 else mx.zeros((batch_size, 0), dtype=mx.complex64)
        Y_resampled = mx.concatenate([Y_pos, Y_neg], axis=-1)

    result = mx.fft.ifft(Y_resampled, axis=-1)
    result = mx.real(result) * scale_factor

    if input_is_1d:
        result = result[0]

    return result


def _force_python_dct(x, n, axis, norm):
    """Pure Python/MLX DCT implementation using matrix multiplication."""
    from scipy.fftpack import dct as scipy_dct

    # Use scipy for reference
    x_np = np.array(x)
    if n <= 0:
        n = x_np.shape[axis]
    norm_arg = "ortho" if norm == "ortho" else None
    result = scipy_dct(x_np, type=2, n=n, axis=axis, norm=norm_arg)
    return mx.array(result.astype(np.float32))


def benchmark_autocorrelation(signal_length: int = 22050) -> list[BenchmarkResult]:
    """Benchmark C++ vs Python autocorrelation."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext

    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    max_lag = 1000

    # C++ implementation
    def cpp_autocorr():
        return _ext.autocorrelation(signal_mx, max_lag, True, True)

    # Python implementation
    def py_autocorr():
        return _force_python_autocorrelation(signal_mx, max_lag, True, True)

    cpp_time = time_function(cpp_autocorr)
    py_time = time_function(py_autocorr)

    cpp_result = np.array(cpp_autocorr())
    py_result = np.array(py_autocorr())
    accuracy = compute_accuracy(cpp_result, py_result)

    results.append(BenchmarkResult(
        name="autocorrelation (C++ vs Python)",
        mlx_time_ms=cpp_time,
        reference_time_ms=py_time,
        speedup=py_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    return results


def benchmark_resample(signal_length: int = 22050) -> list[BenchmarkResult]:
    """Benchmark C++ vs Python resampling."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext

    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    target_samples = 16000

    # C++ implementation
    def cpp_resample():
        return _ext.resample_fft(signal_mx, target_samples)

    # Python implementation
    def py_resample():
        return _force_python_resample_fft(signal_mx, target_samples)

    cpp_time = time_function(cpp_resample)
    py_time = time_function(py_resample)

    cpp_result = np.array(cpp_resample())
    py_result = np.array(py_resample())
    accuracy = compute_accuracy(cpp_result, py_result)

    results.append(BenchmarkResult(
        name="resample_fft (C++ vs Python)",
        mlx_time_ms=cpp_time,
        reference_time_ms=py_time,
        speedup=py_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    return results


def benchmark_dct(n_coeffs: int = 128) -> list[BenchmarkResult]:
    """Benchmark C++ vs Python/scipy DCT."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext

    # Test signal
    np.random.seed(42)
    x_np = np.random.randn(10, n_coeffs).astype(np.float32)
    x_mx = mx.array(x_np)

    # C++ implementation
    def cpp_dct():
        return _ext.dct(x_mx, -1, -1, "ortho")

    # Python/scipy implementation
    def py_dct():
        return _force_python_dct(x_mx, -1, -1, "ortho")

    cpp_time = time_function(cpp_dct)
    py_time = time_function(py_dct)

    cpp_result = np.array(cpp_dct())
    py_result = np.array(py_dct())
    accuracy = compute_accuracy(cpp_result, py_result)

    results.append(BenchmarkResult(
        name=f"DCT (C++ vs scipy, n={n_coeffs})",
        mlx_time_ms=cpp_time,
        reference_time_ms=py_time,
        speedup=py_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    return results


def benchmark_spectral_features(signal_length: int = 22050) -> list[BenchmarkResult]:
    """Benchmark C++ vs Python spectral features."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext
    from mlx_audio_primitives.stft import stft, magnitude

    # Generate spectrogram
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    S = magnitude(stft(signal_mx, n_fft=2048, hop_length=512))
    frequencies = mx.linspace(0, 11025, S.shape[0])

    # Spectral centroid
    def cpp_centroid():
        return _ext.spectral_centroid(S, frequencies)

    def py_centroid():
        S_3d = S[None, :, :]
        freqs_bc = frequencies[:, None]
        weighted = mx.sum(freqs_bc * S_3d, axis=1, keepdims=True)
        total = mx.maximum(mx.sum(S_3d, axis=1, keepdims=True), mx.array(1e-10))
        return weighted / total

    cpp_time = time_function(cpp_centroid)
    py_time = time_function(py_centroid)
    cpp_result = np.array(cpp_centroid())
    py_result = np.array(py_centroid())
    accuracy = compute_accuracy(cpp_result, py_result)

    results.append(BenchmarkResult(
        name="spectral_centroid (C++ vs Python)",
        mlx_time_ms=cpp_time,
        reference_time_ms=py_time,
        speedup=py_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    # Spectral flatness
    def cpp_flatness():
        return _ext.spectral_flatness(S, 1e-10)

    def py_flatness():
        S_3d = mx.maximum(S[None, :, :], mx.array(1e-10))
        log_mean = mx.mean(mx.log(S_3d), axis=1, keepdims=True)
        geo_mean = mx.exp(log_mean)
        arith_mean = mx.mean(S_3d, axis=1, keepdims=True)
        return geo_mean / arith_mean

    cpp_time = time_function(cpp_flatness)
    py_time = time_function(py_flatness)
    cpp_result = np.array(cpp_flatness())
    py_result = np.array(py_flatness())
    accuracy = compute_accuracy(cpp_result, py_result)

    results.append(BenchmarkResult(
        name="spectral_flatness (C++ vs Python)",
        mlx_time_ms=cpp_time,
        reference_time_ms=py_time,
        speedup=py_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    return results


def benchmark_mel_filterbank() -> list[BenchmarkResult]:
    """Benchmark C++ vs Python mel filterbank generation."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext
    import librosa

    sr = 22050
    n_fft = 2048
    n_mels = 128

    # C++ implementation
    def cpp_mel():
        return _ext.mel_filterbank(sr, n_fft, n_mels, 0.0, None, False, "slaney")

    # librosa reference
    def librosa_mel():
        return mx.array(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm="slaney"))

    cpp_time = time_function(cpp_mel)
    librosa_time = time_function(librosa_mel)

    cpp_result = np.array(cpp_mel())
    librosa_result = np.array(librosa_mel())
    accuracy = compute_accuracy(cpp_result, librosa_result)

    results.append(BenchmarkResult(
        name=f"mel_filterbank (C++ vs librosa, n_mels={n_mels})",
        mlx_time_ms=cpp_time,
        reference_time_ms=librosa_time,
        speedup=librosa_time / cpp_time if cpp_time > 0 else 0,
        **accuracy,
    ))

    return results


def benchmark_window_functions() -> list[BenchmarkResult]:
    """Benchmark C++ vs scipy window functions."""
    results = []

    if not HAS_CPP_EXT:
        return results

    import mlx_audio_primitives._ext as _ext
    from scipy.signal.windows import hann, hamming, blackman

    window_length = 2048

    for window_name, scipy_fn in [("hann", hann), ("hamming", hamming), ("blackman", blackman)]:
        def cpp_window(name=window_name):
            return _ext.generate_window(name, window_length, True)

        def scipy_window(fn=scipy_fn):
            return mx.array(fn(window_length, sym=False).astype(np.float32))

        cpp_time = time_function(cpp_window)
        scipy_time = time_function(scipy_window)

        cpp_result = np.array(cpp_window())
        scipy_result = np.array(scipy_window())
        accuracy = compute_accuracy(cpp_result, scipy_result)

        results.append(BenchmarkResult(
            name=f"{window_name} window (C++ vs scipy)",
            mlx_time_ms=cpp_time,
            reference_time_ms=scipy_time,
            speedup=scipy_time / cpp_time if cpp_time > 0 else 0,
            **accuracy,
        ))

    return results


def benchmark_all_cpp(signal_length: int = 22050) -> list[BenchmarkResult]:
    """Run all C++ extension benchmarks."""
    results = []

    results.extend(benchmark_autocorrelation(signal_length))
    results.extend(benchmark_resample(signal_length))
    results.extend(benchmark_dct())
    results.extend(benchmark_spectral_features(signal_length))
    results.extend(benchmark_mel_filterbank())
    results.extend(benchmark_window_functions())

    return results


if __name__ == "__main__":
    from .run import format_results

    print("C++ Extension Benchmarks")
    print("=" * 80)

    if not HAS_CPP_EXT:
        print("C++ extension not available. Skipping benchmarks.")
    else:
        results = benchmark_all_cpp()
        print(format_results(results, verbose=True))

        print("\n[Summary]")
        speedups = [r.speedup for r in results if r.speedup > 0]
        if speedups:
            print(f"Average speedup: {sum(speedups) / len(speedups):.2f}x")
            print(f"Min speedup: {min(speedups):.2f}x")
            print(f"Max speedup: {max(speedups):.2f}x")
