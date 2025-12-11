"""
Spectral features benchmarks.

Compares mlx_audio_primitives spectral feature implementations against librosa:
- spectral_centroid: Weighted mean of frequencies (brightness)
- spectral_bandwidth: Spread around centroid
- spectral_rolloff: Frequency below which X% energy lies
- spectral_flatness: Geometric/arithmetic mean ratio (tonality measure)
- zero_crossing_rate: Sign changes per frame

Expected speedup: ~1.2-1.5x vs librosa (operations are similar complexity,
but MLX keeps data on GPU avoiding transfers in pipelines).

Run: mlx-audio-bench --suite features
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio

from .utils import (
    BenchmarkResult,
    compute_accuracy,
    generate_test_signal,
    time_function,
)


def benchmark_spectral_centroid(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """
    Benchmark spectral centroid computation.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length between frames.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results.
    """
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    # MLX
    mlx_time = time_function(
        lambda: mlx_audio.spectral_centroid(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.spectral_centroid(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )

    # librosa
    librosa_time = time_function(
        lambda: librosa.feature.spectral_centroid(
            y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.spectral_centroid(
        y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name="spectral_centroid",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_spectral_bandwidth(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """Benchmark spectral bandwidth computation."""
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    mlx_time = time_function(
        lambda: mlx_audio.spectral_bandwidth(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.spectral_bandwidth(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )

    librosa_time = time_function(
        lambda: librosa.feature.spectral_bandwidth(
            y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.spectral_bandwidth(
        y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name="spectral_bandwidth",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_spectral_rolloff(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """Benchmark spectral rolloff computation."""
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    mlx_time = time_function(
        lambda: mlx_audio.spectral_rolloff(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.spectral_rolloff(
            signal_mx, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )

    librosa_time = time_function(
        lambda: librosa.feature.spectral_rolloff(
            y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.spectral_rolloff(
        y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name="spectral_rolloff",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_spectral_flatness(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """Benchmark spectral flatness computation."""
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    mlx_time = time_function(
        lambda: mlx_audio.spectral_flatness(
            signal_mx, n_fft=n_fft, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.spectral_flatness(
            signal_mx, n_fft=n_fft, hop_length=hop_length
        )
    )

    librosa_time = time_function(
        lambda: librosa.feature.spectral_flatness(
            y=signal_np, n_fft=n_fft, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.spectral_flatness(
        y=signal_np, n_fft=n_fft, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name="spectral_flatness",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_zero_crossing_rate(
    signal_length: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """Benchmark zero crossing rate computation."""
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    mlx_time = time_function(
        lambda: mlx_audio.zero_crossing_rate(
            signal_mx, frame_length=frame_length, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.zero_crossing_rate(
            signal_mx, frame_length=frame_length, hop_length=hop_length
        )
    )

    librosa_time = time_function(
        lambda: librosa.feature.zero_crossing_rate(
            y=signal_np, frame_length=frame_length, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.zero_crossing_rate(
        y=signal_np, frame_length=frame_length, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name="zero_crossing_rate",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_all_features(
    signal_length: int = 22050,
) -> list[BenchmarkResult]:
    """Run all spectral feature benchmarks."""
    results = []
    results.extend(benchmark_spectral_centroid(signal_length))
    results.extend(benchmark_spectral_bandwidth(signal_length))
    results.extend(benchmark_spectral_rolloff(signal_length))
    results.extend(benchmark_spectral_flatness(signal_length))
    results.extend(benchmark_zero_crossing_rate(signal_length))
    return results
