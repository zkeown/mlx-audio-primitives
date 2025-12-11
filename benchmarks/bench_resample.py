"""
Audio resampling benchmarks.

Compares mlx_audio_primitives resampling against scipy.signal:
- resample: FFT-based resampling (best quality, arbitrary ratios)
- resample_poly: Polyphase FIR filter (efficient for integer ratios)

Expected speedup: ~1.0-1.2x vs scipy. Both use FFT internally, so performance
is similar. The win comes when resampling is part of a larger MLX pipeline.

Common conversions tested:
- 44100 -> 16000 Hz: CD quality to speech model input
- 22050 -> 16000 Hz: Common audio to speech model input
- 16000 -> 22050 Hz: Upsampling for synthesis

Run: mlx-audio-bench --suite resample
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


def benchmark_resample(
    signal_length: int = 22050,
    orig_sr: int = 22050,
    target_sr: int = 16000,
) -> list[BenchmarkResult]:
    """
    Benchmark resampling.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    orig_sr : int, default=22050
        Original sample rate.
    target_sr : int, default=16000
        Target sample rate.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results.
    """
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    # MLX (uses scipy.signal.resample internally, same as librosa res_type='fft')
    mlx_time = time_function(
        lambda: mlx_audio.resample(signal_mx, orig_sr=orig_sr, target_sr=target_sr)
    )
    mlx_result = np.array(
        mlx_audio.resample(signal_mx, orig_sr=orig_sr, target_sr=target_sr)
    )

    # librosa with res_type='fft' to match our implementation
    librosa_time = time_function(
        lambda: librosa.resample(
            signal_np, orig_sr=orig_sr, target_sr=target_sr, res_type="fft"
        )
    )
    librosa_result = librosa.resample(
        signal_np, orig_sr=orig_sr, target_sr=target_sr, res_type="fft"
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name=f"resample ({orig_sr}->{target_sr})",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_resample_ratios(
    signal_length: int = 22050,
    ratios: list[tuple[int, int]] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark resampling with various rate ratios.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    ratios : list[tuple[int, int]], optional
        List of (orig_sr, target_sr) pairs to benchmark.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each ratio.
    """
    import librosa

    if ratios is None:
        ratios = [
            (44100, 22050),  # 2x downsample
            (22050, 44100),  # 2x upsample
            (48000, 16000),  # 3x downsample
            (16000, 48000),  # 3x upsample
        ]

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    for orig_sr, target_sr in ratios:
        mlx_time = time_function(
            lambda o=orig_sr, t=target_sr: mlx_audio.resample(
                signal_mx, orig_sr=o, target_sr=t
            )
        )
        mlx_result = np.array(
            mlx_audio.resample(signal_mx, orig_sr=orig_sr, target_sr=target_sr)
        )

        librosa_time = time_function(
            lambda o=orig_sr, t=target_sr: librosa.resample(
                signal_np, orig_sr=o, target_sr=t, res_type="fft"
            )
        )
        librosa_result = librosa.resample(
            signal_np, orig_sr=orig_sr, target_sr=target_sr, res_type="fft"
        )

        accuracy = compute_accuracy(mlx_result, librosa_result)
        results.append(
            BenchmarkResult(
                name=f"resample ({orig_sr}->{target_sr})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    return results


def benchmark_resample_poly(
    signal_length: int = 22050,
    up: int = 1,
    down: int = 2,
) -> list[BenchmarkResult]:
    """
    Benchmark polyphase resampling.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    up : int, default=1
        Upsampling factor.
    down : int, default=2
        Downsampling factor.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results.
    """
    from scipy.signal import resample_poly as scipy_resample_poly

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    mlx_time = time_function(
        lambda: mlx_audio.resample_poly(signal_mx, up=up, down=down)
    )
    mlx_result = np.array(mlx_audio.resample_poly(signal_mx, up=up, down=down))

    scipy_time = time_function(
        lambda: scipy_resample_poly(signal_np, up, down)
    )
    scipy_result = scipy_resample_poly(signal_np, up, down)

    accuracy = compute_accuracy(mlx_result, scipy_result)
    results.append(
        BenchmarkResult(
            name=f"resample_poly (up={up}, down={down})",
            mlx_time_ms=mlx_time,
            reference_time_ms=scipy_time,
            speedup=scipy_time / mlx_time,
            **accuracy,
        )
    )

    return results
