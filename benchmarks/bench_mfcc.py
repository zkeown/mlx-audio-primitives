"""
MFCC and cepstral features benchmarks.

Compares mlx_audio_primitives MFCC implementation against librosa:
- mfcc: Full pipeline (mel spectrogram -> log -> DCT -> lifter)
- delta: Temporal derivatives via Savitzky-Golay filter

Expected speedup: ~1.5-2x vs librosa. The mel spectrogram portion benefits
from MLX caching; DCT is similar speed across implementations.

Common configurations tested:
- n_mfcc=13: Speech recognition standard
- n_mfcc=40: Higher resolution for music/general audio

Run: mlx-audio-bench --suite mfcc
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


def benchmark_mfcc(
    signal_length: int = 22050,
    n_mfcc: int = 13,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """
    Benchmark MFCC computation.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_mfcc : int, default=13
        Number of MFCCs to compute.
    n_mels : int, default=128
        Number of mel bands.
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
        lambda: mlx_audio.mfcc(
            signal_mx, sr=22050, n_mfcc=n_mfcc, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
    )
    mlx_result = np.array(
        mlx_audio.mfcc(
            signal_mx, sr=22050, n_mfcc=n_mfcc, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
    )

    # librosa
    librosa_time = time_function(
        lambda: librosa.feature.mfcc(
            y=signal_np, sr=22050, n_mfcc=n_mfcc, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
    )
    librosa_result = librosa.feature.mfcc(
        y=signal_np, sr=22050, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length
    )

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name=f"mfcc (n_mfcc={n_mfcc})",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_delta(
    signal_length: int = 22050,
    n_mfcc: int = 13,
    width: int = 9,
) -> list[BenchmarkResult]:
    """
    Benchmark delta feature computation.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_mfcc : int, default=13
        Number of MFCCs.
    width : int, default=9
        Delta filter width.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results.
    """
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)

    # Pre-compute MFCCs
    mfccs_np = librosa.feature.mfcc(y=signal_np, sr=22050, n_mfcc=n_mfcc)
    mfccs_mx = mx.array(mfccs_np.astype(np.float32))

    # MLX delta
    mlx_time = time_function(
        lambda: mlx_audio.delta(mfccs_mx, width=width, order=1)
    )
    mlx_result = np.array(mlx_audio.delta(mfccs_mx, width=width, order=1))

    # librosa delta
    librosa_time = time_function(
        lambda: librosa.feature.delta(mfccs_np, width=width, order=1)
    )
    librosa_result = librosa.feature.delta(mfccs_np, width=width, order=1)

    accuracy = compute_accuracy(mlx_result, librosa_result)
    results.append(
        BenchmarkResult(
            name=f"delta (width={width})",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            **accuracy,
        )
    )

    return results


def benchmark_mfcc_scaling(
    n_mfcc_values: list[int] | None = None,
    signal_length: int = 22050,
) -> list[BenchmarkResult]:
    """
    Benchmark MFCC performance across different coefficient counts.

    Parameters
    ----------
    n_mfcc_values : list[int], optional
        MFCC counts to benchmark. Default: [13, 20, 40].
    signal_length : int, default=22050
        Length of test signal in samples.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each n_mfcc value.
    """
    import librosa

    if n_mfcc_values is None:
        n_mfcc_values = [13, 20, 40]

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    for n_mfcc in n_mfcc_values:
        mlx_time = time_function(
            lambda n=n_mfcc: mlx_audio.mfcc(signal_mx, sr=22050, n_mfcc=n)
        )
        mlx_result = np.array(mlx_audio.mfcc(signal_mx, sr=22050, n_mfcc=n_mfcc))

        librosa_time = time_function(
            lambda n=n_mfcc: librosa.feature.mfcc(y=signal_np, sr=22050, n_mfcc=n)
        )
        librosa_result = librosa.feature.mfcc(y=signal_np, sr=22050, n_mfcc=n_mfcc)

        accuracy = compute_accuracy(mlx_result, librosa_result)
        results.append(
            BenchmarkResult(
                name=f"mfcc scaling (n_mfcc={n_mfcc})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    return results
