"""
Multi-dimensional parameter scaling benchmarks.

Tests performance across:
- Signal lengths: [8000, 22050, 44100, 88200, 176400]
- Batch sizes: [1, 2, 4, 8, 16, 32]
- FFT sizes: [256, 512, 1024, 2048, 4096, 8192]
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio

from .utils import BenchmarkResult, generate_test_signal, time_function

if TYPE_CHECKING:
    from collections.abc import Iterator

# Default parameter ranges
SIGNAL_LENGTHS = [8000, 22050, 44100, 88200, 176400]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
FFT_SIZES = [256, 512, 1024, 2048, 4096, 8192]


def benchmark_stft_scaling(
    signal_lengths: list[int] | None = None,
    n_fft_values: list[int] | None = None,
    verbose: bool = False,
) -> Iterator[BenchmarkResult]:
    """
    Benchmark STFT across signal lengths and FFT sizes.

    Parameters
    ----------
    signal_lengths : list[int], optional
        Signal lengths to test. Default: SIGNAL_LENGTHS
    n_fft_values : list[int], optional
        FFT sizes to test. Default: FFT_SIZES
    verbose : bool, default=False
        Print progress during benchmarking.

    Yields
    ------
    BenchmarkResult
        Result for each parameter combination.
    """
    if signal_lengths is None:
        signal_lengths = SIGNAL_LENGTHS
    if n_fft_values is None:
        n_fft_values = FFT_SIZES

    for signal_length, n_fft in itertools.product(signal_lengths, n_fft_values):
        if n_fft > signal_length:
            continue  # Skip invalid combinations

        hop_length = n_fft // 4
        signal_np = generate_test_signal(signal_length)
        signal_mx = mx.array(signal_np)

        if verbose:
            print(f"  STFT scaling: len={signal_length}, n_fft={n_fft}")

        def run_stft() -> mx.array:
            return mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)

        mlx_time = time_function(run_stft)

        yield BenchmarkResult(
            name=f"stft_scale(len={signal_length}, n_fft={n_fft})",
            mlx_time_ms=mlx_time,
            reference_time_ms=0.0,  # Skip reference for scaling tests
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )


def benchmark_batch_scaling(
    batch_sizes: list[int] | None = None,
    signal_length: int = 22050,
    n_fft: int = 2048,
    verbose: bool = False,
) -> Iterator[BenchmarkResult]:
    """
    Benchmark operations with different batch sizes.

    Parameters
    ----------
    batch_sizes : list[int], optional
        Batch sizes to test. Default: BATCH_SIZES
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress during benchmarking.

    Yields
    ------
    BenchmarkResult
        Result for each batch size and operation.
    """
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES

    hop_length = n_fft // 4

    for batch_size in batch_sizes:
        # Generate batch signal
        signal_np = generate_test_signal(signal_length)
        batch_np = np.tile(signal_np, (batch_size, 1))
        batch_mx = mx.array(batch_np)

        if verbose:
            print(f"  Batch scaling: batch={batch_size}")

        # STFT batch
        def run_stft() -> mx.array:
            return mlx_audio.stft(batch_mx, n_fft=n_fft, hop_length=hop_length)

        stft_time = time_function(run_stft)

        yield BenchmarkResult(
            name=f"stft_batch(batch={batch_size})",
            mlx_time_ms=stft_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )

        # Melspectrogram batch
        def run_mel() -> mx.array:
            return mlx_audio.melspectrogram(batch_mx, n_fft=n_fft, hop_length=hop_length)

        mel_time = time_function(run_mel)

        yield BenchmarkResult(
            name=f"melspec_batch(batch={batch_size})",
            mlx_time_ms=mel_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )


def benchmark_mel_scaling(
    n_mels_values: list[int] | None = None,
    signal_length: int = 22050,
    n_fft: int = 2048,
    verbose: bool = False,
) -> Iterator[BenchmarkResult]:
    """
    Benchmark mel spectrogram across different n_mels values.

    Parameters
    ----------
    n_mels_values : list[int], optional
        Number of mel bands to test. Default: [40, 80, 128, 256]
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress during benchmarking.

    Yields
    ------
    BenchmarkResult
        Result for each n_mels value.
    """
    if n_mels_values is None:
        n_mels_values = [40, 80, 128, 256]

    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    for n_mels in n_mels_values:
        if verbose:
            print(f"  Mel scaling: n_mels={n_mels}")

        def run_mel() -> mx.array:
            return mlx_audio.melspectrogram(
                signal_mx, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )

        mlx_time = time_function(run_mel)

        yield BenchmarkResult(
            name=f"melspec_scale(n_mels={n_mels})",
            mlx_time_ms=mlx_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )


def benchmark_mfcc_scaling(
    n_mfcc_values: list[int] | None = None,
    signal_length: int = 22050,
    n_fft: int = 2048,
    verbose: bool = False,
) -> Iterator[BenchmarkResult]:
    """
    Benchmark MFCC across different n_mfcc values.

    Parameters
    ----------
    n_mfcc_values : list[int], optional
        Number of MFCC coefficients to test. Default: [13, 20, 40]
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress during benchmarking.

    Yields
    ------
    BenchmarkResult
        Result for each n_mfcc value.
    """
    if n_mfcc_values is None:
        n_mfcc_values = [13, 20, 40]

    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    for n_mfcc in n_mfcc_values:
        if verbose:
            print(f"  MFCC scaling: n_mfcc={n_mfcc}")

        def run_mfcc() -> mx.array:
            return mlx_audio.mfcc(
                signal_mx, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
            )

        mlx_time = time_function(run_mfcc)

        yield BenchmarkResult(
            name=f"mfcc_scale(n_mfcc={n_mfcc})",
            mlx_time_ms=mlx_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )


def benchmark_griffinlim_scaling(
    n_iter_values: list[int] | None = None,
    signal_length: int = 22050,
    n_fft: int = 2048,
    verbose: bool = False,
) -> Iterator[BenchmarkResult]:
    """
    Benchmark Griffin-Lim across different iteration counts.

    Parameters
    ----------
    n_iter_values : list[int], optional
        Iteration counts to test. Default: [8, 16, 32, 64]
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress during benchmarking.

    Yields
    ------
    BenchmarkResult
        Result for each iteration count.
    """
    if n_iter_values is None:
        n_iter_values = [8, 16, 32, 64]

    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    # Pre-compute magnitude spectrogram
    S = mlx_audio.magnitude(mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length))
    mx.eval(S)

    for n_iter in n_iter_values:
        if verbose:
            print(f"  Griffin-Lim scaling: n_iter={n_iter}")

        def run_gl() -> mx.array:
            return mlx_audio.griffinlim(
                S, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft
            )

        mlx_time = time_function(run_gl, warmup=1, runs=3)

        yield BenchmarkResult(
            name=f"griffinlim_scale(n_iter={n_iter})",
            mlx_time_ms=mlx_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )


def run_all_scaling_benchmarks(verbose: bool = False) -> list[BenchmarkResult]:
    """
    Run all scaling benchmarks.

    Parameters
    ----------
    verbose : bool, default=False
        Print progress during benchmarking.

    Returns
    -------
    list[BenchmarkResult]
        All scaling benchmark results.
    """
    results = []

    if verbose:
        print("Running STFT scaling benchmarks...")
    results.extend(benchmark_stft_scaling(verbose=verbose))

    if verbose:
        print("Running batch scaling benchmarks...")
    results.extend(benchmark_batch_scaling(verbose=verbose))

    if verbose:
        print("Running mel scaling benchmarks...")
    results.extend(benchmark_mel_scaling(verbose=verbose))

    if verbose:
        print("Running MFCC scaling benchmarks...")
    results.extend(benchmark_mfcc_scaling(verbose=verbose))

    if verbose:
        print("Running Griffin-Lim scaling benchmarks...")
    results.extend(benchmark_griffinlim_scaling(verbose=verbose))

    return results


def benchmark_throughput(
    signal_lengths: list[int] | None = None,
    n_fft: int = 2048,
    verbose: bool = False,
) -> list[dict]:
    """
    Calculate throughput (samples/second) for different signal lengths.

    Parameters
    ----------
    signal_lengths : list[int], optional
        Signal lengths to test. Default: SIGNAL_LENGTHS
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress during benchmarking.

    Returns
    -------
    list[dict]
        Throughput data for each signal length.
    """
    if signal_lengths is None:
        signal_lengths = SIGNAL_LENGTHS

    results = []
    hop_length = n_fft // 4

    for signal_length in signal_lengths:
        signal_np = generate_test_signal(signal_length)
        signal_mx = mx.array(signal_np)

        if verbose:
            print(f"  Throughput: len={signal_length}")

        def run_stft() -> mx.array:
            return mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)

        time_ms = time_function(run_stft)
        samples_per_sec = signal_length / (time_ms / 1000)

        results.append(
            {
                "signal_length": signal_length,
                "time_ms": time_ms,
                "throughput_samples_per_sec": samples_per_sec,
                "throughput_mb_per_sec": (signal_length * 4) / (time_ms / 1000) / (1024 * 1024),
            }
        )

    return results
