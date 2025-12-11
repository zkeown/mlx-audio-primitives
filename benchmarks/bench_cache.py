"""
Cache impact analysis for MLX audio primitives.

Measures:
- Cold start time (forced cache clear)
- Warm time (pre-populated cache)
- Cache effectiveness ratio
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio

from .utils import BenchmarkResult, generate_test_signal

if TYPE_CHECKING:
    from collections.abc import Callable


def clear_all_caches() -> None:
    """Clear all known caches in mlx_audio_primitives."""
    # Clear mel filterbank caches
    try:
        from mlx_audio_primitives import mel as mel_module

        mel_module._compute_mel_filterbank_np.cache_clear()
        if hasattr(mel_module, "_mlx_filterbank_cache"):
            mel_module._mlx_filterbank_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Clear window caches
    try:
        from mlx_audio_primitives import windows as windows_module

        windows_module._get_window_cached.cache_clear()
        if hasattr(windows_module, "_mlx_window_cache"):
            windows_module._mlx_window_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Clear STFT caches
    try:
        from mlx_audio_primitives import stft as stft_module

        if hasattr(stft_module, "_padded_window_cache"):
            stft_module._padded_window_cache.clear()
        if hasattr(stft_module, "_get_compiled_stft_fn"):
            stft_module._get_compiled_stft_fn.cache_clear()
    except (ImportError, AttributeError):
        pass

    # Clear MFCC caches
    try:
        from mlx_audio_primitives import mfcc as mfcc_module

        if hasattr(mfcc_module, "_compute_dct_matrix_np"):
            mfcc_module._compute_dct_matrix_np.cache_clear()
        if hasattr(mfcc_module, "_mlx_dct_cache"):
            mfcc_module._mlx_dct_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Clear MLX metal cache if available
    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass

    # Force garbage collection
    gc.collect()


def warmup_caches(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    n_mfcc: int = 20,
) -> None:
    """
    Pre-populate commonly used caches.

    Parameters
    ----------
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    n_mels : int, default=128
        Number of mel bands.
    n_mfcc : int, default=20
        Number of MFCC coefficients.
    """
    # Warm mel filterbank cache
    try:
        _ = mlx_audio.mel_filterbank(sr, n_fft, n_mels=n_mels)
    except Exception:
        pass

    # Warm window cache
    try:
        _ = mlx_audio.get_window("hann", n_fft)
    except Exception:
        pass


def time_function_once(fn: Callable[[], Any]) -> float:
    """
    Time a function once without warmup (for cold measurements).

    Parameters
    ----------
    fn : Callable
        Function to time.

    Returns
    -------
    float
        Execution time in milliseconds.
    """
    start = time.perf_counter()
    result = fn()
    if isinstance(result, mx.array):
        mx.eval(result)
    end = time.perf_counter()
    return (end - start) * 1000


def time_function_warm(fn: Callable[[], Any], warmup: int = 3, runs: int = 10) -> float:
    """
    Time a function with warmup (for warm measurements).

    Parameters
    ----------
    fn : Callable
        Function to time.
    warmup : int, default=3
        Number of warmup runs.
    runs : int, default=10
        Number of timed runs.

    Returns
    -------
    float
        Median execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
        times.append((time.perf_counter() - start) * 1000)

    return float(np.median(times))


def benchmark_cache_impact(
    fn: Callable[[], Any],
    name: str,
    clear_fn: Callable[[], None] = clear_all_caches,
) -> list[BenchmarkResult]:
    """
    Measure cold vs warm cache performance.

    Parameters
    ----------
    fn : Callable
        Function to benchmark.
    name : str
        Benchmark name.
    clear_fn : Callable
        Function to clear relevant caches.

    Returns
    -------
    list[BenchmarkResult]
        Results for cold and warm scenarios.
    """
    results = []

    # Cold start measurement
    clear_fn()
    cold_time = time_function_once(fn)

    # Warm measurement (cache already populated from cold run)
    warm_time = time_function_warm(fn, warmup=3, runs=10)

    # Calculate cache effectiveness
    cache_speedup = cold_time / warm_time if warm_time > 0 else 0.0

    results.append(
        BenchmarkResult(
            name=f"{name} (cold)",
            mlx_time_ms=cold_time,
            reference_time_ms=0.0,
            speedup=0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )
    )

    results.append(
        BenchmarkResult(
            name=f"{name} (warm)",
            mlx_time_ms=warm_time,
            reference_time_ms=cold_time,  # Use cold time as "reference"
            speedup=cache_speedup,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )
    )

    return results


def benchmark_window_cache(
    window_types: list[str] | None = None,
    n_fft: int = 2048,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark window function cache effectiveness.

    Parameters
    ----------
    window_types : list[str], optional
        Window types to test. Default: ["hann", "hamming", "blackman"]
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    list[BenchmarkResult]
        Cache effectiveness results.
    """
    if window_types is None:
        window_types = ["hann", "hamming", "blackman"]

    results = []

    for window_type in window_types:
        if verbose:
            print(f"  Window cache: {window_type}")

        def get_window() -> mx.array:
            return mlx_audio.get_window(window_type, n_fft)

        results.extend(
            benchmark_cache_impact(get_window, f"window_{window_type}")
        )

    return results


def benchmark_mel_filterbank_cache(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels_values: list[int] | None = None,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark mel filterbank cache effectiveness.

    Parameters
    ----------
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    n_mels_values : list[int], optional
        Number of mel bands to test. Default: [40, 80, 128]
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    list[BenchmarkResult]
        Cache effectiveness results.
    """
    if n_mels_values is None:
        n_mels_values = [40, 80, 128]

    results = []

    for n_mels in n_mels_values:
        if verbose:
            print(f"  Mel filterbank cache: n_mels={n_mels}")

        def get_filterbank() -> mx.array:
            return mlx_audio.mel_filterbank(sr, n_fft, n_mels=n_mels)

        results.extend(
            benchmark_cache_impact(get_filterbank, f"mel_filterbank_n_mels={n_mels}")
        )

    return results


def benchmark_stft_cache(
    signal_length: int = 22050,
    n_fft: int = 2048,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark STFT pipeline cache effectiveness (window + compiled function).

    Parameters
    ----------
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    list[BenchmarkResult]
        Cache effectiveness results.
    """
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    if verbose:
        print("  STFT cache effectiveness")

    def run_stft() -> mx.array:
        return mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)

    return benchmark_cache_impact(run_stft, "stft_pipeline")


def benchmark_melspectrogram_cache(
    signal_length: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark melspectrogram pipeline cache effectiveness.

    Parameters
    ----------
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    n_mels : int, default=128
        Number of mel bands.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    list[BenchmarkResult]
        Cache effectiveness results.
    """
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    if verbose:
        print("  Melspectrogram cache effectiveness")

    def run_mel() -> mx.array:
        return mlx_audio.melspectrogram(
            signal_mx, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    return benchmark_cache_impact(run_mel, "melspectrogram_pipeline")


def benchmark_mfcc_cache(
    signal_length: int = 22050,
    n_fft: int = 2048,
    n_mfcc: int = 20,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """
    Benchmark MFCC pipeline cache effectiveness.

    Parameters
    ----------
    signal_length : int, default=22050
        Signal length in samples.
    n_fft : int, default=2048
        FFT size.
    n_mfcc : int, default=20
        Number of MFCC coefficients.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    list[BenchmarkResult]
        Cache effectiveness results.
    """
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)
    hop_length = n_fft // 4

    if verbose:
        print("  MFCC cache effectiveness")

    def run_mfcc() -> mx.array:
        return mlx_audio.mfcc(
            signal_mx, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
        )

    return benchmark_cache_impact(run_mfcc, "mfcc_pipeline")


def run_all_cache_benchmarks(verbose: bool = False) -> list[BenchmarkResult]:
    """
    Run all cache impact analysis benchmarks.

    Parameters
    ----------
    verbose : bool, default=False
        Print progress during benchmarking.

    Returns
    -------
    list[BenchmarkResult]
        All cache benchmark results.
    """
    results = []

    if verbose:
        print("Running window cache benchmarks...")
    results.extend(benchmark_window_cache(verbose=verbose))

    if verbose:
        print("Running mel filterbank cache benchmarks...")
    results.extend(benchmark_mel_filterbank_cache(verbose=verbose))

    if verbose:
        print("Running STFT cache benchmarks...")
    results.extend(benchmark_stft_cache(verbose=verbose))

    if verbose:
        print("Running melspectrogram cache benchmarks...")
    results.extend(benchmark_melspectrogram_cache(verbose=verbose))

    if verbose:
        print("Running MFCC cache benchmarks...")
    results.extend(benchmark_mfcc_cache(verbose=verbose))

    return results


def get_cache_statistics() -> dict[str, dict[str, int]]:
    """
    Get current cache statistics.

    Returns
    -------
    dict
        Cache statistics with hits/misses for each cache.
    """
    stats = {}

    # Window cache
    try:
        from mlx_audio_primitives import windows as windows_module

        cache_info = windows_module._get_window_cached.cache_info()
        stats["window_lru"] = {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize or 0,
        }
    except (ImportError, AttributeError):
        pass

    # Mel filterbank cache
    try:
        from mlx_audio_primitives import mel as mel_module

        cache_info = mel_module._compute_mel_filterbank_np.cache_info()
        stats["mel_filterbank_lru"] = {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize or 0,
        }
    except (ImportError, AttributeError):
        pass

    return stats
