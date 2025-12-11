"""Window function benchmarks."""
from __future__ import annotations

import numpy as np

import mlx_audio_primitives as mlx_audio
from mlx_audio_primitives import windows as windows_module

from .utils import BenchmarkResult, compute_accuracy, time_function


def benchmark_windows(
    lengths: list[int] | None = None,
    window_types: list[str] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark window function generation.

    Parameters
    ----------
    lengths : list[int], optional
        Window lengths to benchmark. Default: [512, 1024, 2048, 4096].
    window_types : list[str], optional
        Window types to benchmark. Default: ["hann", "hamming", "blackman"].

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each window configuration.
    """
    if lengths is None:
        lengths = [512, 1024, 2048, 4096]
    if window_types is None:
        window_types = ["hann", "hamming", "blackman"]

    results = []

    for length in lengths:
        for window_type in window_types:
            # Clear cache for fair comparison
            windows_module._get_window_cached.cache_clear()

            # MLX timing
            mlx_time = time_function(
                lambda wt=window_type, ln=length: mlx_audio.get_window(wt, ln)
            )
            mlx_win = np.array(mlx_audio.get_window(window_type, length))

            # scipy (reference)
            import scipy.signal

            scipy_time = time_function(
                lambda wt=window_type, ln=length: scipy.signal.get_window(
                    wt, ln, fftbins=True
                )
            )
            scipy_win = scipy.signal.get_window(
                window_type, length, fftbins=True
            ).astype(np.float32)

            accuracy = compute_accuracy(mlx_win, scipy_win)
            results.append(
                BenchmarkResult(
                    name=f"{window_type} ({length})",
                    mlx_time_ms=mlx_time,
                    reference_time_ms=scipy_time,
                    speedup=scipy_time / mlx_time,
                    **accuracy,
                )
            )

    return results


def benchmark_window_caching(
    length: int = 2048,
    window_type: str = "hann",
    iterations: int = 100,
) -> list[BenchmarkResult]:
    """
    Benchmark window function caching effectiveness.

    Compares cold start (cache miss) vs warm (cache hit) performance.

    Parameters
    ----------
    length : int, default=2048
        Window length.
    window_type : str, default="hann"
        Window type.
    iterations : int, default=100
        Number of iterations to measure cached performance.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for cold and warm cache scenarios.
    """
    results = []

    # Cold start (cache cleared)
    windows_module._get_window_cached.cache_clear()
    cold_time = time_function(
        lambda: mlx_audio.get_window(window_type, length),
        warmup=0,  # No warmup for cold start
        runs=1,
    )

    # Warm cache (repeated calls)
    warm_time = time_function(
        lambda: mlx_audio.get_window(window_type, length),
        warmup=3,
        runs=iterations,
    )

    results.append(
        BenchmarkResult(
            name=f"{window_type} ({length}) cold",
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
            name=f"{window_type} ({length}) warm",
            mlx_time_ms=warm_time,
            reference_time_ms=cold_time,
            speedup=cold_time / warm_time if warm_time > 0 else 0.0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            correlation=1.0,
        )
    )

    return results
