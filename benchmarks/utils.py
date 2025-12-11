"""Shared utilities for benchmarking."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    mlx_time_ms: float
    reference_time_ms: float
    speedup: float
    max_abs_error: float
    mean_abs_error: float
    correlation: float  # Pearson correlation coefficient


def time_function(fn: Callable, warmup: int = 3, runs: int = 10) -> float:
    """
    Time a function with warmup runs, return median time in ms.

    Parameters
    ----------
    fn : Callable
        Function to time.
    warmup : int, default=3
        Number of warmup iterations before timing.
    runs : int, default=10
        Number of timed iterations.

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


def compute_accuracy(mlx_result: np.ndarray, ref_result: np.ndarray) -> dict:
    """
    Compute accuracy metrics between MLX and reference output.

    Parameters
    ----------
    mlx_result : np.ndarray
        MLX output array.
    ref_result : np.ndarray
        Reference implementation output array.

    Returns
    -------
    dict
        Dictionary with max_abs_error, mean_abs_error, and correlation.
    """
    diff = np.abs(mlx_result - ref_result)
    return {
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "correlation": float(
            np.corrcoef(mlx_result.ravel(), ref_result.ravel())[0, 1]
        ),
    }


def generate_test_signal(length: int, sr: int = 22050, seed: int = 42) -> np.ndarray:
    """
    Generate reproducible test signal with realistic audio characteristics.

    Parameters
    ----------
    length : int
        Signal length in samples.
    sr : int, default=22050
        Sample rate (used for chirp timing).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Test signal as float32 array.
    """
    rng = np.random.default_rng(seed)
    # Mix of chirp + noise for realistic spectral content
    t = np.linspace(0, length / sr, length, dtype=np.float32)
    chirp = np.sin(2 * np.pi * (100 + 2000 * t / 2) * t).astype(np.float32)
    noise = rng.standard_normal(length).astype(np.float32) * 0.1
    return chirp + noise
