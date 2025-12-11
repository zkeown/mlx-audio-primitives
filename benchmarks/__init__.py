"""Benchmarking suite for mlx-audio-primitives."""
from .bench_cpp_extension import benchmark_all_cpp
from .bench_mel import (
    benchmark_mel_filterbank,
    benchmark_mel_scaling,
    benchmark_melspectrogram,
)
from .bench_stft import benchmark_istft, benchmark_stft, benchmark_stft_scaling
from .bench_windows import benchmark_window_caching, benchmark_windows
from .run import format_results, main, run_all
from .utils import (
    BenchmarkResult,
    compute_accuracy,
    generate_test_signal,
    time_function,
)

__all__ = [
    # Data classes
    "BenchmarkResult",
    # Utilities
    "time_function",
    "compute_accuracy",
    "generate_test_signal",
    "format_results",
    # STFT benchmarks
    "benchmark_stft",
    "benchmark_istft",
    "benchmark_stft_scaling",
    # Mel benchmarks
    "benchmark_mel_filterbank",
    "benchmark_melspectrogram",
    "benchmark_mel_scaling",
    # Window benchmarks
    "benchmark_windows",
    "benchmark_window_caching",
    # C++ extension benchmarks
    "benchmark_all_cpp",
    # CLI
    "run_all",
    "main",
]
