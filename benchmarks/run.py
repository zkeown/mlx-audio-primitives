"""CLI for running benchmarks."""
from __future__ import annotations

import argparse
import sys

from .bench_mel import benchmark_mel_filterbank, benchmark_melspectrogram
from .bench_stft import benchmark_istft, benchmark_stft
from .bench_windows import benchmark_windows
from .utils import BenchmarkResult


def format_results(results: list[BenchmarkResult], verbose: bool = False) -> str:
    """
    Format benchmark results as a table.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results to format.
    verbose : bool, default=False
        If True, include accuracy metrics.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"{'Benchmark':<40} {'MLX (ms)':<10} {'Ref (ms)':<10} {'Speedup':<10}"
    )
    lines.append("-" * 80)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x" if r.speedup > 0 else "N/A"
        lines.append(
            f"{r.name:<40} {r.mlx_time_ms:<10.3f} "
            f"{r.reference_time_ms:<10.3f} {speedup_str:<10}"
        )
        if verbose:
            lines.append(
                f"    Max error: {r.max_abs_error:.2e}, "
                f"Mean error: {r.mean_abs_error:.2e}, "
                f"Corr: {r.correlation:.6f}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def run_all(verbose: bool = False) -> list[BenchmarkResult]:
    """
    Run all benchmarks.

    Parameters
    ----------
    verbose : bool, default=False
        If True, show accuracy metrics.

    Returns
    -------
    list[BenchmarkResult]
        All benchmark results.
    """
    all_results = []

    print("\n[STFT Benchmarks]")
    stft_results = benchmark_stft()
    all_results.extend(stft_results)
    print(format_results(stft_results, verbose))

    print("\n[ISTFT Round-Trip]")
    istft_results = benchmark_istft()
    all_results.extend(istft_results)
    print(format_results(istft_results, verbose))

    print("\n[Mel Filterbank]")
    mel_fb_results = benchmark_mel_filterbank()
    all_results.extend(mel_fb_results)
    print(format_results(mel_fb_results, verbose))

    print("\n[Mel Spectrogram]")
    mel_results = benchmark_melspectrogram()
    all_results.extend(mel_results)
    print(format_results(mel_results, verbose))

    print("\n[Window Functions]")
    win_results = benchmark_windows()
    all_results.extend(win_results)
    print(format_results(win_results, verbose))

    # Summary
    print("\n[Summary]")
    speedups = [r.speedup for r in all_results if r.speedup > 0]
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")

    return all_results


def main(args: list[str] | None = None) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    args : list[str], optional
        Command line arguments. Uses sys.argv if None.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="MLX Audio Primitives Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlx-audio-bench                    # Run all benchmarks
  mlx-audio-bench --verbose          # Show accuracy metrics
  mlx-audio-bench --suite stft       # Run only STFT benchmarks
  mlx-audio-bench --n-fft 4096       # Use custom FFT size
        """,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show accuracy metrics"
    )
    parser.add_argument(
        "--suite",
        choices=["all", "stft", "mel", "windows"],
        default="all",
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "--signal-length",
        type=int,
        default=22050,
        help="Test signal length in samples (default: 22050)",
    )
    parser.add_argument(
        "--n-fft", type=int, default=2048, help="FFT size (default: 2048)"
    )

    opts = parser.parse_args(args)

    print("MLX Audio Primitives Benchmarks")
    print(f"Signal length: {opts.signal_length} samples")
    print(f"FFT size: {opts.n_fft}")

    try:
        if opts.suite == "all":
            run_all(opts.verbose)
        elif opts.suite == "stft":
            results = benchmark_stft(opts.signal_length, opts.n_fft)
            results.extend(benchmark_istft(opts.signal_length, opts.n_fft))
            print(format_results(results, opts.verbose))
        elif opts.suite == "mel":
            results = benchmark_mel_filterbank()
            results.extend(benchmark_melspectrogram(opts.signal_length, opts.n_fft))
            print(format_results(results, opts.verbose))
        elif opts.suite == "windows":
            results = benchmark_windows()
            print(format_results(results, opts.verbose))
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Install benchmark dependencies with: pip install -e .[bench]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
