"""
Benchmark CLI for mlx-audio-primitives.

This module provides the `mlx-audio-bench` command-line tool for comparing
performance against librosa and torchaudio reference implementations.

Usage:
    mlx-audio-bench                    # Run all benchmarks
    mlx-audio-bench --verbose          # Include accuracy metrics
    mlx-audio-bench --suite stft       # Run only STFT benchmarks
    mlx-audio-bench --suite mel        # Run only mel benchmarks
    mlx-audio-bench --n-fft 4096       # Custom FFT size
    mlx-audio-bench --scaling          # Run scaling benchmarks
    mlx-audio-bench --cache-analysis   # Run cache impact analysis
    mlx-audio-bench --save-baseline    # Save results as baseline
    mlx-audio-bench --compare-baseline # Compare to stored baseline

The benchmarks measure:
    - Wall-clock execution time (ms)
    - Speedup ratio vs reference implementations
    - Numerical accuracy (max/mean error, correlation)
    - Cold/warm cache performance (with --cache-analysis)
    - Memory usage (with --memory)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .bench_features import benchmark_all_features
from .bench_griffinlim import benchmark_griffinlim
from .bench_mel import benchmark_mel_filterbank, benchmark_melspectrogram
from .bench_mfcc import benchmark_delta, benchmark_mfcc
from .bench_resample import benchmark_resample, benchmark_resample_ratios
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
            if r.peak_memory_mb is not None:
                lines.append(f"    Peak memory: {r.peak_memory_mb:.2f} MB")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_results_json(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as JSON."""
    from .baseline import create_benchmark_run

    run = create_benchmark_run(results)
    return json.dumps(run.to_dict(), indent=2)


def format_results_markdown(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as markdown table."""
    lines = []
    lines.append("| Benchmark | MLX (ms) | Ref (ms) | Speedup |")
    lines.append("|-----------|----------|----------|---------|")

    for r in results:
        speedup_str = f"{r.speedup:.2f}x" if r.speedup > 0 else "N/A"
        lines.append(
            f"| {r.name} | {r.mlx_time_ms:.3f} | "
            f"{r.reference_time_ms:.3f} | {speedup_str} |"
        )

    return "\n".join(lines)


def format_results_csv(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as CSV."""
    lines = []
    lines.append("name,mlx_time_ms,reference_time_ms,speedup,max_abs_error,mean_abs_error,correlation")

    for r in results:
        lines.append(
            f"{r.name},{r.mlx_time_ms:.6f},{r.reference_time_ms:.6f},"
            f"{r.speedup:.4f},{r.max_abs_error:.2e},{r.mean_abs_error:.2e},{r.correlation:.6f}"
        )

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

    print("\n[Spectral Features]")
    features_results = benchmark_all_features()
    all_results.extend(features_results)
    print(format_results(features_results, verbose))

    print("\n[MFCC]")
    mfcc_results = benchmark_mfcc()
    mfcc_results.extend(benchmark_delta())
    all_results.extend(mfcc_results)
    print(format_results(mfcc_results, verbose))

    print("\n[Resampling]")
    resample_results = benchmark_resample()
    all_results.extend(resample_results)
    print(format_results(resample_results, verbose))

    print("\n[Griffin-Lim]")
    griffinlim_results = benchmark_griffinlim()
    all_results.extend(griffinlim_results)
    print(format_results(griffinlim_results, verbose))

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
  mlx-audio-bench --scaling          # Run scaling benchmarks
  mlx-audio-bench --cache-analysis   # Run cache impact analysis
  mlx-audio-bench --save-baseline    # Save results as baseline
  mlx-audio-bench --compare-baseline # Compare to stored baseline
        """,
    )

    # Existing arguments
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show accuracy metrics"
    )
    parser.add_argument(
        "--suite",
        choices=["all", "stft", "mel", "windows", "features", "mfcc", "resample", "griffinlim"],
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

    # New arguments for extended benchmarking
    parser.add_argument(
        "--output",
        choices=["table", "json", "markdown", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling benchmarks (signal length, batch size, FFT size)",
    )
    parser.add_argument(
        "--cache-analysis",
        action="store_true",
        help="Run cache impact analysis (cold vs warm)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Include memory profiling (requires MLX >= 0.5)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline for current platform",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results to stored baseline",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regression detected",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.10,
        help="Regression threshold (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--platform-info",
        action="store_true",
        help="Show platform information only",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List stored baselines",
    )

    opts = parser.parse_args(args)

    # Import platform info
    try:
        from .platform import format_platform_header, get_platform_info
    except ImportError:
        format_platform_header = lambda: "Platform info not available"
        get_platform_info = None

    # Platform info only
    if opts.platform_info:
        print(format_platform_header())
        return 0

    # List baselines
    if opts.list_baselines:
        try:
            from .baseline import list_baselines

            baselines = list_baselines()
            if baselines:
                print("Stored baselines:")
                for key, timestamp in baselines:
                    print(f"  {key}: {timestamp}")
            else:
                print("No baselines stored.")
        except ImportError:
            print("Baseline module not available.")
        return 0

    # Print header
    if opts.output == "table":
        print(format_platform_header())
        print(f"Signal length: {opts.signal_length} samples")
        print(f"FFT size: {opts.n_fft}")

    all_results = []

    try:
        # Run scaling benchmarks
        if opts.scaling:
            from .bench_scaling import run_all_scaling_benchmarks

            if opts.output == "table":
                print("\n[Scaling Benchmarks]")
            scaling_results = run_all_scaling_benchmarks(verbose=opts.verbose)
            all_results.extend(scaling_results)
            if opts.output == "table":
                print(format_results(scaling_results, opts.verbose))

        # Run cache analysis
        if opts.cache_analysis:
            from .bench_cache import run_all_cache_benchmarks

            if opts.output == "table":
                print("\n[Cache Analysis]")
            cache_results = run_all_cache_benchmarks(verbose=opts.verbose)
            all_results.extend(cache_results)
            if opts.output == "table":
                print(format_results(cache_results, opts.verbose))

        # Run standard benchmark suites
        if not opts.scaling and not opts.cache_analysis:
            if opts.suite == "all":
                all_results = run_all(opts.verbose)
            elif opts.suite == "stft":
                results = benchmark_stft(opts.signal_length, opts.n_fft)
                results.extend(benchmark_istft(opts.signal_length, opts.n_fft))
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "mel":
                results = benchmark_mel_filterbank()
                results.extend(benchmark_melspectrogram(opts.signal_length, opts.n_fft))
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "windows":
                results = benchmark_windows()
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "features":
                results = benchmark_all_features(opts.signal_length)
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "mfcc":
                results = benchmark_mfcc(opts.signal_length)
                results.extend(benchmark_delta(opts.signal_length))
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "resample":
                results = benchmark_resample(opts.signal_length)
                results.extend(benchmark_resample_ratios(opts.signal_length))
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))
            elif opts.suite == "griffinlim":
                results = benchmark_griffinlim(opts.signal_length, opts.n_fft)
                all_results.extend(results)
                if opts.output == "table":
                    print(format_results(results, opts.verbose))

        # Output formatting
        if opts.output == "json":
            print(format_results_json(all_results))
        elif opts.output == "markdown":
            print(format_results_markdown(all_results))
        elif opts.output == "csv":
            print(format_results_csv(all_results))

        # Save baseline
        if opts.save_baseline:
            try:
                from .baseline import create_benchmark_run, save_baseline

                run = create_benchmark_run(all_results)
                platform_key = get_platform_info().key if get_platform_info else "unknown"
                save_baseline(run, platform_key)
                print(f"\nBaseline saved for platform: {platform_key}")
            except ImportError as e:
                print(f"\nCould not save baseline: {e}")

        # Compare to baseline
        if opts.compare_baseline:
            try:
                from .baseline import compare_to_baseline

                report = compare_to_baseline(
                    all_results,
                    threshold=opts.regression_threshold,
                )
                if report is None:
                    print("\nNo baseline found for current platform.")
                else:
                    print(f"\n{report.format_summary()}")
                    if report.has_regressions and opts.fail_on_regression:
                        return 1
            except ImportError as e:
                print(f"\nCould not compare to baseline: {e}")

    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Install benchmark dependencies with: pip install -e .[bench]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
