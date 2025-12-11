"""
Baseline storage and regression detection for benchmarks.

Provides:
- JSON baseline storage
- Comparison against stored baselines
- Regression detection with configurable thresholds
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .platform import get_platform_info
from .schemas import (
    BaselineFile,
    BenchmarkMetric,
    BenchmarkRun,
    ComparisonReport,
    PlatformSnapshot,
    RegressionResult,
)

if TYPE_CHECKING:
    from .utils import BenchmarkResult

BASELINE_PATH = Path(__file__).parent / "baselines.json"
DEFAULT_REGRESSION_THRESHOLD = 0.10  # 10% slowdown triggers warning


def get_git_sha() -> str:
    """
    Get current git commit SHA.

    Returns
    -------
    str
        Git commit SHA or "unknown" if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def get_version() -> str:
    """
    Get package version.

    Returns
    -------
    str
        Package version string.
    """
    try:
        import mlx_audio_primitives

        return getattr(mlx_audio_primitives, "__version__", "0.0.0")
    except ImportError:
        return "0.0.0"


def load_baselines() -> BaselineFile:
    """
    Load baseline file, creating empty if missing.

    Returns
    -------
    BaselineFile
        Loaded or empty baseline file.
    """
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            data = json.load(f)
            return BaselineFile.from_dict(data)
    return BaselineFile()


def save_baselines(baselines: BaselineFile) -> None:
    """
    Save baselines to file.

    Parameters
    ----------
    baselines : BaselineFile
        Baselines to save.
    """
    with open(BASELINE_PATH, "w") as f:
        json.dump(baselines.to_dict(), f, indent=2)


def save_baseline(run: BenchmarkRun, platform_key: str | None = None) -> None:
    """
    Save benchmark run as baseline for platform.

    Parameters
    ----------
    run : BenchmarkRun
        Benchmark run to save.
    platform_key : str, optional
        Platform key. If None, uses key from run's platform.
    """
    baselines = load_baselines()
    key = platform_key or run.platform.key
    baselines.set_baseline(key, run)
    save_baselines(baselines)


def results_to_metrics(results: list[BenchmarkResult]) -> list[BenchmarkMetric]:
    """
    Convert BenchmarkResult list to BenchmarkMetric list.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Results from benchmark run.

    Returns
    -------
    list[BenchmarkMetric]
        Converted metrics.
    """
    return [
        BenchmarkMetric(
            name=r.name,
            mlx_time_ms=r.mlx_time_ms,
            reference_time_ms=r.reference_time_ms,
            speedup=r.speedup,
            max_abs_error=r.max_abs_error,
            mean_abs_error=r.mean_abs_error,
            correlation=r.correlation,
            cold_time_ms=getattr(r, "cold_time_ms", None),
            warm_time_ms=getattr(r, "warm_time_ms", None),
            peak_memory_mb=getattr(r, "peak_memory_mb", None),
            memory_efficiency=getattr(r, "memory_efficiency", None),
        )
        for r in results
    ]


def create_benchmark_run(results: list[BenchmarkResult]) -> BenchmarkRun:
    """
    Create a BenchmarkRun from results and current platform.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Results from benchmark run.

    Returns
    -------
    BenchmarkRun
        Complete benchmark run with platform context.
    """
    info = get_platform_info()
    platform = PlatformSnapshot(
        chip=info.chip,
        chip_family=info.chip_family,
        chip_variant=info.chip_variant,
        memory_gb=info.memory_gb,
        macos_version=info.macos_version,
        python_version=info.python_version,
        mlx_version=info.mlx_version,
    )

    return BenchmarkRun.create_now(
        version=get_version(),
        commit_sha=get_git_sha(),
        platform=platform,
        metrics=results_to_metrics(results),
    )


def compare_to_baseline(
    current: list[BenchmarkResult],
    platform_key: str | None = None,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> ComparisonReport | None:
    """
    Compare current results to baseline, detecting regressions.

    Parameters
    ----------
    current : list[BenchmarkResult]
        Current benchmark results.
    platform_key : str, optional
        Platform key to compare against. If None, uses current platform.
    threshold : float, default=0.10
        Regression threshold (0.10 = 10% slowdown is a regression).

    Returns
    -------
    ComparisonReport or None
        Comparison report, or None if no baseline exists.
    """
    baselines = load_baselines()

    if platform_key is None:
        platform_key = get_platform_info().key

    baseline_run = baselines.get_baseline(platform_key)
    if baseline_run is None:
        return None

    baseline_map = {m.name: m for m in baseline_run.metrics}

    regressions = []
    improvements = []
    unchanged = []

    for result in current:
        if result.name not in baseline_map:
            continue

        baseline = baseline_map[result.name]
        ratio = result.mlx_time_ms / baseline.mlx_time_ms
        percent_change = (ratio - 1) * 100

        if ratio > 1 + threshold:
            regressions.append(
                RegressionResult(
                    name=result.name,
                    baseline_ms=baseline.mlx_time_ms,
                    current_ms=result.mlx_time_ms,
                    regression_percent=percent_change,
                    is_regression=True,
                )
            )
        elif ratio < 1 - threshold:
            improvements.append(
                RegressionResult(
                    name=result.name,
                    baseline_ms=baseline.mlx_time_ms,
                    current_ms=result.mlx_time_ms,
                    regression_percent=percent_change,
                    is_regression=False,
                )
            )
        else:
            unchanged.append(result.name)

    return ComparisonReport(
        platform_key=platform_key,
        baseline_timestamp=baseline_run.timestamp,
        current_timestamp=datetime.now().isoformat(),
        threshold_percent=threshold * 100,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
    )


def list_baselines() -> list[tuple[str, str]]:
    """
    List all stored baselines.

    Returns
    -------
    list[tuple[str, str]]
        List of (platform_key, timestamp) pairs.
    """
    baselines = load_baselines()
    result = []
    for key, data in baselines.baselines.items():
        timestamp = data.get("timestamp", "unknown")
        result.append((key, timestamp))
    return result


def delete_baseline(platform_key: str) -> bool:
    """
    Delete a baseline for a specific platform.

    Parameters
    ----------
    platform_key : str
        Platform key to delete.

    Returns
    -------
    bool
        True if baseline was deleted, False if not found.
    """
    baselines = load_baselines()
    if platform_key in baselines.baselines:
        del baselines.baselines[platform_key]
        save_baselines(baselines)
        return True
    return False
