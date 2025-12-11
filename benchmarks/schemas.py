"""
Data schemas for benchmark results and baselines.

Provides structured data types for:
- Individual benchmark metrics
- Complete benchmark runs
- Baseline storage format
- Regression detection results
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkMetric:
    """Single benchmark measurement with optional memory/cache data."""

    name: str
    mlx_time_ms: float
    reference_time_ms: float
    speedup: float
    max_abs_error: float
    mean_abs_error: float
    correlation: float
    # Optional extended fields
    cold_time_ms: float | None = None
    warm_time_ms: float | None = None
    peak_memory_mb: float | None = None
    memory_efficiency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkMetric:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PlatformSnapshot:
    """Platform information snapshot for benchmark context."""

    chip: str
    chip_family: str
    chip_variant: str
    memory_gb: int
    macos_version: str
    python_version: str
    mlx_version: str

    @property
    def key(self) -> str:
        """Generate unique platform key."""
        return f"{self.chip_family.lower()}-{self.chip_variant.lower()}-{self.memory_gb}gb"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformSnapshot:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkRun:
    """Complete benchmark run with platform context."""

    version: str
    commit_sha: str
    timestamp: str
    platform: PlatformSnapshot
    metrics: list[BenchmarkMetric]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp,
            "platform": self.platform.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRun:
        """Create from dictionary."""
        return cls(
            version=data["version"],
            commit_sha=data["commit_sha"],
            timestamp=data["timestamp"],
            platform=PlatformSnapshot.from_dict(data["platform"]),
            metrics=[BenchmarkMetric.from_dict(m) for m in data["metrics"]],
        )

    @classmethod
    def create_now(
        cls,
        version: str,
        commit_sha: str,
        platform: PlatformSnapshot,
        metrics: list[BenchmarkMetric],
    ) -> BenchmarkRun:
        """Create a new benchmark run with current timestamp."""
        return cls(
            version=version,
            commit_sha=commit_sha,
            timestamp=datetime.now().isoformat(),
            platform=platform,
            metrics=metrics,
        )


@dataclass
class BaselineFile:
    """Baseline storage format with multiple platform entries."""

    schema_version: str = "1.0"
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "baselines": self.baselines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineFile:
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            baselines=data.get("baselines", {}),
        )

    def get_baseline(self, platform_key: str) -> BenchmarkRun | None:
        """Get baseline for a specific platform."""
        if platform_key in self.baselines:
            return BenchmarkRun.from_dict(self.baselines[platform_key])
        return None

    def set_baseline(self, platform_key: str, run: BenchmarkRun) -> None:
        """Set baseline for a specific platform."""
        self.baselines[platform_key] = run.to_dict()


@dataclass
class RegressionResult:
    """Result from comparing current run to baseline."""

    name: str
    baseline_ms: float
    current_ms: float
    regression_percent: float
    is_regression: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComparisonReport:
    """Complete comparison report between current run and baseline."""

    platform_key: str
    baseline_timestamp: str
    current_timestamp: str
    threshold_percent: float
    regressions: list[RegressionResult]
    improvements: list[RegressionResult]
    unchanged: list[str]

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.regressions) > 0

    @property
    def regression_count(self) -> int:
        """Count of regressions."""
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        """Count of improvements."""
        return len(self.improvements)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform_key": self.platform_key,
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "threshold_percent": self.threshold_percent,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "unchanged": self.unchanged,
            "has_regressions": self.has_regressions,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            f"Comparison Report for {self.platform_key}",
            f"Baseline: {self.baseline_timestamp}",
            f"Current: {self.current_timestamp}",
            f"Threshold: {self.threshold_percent:.1f}%",
            "",
        ]

        if self.regressions:
            lines.append(f"REGRESSIONS ({len(self.regressions)}):")
            for r in self.regressions:
                lines.append(
                    f"  {r.name}: {r.baseline_ms:.2f}ms -> {r.current_ms:.2f}ms "
                    f"(+{r.regression_percent:.1f}%)"
                )
            lines.append("")

        if self.improvements:
            lines.append(f"IMPROVEMENTS ({len(self.improvements)}):")
            for i in self.improvements:
                lines.append(
                    f"  {i.name}: {i.baseline_ms:.2f}ms -> {i.current_ms:.2f}ms "
                    f"({i.regression_percent:.1f}%)"
                )
            lines.append("")

        lines.append(f"Unchanged: {len(self.unchanged)} benchmarks")

        return "\n".join(lines)
