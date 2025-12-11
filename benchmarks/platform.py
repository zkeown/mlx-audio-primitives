"""
Platform detection and reporting for Apple Silicon.

Detects:
- Chip family (M1, M2, M3, M4)
- Chip variant (base, Pro, Max, Ultra)
- Memory configuration
- Software versions
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PlatformInfo:
    """Complete platform identification."""

    chip: str  # Full chip name: "Apple M4 Max"
    chip_family: str  # M1, M2, M3, M4
    chip_variant: str  # base, Pro, Max, Ultra
    memory_gb: int
    macos_version: str
    python_version: str
    mlx_version: str

    @property
    def key(self) -> str:
        """Generate unique platform key for baseline storage."""
        return f"{self.chip_family.lower()}-{self.chip_variant.lower()}-{self.memory_gb}gb"


def detect_apple_silicon() -> tuple[str, str, str]:
    """
    Detect Apple Silicon chip details.

    Returns
    -------
    tuple[str, str, str]
        (full_chip_name, chip_family, chip_variant)
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        chip_name = result.stdout.strip()  # "Apple M4 Max"

        # Parse chip family and variant
        match = re.match(r"Apple (M\d+)(?: (Pro|Max|Ultra))?", chip_name)
        if match:
            family = match.group(1)  # M1, M2, M3, M4
            variant = match.group(2) or "base"  # Pro, Max, Ultra, or base
            return chip_name, family, variant

        return chip_name, "Unknown", "Unknown"
    except Exception:
        return "Unknown", "Unknown", "Unknown"


def detect_memory() -> int:
    """
    Detect system memory in GB.

    Returns
    -------
    int
        System memory in gigabytes.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        bytes_val = int(result.stdout.strip())
        return bytes_val // (1024**3)
    except Exception:
        return 0


def detect_macos_version() -> str:
    """
    Detect macOS version.

    Returns
    -------
    str
        macOS version string.
    """
    return platform.mac_ver()[0]


def get_platform_info() -> PlatformInfo:
    """
    Get complete platform information.

    Returns
    -------
    PlatformInfo
        Complete platform identification.
    """
    chip, family, variant = detect_apple_silicon()

    return PlatformInfo(
        chip=chip,
        chip_family=family,
        chip_variant=variant,
        memory_gb=detect_memory(),
        macos_version=detect_macos_version(),
        python_version=platform.python_version(),
        mlx_version=mx.__version__,
    )


def format_platform_header() -> str:
    """
    Format platform info as benchmark header.

    Returns
    -------
    str
        Formatted header string.
    """
    info = get_platform_info()
    lines = [
        "=" * 60,
        "MLX Audio Primitives Benchmark",
        "=" * 60,
        f"Platform: {info.chip}",
        f"Memory: {info.memory_gb} GB",
        f"macOS: {info.macos_version}",
        f"Python: {info.python_version}",
        f"MLX: {info.mlx_version}",
        "=" * 60,
    ]
    return "\n".join(lines)


# Reference performance expectations by platform
# These are relative multipliers compared to M1 base
REFERENCE_PLATFORMS = {
    "m1-base": {"expected_speedup": 1.0},
    "m1-pro": {"expected_speedup": 1.15},
    "m1-max": {"expected_speedup": 1.25},
    "m1-ultra": {"expected_speedup": 1.4},
    "m2-base": {"expected_speedup": 1.3},
    "m2-pro": {"expected_speedup": 1.45},
    "m2-max": {"expected_speedup": 1.55},
    "m2-ultra": {"expected_speedup": 1.7},
    "m3-base": {"expected_speedup": 1.5},
    "m3-pro": {"expected_speedup": 1.65},
    "m3-max": {"expected_speedup": 1.8},
    "m4-base": {"expected_speedup": 1.7},
    "m4-pro": {"expected_speedup": 1.85},
    "m4-max": {"expected_speedup": 2.0},
}


def get_expected_performance_multiplier() -> float:
    """
    Get expected performance multiplier for current platform.

    Returns
    -------
    float
        Expected speedup relative to M1 base.
    """
    info = get_platform_info()
    return REFERENCE_PLATFORMS.get(info.key, {}).get("expected_speedup", 1.0)
