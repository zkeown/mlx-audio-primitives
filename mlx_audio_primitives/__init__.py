"""
mlx-audio-primitives: Foundational audio DSP primitives for Apple's MLX framework.

This library provides librosa-compatible audio processing primitives for MLX,
enabling efficient audio ML pipelines on Apple Silicon.

Core Operations
---------------
stft : Short-Time Fourier Transform
istft : Inverse Short-Time Fourier Transform
magnitude : Compute magnitude from complex STFT
phase : Compute phase from complex STFT

Window Functions
----------------
get_window : Get a window function (hann, hamming, blackman, etc.)

Mel-Scale Operations
--------------------
mel_filterbank : Create mel-scale filterbank matrix
melspectrogram : Compute mel spectrogram from waveform

Decibel Conversions
-------------------
power_to_db : Convert power spectrogram to decibels
db_to_power : Convert decibels to power spectrogram
amplitude_to_db : Convert amplitude spectrogram to decibels
db_to_amplitude : Convert decibels to amplitude spectrogram

Utilities
---------
check_nola : Check NOLA constraint for STFT invertibility
hz_to_mel : Convert Hz to mel scale
mel_to_hz : Convert mel scale to Hz
"""

# Import MLX first to ensure library paths are set up for C++ extension
import mlx.core as _mx  # noqa: F401

# Get version from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import version as _get_version, PackageNotFoundError
    __version__ = _get_version("mlx-audio-primitives")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"  # Fallback for editable installs

# Import C++ extension availability flag for external use
from ._extension import HAS_CPP_EXT as _HAS_CPP_EXT  # noqa: F401

# Core STFT operations
from .stft import (
    stft,
    istft,
    magnitude,
    phase,
    check_nola,
)

# Window functions
from .windows import get_window

# Mel-scale operations
from .mel import (
    mel_filterbank,
    melspectrogram,
    hz_to_mel,
    mel_to_hz,
)

# Decibel conversions
from .convert import (
    power_to_db,
    db_to_power,
    amplitude_to_db,
    db_to_amplitude,
)

__all__ = [
    # Version
    "__version__",
    # STFT
    "stft",
    "istft",
    "magnitude",
    "phase",
    "check_nola",
    # Windows
    "get_window",
    # Mel
    "mel_filterbank",
    "melspectrogram",
    "hz_to_mel",
    "mel_to_hz",
    # Conversions
    "power_to_db",
    "db_to_power",
    "amplitude_to_db",
    "db_to_amplitude",
]
