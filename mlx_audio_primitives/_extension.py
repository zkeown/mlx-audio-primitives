"""
C++ extension loader with graceful fallback.

This module provides a single source of truth for C++ extension availability,
avoiding the need for repeated try/except blocks throughout the codebase.
"""
from typing import Any

try:
    from . import _ext as _ext_module
    HAS_CPP_EXT: bool = True
    _ext: Any | None = _ext_module
except ImportError:
    HAS_CPP_EXT = False
    _ext = None

__all__ = ["_ext", "HAS_CPP_EXT"]
