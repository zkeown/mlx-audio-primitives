"""
C++ extension loader with graceful fallback.

This module provides a single source of truth for C++ extension availability,
avoiding the need for repeated try/except blocks throughout the codebase.

Usage in other modules:
    from ._extension import HAS_CPP_EXT, _ext

    def some_function(...):
        if HAS_CPP_EXT and _ext is not None:
            return _ext.some_function(...)
        # Python fallback
        ...

The extension is optional - all functionality works without it, just potentially
slower for certain operations (overlap-add, signal framing).
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
