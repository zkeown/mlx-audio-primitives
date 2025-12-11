"""
Window functions for audio signal processing.

Provides window functions compatible with librosa/scipy conventions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import mlx.core as mx


@lru_cache(maxsize=64)
def _get_window_cached(
    window_name: str,
    n_fft: int,
    fftbins: bool,
) -> Tuple[bytes, int]:
    """
    Compute window function with caching.

    Returns window data as bytes for efficient caching.
    """
    window_name = window_name.lower()

    # For periodic (fftbins=True), we compute n_fft+1 points and drop the last
    # This matches scipy/librosa behavior for DFT-even windows
    if fftbins:
        n = n_fft + 1
    else:
        n = n_fft

    if window_name in ("hann", "hanning"):
        w = _hann(n)
    elif window_name == "hamming":
        w = _hamming(n)
    elif window_name == "blackman":
        w = _blackman(n)
    elif window_name in ("bartlett", "triangular"):
        w = _bartlett(n)
    elif window_name in ("rectangular", "boxcar", "ones"):
        w = mx.ones(n, dtype=mx.float32)
    else:
        raise ValueError(
            f"Unknown window type: '{window_name}'. "
            f"Supported: hann, hamming, blackman, bartlett, rectangular"
        )

    # For periodic windows, drop the last sample
    if fftbins and n > n_fft:
        w = w[:n_fft]

    # Convert to bytes for caching
    w_np = np.array(w, dtype=np.float32)
    return w_np.tobytes(), len(w_np)


def get_window(
    window: Union[str, mx.array],
    n_fft: int,
    fftbins: bool = True,
) -> mx.array:
    """
    Get a window function.

    Results are cached for repeated calls with identical parameters.

    Parameters
    ----------
    window : str or mx.array
        Window specification. If string, one of:
        - 'hann' or 'hanning': Hann window
        - 'hamming': Hamming window
        - 'blackman': Blackman window
        - 'bartlett': Bartlett (triangular) window
        - 'rectangular' or 'boxcar' or 'ones': Rectangular window (all ones)
        If mx.array, used directly (must have length n_fft).
    n_fft : int
        Length of the window.
    fftbins : bool, default=True
        If True, create a periodic window for use with FFT (DFT-even).
        If False, create a symmetric window.

    Returns
    -------
    mx.array
        Window of shape (n_fft,) with dtype float32.

    Raises
    ------
    ValueError
        If window string is not recognized or array has wrong length.

    Examples
    --------
    >>> window = get_window('hann', 2048)
    >>> window.shape
    (2048,)
    """
    if isinstance(window, mx.array):
        if window.shape[0] != n_fft:
            raise ValueError(
                f"Window array length ({window.shape[0]}) "
                f"must match n_fft ({n_fft})"
            )
        return window.astype(mx.float32)

    if not isinstance(window, str):
        raise TypeError(
            f"window must be str or mx.array, got {type(window).__name__}"
        )

    # Get cached window data
    window_bytes, length = _get_window_cached(window, n_fft, fftbins)

    # Convert from bytes back to MLX array
    w = np.frombuffer(window_bytes, dtype=np.float32)
    return mx.array(w)


def _hann(n: int) -> mx.array:
    """
    Hann (raised cosine) window.

    w[k] = 0.5 - 0.5 * cos(2 * pi * k / (n - 1))

    Computed in float64 for precision, then cast to float32.
    This ensures perfect symmetry matching scipy.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    return mx.array(window.astype(np.float32))


def _hamming(n: int) -> mx.array:
    """
    Hamming window.

    w[k] = 0.54 - 0.46 * cos(2 * pi * k / (n - 1))

    Computed in float64 for precision, then cast to float32.
    This ensures perfect symmetry matching scipy.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))
    return mx.array(window.astype(np.float32))


def _blackman(n: int) -> mx.array:
    """
    Blackman window.

    w[k] = 0.42 - 0.5 * cos(2 * pi * k / (n - 1)) + 0.08 * cos(4 * pi * k / (n - 1))

    Computed in float64 for precision, then cast to float32.
    This ensures perfect symmetry and non-negative values matching scipy.

    Note: The theoretical value at endpoints is exactly 0 (0.42 - 0.5 + 0.08 = 0),
    but floating-point computation can produce tiny negative values (~1e-17).
    We clamp to ensure non-negativity.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    window = (
        0.42
        - 0.5 * np.cos(2 * np.pi * k / (n - 1))
        + 0.08 * np.cos(4 * np.pi * k / (n - 1))
    )
    # Clamp to non-negative (theoretical min is 0 at endpoints)
    window = np.maximum(window, 0.0)
    return mx.array(window.astype(np.float32))


def _bartlett(n: int) -> mx.array:
    """
    Bartlett (triangular) window.

    w[k] = 1 - |2 * k / (n - 1) - 1|

    Computed in float64 for precision, then cast to float32.
    This ensures perfect symmetry matching scipy.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    window = 1 - np.abs(2 * k / (n - 1) - 1)
    return mx.array(window.astype(np.float32))
