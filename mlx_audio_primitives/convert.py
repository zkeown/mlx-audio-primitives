"""
Decibel conversion utilities.

Provides functions to convert between power/amplitude and decibels.
"""
from __future__ import annotations

from typing import Optional, Union, Callable

import mlx.core as mx


def power_to_db(
    S: mx.array,
    ref: Union[float, Callable[[mx.array], mx.array]] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> mx.array:
    """
    Convert a power spectrogram to decibel (dB) units.

    This computes: 10 * log10(S / ref)

    Parameters
    ----------
    S : mx.array
        Input power spectrogram (non-negative).
    ref : float or callable, default=1.0
        Reference power. If callable, computed as ref(S).
        For amplitude spectrograms squared, use ref=mx.max.
    amin : float, default=1e-10
        Minimum threshold for S. Values below this are clipped.
    top_db : float or None, default=80.0
        Maximum dynamic range in dB. If not None, the output is clipped to
        (max(S_db) - top_db, max(S_db)).

    Returns
    -------
    mx.array
        Power spectrogram in decibels.

    Examples
    --------
    >>> S = magnitude(stft(y)) ** 2  # Power spectrogram
    >>> S_db = power_to_db(S)
    """
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = mx.array(ref, dtype=S.dtype)

    # Ensure positive values
    S = mx.maximum(S, amin)
    ref_value = mx.maximum(ref_value, amin)

    # Convert to dB
    S_db = 10.0 * mx.log10(S / ref_value)

    # Apply top_db threshold
    if top_db is not None:
        if top_db <= 0:
            raise ValueError(f"top_db must be positive, got {top_db}")
        S_db = mx.maximum(S_db, mx.max(S_db) - top_db)

    return S_db


def db_to_power(
    S_db: mx.array,
    ref: float = 1.0,
) -> mx.array:
    """
    Convert a decibel spectrogram back to power.

    This computes: ref * 10^(S_db / 10)

    Parameters
    ----------
    S_db : mx.array
        Input spectrogram in decibels.
    ref : float, default=1.0
        Reference power used in power_to_db.

    Returns
    -------
    mx.array
        Power spectrogram.

    Examples
    --------
    >>> S_db = power_to_db(S)
    >>> S_reconstructed = db_to_power(S_db)
    """
    return ref * mx.power(10.0, S_db / 10.0)


def amplitude_to_db(
    S: mx.array,
    ref: Union[float, Callable[[mx.array], mx.array]] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
) -> mx.array:
    """
    Convert an amplitude spectrogram to decibel (dB) units.

    This computes: 20 * log10(S / ref)

    Parameters
    ----------
    S : mx.array
        Input amplitude spectrogram (non-negative).
    ref : float or callable, default=1.0
        Reference amplitude. If callable, computed as ref(S).
        Common choice: ref=mx.max for normalization.
    amin : float, default=1e-5
        Minimum threshold for S. Values below this are clipped.
    top_db : float or None, default=80.0
        Maximum dynamic range in dB. If not None, the output is clipped to
        (max(S_db) - top_db, max(S_db)).

    Returns
    -------
    mx.array
        Amplitude spectrogram in decibels.

    Examples
    --------
    >>> S = magnitude(stft(y))  # Amplitude spectrogram
    >>> S_db = amplitude_to_db(S)
    """
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = mx.array(ref, dtype=S.dtype)

    # Ensure positive values
    S = mx.maximum(S, amin)
    ref_value = mx.maximum(ref_value, amin)

    # Convert to dB (amplitude uses 20 * log10)
    S_db = 20.0 * mx.log10(S / ref_value)

    # Apply top_db threshold
    if top_db is not None:
        if top_db <= 0:
            raise ValueError(f"top_db must be positive, got {top_db}")
        S_db = mx.maximum(S_db, mx.max(S_db) - top_db)

    return S_db


def db_to_amplitude(
    S_db: mx.array,
    ref: float = 1.0,
) -> mx.array:
    """
    Convert a decibel spectrogram back to amplitude.

    This computes: ref * 10^(S_db / 20)

    Parameters
    ----------
    S_db : mx.array
        Input spectrogram in decibels.
    ref : float, default=1.0
        Reference amplitude used in amplitude_to_db.

    Returns
    -------
    mx.array
        Amplitude spectrogram.

    Examples
    --------
    >>> S_db = amplitude_to_db(S)
    >>> S_reconstructed = db_to_amplitude(S_db)
    """
    return ref * mx.power(10.0, S_db / 20.0)
