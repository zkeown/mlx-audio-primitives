"""
Mel-scale filterbank and mel spectrogram.

Provides mel filterbank construction and mel spectrogram computation.

Note: hz_to_mel() and mel_to_hz() are NumPy-based utilities used internally
for precision in filterbank construction. They accept and return np.ndarray.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .stft import stft, magnitude
from ._validation import validate_positive, validate_non_negative

# Import C++ extension with graceful fallback
# noqa: F401 - reserved for future use
from ._extension import _ext, HAS_CPP_EXT

# Slaney mel scale constants (librosa default)
# These define the piecewise-linear/log mel scale used by librosa
_SLANEY_F_MIN = 0.0  # Minimum frequency for linear region
_SLANEY_F_SP = 200.0 / 3  # Linear region spacing in Hz (~66.67 Hz)
_SLANEY_MIN_LOG_HZ = 1000.0  # Boundary between linear and log regions
_SLANEY_MIN_LOG_MEL = (_SLANEY_MIN_LOG_HZ - _SLANEY_F_MIN) / _SLANEY_F_SP
_SLANEY_LOGSTEP = np.log(6.4) / 27.0  # Logarithmic step size


def hz_to_mel(frequencies: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert Hz to mel scale.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in Hz.
    htk : bool, default=False
        If True, use HTK formula. If False, use Slaney (librosa default).

    Returns
    -------
    np.ndarray
        Frequencies in mel scale.
    """
    frequencies = np.asarray(frequencies)

    if htk:
        # HTK formula: mel = 2595 * log10(1 + f / 700)
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    else:
        # Slaney formula (librosa default)
        # Linear below 1000 Hz, logarithmic above
        # Use np.where with safe computation to avoid divide by zero
        # For frequencies at 0 Hz, the linear formula applies, so log isn't used
        with np.errstate(divide='ignore', invalid='ignore'):
            mels = np.where(
                frequencies < _SLANEY_MIN_LOG_HZ,
                (frequencies - _SLANEY_F_MIN) / _SLANEY_F_SP,
                _SLANEY_MIN_LOG_MEL
                + np.log(frequencies / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP
            )
        return mels


def mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert mel scale to Hz.

    Parameters
    ----------
    mels : np.ndarray
        Frequencies in mel scale.
    htk : bool, default=False
        If True, use HTK formula. If False, use Slaney (librosa default).

    Returns
    -------
    np.ndarray
        Frequencies in Hz.
    """
    mels = np.asarray(mels)

    if htk:
        # HTK formula: f = 700 * (10^(mel / 2595) - 1)
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        # Slaney formula (inverse)
        freqs = np.where(
            mels < _SLANEY_MIN_LOG_MEL,
            _SLANEY_F_MIN + _SLANEY_F_SP * mels,
            _SLANEY_MIN_LOG_HZ
            * np.exp(_SLANEY_LOGSTEP * (mels - _SLANEY_MIN_LOG_MEL))
        )
        return freqs


@lru_cache(maxsize=32)
def _compute_mel_filterbank_np(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    htk: bool,
    norm: Optional[str],
) -> Tuple[bytes, Tuple[int, int]]:
    """
    Compute mel filterbank as a cacheable tuple structure.

    Returns the filterbank as bytes (hashable) along with shape info.
    This allows caching with lru_cache while still returning MLX arrays.

    Note: We use pure NumPy here instead of the C++ extension because:
    1. Mel filterbanks are computed once and cached, so startup cost is negligible
    2. NumPy's linspace has better precision than MLX's, matching librosa exactly
    3. The filterbank is small (~500KB) so memory copy cost is negligible
    """
    # Pure NumPy implementation for precision (matches librosa exactly)
    # Number of frequency bins
    n_freqs = 1 + n_fft // 2

    # Frequencies of FFT bins
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Mel scale boundaries
    mel_min = hz_to_mel(fmin, htk=htk)
    mel_max = hz_to_mel(fmax, htk=htk)

    # Mel points: n_mels + 2 points (including edges)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points, htk=htk)

    # Create filterbank using vectorized operations
    # Each filter is a triangular window centered at hz_points[i+1]
    # with edges at hz_points[i] and hz_points[i+2]

    # Extract lower, center, upper frequencies for each mel band
    # Shape: (n_mels, 1) for broadcasting with (1, n_freqs)
    f_lower = hz_points[:-2, np.newaxis]    # (n_mels, 1)
    f_center = hz_points[1:-1, np.newaxis]  # (n_mels, 1)
    f_upper = hz_points[2:, np.newaxis]     # (n_mels, 1)
    freqs = fft_freqs[np.newaxis, :]        # (1, n_freqs)

    # Compute slopes for the triangular filters
    # Lower slope: (freq - f_lower) / (f_center - f_lower)
    # Upper slope: (f_upper - freq) / (f_upper - f_center)
    # Add small epsilon to avoid division by zero
    lower_slope = (freqs - f_lower) / (f_center - f_lower + 1e-10)
    upper_slope = (f_upper - freqs) / (f_upper - f_center + 1e-10)

    # Triangular filter: min of slopes, clipped to [0, inf)
    # This creates the triangular shape: rises from f_lower to f_center,
    # then falls from f_center to f_upper
    filterbank = np.maximum(0, np.minimum(lower_slope, upper_slope)).astype(
        np.float32
    )

    # Normalize
    if norm == "slaney":
        # Normalize by bandwidth (area under each filter = 1)
        enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
        filterbank *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unknown norm: '{norm}'. Supported: 'slaney', None")

    # Return as bytes for efficient caching (avoids tuple conversion overhead)
    return filterbank.tobytes(), filterbank.shape


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[str] = "slaney",
) -> mx.array:
    """
    Create a mel-scale filterbank matrix.

    Results are cached for repeated calls with identical parameters.

    Parameters
    ----------
    sr : int
        Sample rate of the audio.
    n_fft : int
        FFT size.
    n_mels : int, default=128
        Number of mel bands.
    fmin : float, default=0.0
        Minimum frequency (Hz).
    fmax : float, optional
        Maximum frequency (Hz). Default: sr / 2.
    htk : bool, default=False
        If True, use HTK formula for mel scale.
        If False, use Slaney formula (librosa default).
    norm : str or None, default='slaney'
        Normalization mode:
        - 'slaney': Divide each filter by its bandwidth (area normalization).
        - None: No normalization.

    Returns
    -------
    mx.array
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).

    Examples
    --------
    >>> mel_fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
    >>> mel_fb.shape
    (128, 1025)
    """
    # Input validation
    validate_positive(n_mels, "n_mels")
    validate_non_negative(fmin, "fmin")

    if fmax is None:
        fmax = sr / 2.0

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
    if fmax > sr / 2.0:
        raise ValueError(
            f"fmax ({fmax}) cannot exceed Nyquist frequency ({sr / 2.0})"
        )

    # Get cached filterbank data
    filterbank_bytes, shape = _compute_mel_filterbank_np(
        sr, n_fft, n_mels, fmin, fmax, htk, norm
    )

    # Convert from bytes back to MLX array
    filterbank = np.frombuffer(filterbank_bytes, dtype=np.float32).reshape(shape)
    return mx.array(filterbank)


def melspectrogram(
    y: mx.array,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Union[str, mx.array] = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[str] = "slaney",
) -> mx.array:
    """
    Compute mel spectrogram from audio waveform.

    Parameters
    ----------
    y : mx.array
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Hop length. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    window : str or mx.array, default='hann'
        Window function.
    center : bool, default=True
        Center padding.
    pad_mode : str, default='constant'
        Padding mode.
    power : float, default=2.0
        Exponent for the magnitude spectrogram.
        1.0 for amplitude, 2.0 for power spectrogram.
    n_mels : int, default=128
        Number of mel bands.
    fmin : float, default=0.0
        Minimum frequency.
    fmax : float, optional
        Maximum frequency. Default: sr / 2.
    htk : bool, default=False
        Use HTK formula for mel scale.
    norm : str or None, default='slaney'
        Mel filterbank normalization.

    Returns
    -------
    mx.array
        Mel spectrogram.
        Shape: (n_mels, n_frames) for 1D input.
        Shape: (batch, n_mels, n_frames) for 2D input.

    Examples
    --------
    >>> mel = melspectrogram(y, sr=22050, n_mels=128)
    >>> mel.shape
    (128, 44)
    """
    # Compute STFT
    S = stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute magnitude spectrogram
    S_mag = magnitude(S)

    # Apply power
    if power != 1.0:
        S_mag = mx.power(S_mag, power)

    # Get mel filterbank
    mel_basis = mel_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
    )

    # Apply mel filterbank
    # S_mag shape: (freq_bins, n_frames) or (batch, freq_bins, n_frames)
    # mel_basis shape: (n_mels, freq_bins)

    if S_mag.ndim == 2:
        # (n_mels, freq_bins) @ (freq_bins, n_frames) -> (n_mels, n_frames)
        mel_spec = mx.matmul(mel_basis, S_mag)
    else:
        # Batch case: (batch, freq_bins, n_frames)
        # Need: (batch, n_mels, n_frames)
        # mel_basis: (n_mels, freq_bins)
        # Use einsum or explicit broadcast
        mel_spec = mx.matmul(mel_basis, S_mag)  # Broadcasting works

    return mel_spec
