"""
Pitch and periodicity analysis primitives.

Provides autocorrelation for pitch detection and periodicity analysis.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._extension import HAS_CPP_EXT, _ext
from ._validation import validate_positive


def autocorrelation(
    y: mx.array,
    max_lag: int | None = None,
    normalize: bool = True,
    center: bool = True,
) -> mx.array:
    """
    Compute autocorrelation of a signal using FFT.

    The autocorrelation is computed efficiently using the Wiener-Khinchin
    theorem: r[k] = IFFT(|FFT(y)|^2).

    Parameters
    ----------
    y : mx.array
        Input signal. Shape: (samples,) or (batch, samples).
    max_lag : int, optional
        Maximum lag to compute. Default: signal length.
    normalize : bool, default=True
        If True, normalize by r[0] so that r[0] = 1.
    center : bool, default=True
        If True, subtract mean before computing autocorrelation.

    Returns
    -------
    mx.array
        Autocorrelation values for lags 0 to max_lag-1.
        Shape: (max_lag,) for 1D input.
        Shape: (batch, max_lag) for batched input.

    Notes
    -----
    For pitch detection, look for the first peak after lag 0.
    The lag of this peak corresponds to the fundamental period.

    Examples
    --------
    >>> y = mx.array(np.sin(2 * np.pi * 440 * np.arange(22050) / 22050))
    >>> r = autocorrelation(y, max_lag=2000)
    >>> # Peak at lag ~50 corresponds to 440 Hz at 22050 sample rate
    """
    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.autocorrelation(
            y,
            max_lag if max_lag is not None else -1,
            normalize,
            center,
        )

    # Python fallback
    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, n = y.shape

    if max_lag is None:
        max_lag = n

    max_lag = min(max_lag, n)

    # Center the signal (subtract mean)
    if center:
        y = y - mx.mean(y, axis=-1, keepdims=True)

    # Use FFT for efficient autocorrelation (Wiener-Khinchin theorem)
    # Zero-pad to avoid circular correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))

    # Convert to numpy for FFT (MLX FFT runs on CPU anyway)
    y_np = np.array(y)

    # FFT
    Y = np.fft.rfft(y_np, n=n_fft, axis=-1)

    # Power spectrum
    power = Y * np.conj(Y)

    # Inverse FFT to get autocorrelation
    r = np.fft.irfft(power, n=n_fft, axis=-1)

    # Take only positive lags up to max_lag
    r = r[:, :max_lag]

    # Normalize if requested
    if normalize:
        # Normalize by r[0] (variance)
        r0 = r[:, :1]
        r0 = np.maximum(r0, 1e-10)  # Avoid division by zero
        r = r / r0

    result = mx.array(r.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    return result


def pitch_detect_acf(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold: float = 0.1,
    center: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Detect pitch using autocorrelation.

    This is a simple pitch detection algorithm that finds the fundamental
    frequency by looking for peaks in the autocorrelation function.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency to detect (Hz).
    fmax : float, default=2000.0
        Maximum frequency to detect (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, default=512
        Hop length between frames.
    threshold : float, default=0.1
        Minimum autocorrelation peak value for voiced detection.
    center : bool, default=True
        Center-pad the signal for framing.

    Returns
    -------
    tuple
        (f0, voiced_flag) where:
        - f0: Detected fundamental frequency for each frame (Hz).
          Shape: (n_frames,) or (batch, n_frames).
        - voiced_flag: Boolean indicating voiced frames.
          Shape: (n_frames,) or (batch, n_frames).

    Examples
    --------
    >>> f0, voiced = pitch_detect_acf(y, sr=22050, fmin=80, fmax=500)
    >>> f0[voiced]  # Get only voiced pitch values
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    # Compute lag bounds from frequency bounds
    # Lag (samples) = sr / frequency, so:
    # - High frequencies → short periods → small lags
    # - Low frequencies → long periods → large lags
    min_lag = int(sr / fmax)  # Maximum frequency → minimum lag
    max_lag = int(sr / fmin)  # Minimum frequency → maximum lag

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.array(y)
        y_np = np.pad(y_np, [(0, 0), (pad_length, pad_length)], mode="constant")
    else:
        y_np = np.array(y)

    # Frame the signal
    padded_length = y_np.shape[1]
    n_frames = 1 + (padded_length - frame_length) // hop_length

    # Process each frame
    f0_np = np.zeros((batch_size, n_frames), dtype=np.float32)
    voiced_np = np.zeros((batch_size, n_frames), dtype=bool)

    for b in range(batch_size):
        for t in range(n_frames):
            start = t * hop_length
            end = start + frame_length
            frame = y_np[b, start:end]

            # Compute autocorrelation for this frame
            frame_centered = frame - np.mean(frame)
            n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
            Y = np.fft.rfft(frame_centered, n=n_fft)
            power = Y * np.conj(Y)
            r = np.fft.irfft(power, n=n_fft)

            # Normalize
            if r[0] > 1e-10:
                r = r / r[0]
            else:
                continue

            # Search for peak in valid lag range
            search_range = r[min_lag:max_lag + 1]
            if len(search_range) == 0:
                continue

            # Find peaks (local maxima) in the autocorrelation
            # A local maximum satisfies: r[i-1] < r[i] > r[i+1]
            # We take the FIRST peak above threshold, not the maximum, because:
            # - The fundamental period produces the first significant peak
            # - Later peaks at 2T, 3T, etc. are harmonically related but less reliable
            peaks = []
            for i in range(1, len(search_range) - 1):
                if search_range[i] > search_range[i - 1] and search_range[i] > search_range[i + 1]:
                    if search_range[i] > threshold:
                        peaks.append((i, search_range[i]))

            # Use the first significant peak (corresponds to fundamental period)
            if len(peaks) > 0:
                peak_idx, peak_value = peaks[0]
                peak_lag = min_lag + peak_idx
                f0_np[b, t] = sr / peak_lag
                voiced_np[b, t] = True
            elif len(search_range) > 0:
                # Fallback: if no local peak found, use global max if above threshold
                peak_idx = np.argmax(search_range)
                peak_value = search_range[peak_idx]
                if peak_value > threshold:
                    peak_lag = min_lag + peak_idx
                    f0_np[b, t] = sr / peak_lag
                    voiced_np[b, t] = True

    f0 = mx.array(f0_np)
    voiced = mx.array(voiced_np)

    # Remove batch dimension if input was 1D
    if input_is_1d:
        f0 = f0[0]
        voiced = voiced[0]

    return f0, voiced


def periodicity(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
) -> mx.array:
    """
    Compute periodicity (autocorrelation strength) per frame.

    Periodicity measures how periodic/harmonic a signal is at each frame.
    Values close to 1 indicate highly periodic (tonal) content,
    while values close to 0 indicate noise-like content.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency for periodicity search (Hz).
    fmax : float, default=2000.0
        Maximum frequency for periodicity search (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, default=512
        Hop length between frames.
    center : bool, default=True
        Center-pad the signal for framing.

    Returns
    -------
    mx.array
        Periodicity strength for each frame (0 to 1).
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> p = periodicity(y, sr=22050)
    >>> # High values indicate voiced/tonal content
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    # Compute lag bounds
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.array(y)
        y_np = np.pad(y_np, [(0, 0), (pad_length, pad_length)], mode="constant")
    else:
        y_np = np.array(y)

    # Frame the signal
    padded_length = y_np.shape[1]
    n_frames = 1 + (padded_length - frame_length) // hop_length

    # Process each frame
    periodicity_np = np.zeros((batch_size, 1, n_frames), dtype=np.float32)

    for b in range(batch_size):
        for t in range(n_frames):
            start = t * hop_length
            end = start + frame_length
            frame = y_np[b, start:end]

            # Compute autocorrelation
            frame_centered = frame - np.mean(frame)
            n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
            Y = np.fft.rfft(frame_centered, n=n_fft)
            power = Y * np.conj(Y)
            r = np.fft.irfft(power, n=n_fft)

            # Normalize
            if r[0] > 1e-10:
                r = r / r[0]

                # Find maximum in valid lag range
                search_range = r[min_lag:max_lag + 1]
                if len(search_range) > 0:
                    periodicity_np[b, 0, t] = np.max(search_range)

    result = mx.array(periodicity_np)

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    return result
