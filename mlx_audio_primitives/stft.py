"""
Short-Time Fourier Transform (STFT) and Inverse STFT.

Provides librosa-compatible STFT and ISTFT implementations for MLX.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

# Import C++ extension with graceful fallback
from ._extension import HAS_CPP_EXT, _ext
from .windows import get_window

# Numerical constants
_WINDOW_SUM_EPSILON = 1e-8  # Minimum window sum for numerical stability

# Cache for padded windows to avoid repeated padding operations
_padded_window_cache: dict[tuple, mx.array] = {}


def _get_padded_window(
    window: str | mx.array, win_length: int, n_fft: int
) -> mx.array:
    """Get window, padding to n_fft if needed, with caching."""
    # Create cache key
    if isinstance(window, str):
        cache_key = (window, win_length, n_fft)
    else:
        # For array windows, use id (caller must ensure consistency)
        cache_key = (id(window), win_length, n_fft)

    if cache_key in _padded_window_cache:
        return _padded_window_cache[cache_key]

    # Get base window
    win = get_window(window, win_length, fftbins=True)

    # Pad if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = mx.pad(win, [(pad_left, pad_right)])

    # Cache and return
    _padded_window_cache[cache_key] = win
    return win


@lru_cache(maxsize=8)
def _get_compiled_stft_fn(
    n_fft: int, hop_length: int, center: bool, pad_mode: str
):
    """
    Get a compiled STFT function for the given parameters.

    Using mx.compile() enables graph-level optimizations including
    kernel fusion and reduced dispatch overhead.
    """

    def _stft_core(y: mx.array, win: mx.array) -> mx.array:
        # Center padding (if enabled)
        if center:
            pad_length = n_fft // 2
            y = _pad_signal(y, pad_length, pad_mode)

        # Frame the signal
        frames = _frame_signal(y, n_fft, hop_length)

        # Apply window and compute FFT in sequence
        # (compiler may fuse these operations)
        frames = frames * win
        return mx.fft.rfft(frames, axis=-1)

    # Compile the function for better performance
    return mx.compile(_stft_core)


def stft(
    y: mx.array,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> mx.array:
    """
    Short-Time Fourier Transform.

    Parameters
    ----------
    y : mx.array
        Input signal. Shape: (samples,) or (batch, samples).
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Number of samples between frames. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    window : str or mx.array, default='hann'
        Window function. See get_window() for options.
    center : bool, default=True
        If True, pad signal so frame t is centered at y[t * hop_length].
    pad_mode : str, default='constant'
        Padding mode if center=True. One of: 'constant', 'reflect', 'edge'.

    Returns
    -------
    mx.array
        Complex STFT matrix.
        Shape: (n_fft//2 + 1, n_frames) for 1D input.
        Shape: (batch, n_fft//2 + 1, n_frames) for 2D input.

    Raises
    ------
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> y = mx.array(np.random.randn(22050).astype(np.float32))
    >>> S = stft(y, n_fft=2048, hop_length=512)
    >>> S.shape
    (1025, 44)
    """
    # Validate and set defaults
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if win_length <= 0:
        raise ValueError(f"win_length must be positive, got {win_length}")
    if win_length > n_fft:
        raise ValueError(
            f"win_length ({win_length}) must be <= n_fft ({n_fft})"
        )
    if hop_length > n_fft:
        raise ValueError(
            f"hop_length ({hop_length}) should typically be <= n_fft ({n_fft})"
        )

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]  # Add batch dimension

    batch_size, signal_length = y.shape

    # Get window with caching (includes padding if needed)
    win = _get_padded_window(window, win_length, n_fft)

    # Use compiled STFT implementation for better performance
    stft_fn = _get_compiled_stft_fn(n_fft, hop_length, center, pad_mode)
    stft_matrix = stft_fn(y, win)

    # Transpose to (batch, freq_bins, n_frames) to match librosa convention
    stft_matrix = mx.transpose(stft_matrix, (0, 2, 1))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        stft_matrix = stft_matrix[0]

    return stft_matrix


def istft(
    stft_matrix: mx.array,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    length: int | None = None,
) -> mx.array:
    """
    Inverse Short-Time Fourier Transform.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix.
        Shape: (n_fft//2 + 1, n_frames) for 1D output.
        Shape: (batch, n_fft//2 + 1, n_frames) for 2D output.
    hop_length : int, optional
        Number of samples between frames. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    n_fft : int, optional
        FFT size. Default: inferred from stft_matrix as 2 * (freq_bins - 1).
    window : str or mx.array, default='hann'
        Window function. Must match the window used in stft().
    center : bool, default=True
        If True, trim the padding added during stft().
    length : int, optional
        If provided, the output is trimmed or zero-padded to this length.

    Returns
    -------
    mx.array
        Reconstructed time-domain signal.
        Shape: (samples,) for 2D input.
        Shape: (batch, samples) for 3D input.

    Examples
    --------
    >>> S = stft(y, n_fft=2048, hop_length=512)
    >>> y_reconstructed = istft(S, hop_length=512)
    """
    # Validate input dimensions
    if stft_matrix.ndim not in (2, 3):
        raise ValueError(
            f"stft_matrix must be 2D or 3D, got {stft_matrix.ndim}D"
        )

    # Handle batched input
    input_is_2d = stft_matrix.ndim == 2
    if input_is_2d:
        stft_matrix = stft_matrix[None, :]  # Add batch dimension

    batch_size, freq_bins, n_frames = stft_matrix.shape

    # Infer n_fft from frequency bins
    if n_fft is None:
        n_fft = 2 * (freq_bins - 1)

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Get window with caching (includes padding if needed)
    win = _get_padded_window(window, win_length, n_fft)

    # Transpose to (batch, n_frames, freq_bins) for irfft
    stft_matrix = mx.transpose(stft_matrix, (0, 2, 1))

    # Inverse FFT: (batch, n_frames, n_fft)
    frames = mx.fft.irfft(stft_matrix, n=n_fft, axis=-1)

    # Determine output length
    # If length is specified with center=True, we need to account for
    # the padding that was added during STFT
    if length is not None:
        if center:
            # The original signal was padded by n_fft//2 on each side
            # So the "padded length" we need to reconstruct is length + n_fft
            padded_length = length + n_fft
        else:
            padded_length = length
    else:
        # Without length, use the natural overlap-add length
        padded_length = n_fft + (n_frames - 1) * hop_length

    # Perform overlap-add reconstruction
    y = _overlap_add(frames, hop_length, win, padded_length)

    # Trim center padding if needed
    if center:
        pad_length = n_fft // 2
        if length is not None:
            # When length is specified, extract exactly 'length' samples
            # from the middle of the reconstructed signal
            y = y[:, pad_length:pad_length + length]
        else:
            # Without length, just trim the padding from both sides
            # Note: negative slice y[:, pad:-pad] doesn't work when empty
            end_idx = y.shape[1] - pad_length
            if end_idx > pad_length:
                y = y[:, pad_length:end_idx]
            else:
                # Edge case: signal shorter than padding
                y = y[:, :0]  # Empty result
    else:
        # No center padding to trim
        if length is not None:
            current_length = y.shape[1]
            if length < current_length:
                y = y[:, :length]
            elif length > current_length:
                pad_amount = length - current_length
                y = mx.pad(y, [(0, 0), (0, pad_amount)])

    # Remove batch dimension if input was 2D
    if input_is_2d:
        y = y[0]

    return y


def magnitude(stft_matrix: mx.array) -> mx.array:
    """
    Compute magnitude spectrogram from complex STFT.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix (output of stft()).

    Returns
    -------
    mx.array
        Magnitude spectrogram (same shape, real-valued).
    """
    # Use mx.abs for complex arrays - more efficient than manual computation
    return mx.abs(stft_matrix)


def phase(stft_matrix: mx.array) -> mx.array:
    """
    Compute phase spectrogram from complex STFT.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix (output of stft()).

    Returns
    -------
    mx.array
        Phase spectrogram in radians (same shape, real-valued).
    """
    return mx.arctan2(stft_matrix.imag, stft_matrix.real)


def check_nola(
    window: str | mx.array,
    hop_length: int,
    n_fft: int,
    tol: float = 1e-10,
) -> bool:
    """
    Check the Nonzero Overlap-Add (NOLA) constraint.

    The NOLA constraint ensures that the ISTFT is invertible. It requires that
    the sum of squared windows at every sample position is nonzero.

    Parameters
    ----------
    window : str or mx.array
        Window function.
    hop_length : int
        Hop length.
    n_fft : int
        FFT size (determines window length for string windows).
    tol : float, default=1e-10
        Tolerance for considering sum as nonzero.

    Returns
    -------
    bool
        True if NOLA constraint is satisfied.

    Examples
    --------
    >>> check_nola('hann', hop_length=512, n_fft=2048)
    True
    """
    win = get_window(window, n_fft, fftbins=True)
    win_np = np.array(win)

    # Following scipy.signal.check_NOLA:
    # Sum the squared window values within each hop-sized bin
    # Check that the minimum bin sum is nonzero
    step = hop_length
    n_bins = n_fft // step

    # Sum squared windows for each position within a hop
    binsums = sum(
        win_np[ii * step:(ii + 1) * step] ** 2
        for ii in range(n_bins)
    )

    # Handle remainder if n_fft is not a multiple of hop_length
    if n_fft % step != 0:
        binsums[:n_fft % step] += win_np[-(n_fft % step):] ** 2

    return bool(np.min(binsums) > tol)


def _pad_signal(y: mx.array, pad_length: int, mode: str) -> mx.array:
    """Pad signal on both sides."""
    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.pad_signal(y, pad_length, mode)

    # Fallback to Python implementation
    if mode == "constant":
        return mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="constant")
    elif mode == "edge":
        return mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="edge")
    elif mode == "reflect":
        # MLX doesn't support reflect padding, implement manually
        # Reflect padding: [d c b a | a b c d e f | f e d c]
        batch_size, signal_length = y.shape

        # Left reflection (excluding the edge element, reversed)
        if pad_length > 0:
            # y[:, 1:pad_length+1] then reverse -> y[:, pad_length:0:-1]
            left_pad = y[:, pad_length:0:-1]
        else:
            left_pad = y[:, :0]  # Empty slice

        # Right reflection (excluding the edge element, reversed)
        if pad_length > 0:
            # y[:, -pad_length-1:-1] then reverse -> y[:, -2:-pad_length-2:-1]
            right_pad = y[:, -2:-pad_length - 2:-1]
        else:
            right_pad = y[:, :0]  # Empty slice

        return mx.concatenate([left_pad, y, right_pad], axis=1)
    else:
        raise ValueError(
            f"Unknown pad_mode: '{mode}'. Supported: reflect, constant, edge"
        )


def _frame_signal(y: mx.array, n_fft: int, hop_length: int) -> mx.array:
    """
    Frame signal into overlapping windows.

    Parameters
    ----------
    y : mx.array
        Signal of shape (batch, samples).
    n_fft : int
        Frame length.
    hop_length : int
        Hop length.

    Returns
    -------
    mx.array
        Framed signal of shape (batch, n_frames, n_fft).

    Raises
    ------
    ValueError
        If parameters are invalid or signal is too short.
    """
    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.frame_signal(y, n_fft, hop_length)

    # Fallback to Python implementation
    batch_size, signal_length = y.shape

    # Validate inputs
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if signal_length < n_fft:
        raise ValueError(
            f"Signal length ({signal_length}) must be >= n_fft ({n_fft}). "
            f"Consider using center=True in stft() or padding the signal."
        )

    n_frames = 1 + (signal_length - n_fft) // hop_length

    # Optimized frame construction using as_strided view when possible,
    # falling back to gather-based approach
    #
    # Key insight: We can use mx.as_strided to create a view of the signal
    # with overlapping frames, avoiding index computation and gather operations.
    # This is significantly faster as it's just a metadata operation.

    # Check if we can use strided view (MLX >= 0.5 has as_strided)
    if hasattr(mx, 'as_strided'):
        # Use strided view for zero-copy framing
        # For batched signals, we need to handle each batch
        # y shape: (batch, signal_length) -> frames: (batch, n_frames, n_fft)
        strides = y.strides
        # New strides: (batch_stride, hop_length, 1) in elements
        # as_strided expects strides in elements, not bytes
        new_shape = (batch_size, n_frames, n_fft)
        new_strides = (strides[0], hop_length, 1)
        frames = mx.as_strided(y, shape=new_shape, strides=new_strides)
    else:
        # Fallback: Use optimized gather with pre-computed indices
        # Create frame start indices: [0, hop_length, 2*hop_length, ...]
        frame_starts = mx.arange(n_frames) * hop_length  # (n_frames,)

        # Create sample offsets within each frame: [0, 1, 2, ..., n_fft-1]
        sample_offsets = mx.arange(n_fft)  # (n_fft,)

        # Broadcast to get all indices: (n_frames, n_fft)
        indices = frame_starts[:, None] + sample_offsets[None, :]

        # Gather frames - use indices directly without extra flatten/reshape
        # by leveraging that take gathers along axis=1 for each batch element
        frames = mx.take(y, indices.flatten(), axis=1).reshape(
            batch_size, n_frames, n_fft
        )

    return frames  # (batch, n_frames, n_fft)


def _overlap_add(
    frames: mx.array,
    hop_length: int,
    window: mx.array,
    output_length: int,
) -> mx.array:
    """
    Overlap-add reconstruction with squared window normalization.

    Uses C++ extension when available, otherwise falls back to custom Metal kernel.

    Parameters
    ----------
    frames : mx.array
        Framed signal of shape (batch, n_frames, n_fft).
    hop_length : int
        Hop length.
    window : mx.array
        Window function.
    output_length : int
        Length of output signal.

    Returns
    -------
    mx.array
        Reconstructed signal of shape (batch, output_length).
    """
    # Use C++ extension if available (handles windowing and normalization internally)
    if HAS_CPP_EXT and _ext is not None:
        return _ext.overlap_add(frames, window, hop_length, output_length)

    # Fallback to Python/Metal implementation
    batch_size, n_frames, n_fft = frames.shape

    # Apply window to frames
    windowed_frames = frames * window  # (batch, n_frames, n_fft)

    # Precompute window squared once
    window_sq = window ** 2  # (n_fft,)

    # Use custom Metal kernel for scatter-add
    output, window_sum = _scatter_add_overlap(
        windowed_frames, window_sq, hop_length, output_length
    )

    # Normalize by squared window sum (avoid division by zero).
    # _WINDOW_SUM_EPSILON (1e-8) is chosen to be:
    # - Far below any audible signal level (~1e-5 for 16-bit audio)
    # - Much smaller than the NOLA constraint tolerance (1e-10)
    # - Large enough to prevent numerical instability
    window_sum = mx.maximum(window_sum, _WINDOW_SUM_EPSILON)
    output = output / window_sum

    return output


# Custom Metal kernel for scatter-add operation
_SCATTER_ADD_SOURCE = """
    // Thread indices
    uint batch_idx = thread_position_in_grid.z;
    uint frame_idx = thread_position_in_grid.y;
    uint sample_idx = thread_position_in_grid.x;

    // Compute output position
    uint out_pos = frame_idx * hop_length + sample_idx;

    // Bounds check
    if (out_pos >= output_length) return;
    if (batch_idx >= batch_size) return;
    if (frame_idx >= n_frames) return;
    if (sample_idx >= n_fft) return;

    // Get input index: frames[batch_idx, frame_idx, sample_idx]
    uint frame_flat_idx = batch_idx * n_frames * n_fft
                        + frame_idx * n_fft
                        + sample_idx;

    // Get output index: output[batch_idx, out_pos]
    uint out_flat_idx = batch_idx * output_length + out_pos;

    // Atomic add to output
    float val = frames[frame_flat_idx];
    atomic_fetch_add_explicit(
        (device atomic_float*)&output[out_flat_idx],
        val,
        memory_order_relaxed
    );

    // Atomic add to window_sum (only once per output position, use batch 0)
    if (batch_idx == 0) {
        float ws_val = window_sq[sample_idx];
        atomic_fetch_add_explicit(
            (device atomic_float*)&window_sum[out_pos],
            ws_val,
            memory_order_relaxed
        );
    }
"""

@lru_cache(maxsize=8)
def _get_scatter_add_kernel() -> Any:
    """Thread-safe lazy initialization of the scatter-add kernel."""
    return mx.fast.metal_kernel(
        name="scatter_add_overlap",
        input_names=["frames", "window_sq"],
        output_names=["output", "window_sum"],
        source=_SCATTER_ADD_SOURCE,
        atomic_outputs=True,  # Enable atomic operations on outputs
    )


def _scatter_add_overlap(
    windowed_frames: mx.array,
    window_sq: mx.array,
    hop_length: int,
    output_length: int,
) -> tuple[mx.array, mx.array]:
    """
    Perform scatter-add for overlap-add using custom Metal kernel.

    Parameters
    ----------
    windowed_frames : mx.array
        Windowed frames of shape (batch, n_frames, n_fft).
    window_sq : mx.array
        Squared window of shape (n_fft,).
    hop_length : int
        Hop length between frames.
    output_length : int
        Length of output signal.

    Returns
    -------
    tuple
        (output, window_sum) arrays.
    """
    batch_size, n_frames, n_fft = windowed_frames.shape

    kernel = _get_scatter_add_kernel()

    # Launch kernel with grid covering all (sample, frame, batch) combinations
    # Grid: (n_fft, n_frames, batch_size)
    grid = (n_fft, n_frames, batch_size)
    threadgroup = (min(n_fft, 256), 1, 1)

    outputs = kernel(
        inputs=[windowed_frames, window_sq],
        output_shapes=[(batch_size, output_length), (output_length,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=grid,
        threadgroup=threadgroup,
        template=[
            ("hop_length", hop_length),
            ("output_length", output_length),
            ("batch_size", batch_size),
            ("n_frames", n_frames),
            ("n_fft", n_fft),
        ],
        init_value=0,
    )

    return outputs[0], outputs[1]
