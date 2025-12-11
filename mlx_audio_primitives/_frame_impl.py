"""
Shared optimized framing implementation.

Provides the core framing logic used by both stft.py and framing.py.
Uses the fastest available method:
1. C++ extension (if available)
2. mx.as_strided (MLX >= 0.5, zero-copy)
3. Gather-based fallback (always works)
"""

from __future__ import annotations

import mlx.core as mx

from ._extension import HAS_CPP_EXT, _ext


def frame_signal_batched(
    y: mx.array,
    frame_length: int,
    hop_length: int,
) -> mx.array:
    """
    Frame a batched signal into overlapping windows.

    This is the core framing implementation used throughout the library.
    It automatically selects the fastest available method.

    Parameters
    ----------
    y : mx.array
        Signal of shape (batch, samples).
    frame_length : int
        Length of each frame in samples.
    hop_length : int
        Number of samples between frame starts.

    Returns
    -------
    mx.array
        Framed signal of shape (batch, n_frames, frame_length).

    Raises
    ------
    ValueError
        If parameters are invalid or signal is too short.
    """
    batch_size, signal_length = y.shape

    # Validate inputs
    if frame_length <= 0:
        raise ValueError(f"frame_length must be positive, got {frame_length}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if signal_length < frame_length:
        raise ValueError(
            f"Signal length ({signal_length}) must be >= frame_length ({frame_length}). "
            f"Consider padding the signal."
        )

    n_frames = 1 + (signal_length - frame_length) // hop_length

    # Priority 1: Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.frame_signal(y, frame_length, hop_length)

    # Priority 2: Use strided view for zero-copy framing (MLX >= 0.5)
    if hasattr(mx, "as_strided"):
        strides = y.strides
        # New strides: (batch_stride, hop_length, 1) in elements
        new_shape = (batch_size, n_frames, frame_length)
        new_strides = (strides[0], hop_length, 1)
        return mx.as_strided(y, shape=new_shape, strides=new_strides)

    # Priority 3: Fallback to gather-based approach
    frame_starts = mx.arange(n_frames) * hop_length
    sample_offsets = mx.arange(frame_length)
    indices = frame_starts[:, None] + sample_offsets[None, :]
    return mx.take(y, indices.flatten(), axis=1).reshape(
        batch_size, n_frames, frame_length
    )
