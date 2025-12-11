"""
Griffin-Lim phase reconstruction benchmarks.

Compares mlx_audio_primitives Griffin-Lim against librosa:
- Iterative algorithm: repeated STFT/ISTFT with magnitude constraint
- Momentum acceleration (Perraudin et al. 2013) for faster convergence

Expected speedup: ~1.2-1.3x vs librosa. Performance is dominated by
FFT operations (n_iter * 2 FFTs per iteration). Keeping operations on
GPU avoids CPU<->GPU transfers between iterations.

Iteration configurations tested:
- n_iter=32: Fast, reasonable quality (default)
- n_iter=64: Higher quality, diminishing returns

Use cases: Vocoder-free audio synthesis, spectrogram inversion for debugging.

Run: mlx-audio-bench --suite griffinlim
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio

from .utils import (
    BenchmarkResult,
    generate_test_signal,
    time_function,
)


def benchmark_griffinlim(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
) -> list[BenchmarkResult]:
    """
    Benchmark Griffin-Lim phase reconstruction.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length between frames.
    n_iter : int, default=32
        Number of Griffin-Lim iterations.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results.
    """
    import librosa

    results = []
    signal_np = generate_test_signal(signal_length)

    # Compute magnitude spectrogram
    S_np = np.abs(librosa.stft(signal_np, n_fft=n_fft, hop_length=hop_length))
    S_mx = mx.array(S_np.astype(np.float32))

    # MLX
    mlx_time = time_function(
        lambda: mlx_audio.griffinlim(
            S_mx, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
            length=signal_length, random_state=42
        )
    )
    mlx_result = np.array(
        mlx_audio.griffinlim(
            S_mx, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
            length=signal_length, random_state=42
        )
    )

    # librosa
    librosa_time = time_function(
        lambda: librosa.griffinlim(
            S_np, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
            length=signal_length, random_state=42
        )
    )
    librosa_result = librosa.griffinlim(
        S_np, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
        length=signal_length, random_state=42
    )

    # Compare reconstruction error on magnitude (not correlation, since phase varies)
    S_mlx_recon = np.abs(librosa.stft(mlx_result, n_fft=n_fft, hop_length=hop_length))
    S_librosa_recon = np.abs(
        librosa.stft(librosa_result, n_fft=n_fft, hop_length=hop_length)
    )

    mlx_error = np.mean((S_np - S_mlx_recon) ** 2)
    librosa_error = np.mean((S_np - S_librosa_recon) ** 2)

    results.append(
        BenchmarkResult(
            name=f"griffinlim (n_iter={n_iter})",
            mlx_time_ms=mlx_time,
            reference_time_ms=librosa_time,
            speedup=librosa_time / mlx_time,
            max_abs_error=float(mlx_error),  # Reconstruction MSE
            mean_abs_error=float(mlx_error),
            correlation=float(mlx_error / librosa_error),  # Error ratio
        )
    )

    return results


def benchmark_griffinlim_iterations(
    signal_length: int = 22050,
    n_iter_values: list[int] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark Griffin-Lim with different iteration counts.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_iter_values : list[int], optional
        Iteration counts to benchmark. Default: [8, 16, 32, 64].

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each iteration count.
    """
    import librosa

    if n_iter_values is None:
        n_iter_values = [8, 16, 32, 64]

    results = []
    signal_np = generate_test_signal(signal_length)
    n_fft = 2048
    hop_length = 512

    # Compute magnitude spectrogram once
    S_np = np.abs(librosa.stft(signal_np, n_fft=n_fft, hop_length=hop_length))
    S_mx = mx.array(S_np.astype(np.float32))

    for n_iter in n_iter_values:
        mlx_time = time_function(
            lambda ni=n_iter: mlx_audio.griffinlim(
                S_mx, n_iter=ni, hop_length=hop_length, n_fft=n_fft,
                length=signal_length, random_state=42
            )
        )

        librosa_time = time_function(
            lambda ni=n_iter: librosa.griffinlim(
                S_np, n_iter=ni, hop_length=hop_length, n_fft=n_fft,
                length=signal_length, random_state=42
            )
        )

        # Compute reconstruction errors
        mlx_result = np.array(
            mlx_audio.griffinlim(
                S_mx, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
                length=signal_length, random_state=42
            )
        )
        S_mlx_recon = np.abs(
            librosa.stft(mlx_result, n_fft=n_fft, hop_length=hop_length)
        )
        mlx_error = np.mean((S_np - S_mlx_recon) ** 2)

        results.append(
            BenchmarkResult(
                name=f"griffinlim scaling (n_iter={n_iter})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                max_abs_error=float(mlx_error),
                mean_abs_error=float(mlx_error),
                correlation=1.0,  # Placeholder
            )
        )

    return results
