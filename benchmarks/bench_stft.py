"""STFT and ISTFT benchmarks comparing MLX vs librosa and torchaudio."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio

from .utils import (
    BenchmarkResult,
    compute_accuracy,
    generate_test_signal,
    time_function,
)


def _warmup_mlx() -> None:
    """Warmup MLX to avoid cold-start overhead in benchmarks."""
    # Run a small STFT to trigger JIT compilation
    dummy = mx.array(np.zeros(1024, dtype=np.float32))
    result = mlx_audio.stft(dummy, n_fft=256, hop_length=64)
    mx.eval(result)


def benchmark_stft(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    compare_librosa: bool = True,
    compare_torch: bool = True,
) -> list[BenchmarkResult]:
    """
    Benchmark STFT against reference implementations.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length between frames.
    compare_librosa : bool, default=True
        Compare against librosa.
    compare_torch : bool, default=True
        Compare against torchaudio.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each comparison.
    """
    # Warmup MLX before benchmarking
    _warmup_mlx()

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    # MLX timing
    mlx_time = time_function(
        lambda: mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)
    )
    mlx_result = np.array(
        mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)
    )

    if compare_librosa:
        import librosa

        librosa_time = time_function(
            lambda: librosa.stft(signal_np, n_fft=n_fft, hop_length=hop_length)
        )
        librosa_result = librosa.stft(signal_np, n_fft=n_fft, hop_length=hop_length)
        accuracy = compute_accuracy(np.abs(mlx_result), np.abs(librosa_result))
        results.append(
            BenchmarkResult(
                name=f"STFT vs librosa (n_fft={n_fft})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    if compare_torch:
        import torch

        signal_torch = torch.from_numpy(signal_np)
        window = torch.hann_window(n_fft)
        # Use pad_mode='constant' to match librosa/MLX defaults
        torch_time = time_function(
            lambda: torch.stft(
                signal_torch,
                n_fft,
                hop_length,
                window=window,
                return_complex=True,
                center=True,
                pad_mode="constant",
            )
        )
        torch_result = torch.stft(
            signal_torch,
            n_fft,
            hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="constant",
        ).numpy()
        accuracy = compute_accuracy(np.abs(mlx_result), np.abs(torch_result))
        results.append(
            BenchmarkResult(
                name=f"STFT vs torch (n_fft={n_fft})",
                mlx_time_ms=mlx_time,
                reference_time_ms=torch_time,
                speedup=torch_time / mlx_time,
                **accuracy,
            )
        )

    return results


def benchmark_istft(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[BenchmarkResult]:
    """
    Benchmark ISTFT round-trip reconstruction.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length between frames.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for ISTFT.
    """
    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    # Compute STFT first
    stft_mx = mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)
    mx.eval(stft_mx)

    # MLX ISTFT timing
    mlx_time = time_function(
        lambda: mlx_audio.istft(stft_mx, hop_length=hop_length, length=signal_length)
    )
    reconstructed = np.array(
        mlx_audio.istft(stft_mx, hop_length=hop_length, length=signal_length)
    )

    # Compute reconstruction error
    error = np.abs(signal_np - reconstructed)
    results.append(
        BenchmarkResult(
            name=f"ISTFT round-trip (n_fft={n_fft})",
            mlx_time_ms=mlx_time,
            reference_time_ms=0.0,  # No reference comparison
            speedup=0.0,
            max_abs_error=float(np.max(error)),
            mean_abs_error=float(np.mean(error)),
            correlation=float(np.corrcoef(signal_np, reconstructed)[0, 1]),
        )
    )

    return results


def benchmark_stft_scaling(
    n_fft_values: list[int] | None = None,
    signal_length: int = 44100,
) -> list[BenchmarkResult]:
    """
    Benchmark STFT performance across different FFT sizes.

    Parameters
    ----------
    n_fft_values : list[int], optional
        FFT sizes to benchmark. Default: [512, 1024, 2048, 4096].
    signal_length : int, default=44100
        Length of test signal in samples (2 seconds at 22050 Hz).

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each FFT size.
    """
    if n_fft_values is None:
        n_fft_values = [512, 1024, 2048, 4096]

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    for n_fft in n_fft_values:
        hop_length = n_fft // 4

        # MLX timing
        mlx_time = time_function(
            lambda nf=n_fft, hl=hop_length: mlx_audio.stft(
                signal_mx, n_fft=nf, hop_length=hl
            )
        )

        # librosa timing
        import librosa

        librosa_time = time_function(
            lambda nf=n_fft, hl=hop_length: librosa.stft(
                signal_np, n_fft=nf, hop_length=hl
            )
        )

        # Accuracy
        mlx_result = np.array(
            mlx_audio.stft(signal_mx, n_fft=n_fft, hop_length=hop_length)
        )
        librosa_result = librosa.stft(signal_np, n_fft=n_fft, hop_length=hop_length)
        accuracy = compute_accuracy(np.abs(mlx_result), np.abs(librosa_result))

        results.append(
            BenchmarkResult(
                name=f"STFT scaling (n_fft={n_fft})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    return results
