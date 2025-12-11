"""Mel spectrogram benchmarks."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

import mlx_audio_primitives as mlx_audio
from mlx_audio_primitives import mel as mel_module

from .utils import (
    BenchmarkResult,
    compute_accuracy,
    generate_test_signal,
    time_function,
)


def benchmark_mel_filterbank(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels_list: list[int] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark mel filterbank construction.

    Parameters
    ----------
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    n_mels_list : list[int], optional
        List of mel band counts to benchmark. Default: [40, 80, 128].

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each n_mels value.
    """
    if n_mels_list is None:
        n_mels_list = [40, 80, 128]

    results = []

    for n_mels in n_mels_list:
        # Clear MLX cache to measure cold start
        mel_module._compute_mel_filterbank_np.cache_clear()

        mlx_time = time_function(
            lambda n=n_mels: mlx_audio.mel_filterbank(sr, n_fft, n_mels=n)
        )
        mlx_fb = np.array(mlx_audio.mel_filterbank(sr, n_fft, n_mels=n_mels))

        # librosa
        import librosa

        librosa_time = time_function(
            lambda n=n_mels: librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n)
        )
        librosa_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

        accuracy = compute_accuracy(mlx_fb, librosa_fb)
        results.append(
            BenchmarkResult(
                name=f"mel_filterbank (n_mels={n_mels})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    return results


def benchmark_melspectrogram(
    signal_length: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    compare_librosa: bool = True,
    compare_torch: bool = True,
) -> list[BenchmarkResult]:
    """
    Benchmark full mel spectrogram pipeline.

    Parameters
    ----------
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length between frames.
    n_mels : int, default=128
        Number of mel bands.
    compare_librosa : bool, default=True
        Compare against librosa.
    compare_torch : bool, default=True
        Compare against torchaudio.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each comparison.
    """
    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    # MLX
    mlx_time = time_function(
        lambda: mlx_audio.melspectrogram(
            signal_mx, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    )
    mlx_mel = np.array(
        mlx_audio.melspectrogram(
            signal_mx, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    )

    if compare_librosa:
        import librosa

        librosa_time = time_function(
            lambda: librosa.feature.melspectrogram(
                y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
        )
        librosa_mel = librosa.feature.melspectrogram(
            y=signal_np, sr=22050, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        accuracy = compute_accuracy(mlx_mel, librosa_mel)
        results.append(
            BenchmarkResult(
                name=f"melspectrogram vs librosa (n_mels={n_mels})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    if compare_torch:
        import torch
        import torchaudio

        signal_torch = torch.from_numpy(signal_np)
        # Use matching parameters: pad_mode='constant', mel_scale='slaney', norm='slaney'
        # to match librosa/MLX defaults for meaningful accuracy comparison
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            pad_mode="constant",
            mel_scale="slaney",
            norm="slaney",
        )
        torch_time = time_function(lambda: mel_transform(signal_torch))
        torch_mel = mel_transform(signal_torch).numpy()

        accuracy = compute_accuracy(mlx_mel, torch_mel)
        results.append(
            BenchmarkResult(
                name=f"melspectrogram vs torch (n_mels={n_mels})",
                mlx_time_ms=mlx_time,
                reference_time_ms=torch_time,
                speedup=torch_time / mlx_time,
                **accuracy,
            )
        )

    return results


def benchmark_mel_scaling(
    n_mels_values: list[int] | None = None,
    signal_length: int = 22050,
    n_fft: int = 2048,
) -> list[BenchmarkResult]:
    """
    Benchmark mel spectrogram performance across different mel band counts.

    Parameters
    ----------
    n_mels_values : list[int], optional
        Mel band counts to benchmark. Default: [40, 80, 128, 256].
    signal_length : int, default=22050
        Length of test signal in samples.
    n_fft : int, default=2048
        FFT size.

    Returns
    -------
    list[BenchmarkResult]
        Benchmark results for each n_mels value.
    """
    if n_mels_values is None:
        n_mels_values = [40, 80, 128, 256]

    results = []
    signal_np = generate_test_signal(signal_length)
    signal_mx = mx.array(signal_np)

    for n_mels in n_mels_values:
        # Clear cache
        mel_module._compute_mel_filterbank_np.cache_clear()

        # MLX timing
        mlx_time = time_function(
            lambda nm=n_mels: mlx_audio.melspectrogram(
                signal_mx, n_fft=n_fft, n_mels=nm
            )
        )

        # librosa timing
        import librosa

        librosa_time = time_function(
            lambda nm=n_mels: librosa.feature.melspectrogram(
                y=signal_np, sr=22050, n_fft=n_fft, n_mels=nm
            )
        )

        # Accuracy
        mlx_mel = np.array(
            mlx_audio.melspectrogram(signal_mx, n_fft=n_fft, n_mels=n_mels)
        )
        librosa_mel = librosa.feature.melspectrogram(
            y=signal_np, sr=22050, n_fft=n_fft, n_mels=n_mels
        )
        accuracy = compute_accuracy(mlx_mel, librosa_mel)

        results.append(
            BenchmarkResult(
                name=f"melspec scaling (n_mels={n_mels})",
                mlx_time_ms=mlx_time,
                reference_time_ms=librosa_time,
                speedup=librosa_time / mlx_time,
                **accuracy,
            )
        )

    return results
