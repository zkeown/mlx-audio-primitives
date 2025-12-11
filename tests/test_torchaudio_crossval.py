"""
Cross-validation tests against torchaudio.

These tests compare our implementations against PyTorch's torchaudio
to provide independent verification beyond librosa/scipy.
"""
import mlx.core as mx
import numpy as np
import pytest
import torch
import torchaudio

from mlx_audio_primitives import (
    get_window,
    istft,
    magnitude,
    melspectrogram,
    stft,
)


class TestSTFTCrossValidation:
    """Cross-validate STFT against torchaudio."""

    def test_stft_matches_torchaudio(self):
        """Test that STFT matches torchaudio output."""
        np.random.seed(42)
        x = np.random.randn(4096).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        # Our implementation
        x_mlx = mx.array(x)
        mlx_result = np.array(stft(x_mlx, n_fft=n_fft, hop_length=hop_length,
                                    center=False))

        # torchaudio
        x_torch = torch.from_numpy(x)
        window = torch.hann_window(n_fft)
        torch_result = torch.stft(
            x_torch, n_fft=n_fft, hop_length=hop_length,
            window=window, center=False, return_complex=True
        ).numpy()

        # Tolerance for cross-library comparison
        np.testing.assert_allclose(mlx_result, torch_result, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n_fft", [512, 1024, 2048])
    def test_stft_various_n_fft(self, n_fft):
        """Test STFT with various n_fft values against torchaudio."""
        np.random.seed(42)
        x = np.random.randn(8192).astype(np.float32)
        hop_length = n_fft // 4

        x_mlx = mx.array(x)
        mlx_result = np.array(stft(x_mlx, n_fft=n_fft, hop_length=hop_length,
                                    center=False))

        x_torch = torch.from_numpy(x)
        window = torch.hann_window(n_fft)
        torch_result = torch.stft(
            x_torch, n_fft=n_fft, hop_length=hop_length,
            window=window, center=False, return_complex=True
        ).numpy()

        np.testing.assert_allclose(mlx_result, torch_result, rtol=1e-4, atol=1e-4)

    def test_stft_magnitude_matches_torchaudio(self):
        """Test that magnitude spectrogram matches torchaudio."""
        np.random.seed(42)
        x = np.random.randn(4096).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        # Our implementation
        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length, center=False)
        mlx_mag = np.array(magnitude(S))

        # torchaudio
        x_torch = torch.from_numpy(x)
        window = torch.hann_window(n_fft)
        torch_stft = torch.stft(
            x_torch, n_fft=n_fft, hop_length=hop_length,
            window=window, center=False, return_complex=True
        )
        torch_mag = torch.abs(torch_stft).numpy()

        np.testing.assert_allclose(mlx_mag, torch_mag, rtol=1e-4, atol=1e-4)


class TestWindowCrossValidation:
    """Cross-validate window functions against PyTorch."""

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048])
    def test_hann_window_periodic(self, n):
        """Test periodic Hann window against PyTorch."""
        mlx_window = np.array(get_window("hann", n, fftbins=True))
        torch_window = torch.hann_window(n, periodic=True).numpy()

        # Note: Small differences due to float32 computation order
        np.testing.assert_allclose(mlx_window, torch_window, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048])
    def test_hann_window_symmetric(self, n):
        """Test symmetric Hann window against PyTorch."""
        mlx_window = np.array(get_window("hann", n, fftbins=False))
        torch_window = torch.hann_window(n, periodic=False).numpy()

        np.testing.assert_allclose(mlx_window, torch_window, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048])
    def test_hamming_window_periodic(self, n):
        """Test periodic Hamming window against PyTorch."""
        mlx_window = np.array(get_window("hamming", n, fftbins=True))
        torch_window = torch.hamming_window(n, periodic=True).numpy()

        np.testing.assert_allclose(mlx_window, torch_window, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048])
    def test_blackman_window_periodic(self, n):
        """Test periodic Blackman window against PyTorch."""
        mlx_window = np.array(get_window("blackman", n, fftbins=True))
        torch_window = torch.blackman_window(n, periodic=True).numpy()

        np.testing.assert_allclose(mlx_window, torch_window, rtol=1e-5, atol=1e-6)


class TestMelSpectrogramCrossValidation:
    """Cross-validate mel spectrogram against torchaudio."""

    def test_melspectrogram_matches_torchaudio(self):
        """Test mel spectrogram against torchaudio."""
        np.random.seed(42)
        x = np.random.randn(22050).astype(np.float32)

        sr = 22050
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        # Our implementation
        x_mlx = mx.array(x)
        mlx_mel = np.array(melspectrogram(
            x_mlx, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0, center=False
        ))

        # torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=False,
            norm='slaney',
            mel_scale='slaney',
        )
        x_torch = torch.from_numpy(x)
        torch_mel = mel_transform(x_torch).numpy()

        # Compare actual values with tight tolerances
        np.testing.assert_allclose(mlx_mel, torch_mel, rtol=1e-3, atol=1e-5)


class TestWindowSymmetryFixed:
    """
    Tests verifying that window symmetry issues have been fixed.

    MLX windows are now computed in float64 and cast to float32,
    matching scipy's approach for perfect symmetry.
    """

    def test_window_symmetry_matches_scipy(self):
        """
        Verify that MLX windows are now perfectly symmetric like scipy.
        """
        import scipy.signal
        n = 1024

        # MLX now computes in float64, casts to float32
        mlx_hann = np.array(get_window("hann", n, fftbins=False))
        mlx_asymmetry = np.max(np.abs(mlx_hann - mlx_hann[::-1]))

        # Scipy computes in float64
        scipy_hann = scipy.signal.get_window("hann", n, fftbins=False)
        scipy_hann_32 = scipy_hann.astype(np.float32)
        scipy_asymmetry = np.max(np.abs(scipy_hann_32 - scipy_hann_32[::-1]))

        # Both should be perfectly symmetric
        assert mlx_asymmetry == 0, f"MLX should be symmetric, got {mlx_asymmetry}"
        assert scipy_asymmetry == 0, "Scipy should be symmetric"

        # MLX should match scipy very closely (within float32 epsilon)
        # Note: Exact bitwise match is not guaranteed due to different computation order,
        # but both use float64 internally so the difference should be minimal
        max_diff = np.max(np.abs(mlx_hann - scipy_hann_32))
        assert max_diff < 2e-7, f"MLX should match scipy closely, diff={max_diff}"

    def test_mlx_more_precise_than_torch(self):
        """
        Verify that MLX windows are now more precise than PyTorch.

        PyTorch computes in float32, so it has asymmetry.
        MLX now computes in float64, so it's perfectly symmetric.
        """
        n = 1024

        mlx_hann = np.array(get_window("hann", n, fftbins=False))
        torch_hann = torch.hann_window(n, periodic=False).numpy()

        mlx_asymmetry = np.max(np.abs(mlx_hann - mlx_hann[::-1]))
        torch_asymmetry = np.max(np.abs(torch_hann - torch_hann[::-1]))

        # MLX should be perfectly symmetric now
        assert mlx_asymmetry == 0, "MLX should be symmetric"

        # PyTorch still has asymmetry (computes in float32)
        assert torch_asymmetry > 0, "PyTorch should have some asymmetry"

        # But they should still be close overall
        max_diff = np.max(np.abs(mlx_hann - torch_hann))
        assert max_diff < 1e-6

    def test_blackman_non_negative(self):
        """
        Verify that Blackman window is now non-negative.

        Blackman window: w[k] = 0.42 - 0.5*cos(...) + 0.08*cos(...)
        At k=0: w[0] = 0.42 - 0.5 + 0.08 = 0.0 exactly

        With float64 computation, this should be exactly 0 or tiny positive.
        """
        n = 1024

        mlx_blackman = np.array(get_window("blackman", n, fftbins=True))
        torch_blackman = torch.blackman_window(n, periodic=True).numpy()

        mlx_min = mlx_blackman.min()
        torch_min = torch_blackman.min()

        # MLX should be non-negative (computed in float64)
        assert mlx_min >= 0, f"MLX Blackman should be non-negative: {mlx_min}"

        # PyTorch may still have tiny negatives (computed in float32)
        assert torch_min > -1e-7, f"Torch Blackman min {torch_min} too negative"


class TestSTFTRoundTrip:
    """Test STFT round-trip against torchaudio."""

    def test_round_trip_matches_torchaudio_quality(self):
        """Test that our STFT round-trip quality matches torchaudio's."""
        np.random.seed(42)
        x = np.random.randn(4096).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        # Our round-trip
        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        x_reconstructed_mlx = np.array(istft(S, hop_length=hop_length, length=len(x)))

        # torchaudio round-trip
        x_torch = torch.from_numpy(x)
        window = torch.hann_window(n_fft)
        S_torch = torch.stft(
            x_torch, n_fft=n_fft, hop_length=hop_length,
            window=window, center=True, return_complex=True
        )
        x_reconstructed_torch = torch.istft(
            S_torch, n_fft=n_fft, hop_length=hop_length,
            window=window, center=True, length=len(x)
        ).numpy()

        # Both should reconstruct well
        mlx_error = np.max(np.abs(x_reconstructed_mlx - x))
        torch_error = np.max(np.abs(x_reconstructed_torch - x))

        assert mlx_error < 1e-4, f"MLX reconstruction error {mlx_error} too large"
        assert torch_error < 1e-4, f"Torch reconstruction error {torch_error} too large"

        # Errors should be similar magnitude
        error_ratio = mlx_error / max(torch_error, 1e-10)
        assert 0.1 < error_ratio < 10, \
            f"MLX/torch error ratio {error_ratio} indicates quality difference"
