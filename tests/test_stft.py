"""
STFT/ISTFT test suite.

Tests cover:
- Librosa compatibility (output shape, values within tolerance)
- Round-trip reconstruction (STFT -> ISTFT ≈ original)
- Edge cases (short signals, various hop lengths, center padding)
- Batch processing (2D inputs)
- Magnitude and phase extraction
- NOLA constraint verification

Cross-references:
- Mathematical properties: test_mathematical_properties.py
- PyTorch validation: test_torchaudio_crossval.py
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import stft, istft, magnitude, phase, check_nola


class TestSTFT:
    """Tests for STFT function."""

    def test_stft_matches_librosa(self, random_signal):
        """Test that STFT matches librosa output."""
        n_fft = 2048
        hop_length = 512

        # librosa reference
        expected = librosa.stft(random_signal, n_fft=n_fft, hop_length=hop_length)

        # Our implementation
        y_mlx = mx.array(random_signal)
        actual = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        actual_np = np.array(actual)

        # Tolerance accounts for MLX vs SciPy FFT implementation differences
        # MLX FFT differs from SciPy FFT by up to ~6e-5 for n_fft=2048
        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n_fft", [512, 1024, 2048])
    @pytest.mark.parametrize("hop_length", [128, 256, 512])
    def test_stft_various_params(self, random_signal, n_fft, hop_length):
        """Test STFT with various parameter combinations."""
        if hop_length > n_fft:
            pytest.skip("hop_length > n_fft is unusual")

        expected = librosa.stft(random_signal, n_fft=n_fft, hop_length=hop_length)

        y_mlx = mx.array(random_signal)
        actual = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        actual_np = np.array(actual)

        # Tolerance accounts for MLX vs SciPy FFT implementation differences
        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    def test_stft_output_shape(self, random_signal):
        """Test STFT output shape."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(random_signal)
        result = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)

        # Expected shape: (n_fft//2 + 1, n_frames)
        n_frames = 1 + (len(random_signal) + n_fft // 2 * 2 - n_fft) // hop_length
        expected_shape = (n_fft // 2 + 1, n_frames)

        assert result.shape == expected_shape

    def test_stft_output_dtype(self, random_signal):
        """Test STFT output is complex."""
        y_mlx = mx.array(random_signal)
        result = stft(y_mlx)
        assert result.dtype == mx.complex64

    def test_stft_center_false(self, random_signal):
        """Test STFT with center=False."""
        n_fft = 2048
        hop_length = 512

        expected = librosa.stft(
            random_signal, n_fft=n_fft, hop_length=hop_length, center=False
        )

        y_mlx = mx.array(random_signal)
        actual = stft(y_mlx, n_fft=n_fft, hop_length=hop_length, center=False)
        actual_np = np.array(actual)

        # Tolerance accounts for MLX vs SciPy FFT implementation differences
        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    def test_stft_batched(self, batch_signals):
        """Test STFT with batched input."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(batch_signals)
        result = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)

        # Check shape
        assert result.ndim == 3
        assert result.shape[0] == batch_signals.shape[0]

        # Check each batch element matches librosa
        for i in range(batch_signals.shape[0]):
            expected = librosa.stft(
                batch_signals[i], n_fft=n_fft, hop_length=hop_length
            )
            actual = np.array(result[i])
            # Tolerance accounts for MLX vs SciPy FFT differences
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestISTFT:
    """Tests for ISTFT function."""

    def test_round_trip_reconstruction(self, random_signal):
        """Test that istft(stft(x)) ≈ x."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(random_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=len(random_signal))
        reconstructed_np = np.array(reconstructed)

        np.testing.assert_allclose(
            reconstructed_np, random_signal, rtol=1e-4, atol=1e-4
        )

    def test_round_trip_chirp(self, chirp_signal):
        """Test round-trip with chirp signal."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(chirp_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=len(chirp_signal))
        reconstructed_np = np.array(reconstructed)

        np.testing.assert_allclose(
            reconstructed_np, chirp_signal, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("hop_length", [256, 512, 1024])
    def test_round_trip_various_hop_lengths(self, random_signal, hop_length):
        """Test round-trip with various hop lengths."""
        n_fft = 2048

        y_mlx = mx.array(random_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=len(random_signal))
        reconstructed_np = np.array(reconstructed)

        np.testing.assert_allclose(
            reconstructed_np, random_signal, rtol=1e-4, atol=1e-4
        )

    def test_istft_batched(self, batch_signals):
        """Test ISTFT with batched input."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(batch_signals)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=batch_signals.shape[1])
        reconstructed_np = np.array(reconstructed)

        # Check shape
        assert reconstructed_np.shape == batch_signals.shape

        # Check reconstruction quality for each batch element
        np.testing.assert_allclose(
            reconstructed_np, batch_signals, rtol=1e-4, atol=1e-4
        )


class TestMagnitudePhase:
    """Tests for magnitude and phase extraction."""

    def test_magnitude(self, random_signal):
        """Test magnitude extraction."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        mag = magnitude(S)

        # Compare with numpy abs
        expected = np.abs(np.array(S))
        np.testing.assert_allclose(np.array(mag), expected, rtol=1e-6)

    def test_phase(self, random_signal):
        """Test phase extraction."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        ph = phase(S)

        # Compare with numpy angle
        expected = np.angle(np.array(S))
        np.testing.assert_allclose(np.array(ph), expected, rtol=1e-5, atol=1e-5)

    def test_magnitude_phase_reconstruction(self, random_signal):
        """Test that magnitude and phase can reconstruct the complex STFT."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        mag = magnitude(S)
        ph = phase(S)

        # Reconstruct: mag * exp(1j * phase)
        reconstructed = mag * mx.exp(1j * ph)

        np.testing.assert_allclose(
            np.array(reconstructed), np.array(S), rtol=1e-5, atol=1e-5
        )


class TestWinLengthVariations:
    """Tests for win_length < n_fft parameter combinations."""

    @pytest.mark.parametrize("win_length", [512, 1024, 1536])
    def test_stft_win_length_less_than_n_fft(self, random_signal, win_length):
        """Test STFT when win_length < n_fft (window is zero-padded)."""
        n_fft = 2048
        hop_length = 256

        expected = librosa.stft(
            random_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        y_mlx = mx.array(random_signal)
        actual = stft(
            y_mlx,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("win_length", [512, 1024, 1536])
    def test_round_trip_win_length_less_than_n_fft(self, random_signal, win_length):
        """Test STFT->ISTFT round trip when win_length < n_fft."""
        n_fft = 2048
        hop_length = 256

        y_mlx = mx.array(random_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        reconstructed = istft(
            S,
            hop_length=hop_length,
            win_length=win_length,
            length=len(random_signal),
        )
        reconstructed_np = np.array(reconstructed)

        np.testing.assert_allclose(
            reconstructed_np, random_signal, rtol=1e-4, atol=1e-4
        )


class TestISTFTEdgeCases:
    """Tests for ISTFT edge cases."""

    def test_istft_batched_with_length(self, batch_signals):
        """Test batched ISTFT with explicit length parameter."""
        n_fft = 2048
        hop_length = 512
        target_length = batch_signals.shape[1]

        y_mlx = mx.array(batch_signals)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=target_length)
        reconstructed_np = np.array(reconstructed)

        assert reconstructed_np.shape == batch_signals.shape
        np.testing.assert_allclose(
            reconstructed_np, batch_signals, rtol=1e-4, atol=1e-4
        )

    def test_istft_center_false_with_length(self, random_signal):
        """Test ISTFT with center=False and explicit length."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(random_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length, center=False)

        # Without center, we can still specify length for trimming/padding
        reconstructed = istft(
            S, hop_length=hop_length, center=False, length=len(random_signal)
        )
        reconstructed_np = np.array(reconstructed)

        # Reconstruction should match (with possible edge effects)
        assert reconstructed_np.shape[0] == len(random_signal)

    @pytest.mark.parametrize("target_length", [20000, 22050, 25000])
    def test_istft_various_lengths(self, random_signal, target_length):
        """Test ISTFT with various target lengths (shorter, exact, longer)."""
        n_fft = 2048
        hop_length = 512

        y_mlx = mx.array(random_signal)
        S = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(S, hop_length=hop_length, length=target_length)
        reconstructed_np = np.array(reconstructed)

        assert reconstructed_np.shape[0] == target_length


class TestReflectPaddingEdgeCases:
    """Tests for reflect padding edge cases."""

    def test_reflect_padding_short_signal(self):
        """Test reflect padding with signal shorter than n_fft but produces frames."""
        # Signal of 1500 samples with n_fft=2048, center=True adds 1024 padding each side
        # Total padded length = 1500 + 2048 = 3548, which gives (3548-2048)//512 + 1 = 3 frames
        short_signal = np.random.randn(1500).astype(np.float32)

        y_mlx = mx.array(short_signal)
        S = stft(y_mlx, n_fft=2048, hop_length=512, pad_mode="reflect")

        # Should not raise an error
        assert S.shape[0] == 1025  # n_fft//2 + 1

        # Round-trip should work
        reconstructed = istft(S, hop_length=512, length=len(short_signal))
        reconstructed_np = np.array(reconstructed)

        # Allow slightly higher tolerance for edge cases
        np.testing.assert_allclose(
            reconstructed_np, short_signal, rtol=1e-3, atol=1e-3
        )

    def test_reflect_padding_matches_librosa(self, random_signal):
        """Test that reflect padding matches librosa exactly."""
        n_fft = 2048
        hop_length = 512

        expected = librosa.stft(
            random_signal, n_fft=n_fft, hop_length=hop_length, pad_mode="reflect"
        )

        y_mlx = mx.array(random_signal)
        actual = stft(
            y_mlx, n_fft=n_fft, hop_length=hop_length, pad_mode="reflect"
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)


class TestCheckNOLA:
    """Tests for NOLA constraint checking."""

    def test_hann_512_hop_satisfies_nola(self):
        """Test that Hann window with 512 hop satisfies NOLA."""
        assert check_nola("hann", hop_length=512, n_fft=2048)

    def test_hann_1024_hop_satisfies_nola(self):
        """Test that Hann window with 1024 hop (50% overlap) satisfies NOLA."""
        assert check_nola("hann", hop_length=1024, n_fft=2048)

    def test_hamming_satisfies_nola(self):
        """Test that Hamming window satisfies NOLA."""
        assert check_nola("hamming", hop_length=512, n_fft=2048)

    def test_rectangular_satisfies_nola(self):
        """Test that rectangular window satisfies NOLA."""
        assert check_nola("rectangular", hop_length=512, n_fft=2048)
