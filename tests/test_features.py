"""
Spectral features test suite.

Tests cover:
- librosa.feature compatibility (values within tolerance)
- Edge cases (short signals, various n_fft/hop_length)
- Batch processing (1D and 2D inputs)
- Physical constraints (centroid in valid frequency range, flatness in [0, 1])

Cross-validates against: librosa.feature.spectral_* functions
Tolerance: rtol=1e-4, atol=1e-4 (log operations and FFT differences)

Note: spectral_contrast and spectral_rolloff use numpy internally for
operations not supported natively in MLX (searchsorted, complex indexing).
"""
import librosa
import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_primitives import (
    magnitude,
    spectral_bandwidth,
    spectral_centroid,
    spectral_contrast,
    spectral_flatness,
    spectral_rolloff,
    stft,
    zero_crossing_rate,
)


class TestSpectralCentroid:
    """Test spectral_centroid() function."""

    def test_basic_centroid(self, random_signal):
        """Test basic spectral centroid matches librosa."""
        y_mx = mx.array(random_signal)
        result = spectral_centroid(y_mx, sr=22050, n_fft=2048, hop_length=512)

        expected = librosa.feature.spectral_centroid(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("n_fft", [1024, 2048])
    @pytest.mark.parametrize("hop_length", [256, 512])
    def test_various_params(self, random_signal, n_fft, hop_length):
        """Test with various parameters."""
        y_mx = mx.array(random_signal)
        result = spectral_centroid(y_mx, sr=22050, n_fft=n_fft, hop_length=hop_length)

        expected = librosa.feature.spectral_centroid(
            y=random_signal, sr=22050, n_fft=n_fft, hop_length=hop_length
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = spectral_centroid(y_mx, sr=22050, n_fft=2048, hop_length=512)

        # Shape should be (1, n_frames)
        assert result.shape[0] == 1

    def test_from_spectrogram(self, random_signal):
        """Test computing from pre-computed spectrogram."""
        y_mx = mx.array(random_signal)
        S = magnitude(stft(y_mx, n_fft=2048, hop_length=512))

        result = spectral_centroid(S=S, sr=22050, n_fft=2048)

        expected = librosa.feature.spectral_centroid(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = spectral_centroid(y_mx, sr=22050, n_fft=2048, hop_length=512)

        assert result.shape[0] == 4  # Batch size preserved

    def test_centroid_range(self, chirp_signal):
        """Test centroid is within valid frequency range."""
        y_mx = mx.array(chirp_signal)
        result = spectral_centroid(y_mx, sr=22050, n_fft=2048, hop_length=512)

        result_np = np.array(result)

        # Centroid should be between 0 and Nyquist
        assert np.all(result_np >= 0)
        assert np.all(result_np <= 22050 / 2)


class TestSpectralBandwidth:
    """Test spectral_bandwidth() function."""

    def test_basic_bandwidth(self, random_signal):
        """Test basic spectral bandwidth matches librosa."""
        y_mx = mx.array(random_signal)
        result = spectral_bandwidth(y_mx, sr=22050, n_fft=2048, hop_length=512)

        expected = librosa.feature.spectral_bandwidth(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_various_p(self, random_signal, p):
        """Test with different power values."""
        y_mx = mx.array(random_signal)
        result = spectral_bandwidth(y_mx, sr=22050, n_fft=2048, hop_length=512, p=p)

        expected = librosa.feature.spectral_bandwidth(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512, p=p
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = spectral_bandwidth(y_mx, sr=22050, n_fft=2048, hop_length=512)

        assert result.shape[0] == 1

    def test_bandwidth_non_negative(self, random_signal):
        """Test bandwidth is non-negative."""
        y_mx = mx.array(random_signal)
        result = spectral_bandwidth(y_mx, sr=22050)

        assert np.all(np.array(result) >= 0)


class TestSpectralRolloff:
    """Test spectral_rolloff() function."""

    def test_basic_rolloff(self, random_signal):
        """Test basic spectral rolloff matches librosa."""
        y_mx = mx.array(random_signal)
        result = spectral_rolloff(y_mx, sr=22050, n_fft=2048, hop_length=512)

        expected = librosa.feature.spectral_rolloff(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=100  # Allow 100 Hz tolerance
        )

    @pytest.mark.parametrize("roll_percent", [0.5, 0.85, 0.95])
    def test_various_percent(self, random_signal, roll_percent):
        """Test with different rolloff percentages."""
        y_mx = mx.array(random_signal)
        result = spectral_rolloff(
            y_mx, sr=22050, n_fft=2048, hop_length=512, roll_percent=roll_percent
        )

        expected = librosa.feature.spectral_rolloff(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512, roll_percent=roll_percent
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=100
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = spectral_rolloff(y_mx, sr=22050, n_fft=2048, hop_length=512)

        assert result.shape[0] == 1

    def test_rolloff_range(self, random_signal):
        """Test rolloff is within valid frequency range."""
        y_mx = mx.array(random_signal)
        result = spectral_rolloff(y_mx, sr=22050)

        result_np = np.array(result)
        assert np.all(result_np >= 0)
        assert np.all(result_np <= 22050 / 2)

    def test_invalid_roll_percent(self, random_signal):
        """Test error for invalid roll_percent."""
        y_mx = mx.array(random_signal)

        with pytest.raises(ValueError):
            spectral_rolloff(y_mx, sr=22050, roll_percent=-0.1)

        with pytest.raises(ValueError):
            spectral_rolloff(y_mx, sr=22050, roll_percent=1.5)


class TestSpectralFlatness:
    """Test spectral_flatness() function."""

    def test_basic_flatness(self, random_signal):
        """Test basic spectral flatness matches librosa."""
        y_mx = mx.array(random_signal)
        result = spectral_flatness(y_mx, n_fft=2048, hop_length=512)

        expected = librosa.feature.spectral_flatness(
            y=random_signal, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = spectral_flatness(y_mx, n_fft=2048, hop_length=512)

        assert result.shape[0] == 1

    def test_flatness_range(self, random_signal):
        """Test flatness is in [0, 1] range."""
        y_mx = mx.array(random_signal)
        result = spectral_flatness(y_mx)

        result_np = np.array(result)
        assert np.all(result_np >= 0)
        assert np.all(result_np <= 1)

    def test_noise_high_flatness(self):
        """Test that white noise has high flatness."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(22050).astype(np.float32)

        result = spectral_flatness(mx.array(noise))
        result_np = np.array(result)

        # White noise should have flatness close to 1
        assert np.mean(result_np) > 0.5

    def test_sine_low_flatness(self):
        """Test that pure sine wave has low flatness."""
        t = np.linspace(0, 1, 22050, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)

        result = spectral_flatness(mx.array(sine))
        result_np = np.array(result)

        # Pure tone should have low flatness
        assert np.mean(result_np) < 0.3


class TestSpectralContrast:
    """Test spectral_contrast() function."""

    def test_basic_contrast(self, random_signal):
        """Test basic spectral contrast matches librosa."""
        y_mx = mx.array(random_signal)
        result = spectral_contrast(
            y_mx, sr=22050, n_fft=2048, hop_length=512, n_bands=6
        )

        expected = librosa.feature.spectral_contrast(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512, n_bands=6
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=0.1, atol=0.5
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = spectral_contrast(y_mx, sr=22050, n_fft=2048, hop_length=512, n_bands=6)

        # Should have n_bands + 1 features
        assert result.shape[0] == 7

    @pytest.mark.parametrize("n_bands", [4, 6, 8])
    def test_various_n_bands(self, random_signal, n_bands):
        """Test with various number of bands."""
        y_mx = mx.array(random_signal)
        result = spectral_contrast(y_mx, sr=22050, n_bands=n_bands)

        assert result.shape[0] == n_bands + 1


class TestZeroCrossingRate:
    """Test zero_crossing_rate() function."""

    def test_basic_zcr(self, random_signal):
        """Test basic ZCR matches librosa."""
        y_mx = mx.array(random_signal)
        result = zero_crossing_rate(y_mx, frame_length=2048, hop_length=512)

        expected = librosa.feature.zero_crossing_rate(
            y=random_signal, frame_length=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("frame_length", [1024, 2048])
    @pytest.mark.parametrize("hop_length", [256, 512])
    def test_various_params(self, random_signal, frame_length, hop_length):
        """Test with various parameters."""
        y_mx = mx.array(random_signal)
        result = zero_crossing_rate(y_mx, frame_length=frame_length, hop_length=hop_length)

        expected = librosa.feature.zero_crossing_rate(
            y=random_signal, frame_length=frame_length, hop_length=hop_length
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = zero_crossing_rate(y_mx, frame_length=2048, hop_length=512)

        # Shape should be (1, n_frames)
        assert result.shape[0] == 1

    def test_zcr_range(self, random_signal):
        """Test ZCR is in [0, 1] range."""
        y_mx = mx.array(random_signal)
        result = zero_crossing_rate(y_mx)

        result_np = np.array(result)
        assert np.all(result_np >= 0)
        assert np.all(result_np <= 1)

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = zero_crossing_rate(y_mx, frame_length=2048, hop_length=512)

        assert result.shape[0] == 4  # Batch size preserved

    def test_high_freq_high_zcr(self):
        """Test that high frequency has high ZCR."""
        t = np.linspace(0, 1, 22050, dtype=np.float32)
        high_freq = np.sin(2 * np.pi * 5000 * t)  # 5000 Hz
        low_freq = np.sin(2 * np.pi * 100 * t)    # 100 Hz

        zcr_high = zero_crossing_rate(mx.array(high_freq))
        zcr_low = zero_crossing_rate(mx.array(low_freq))

        # High frequency should have higher ZCR
        assert np.mean(np.array(zcr_high)) > np.mean(np.array(zcr_low))
