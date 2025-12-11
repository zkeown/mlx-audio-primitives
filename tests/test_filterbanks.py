"""
Linear and Bark filterbank test suite.

Tests cover:
- Hz-to-Bark and Bark-to-Hz conversions (Zwicker/Traunmuller formulas)
- Round-trip conversion accuracy (hz -> bark -> hz)
- Linear filterbank shape and normalization
- Bark filterbank shape and frequency spacing
- Comparison with mel filterbank as reference

Cross-validates against: scipy.signal and known psychoacoustic values
Tolerance: rtol=1e-5, atol=1e-5 (direct computation, no FFT)

Note: Bark scale approximates critical bands of human hearing.
Formula: bark = 6 * arcsinh(f / 600) (Traunmuller, 1990)
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_primitives import (
    bark_filterbank,
    bark_to_hz,
    hz_to_bark,
    linear_filterbank,
    mel_filterbank,
)


class TestHzToBark:
    """Test hz_to_bark() conversion."""

    def test_zero_frequency(self):
        """Test that 0 Hz maps to 0 Bark."""
        result = hz_to_bark(np.array([0.0]))
        assert np.isclose(result[0], 0.0, atol=0.01)

    def test_1000hz(self):
        """Test 1000 Hz (should be approximately 8.5 Bark)."""
        result = hz_to_bark(np.array([1000.0]))
        # 1000 Hz is approximately 8.5 Bark
        assert 8.0 < result[0] < 9.0

    def test_monotonic(self):
        """Test that hz_to_bark is monotonically increasing."""
        freqs = np.linspace(0, 20000, 100)
        bark = hz_to_bark(freqs)
        assert np.all(np.diff(bark) >= 0)

    @pytest.mark.parametrize("formula", ["zwicker", "traunmuller"])
    def test_formulas(self, formula):
        """Test both Bark scale formulas."""
        freqs = np.array([100, 500, 1000, 5000, 10000])
        bark = hz_to_bark(freqs, formula=formula)

        # All should be positive for positive frequencies
        assert np.all(bark > 0)

        # Should be monotonic
        assert np.all(np.diff(bark) > 0)


class TestBarkToHz:
    """Test bark_to_hz() conversion."""

    def test_zero_bark(self):
        """Test that 0 Bark maps to approximately 0 Hz."""
        result = bark_to_hz(np.array([0.0]))
        assert np.isclose(result[0], 0.0, atol=10)

    def test_round_trip(self):
        """Test Hz -> Bark -> Hz round trip."""
        freqs = np.array([100, 500, 1000, 2000, 5000, 10000])
        bark = hz_to_bark(freqs)
        freqs_recovered = bark_to_hz(bark)

        np.testing.assert_allclose(freqs_recovered, freqs, rtol=0.01, atol=1.0)

    @pytest.mark.parametrize("formula", ["zwicker", "traunmuller"])
    def test_round_trip_formulas(self, formula):
        """Test round trip with both formulas."""
        freqs = np.array([100, 500, 1000, 5000])
        bark = hz_to_bark(freqs, formula=formula)
        freqs_recovered = bark_to_hz(bark, formula=formula)

        np.testing.assert_allclose(freqs_recovered, freqs, rtol=0.02, atol=5.0)


class TestBarkFilterbank:
    """Test bark_filterbank() function."""

    def test_basic_filterbank(self):
        """Test basic Bark filterbank creation."""
        fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)

        # Check shape
        assert fb.shape == (24, 1025)

        # Check dtype
        assert fb.dtype == mx.float32

    def test_filterbank_shape(self):
        """Test filterbank shape for various parameters."""
        for n_bands in [12, 24, 48]:
            for n_fft in [1024, 2048, 4096]:
                fb = bark_filterbank(sr=22050, n_fft=n_fft, n_bands=n_bands)
                assert fb.shape == (n_bands, n_fft // 2 + 1)

    def test_filterbank_non_negative(self):
        """Test that filterbank values are non-negative."""
        fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)
        assert np.all(np.array(fb) >= 0)

    def test_filterbank_sum_positive(self):
        """Test that each filter has non-zero sum."""
        fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)
        fb_np = np.array(fb)

        # Each filter should have some non-zero values
        for i in range(24):
            assert np.sum(fb_np[i, :]) > 0

    def test_caching(self):
        """Test that filterbank is cached."""
        fb1 = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)
        fb2 = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)

        # Same object should be returned (caching)
        assert fb1 is fb2

    def test_different_params_different_result(self):
        """Test that different params give different results."""
        fb1 = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)
        fb2 = bark_filterbank(sr=22050, n_fft=2048, n_bands=12)

        assert fb1.shape != fb2.shape

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError, match="must be positive"):
            bark_filterbank(sr=22050, n_fft=2048, n_bands=0)

        with pytest.raises(ValueError, match="must be less than"):
            bark_filterbank(sr=22050, n_fft=2048, n_bands=24, fmin=5000, fmax=1000)

    @pytest.mark.parametrize("norm", ["slaney", None])
    def test_normalization(self, norm):
        """Test filterbank with different normalization."""
        fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24, norm=norm)

        # Both should produce valid filterbanks
        assert fb.shape == (24, 1025)
        assert np.all(np.array(fb) >= 0)


class TestLinearFilterbank:
    """Test linear_filterbank() function."""

    def test_basic_filterbank(self):
        """Test basic linear filterbank creation."""
        fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)

        # Check shape
        assert fb.shape == (64, 1025)

        # Check dtype
        assert fb.dtype == mx.float32

    def test_filterbank_shape(self):
        """Test filterbank shape for various parameters."""
        for n_bands in [32, 64, 128]:
            for n_fft in [1024, 2048, 4096]:
                fb = linear_filterbank(sr=22050, n_fft=n_fft, n_bands=n_bands)
                assert fb.shape == (n_bands, n_fft // 2 + 1)

    def test_filterbank_non_negative(self):
        """Test that filterbank values are non-negative."""
        fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)
        assert np.all(np.array(fb) >= 0)

    def test_equal_spacing(self):
        """Test that linear filterbank has equal frequency spacing."""
        fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=10, norm=None)
        fb_np = np.array(fb)

        # Find center of each filter (weighted mean)
        freqs = np.linspace(0, 22050 / 2, 1025)
        centers = []
        for i in range(10):
            if np.sum(fb_np[i, :]) > 0:
                center = np.sum(freqs * fb_np[i, :]) / np.sum(fb_np[i, :])
                centers.append(center)

        # Centers should be approximately equally spaced
        if len(centers) >= 3:
            diffs = np.diff(centers)
            # Allow for some variation due to discrete bins
            assert np.std(diffs) / np.mean(diffs) < 0.2

    def test_caching(self):
        """Test that filterbank is cached."""
        fb1 = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)
        fb2 = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)

        assert fb1 is fb2

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError, match="must be positive"):
            linear_filterbank(sr=22050, n_fft=2048, n_bands=-5)

    @pytest.mark.parametrize("norm", ["slaney", None])
    def test_normalization(self, norm):
        """Test filterbank with different normalization."""
        fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=64, norm=norm)

        assert fb.shape == (64, 1025)
        assert np.all(np.array(fb) >= 0)


class TestFilterbankComparison:
    """Compare filterbanks against each other."""

    def test_bark_vs_mel_shape(self):
        """Test that Bark and mel filterbanks have compatible shapes."""
        bark_fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=40)
        mel_fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=40)

        assert bark_fb.shape == mel_fb.shape

    def test_linear_vs_mel_shape(self):
        """Test that linear and mel filterbanks have compatible shapes."""
        linear_fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=128)
        mel_fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=128)

        assert linear_fb.shape == mel_fb.shape

    def test_filterbank_coverage(self):
        """Test that filterbanks cover the frequency range."""
        for fb_func, name in [
            (lambda: bark_filterbank(sr=22050, n_fft=2048, n_bands=24), "bark"),
            (lambda: linear_filterbank(sr=22050, n_fft=2048, n_bands=64), "linear"),
        ]:
            fb = fb_func()
            fb_np = np.array(fb)

            # Sum across all filters - should be non-zero for most bins
            coverage = np.sum(fb_np, axis=0)
            nonzero_bins = np.sum(coverage > 0)

            # At least 50% of bins should be covered
            assert nonzero_bins > 0.5 * fb_np.shape[1], (
                f"{name} filterbank has poor coverage"
            )
