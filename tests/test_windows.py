"""
Window functions test suite.

Tests cover:
- Scipy compatibility for Hann, Hamming, Blackman, Bartlett windows
- Periodic (DFT-even) vs symmetric window modes
- Custom window array passthrough
- Window aliases (hanning, triangular, boxcar, ones)
- Shape and dtype validation
- Error handling for unknown window types
"""
import numpy as np
import pytest
import scipy.signal
import mlx.core as mx

from mlx_audio_primitives import get_window


class TestGetWindow:
    """Tests for get_window function."""

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman", "bartlett"])
    @pytest.mark.parametrize("n_fft", [256, 512, 1024, 2048])
    def test_window_matches_scipy_periodic(self, window_type, n_fft):
        """Test that windows match scipy output for periodic (fftbins=True)."""
        # Our implementation
        mlx_window = get_window(window_type, n_fft, fftbins=True)
        mlx_np = np.array(mlx_window)

        # scipy reference (periodic)
        scipy_window = scipy.signal.get_window(window_type, n_fft, fftbins=True)

        np.testing.assert_allclose(mlx_np, scipy_window, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman", "bartlett"])
    @pytest.mark.parametrize("n_fft", [256, 512, 1024, 2048])
    def test_window_matches_scipy_symmetric(self, window_type, n_fft):
        """Test that windows match scipy output for symmetric (fftbins=False)."""
        # Our implementation
        mlx_window = get_window(window_type, n_fft, fftbins=False)
        mlx_np = np.array(mlx_window)

        # scipy reference (symmetric)
        scipy_window = scipy.signal.get_window(window_type, n_fft, fftbins=False)

        np.testing.assert_allclose(mlx_np, scipy_window, rtol=1e-5, atol=1e-5)

    def test_rectangular_window(self):
        """Test rectangular/boxcar window."""
        n_fft = 1024
        window = get_window("rectangular", n_fft)
        expected = np.ones(n_fft, dtype=np.float32)
        np.testing.assert_allclose(np.array(window), expected, rtol=1e-6)

    def test_window_array_passthrough(self):
        """Test that custom window arrays are passed through."""
        custom_window = mx.array(np.ones(1024, dtype=np.float32) * 0.5)
        result = get_window(custom_window, 1024)
        np.testing.assert_allclose(np.array(result), np.array(custom_window))

    def test_window_array_wrong_length_raises(self):
        """Test that wrong-length window arrays raise ValueError."""
        custom_window = mx.array(np.ones(512, dtype=np.float32))
        with pytest.raises(ValueError, match="must match n_fft"):
            get_window(custom_window, 1024)

    def test_unknown_window_raises(self):
        """Test that unknown window types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown window type"):
            get_window("unknown_window", 1024)

    def test_window_shape(self):
        """Test that window has correct shape."""
        n_fft = 2048
        window = get_window("hann", n_fft)
        assert window.shape == (n_fft,)

    def test_window_dtype(self):
        """Test that window has float32 dtype."""
        window = get_window("hann", 1024)
        assert window.dtype == mx.float32

    def test_window_aliases(self):
        """Test window name aliases."""
        n_fft = 1024

        # hann/hanning should be the same
        hann = get_window("hann", n_fft)
        hanning = get_window("hanning", n_fft)
        np.testing.assert_allclose(np.array(hann), np.array(hanning))

        # rectangular/boxcar/ones should be the same
        rect = get_window("rectangular", n_fft)
        boxcar = get_window("boxcar", n_fft)
        ones = get_window("ones", n_fft)
        np.testing.assert_allclose(np.array(rect), np.array(boxcar))
        np.testing.assert_allclose(np.array(rect), np.array(ones))
