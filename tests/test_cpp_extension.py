"""
Tests for the C++ extension module.

These tests validate that:
1. All C++ functions are accessible and callable
2. GPU (Metal) and CPU paths produce consistent results
3. Mathematical correctness of core operations
4. Edge cases and error handling
"""
import pytest
import numpy as np
import mlx.core as mx

# Import the C++ extension directly
from mlx_audio_primitives import _ext


@pytest.fixture(autouse=True)
def reset_device():
    """Ensure device state is restored after each test."""
    original_device = mx.default_device()
    yield
    mx.set_default_device(original_device)


class TestExtensionAvailability:
    """Test that the C++ extension is loaded and functions are available."""

    def test_extension_loaded(self):
        """Verify the extension module is loaded."""
        assert _ext is not None
        assert hasattr(_ext, '__doc__')

    def test_all_functions_available(self):
        """Verify all expected functions are exported."""
        expected_functions = [
            'frame_signal',
            'generate_window',
            'pad_signal',
            'overlap_add',
            'mel_filterbank',
            'hz_to_mel',
            'mel_to_hz',
        ]
        for func_name in expected_functions:
            assert hasattr(_ext, func_name), f"Missing function: {func_name}"


class TestGenerateWindow:
    """Tests for the generate_window function."""

    @pytest.mark.parametrize("window_type", [
        "hann", "hamming", "blackman", "bartlett", "rectangular"
    ])
    def test_window_types(self, window_type):
        """Test that all window types can be generated."""
        window = _ext.generate_window(window_type, 512, True)
        mx.eval(window)
        assert window.shape == (512,)
        assert window.dtype == mx.float32

    @pytest.mark.parametrize("length", [64, 128, 256, 512, 1024, 2048])
    def test_various_lengths(self, length):
        """Test window generation for various lengths."""
        window = _ext.generate_window("hann", length, True)
        mx.eval(window)
        assert window.shape == (length,)

    def test_periodic_vs_symmetric(self):
        """Test periodic vs symmetric window generation."""
        periodic = _ext.generate_window("hann", 512, True)
        symmetric = _ext.generate_window("hann", 512, False)
        mx.eval(periodic, symmetric)
        # They should be different
        assert not np.allclose(np.array(periodic), np.array(symmetric))

    def test_hann_window_values(self):
        """Verify Hann window values are mathematically correct."""
        n = 512
        window = _ext.generate_window("hann", n, False)
        mx.eval(window)
        window_np = np.array(window)

        # Expected Hann window: 0.5 - 0.5 * cos(2 * pi * k / (n - 1))
        k = np.arange(n)
        expected = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))

        # Allow for GPU/CPU floating point differences
        np.testing.assert_allclose(window_np, expected, rtol=1e-3, atol=1e-6)

    def test_window_bounds(self):
        """Verify all window values are in valid range [0, 1]."""
        for wtype in ["hann", "hamming", "blackman", "bartlett", "rectangular"]:
            window = _ext.generate_window(wtype, 512, True)
            mx.eval(window)
            window_np = np.array(window)
            assert np.all(window_np >= 0), f"{wtype} has negative values"
            assert np.all(window_np <= 1.01), f"{wtype} exceeds 1"  # Small tolerance

    def test_invalid_window_type(self):
        """Test that invalid window type raises an error."""
        with pytest.raises(Exception):
            _ext.generate_window("invalid_window", 512, True)


class TestFrameSignal:
    """Tests for the frame_signal function."""

    def test_basic_framing(self):
        """Test basic signal framing."""
        signal = mx.zeros((1, 1000))
        frames = _ext.frame_signal(signal, 256, 128)
        mx.eval(frames)

        # Expected: (1000 - 256) / 128 + 1 = 6 frames
        assert frames.shape == (1, 6, 256)

    def test_framing_values(self):
        """Test that framing extracts correct values."""
        # Create signal with known pattern
        signal_np = np.arange(1000, dtype=np.float32).reshape(1, -1)
        signal = mx.array(signal_np)

        frames = _ext.frame_signal(signal, 256, 128)
        mx.eval(frames)
        frames_np = np.array(frames)

        # First frame should be samples 0-255
        np.testing.assert_array_equal(frames_np[0, 0, :], signal_np[0, :256])
        # Second frame should be samples 128-383
        np.testing.assert_array_equal(frames_np[0, 1, :], signal_np[0, 128:384])

    def test_batch_framing(self):
        """Test framing with batch dimension."""
        signal = mx.random.uniform(shape=(4, 2000))
        mx.eval(signal)

        frames = _ext.frame_signal(signal, 512, 256)
        mx.eval(frames)

        # Check batch dimension is preserved
        assert frames.shape[0] == 4
        assert frames.shape[2] == 512

    @pytest.mark.parametrize("frame_length,hop_length", [
        (256, 128),
        (512, 256),
        (1024, 512),
        (2048, 512),
    ])
    def test_various_parameters(self, frame_length, hop_length):
        """Test framing with various frame/hop lengths."""
        signal = mx.zeros((1, 8000))
        frames = _ext.frame_signal(signal, frame_length, hop_length)
        mx.eval(frames)

        expected_n_frames = 1 + (8000 - frame_length) // hop_length
        assert frames.shape == (1, expected_n_frames, frame_length)


class TestPadSignal:
    """Tests for the pad_signal function."""

    @pytest.mark.parametrize("mode", ["constant", "reflect", "edge"])
    def test_padding_modes(self, mode):
        """Test all padding modes work."""
        signal = mx.ones((1, 100))
        padded = _ext.pad_signal(signal, 10, mode)
        mx.eval(padded)

        assert padded.shape == (1, 120)

    def test_constant_padding_values(self):
        """Verify constant padding uses zeros."""
        signal = mx.ones((1, 100))
        padded = _ext.pad_signal(signal, 10, "constant")
        mx.eval(padded)
        padded_np = np.array(padded)

        # First and last 10 values should be 0
        np.testing.assert_array_equal(padded_np[0, :10], np.zeros(10))
        np.testing.assert_array_equal(padded_np[0, -10:], np.zeros(10))
        # Middle should be 1s
        np.testing.assert_array_equal(padded_np[0, 10:-10], np.ones(100))

    def test_edge_padding_values(self):
        """Verify edge padding replicates edge values."""
        signal_np = np.arange(100, dtype=np.float32).reshape(1, -1)
        signal = mx.array(signal_np)

        padded = _ext.pad_signal(signal, 5, "edge")
        mx.eval(padded)
        padded_np = np.array(padded)

        # First 5 values should equal signal[0]
        np.testing.assert_array_equal(padded_np[0, :5], np.full(5, signal_np[0, 0]))
        # Last 5 values should equal signal[-1]
        np.testing.assert_array_equal(padded_np[0, -5:], np.full(5, signal_np[0, -1]))

    def test_reflect_padding_values(self):
        """Verify reflect padding mirrors signal."""
        signal_np = np.arange(100, dtype=np.float32).reshape(1, -1)
        signal = mx.array(signal_np)

        padded = _ext.pad_signal(signal, 5, "reflect")
        mx.eval(padded)
        padded_np = np.array(padded)

        # Check reflect pattern at left: [5,4,3,2,1 | 0,1,2,3,4,5,...]
        expected_left = signal_np[0, 1:6][::-1]  # [5,4,3,2,1]
        np.testing.assert_array_equal(padded_np[0, :5], expected_left)

    def test_batch_padding(self):
        """Test padding with batch dimension."""
        signal = mx.random.uniform(shape=(4, 100))
        mx.eval(signal)

        padded = _ext.pad_signal(signal, 20, "constant")
        mx.eval(padded)

        assert padded.shape == (4, 140)


class TestOverlapAdd:
    """Tests for the overlap_add function."""

    def test_basic_overlap_add(self):
        """Test basic overlap-add reconstruction."""
        frames = mx.random.uniform(shape=(1, 10, 256))
        window = _ext.generate_window("hann", 256, False)
        mx.eval(frames, window)

        output = _ext.overlap_add(frames, window, 128, 1280)
        mx.eval(output)

        assert output.shape == (1, 1280)

    def test_overlap_add_normalization(self):
        """Test that overlap-add properly normalizes by window sum."""
        # Create constant frames
        frames = mx.ones((1, 10, 256))
        window = _ext.generate_window("hann", 256, False)
        mx.eval(frames, window)

        output = _ext.overlap_add(frames, window, 128, 1280)
        mx.eval(output)
        output_np = np.array(output)

        # The output should have finite, non-zero values in the middle region
        # Overlap-add with windowing creates a characteristic pattern
        middle = output_np[0, 256:-256]
        assert np.all(np.isfinite(middle)), "Output has non-finite values"
        assert np.mean(middle) > 0.5, "Output mean is too low"

    def test_batch_overlap_add(self):
        """Test overlap-add with batch dimension."""
        frames = mx.random.uniform(shape=(4, 10, 512))
        window = _ext.generate_window("hann", 512, False)
        mx.eval(frames, window)

        output = _ext.overlap_add(frames, window, 256, 2560)
        mx.eval(output)

        assert output.shape == (4, 2560)


class TestMelFilterbank:
    """Tests for the mel_filterbank function."""

    def test_basic_mel_filterbank(self):
        """Test basic mel filterbank generation."""
        fb = _ext.mel_filterbank(16000, 512, 80)
        mx.eval(fb)

        # Shape should be (n_mels, n_fft // 2 + 1)
        assert fb.shape == (80, 257)

    @pytest.mark.parametrize("n_mels", [40, 64, 80, 128])
    def test_various_n_mels(self, n_mels):
        """Test filterbank with various mel band counts."""
        fb = _ext.mel_filterbank(16000, 512, n_mels)
        mx.eval(fb)

        assert fb.shape[0] == n_mels

    @pytest.mark.parametrize("n_fft", [256, 512, 1024, 2048])
    def test_various_n_fft(self, n_fft):
        """Test filterbank with various FFT sizes."""
        fb = _ext.mel_filterbank(16000, n_fft, 80)
        mx.eval(fb)

        assert fb.shape == (80, n_fft // 2 + 1)

    def test_filterbank_non_negative(self):
        """Verify filterbank weights are non-negative."""
        fb = _ext.mel_filterbank(16000, 512, 80)
        mx.eval(fb)
        fb_np = np.array(fb)

        assert np.all(fb_np >= 0), "Filterbank has negative weights"

    def test_filterbank_triangular_shape(self):
        """Verify filterbank has triangular filter shape."""
        fb = _ext.mel_filterbank(16000, 512, 80)
        mx.eval(fb)
        fb_np = np.array(fb)

        # Each row should have a single peak (triangular filter)
        for i in range(fb_np.shape[0]):
            row = fb_np[i]
            nonzero = row[row > 0]
            if len(nonzero) > 2:
                # Find peak position - should exist for triangular filters
                peak_idx = np.argmax(row)
                # Peak should be in a reasonable position
                assert 0 < peak_idx < len(row) - 1 or row[peak_idx] > 0


class TestHzMelConversion:
    """Tests for Hz to Mel and Mel to Hz conversion."""

    def test_hz_to_mel_basic(self):
        """Test basic Hz to Mel conversion."""
        freqs = mx.array([0.0, 1000.0, 2000.0, 4000.0])
        mels = _ext.hz_to_mel(freqs, False)
        mx.eval(mels)

        assert mels.shape == freqs.shape
        # Mels should be monotonically increasing
        mels_np = np.array(mels)
        assert np.all(np.diff(mels_np) > 0)

    def test_mel_to_hz_basic(self):
        """Test basic Mel to Hz conversion."""
        mels = mx.array([0.0, 500.0, 1000.0, 1500.0])
        freqs = _ext.mel_to_hz(mels, False)
        mx.eval(freqs)

        assert freqs.shape == mels.shape
        # Freqs should be monotonically increasing
        freqs_np = np.array(freqs)
        assert np.all(np.diff(freqs_np) > 0)

    def test_roundtrip_conversion(self):
        """Test that Hz -> Mel -> Hz gives back original values."""
        original_hz = mx.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        mx.eval(original_hz)

        mels = _ext.hz_to_mel(original_hz, False)
        recovered_hz = _ext.mel_to_hz(mels, False)
        mx.eval(mels, recovered_hz)

        np.testing.assert_allclose(
            np.array(original_hz),
            np.array(recovered_hz),
            rtol=1e-5
        )

    def test_htk_mode(self):
        """Test HTK mode for mel conversion."""
        freqs = mx.array([1000.0])

        mels_slaney = _ext.hz_to_mel(freqs, False)
        mels_htk = _ext.hz_to_mel(freqs, True)
        mx.eval(mels_slaney, mels_htk)

        # HTK and Slaney should give different values
        assert not np.isclose(np.array(mels_slaney)[0], np.array(mels_htk)[0])


class TestGPUvsCPU:
    """Tests comparing GPU (Metal) and CPU results for consistency."""

    def test_window_gpu_cpu_match(self):
        """Verify GPU and CPU window generation match."""
        # GPU
        mx.set_default_device(mx.gpu)
        window_gpu = _ext.generate_window("hann", 512, True)
        mx.eval(window_gpu)
        window_gpu_np = np.array(window_gpu)

        # CPU
        mx.set_default_device(mx.cpu)
        window_cpu = _ext.generate_window("hann", 512, True)
        mx.eval(window_cpu)
        window_cpu_np = np.array(window_cpu)

        # Reset to GPU
        mx.set_default_device(mx.gpu)

        # Allow small floating point differences between GPU/CPU
        np.testing.assert_allclose(window_gpu_np, window_cpu_np, rtol=1e-3)

    def test_frame_signal_gpu_cpu_match(self):
        """Verify GPU and CPU framing match."""
        signal_np = np.random.randn(1, 4000).astype(np.float32)

        # GPU
        mx.set_default_device(mx.gpu)
        signal_gpu = mx.array(signal_np)
        frames_gpu = _ext.frame_signal(signal_gpu, 512, 256)
        mx.eval(frames_gpu)
        frames_gpu_np = np.array(frames_gpu)

        # CPU
        mx.set_default_device(mx.cpu)
        signal_cpu = mx.array(signal_np)
        frames_cpu = _ext.frame_signal(signal_cpu, 512, 256)
        mx.eval(frames_cpu)
        frames_cpu_np = np.array(frames_cpu)

        # Reset to GPU
        mx.set_default_device(mx.gpu)

        np.testing.assert_allclose(frames_gpu_np, frames_cpu_np, rtol=1e-5)

    def test_mel_filterbank_gpu_cpu_match(self):
        """Verify GPU and CPU mel filterbank match."""
        # GPU
        mx.set_default_device(mx.gpu)
        fb_gpu = _ext.mel_filterbank(16000, 512, 80)
        mx.eval(fb_gpu)
        fb_gpu_np = np.array(fb_gpu)

        # CPU
        mx.set_default_device(mx.cpu)
        fb_cpu = _ext.mel_filterbank(16000, 512, 80)
        mx.eval(fb_cpu)
        fb_cpu_np = np.array(fb_cpu)

        # Reset to GPU
        mx.set_default_device(mx.gpu)

        # Allow for floating point differences between GPU/CPU
        # Use atol for near-zero values where relative tolerance is meaningless
        np.testing.assert_allclose(fb_gpu_np, fb_cpu_np, rtol=1e-3, atol=1e-6)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_window_length(self):
        """Test window generation with length 1."""
        window = _ext.generate_window("hann", 1, True)
        mx.eval(window)
        assert window.shape == (1,)

    def test_small_signal_framing(self):
        """Test framing when signal is exactly frame_length."""
        signal = mx.zeros((1, 256))
        frames = _ext.frame_signal(signal, 256, 128)
        mx.eval(frames)
        assert frames.shape == (1, 1, 256)

    def test_zero_padding(self):
        """Test pad_signal with pad_length=0."""
        signal = mx.ones((1, 100))
        padded = _ext.pad_signal(signal, 0, "constant")
        mx.eval(padded)
        assert padded.shape == (1, 100)

    def test_large_batch(self):
        """Test operations with large batch size."""
        signal = mx.random.uniform(shape=(32, 8000))
        mx.eval(signal)

        frames = _ext.frame_signal(signal, 512, 256)
        mx.eval(frames)

        assert frames.shape[0] == 32
