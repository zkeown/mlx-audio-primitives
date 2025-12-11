"""
Audio resampling test suite.

Tests cover:
- Same-rate resampling (identity operation)
- Downsampling (44.1kHz -> 16kHz, 22.05kHz)
- Upsampling (16kHz -> 22.05kHz, 44.1kHz)
- Integer ratio resampling (polyphase method)
- Signal length preservation and energy conservation
- Batch processing

Cross-validates against: librosa.resample (res_type='fft')
Tolerance: rtol=1e-4, atol=1e-4 (FFT-based resampling differences)

Note: resample() uses scipy.signal.resample (FFT method).
resample_poly() uses scipy.signal.resample_poly (polyphase FIR method).
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import resample, resample_poly


class TestResample:
    """Test resample() function."""

    def test_same_rate(self, random_signal):
        """Test that resampling to same rate returns input."""
        y_mx = mx.array(random_signal)
        result = resample(y_mx, orig_sr=22050, target_sr=22050)

        np.testing.assert_allclose(
            np.array(result), random_signal, rtol=1e-5, atol=1e-5
        )

    def test_downsample_2x(self, random_signal):
        """Test 2x downsampling matches librosa with res_type='fft'."""
        y_mx = mx.array(random_signal)
        result = resample(y_mx, orig_sr=44100, target_sr=22050)

        # Our implementation uses scipy.signal.resample which matches librosa's 'fft' mode
        expected = librosa.resample(
            random_signal, orig_sr=44100, target_sr=22050, res_type="fft"
        )

        # Check length is correct
        assert len(np.array(result)) == len(expected)

        # Check values match exactly (same algorithm)
        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_upsample_2x(self, random_signal):
        """Test 2x upsampling matches librosa with res_type='fft'."""
        y_mx = mx.array(random_signal)
        result = resample(y_mx, orig_sr=22050, target_sr=44100)

        # Our implementation uses scipy.signal.resample which matches librosa's 'fft' mode
        expected = librosa.resample(
            random_signal, orig_sr=22050, target_sr=44100, res_type="fft"
        )

        # Check length is correct
        assert len(np.array(result)) == len(expected)

        # Check values match exactly (same algorithm)
        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("orig_sr,target_sr", [
        (44100, 22050),
        (22050, 16000),
        (16000, 8000),
        (8000, 16000),
        (22050, 44100),
    ])
    def test_various_rates(self, random_signal, orig_sr, target_sr):
        """Test resampling with various rate combinations."""
        y_mx = mx.array(random_signal)
        result = resample(y_mx, orig_sr=orig_sr, target_sr=target_sr)

        expected = librosa.resample(random_signal, orig_sr=orig_sr, target_sr=target_sr)

        # Check length is approximately correct
        expected_length = int(np.round(len(random_signal) * target_sr / orig_sr))
        assert abs(len(np.array(result)) - expected_length) <= 1

    def test_output_length(self, random_signal):
        """Test that output length is correct."""
        y_mx = mx.array(random_signal)

        # 2x downsample
        result = resample(y_mx, orig_sr=44100, target_sr=22050)
        expected_length = int(np.round(len(random_signal) * 22050 / 44100))
        assert len(np.array(result)) == expected_length

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = resample(y_mx, orig_sr=44100, target_sr=22050)

        assert result.shape[0] == 4  # Batch size preserved

    def test_preserves_frequency_content(self):
        """Test that resampling preserves frequency content within Nyquist."""
        # Create a signal with known frequency content
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        freq = 1000  # 1000 Hz
        y = np.sin(2 * np.pi * freq * t)

        y_mx = mx.array(y)
        y_resampled = resample(y_mx, orig_sr=sr, target_sr=22050)

        # The resampled signal should still have the 1000 Hz component
        # (which is below the new Nyquist of 11025 Hz)
        y_resampled_np = np.array(y_resampled)

        # Compute FFT and check peak is near 1000 Hz
        fft_result = np.abs(np.fft.rfft(y_resampled_np))
        freqs = np.fft.rfftfreq(len(y_resampled_np), 1/22050)
        peak_freq = freqs[np.argmax(fft_result)]

        assert abs(peak_freq - freq) < 50  # Within 50 Hz

    def test_linear_resampling(self, random_signal):
        """Test linear interpolation resampling."""
        y_mx = mx.array(random_signal)
        result = resample(y_mx, orig_sr=44100, target_sr=22050, res_type="linear")

        # Should produce output of correct length
        expected_length = int(np.round(len(random_signal) * 22050 / 44100))
        assert len(np.array(result)) == expected_length

    def test_invalid_res_type(self, random_signal):
        """Test error for invalid res_type."""
        y_mx = mx.array(random_signal)

        with pytest.raises(ValueError, match="Unknown res_type"):
            resample(y_mx, orig_sr=22050, target_sr=16000, res_type="invalid")


class TestResamplePoly:
    """Test resample_poly() function."""

    def test_downsample_2_1(self, random_signal):
        """Test 2:1 downsampling."""
        y_mx = mx.array(random_signal)
        result = resample_poly(y_mx, up=1, down=2)

        # Output should be approximately half the length
        expected_length = len(random_signal) // 2
        assert abs(len(np.array(result)) - expected_length) <= 10

    def test_upsample_1_2(self, random_signal):
        """Test 1:2 upsampling."""
        y_mx = mx.array(random_signal)
        result = resample_poly(y_mx, up=2, down=1)

        # Output should be approximately twice the length
        expected_length = len(random_signal) * 2
        assert abs(len(np.array(result)) - expected_length) <= 10

    def test_same_ratio(self, random_signal):
        """Test 1:1 ratio returns similar input."""
        y_mx = mx.array(random_signal)
        result = resample_poly(y_mx, up=1, down=1)

        np.testing.assert_allclose(
            np.array(result), random_signal, rtol=1e-5, atol=1e-5
        )

    def test_ratio_simplification(self, random_signal):
        """Test that equivalent ratios give same result."""
        y_mx = mx.array(random_signal)

        result_2_4 = resample_poly(y_mx, up=2, down=4)
        result_1_2 = resample_poly(y_mx, up=1, down=2)

        np.testing.assert_allclose(
            np.array(result_2_4), np.array(result_1_2), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("up,down", [(1, 2), (2, 1), (3, 1), (1, 3), (2, 3), (3, 2)])
    def test_various_ratios(self, short_signal, up, down):
        """Test with various up/down ratios."""
        y_mx = mx.array(short_signal)
        result = resample_poly(y_mx, up=up, down=down)

        # Just check it produces output
        assert len(np.array(result)) > 0

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = resample_poly(y_mx, up=1, down=2)

        assert result.shape[0] == 4  # Batch size preserved


class TestResamplingQuality:
    """Test resampling quality and properties."""

    def test_energy_preservation(self, random_signal):
        """Test that resampling approximately preserves signal energy."""
        y_mx = mx.array(random_signal)

        # Original energy
        original_energy = np.sum(random_signal ** 2) / len(random_signal)

        # Downsample and compute energy
        y_down = resample(y_mx, orig_sr=44100, target_sr=22050)
        y_down_np = np.array(y_down)
        down_energy = np.sum(y_down_np ** 2) / len(y_down_np)

        # Energy should be similar (within 50%)
        assert abs(down_energy - original_energy) / original_energy < 0.5

    def test_no_aliasing_in_fft_resample(self):
        """Test that FFT resampling doesn't introduce aliasing."""
        # Create signal with frequency above target Nyquist
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        high_freq = 15000  # Above 22050/2 = 11025 Hz Nyquist
        y = np.sin(2 * np.pi * high_freq * t)

        y_mx = mx.array(y)
        y_resampled = resample(y_mx, orig_sr=sr, target_sr=22050, res_type="fft")

        # The high frequency should be filtered out (no aliasing)
        y_resampled_np = np.array(y_resampled)

        # Energy should be much lower (high freq was filtered)
        original_energy = np.sum(y ** 2)
        resampled_energy = np.sum(y_resampled_np ** 2)

        # Resampled should have much less energy (high freq removed)
        assert resampled_energy < 0.1 * original_energy

    def test_round_trip(self, chirp_signal):
        """Test downsample then upsample recovers approximate signal."""
        y_mx = mx.array(chirp_signal)

        # Downsample to half rate
        y_down = resample(y_mx, orig_sr=22050, target_sr=11025)

        # Upsample back
        y_up = resample(y_down, orig_sr=11025, target_sr=22050)

        y_up_np = np.array(y_up)

        # Length should match (approximately)
        assert abs(len(y_up_np) - len(chirp_signal)) <= 2

        # Low frequency content should be preserved
        # (high freq above 5512.5 Hz will be lost)
        # Just check correlation is reasonable
        min_len = min(len(y_up_np), len(chirp_signal))
        corr = np.corrcoef(y_up_np[:min_len], chirp_signal[:min_len])[0, 1]
        assert corr > 0.5
