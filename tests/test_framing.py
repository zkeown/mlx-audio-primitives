"""
Time-domain primitives test suite.

Tests cover:
- Signal framing (frame_length, hop_length combinations)
- RMS energy computation per frame
- Preemphasis filter (high-frequency boost, coef=0.97 typical)
- Deemphasis filter (inverse of preemphasis)
- Round-trip: preemphasis -> deemphasis recovers original
- Batch processing

Cross-validates against: librosa.util.frame, librosa.effects.preemphasis
Tolerance: rtol=1e-5, atol=1e-5 (simple arithmetic operations)

Note: frame() returns (n_frames, frame_length), transposed from librosa convention.
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import frame, rms, preemphasis, deemphasis


class TestFrame:
    """Test frame() function."""

    def test_basic_framing(self, random_signal):
        """Test basic framing matches librosa."""
        y_mx = mx.array(random_signal)
        frames = frame(y_mx, frame_length=2048, hop_length=512)

        # librosa.util.frame returns (frame_length, n_frames), we return (n_frames, frame_length)
        expected = librosa.util.frame(random_signal, frame_length=2048, hop_length=512)
        expected = expected.T  # Transpose to match our convention

        np.testing.assert_allclose(
            np.array(frames), expected, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("frame_length", [512, 1024, 2048])
    @pytest.mark.parametrize("hop_length", [128, 256, 512])
    def test_various_params(self, random_signal, frame_length, hop_length):
        """Test framing with various parameters."""
        y_mx = mx.array(random_signal)
        frames = frame(y_mx, frame_length=frame_length, hop_length=hop_length)

        expected = librosa.util.frame(random_signal, frame_length=frame_length, hop_length=hop_length)
        expected = expected.T

        np.testing.assert_allclose(
            np.array(frames), expected, rtol=1e-5, atol=1e-5
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        frames = frame(y_mx, frame_length=2048, hop_length=512)

        n_frames = 1 + (len(random_signal) - 2048) // 512
        assert frames.shape == (n_frames, 2048)

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        frames = frame(y_mx, frame_length=2048, hop_length=512)

        assert frames.shape[0] == 4  # Batch size preserved
        assert frames.shape[2] == 2048  # Frame length

    def test_invalid_parameters(self, random_signal):
        """Test error handling for invalid parameters."""
        y_mx = mx.array(random_signal)

        with pytest.raises(ValueError, match="must be positive"):
            frame(y_mx, frame_length=-1, hop_length=512)

        with pytest.raises(ValueError, match="must be positive"):
            frame(y_mx, frame_length=2048, hop_length=0)

    def test_signal_too_short(self, short_signal):
        """Test error when signal is shorter than frame length."""
        y_mx = mx.array(short_signal[:500])  # 500 samples

        with pytest.raises(ValueError, match="must be >= frame_length"):
            frame(y_mx, frame_length=2048, hop_length=512)


class TestRMS:
    """Test rms() function."""

    def test_basic_rms(self, random_signal):
        """Test basic RMS matches librosa."""
        y_mx = mx.array(random_signal)
        result = rms(y_mx, frame_length=2048, hop_length=512)

        expected = librosa.feature.rms(y=random_signal, frame_length=2048, hop_length=512)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("frame_length", [512, 1024, 2048])
    @pytest.mark.parametrize("hop_length", [128, 256, 512])
    def test_various_params(self, random_signal, frame_length, hop_length):
        """Test RMS with various parameters."""
        y_mx = mx.array(random_signal)
        result = rms(y_mx, frame_length=frame_length, hop_length=hop_length)

        expected = librosa.feature.rms(y=random_signal, frame_length=frame_length, hop_length=hop_length)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape matches librosa convention."""
        y_mx = mx.array(random_signal)
        result = rms(y_mx, frame_length=2048, hop_length=512)

        # librosa returns (1, n_frames)
        assert result.shape[0] == 1

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = rms(y_mx, frame_length=2048, hop_length=512)

        assert result.shape[0] == 4  # Batch size preserved
        assert result.shape[1] == 1  # Feature dimension

    def test_rms_values_positive(self, random_signal):
        """Test that RMS values are non-negative."""
        y_mx = mx.array(random_signal)
        result = rms(y_mx)

        assert np.all(np.array(result) >= 0)


class TestPreemphasis:
    """Test preemphasis() function."""

    def test_basic_preemphasis(self, random_signal):
        """Test basic pre-emphasis matches librosa."""
        y_mx = mx.array(random_signal)
        result = preemphasis(y_mx, coef=0.97)

        expected = librosa.effects.preemphasis(random_signal, coef=0.97)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("coef", [0.0, 0.5, 0.95, 0.97, 1.0])
    def test_various_coef(self, random_signal, coef):
        """Test pre-emphasis with various coefficients."""
        y_mx = mx.array(random_signal)
        result = preemphasis(y_mx, coef=coef)

        expected = librosa.effects.preemphasis(random_signal, coef=coef)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-5, atol=1e-5
        )

    def test_output_shape(self, random_signal):
        """Test output shape matches input."""
        y_mx = mx.array(random_signal)
        result = preemphasis(y_mx)

        assert result.shape == (len(random_signal),)

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = preemphasis(y_mx, coef=0.97)

        assert result.shape == batch_signals.shape

    def test_invalid_coef(self, random_signal):
        """Test error handling for invalid coefficient."""
        y_mx = mx.array(random_signal)

        with pytest.raises(ValueError, match="must be in"):
            preemphasis(y_mx, coef=-0.1)

        with pytest.raises(ValueError, match="must be in"):
            preemphasis(y_mx, coef=1.5)


class TestDeemphasis:
    """Test deemphasis() function."""

    def test_basic_deemphasis(self, random_signal):
        """Test basic de-emphasis matches librosa."""
        y_mx = mx.array(random_signal)
        result = deemphasis(y_mx, coef=0.97)

        expected = librosa.effects.deemphasis(random_signal, coef=0.97)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_round_trip(self, random_signal):
        """Test that preemphasis -> deemphasis recovers original."""
        y_mx = mx.array(random_signal)
        coef = 0.97

        y_emph = preemphasis(y_mx, coef=coef)
        y_recovered = deemphasis(y_emph, coef=coef)

        # The round-trip should be close to original (except first sample)
        np.testing.assert_allclose(
            np.array(y_recovered)[1:], random_signal[1:], rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape matches input."""
        y_mx = mx.array(random_signal)
        result = deemphasis(y_mx)

        assert result.shape == (len(random_signal),)

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = deemphasis(y_mx, coef=0.97)

        assert result.shape == batch_signals.shape
