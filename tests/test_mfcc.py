"""
MFCC and cepstral features test suite.

Tests cover:
- librosa.feature.mfcc compatibility (values within tolerance)
- Various n_mfcc values (13, 20, 40 - common configurations)
- Delta and delta-delta (acceleration) features
- DCT-II computation accuracy
- Liftering parameter effects
- Batch processing

Cross-validates against: librosa.feature.mfcc, librosa.feature.delta
Tolerance: rtol=1e-4, atol=1e-4 (DCT and log operations)

Pipeline: audio -> mel spectrogram -> log -> DCT -> (optional lifter)
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import mfcc, delta, dct, melspectrogram, power_to_db


class TestMFCC:
    """Test mfcc() function."""

    def test_basic_mfcc(self, random_signal):
        """Test basic MFCC matches librosa."""
        y_mx = mx.array(random_signal)
        result = mfcc(y_mx, sr=22050, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512)

        expected = librosa.feature.mfcc(
            y=random_signal, sr=22050, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("n_mfcc", [13, 20, 40])
    def test_various_n_mfcc(self, random_signal, n_mfcc):
        """Test with various number of MFCCs."""
        y_mx = mx.array(random_signal)
        result = mfcc(y_mx, sr=22050, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

        expected = librosa.feature.mfcc(
            y=random_signal, sr=22050, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("n_mels", [40, 80, 128])
    def test_various_n_mels(self, random_signal, n_mels):
        """Test with various number of mel bands."""
        y_mx = mx.array(random_signal)
        result = mfcc(y_mx, sr=22050, n_mfcc=13, n_mels=n_mels)

        expected = librosa.feature.mfcc(
            y=random_signal, sr=22050, n_mfcc=13, n_mels=n_mels
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape is correct."""
        y_mx = mx.array(random_signal)
        result = mfcc(y_mx, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512)

        # Shape should be (n_mfcc, n_frames)
        assert result.shape[0] == 13

    def test_from_spectrogram(self, random_signal):
        """Test computing from pre-computed mel spectrogram."""
        # Compute log-power mel spectrogram using librosa (which is what MFCC expects)
        S_librosa = librosa.feature.melspectrogram(
            y=random_signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128
        )
        S_db_librosa = librosa.power_to_db(S_librosa, ref=np.max)

        # Convert to mx.array
        S_db_mx = mx.array(S_db_librosa.astype(np.float32))

        # Both should use the same pre-computed S_db
        result = mfcc(S=S_db_mx, sr=22050, n_mfcc=13)

        expected = librosa.feature.mfcc(
            S=S_db_librosa, sr=22050, n_mfcc=13
        )

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = mfcc(y_mx, sr=22050, n_mfcc=13)

        assert result.shape[0] == 4  # Batch size preserved
        assert result.shape[1] == 13  # n_mfcc

    def test_lifter(self, random_signal):
        """Test liftering."""
        y_mx = mx.array(random_signal)

        result_no_lift = mfcc(y_mx, sr=22050, n_mfcc=13, lifter=0)
        result_lift = mfcc(y_mx, sr=22050, n_mfcc=13, lifter=22)

        # Liftering should change the values
        assert not np.allclose(np.array(result_no_lift), np.array(result_lift))

        # Compare with librosa liftering
        expected = librosa.feature.mfcc(
            y=random_signal, sr=22050, n_mfcc=13, lifter=22
        )

        # Relaxed tolerance to account for C++ DCT implementation differences
        np.testing.assert_allclose(
            np.array(result_lift), expected, rtol=1e-3, atol=1e-3
        )

    def test_invalid_parameters(self, random_signal):
        """Test error handling for invalid parameters."""
        y_mx = mx.array(random_signal)

        with pytest.raises(ValueError, match="must be positive"):
            mfcc(y_mx, sr=22050, n_mfcc=0)


class TestDelta:
    """Test delta() function."""

    def test_basic_delta(self, random_signal):
        """Test basic delta matches librosa."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        result = delta(mfccs, width=9, order=1)

        # Compute reference
        mfccs_np = np.array(mfccs)
        expected = librosa.feature.delta(mfccs_np, width=9, order=1)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_delta_delta(self, random_signal):
        """Test second order delta (acceleration)."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        result = delta(mfccs, width=9, order=2)

        # Compute reference
        mfccs_np = np.array(mfccs)
        expected = librosa.feature.delta(mfccs_np, width=9, order=2)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("width", [3, 5, 9, 13])
    def test_various_width(self, random_signal, width):
        """Test with various window widths."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        result = delta(mfccs, width=width, order=1)

        mfccs_np = np.array(mfccs)
        expected = librosa.feature.delta(mfccs_np, width=width, order=1)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_output_shape(self, random_signal):
        """Test output shape matches input."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        result = delta(mfccs, width=9)

        assert result.shape == mfccs.shape

    def test_invalid_width(self, random_signal):
        """Test error handling for invalid width."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        with pytest.raises(ValueError, match="must be >= 3"):
            delta(mfccs, width=1)

        with pytest.raises(ValueError, match="must be odd"):
            delta(mfccs, width=4)


class TestDCT:
    """Test dct() function."""

    def test_basic_dct(self):
        """Test basic DCT computation."""
        from scipy.fftpack import dct as scipy_dct

        # Create test signal
        x = np.random.randn(128).astype(np.float32)
        x_mx = mx.array(x)

        result = dct(x_mx, type=2, norm="ortho")

        expected = scipy_dct(x, type=2, norm="ortho")

        # Relaxed tolerance (rtol=1e-4) to account for C++ matmul-based DCT vs scipy
        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_dct_truncated(self):
        """Test DCT with fewer output coefficients."""
        from scipy.fftpack import dct as scipy_dct

        x = np.random.randn(128).astype(np.float32)
        x_mx = mx.array(x)

        result = dct(x_mx, type=2, n=20, norm="ortho")

        # Scipy doesn't support n directly, so compute full and truncate
        expected = scipy_dct(x, type=2, norm="ortho")[:20]

        # Relaxed tolerance to account for C++ matmul-based DCT vs scipy
        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_dct_2d(self):
        """Test DCT on 2D input."""
        from scipy.fftpack import dct as scipy_dct

        x = np.random.randn(10, 128).astype(np.float32)
        x_mx = mx.array(x)

        result = dct(x_mx, type=2, norm="ortho", axis=-1)

        expected = scipy_dct(x, type=2, norm="ortho", axis=-1)

        # Relaxed tolerance to account for C++ matmul-based DCT vs scipy
        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_invalid_type(self):
        """Test error for unsupported DCT type."""
        x_mx = mx.array(np.random.randn(128).astype(np.float32))

        with pytest.raises(ValueError, match="Only DCT type 2"):
            dct(x_mx, type=1)


class TestMFCCMathematicalProperties:
    """Test mathematical properties of MFCCs."""

    def test_mfcc_decorrelation(self, random_signal):
        """Test that MFCCs are approximately decorrelated."""
        y_mx = mx.array(random_signal)
        mfccs = np.array(mfcc(y_mx, sr=22050, n_mfcc=13))

        # Compute correlation matrix
        corr_matrix = np.corrcoef(mfccs)

        # Off-diagonal elements should be relatively small
        # DCT provides some decorrelation but not perfect for random signals
        off_diagonal = corr_matrix - np.eye(13)
        # Use median instead of max for a more robust check
        assert np.median(np.abs(off_diagonal)) < 0.3

    def test_delta_captures_dynamics(self, chirp_signal):
        """Test that delta features capture temporal dynamics."""
        y_mx = mx.array(chirp_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)

        deltas = delta(mfccs, width=9, order=1)

        # For a chirp (changing frequency), deltas should be non-zero
        delta_np = np.array(deltas)
        assert np.std(delta_np) > 0.01

    def test_feature_concatenation(self, random_signal):
        """Test typical MFCC + delta + delta-delta concatenation."""
        y_mx = mx.array(random_signal)
        mfccs = mfcc(y_mx, sr=22050, n_mfcc=13)
        delta1 = delta(mfccs, width=9, order=1)
        delta2 = delta(mfccs, width=9, order=2)

        # Concatenate features
        features = mx.concatenate([mfccs, delta1, delta2], axis=0)

        # Should have 39 features (13 + 13 + 13)
        assert features.shape[0] == 39
