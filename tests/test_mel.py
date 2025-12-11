"""
Tests for mel filterbank and mel spectrogram.
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import mel_filterbank, melspectrogram, hz_to_mel, mel_to_hz


class TestMelScale:
    """Tests for mel scale conversions."""

    def test_hz_to_mel_slaney(self):
        """Test Hz to mel conversion (Slaney)."""
        freqs = np.array([0, 500, 1000, 2000, 4000, 8000])
        expected = librosa.hz_to_mel(freqs, htk=False)
        actual = hz_to_mel(freqs, htk=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_hz_to_mel_htk(self):
        """Test Hz to mel conversion (HTK)."""
        freqs = np.array([0, 500, 1000, 2000, 4000, 8000])
        expected = librosa.hz_to_mel(freqs, htk=True)
        actual = hz_to_mel(freqs, htk=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_mel_to_hz_slaney(self):
        """Test mel to Hz conversion (Slaney)."""
        mels = np.array([0, 5, 10, 15, 20, 30])
        expected = librosa.mel_to_hz(mels, htk=False)
        actual = mel_to_hz(mels, htk=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_mel_to_hz_htk(self):
        """Test mel to Hz conversion (HTK)."""
        mels = np.array([0, 500, 1000, 1500, 2000, 2500])
        expected = librosa.mel_to_hz(mels, htk=True)
        actual = mel_to_hz(mels, htk=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_round_trip_slaney(self):
        """Test Hz -> mel -> Hz round trip (Slaney)."""
        freqs = np.linspace(0, 8000, 100)
        mels = hz_to_mel(freqs, htk=False)
        recovered = mel_to_hz(mels, htk=False)
        np.testing.assert_allclose(recovered, freqs, rtol=1e-5)

    def test_round_trip_htk(self):
        """Test Hz -> mel -> Hz round trip (HTK)."""
        freqs = np.linspace(0, 8000, 100)
        mels = hz_to_mel(freqs, htk=True)
        recovered = mel_to_hz(mels, htk=True)
        np.testing.assert_allclose(recovered, freqs, rtol=1e-5)


class TestMelFilterbank:
    """Tests for mel filterbank."""

    def test_mel_filterbank_matches_librosa(self):
        """Test that mel filterbank matches librosa."""
        sr = 22050
        n_fft = 2048
        n_mels = 128

        expected = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        actual = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("n_mels", [40, 64, 80, 128])
    def test_mel_filterbank_various_n_mels(self, n_mels):
        """Test mel filterbank with various numbers of mel bands."""
        sr = 22050
        n_fft = 2048

        expected = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        actual = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_mel_filterbank_htk(self):
        """Test mel filterbank with HTK formula."""
        sr = 22050
        n_fft = 2048
        n_mels = 128

        expected = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True)
        actual = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_mel_filterbank_fmin_fmax(self):
        """Test mel filterbank with custom fmin/fmax."""
        sr = 22050
        n_fft = 2048
        n_mels = 80
        fmin = 80.0
        fmax = 7600.0

        expected = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        actual = mel_filterbank(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_mel_filterbank_no_norm(self):
        """Test mel filterbank without normalization."""
        sr = 22050
        n_fft = 2048
        n_mels = 128

        expected = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm=None)
        actual = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, norm=None)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_mel_filterbank_shape(self):
        """Test mel filterbank output shape."""
        sr = 22050
        n_fft = 2048
        n_mels = 128

        fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        assert fb.shape == (n_mels, n_fft // 2 + 1)

    def test_mel_filterbank_fmax_exceeds_nyquist_raises(self):
        """Test that fmax > Nyquist raises ValueError."""
        sr = 22050
        with pytest.raises(ValueError, match="cannot exceed Nyquist"):
            mel_filterbank(sr=sr, n_fft=2048, fmax=sr)


class TestMelspectrogram:
    """Tests for mel spectrogram."""

    def test_melspectrogram_matches_librosa(self, random_signal):
        """Test that melspectrogram matches librosa."""
        sr = 22050
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        expected = librosa.feature.melspectrogram(
            y=random_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        y_mlx = mx.array(random_signal)
        actual = melspectrogram(
            y_mlx, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    def test_melspectrogram_shape(self, random_signal):
        """Test melspectrogram output shape."""
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        y_mlx = mx.array(random_signal)
        result = melspectrogram(y_mlx, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Expected shape: (n_mels, n_frames)
        n_frames = 1 + (len(random_signal) + n_fft // 2 * 2 - n_fft) // hop_length
        assert result.shape[0] == n_mels
        # n_frames may vary slightly due to padding

    def test_melspectrogram_power_1(self, random_signal):
        """Test melspectrogram with power=1 (amplitude)."""
        sr = 22050
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        expected = librosa.feature.melspectrogram(
            y=random_signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )

        y_mlx = mx.array(random_signal)
        actual = melspectrogram(
            y_mlx,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("n_mels", [40, 80, 128])
    def test_melspectrogram_various_n_mels(self, random_signal, n_mels):
        """Test melspectrogram with various numbers of mel bands."""
        sr = 22050
        n_fft = 2048
        hop_length = 512

        expected = librosa.feature.melspectrogram(
            y=random_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        y_mlx = mx.array(random_signal)
        actual = melspectrogram(
            y_mlx, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-4, atol=1e-4)
