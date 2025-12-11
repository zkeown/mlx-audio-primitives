"""
Decibel conversion functions test suite.

Tests cover:
- power_to_db and db_to_power round-trip accuracy
- amplitude_to_db and db_to_amplitude round-trip accuracy
- Librosa compatibility for all conversion functions
- Custom reference values (scalar and callable)
- top_db dynamic range clipping
- Custom amin (amplitude minimum) handling
"""
import librosa
import mlx.core as mx
import numpy as np

from mlx_audio_primitives import (
    amplitude_to_db,
    db_to_amplitude,
    db_to_power,
    magnitude,
    power_to_db,
    stft,
)


class TestPowerToDb:
    """Tests for power_to_db function."""

    def test_power_to_db_matches_librosa(self, random_signal):
        """Test that power_to_db matches librosa."""
        # Create power spectrogram
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2

        expected = librosa.power_to_db(np.array(S_power))
        actual = power_to_db(S_power)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_power_to_db_custom_ref(self, random_signal):
        """Test power_to_db with custom reference."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2

        ref = 0.5
        expected = librosa.power_to_db(np.array(S_power), ref=ref)
        actual = power_to_db(S_power, ref=ref)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_power_to_db_no_top_db(self, random_signal):
        """Test power_to_db without top_db clipping."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2

        expected = librosa.power_to_db(np.array(S_power), top_db=None)
        actual = power_to_db(S_power, top_db=None)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_power_to_db_custom_amin(self, random_signal):
        """Test power_to_db with custom amin."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2

        amin = 1e-5
        expected = librosa.power_to_db(np.array(S_power), amin=amin)
        actual = power_to_db(S_power, amin=amin)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)


class TestDbToPower:
    """Tests for db_to_power function."""

    def test_db_to_power_matches_librosa(self, random_signal):
        """Test that db_to_power matches librosa."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2
        S_db = power_to_db(S_power, top_db=None)

        expected = librosa.db_to_power(np.array(S_db))
        actual = db_to_power(S_db)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_power_round_trip(self, random_signal):
        """Test power_to_db -> db_to_power round trip."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_power = magnitude(S) ** 2

        # Round trip (without top_db clipping for exact round trip)
        S_db = power_to_db(S_power, top_db=None)
        S_recovered = db_to_power(S_db)

        np.testing.assert_allclose(
            np.array(S_recovered), np.array(S_power), rtol=1e-5, atol=1e-5
        )


class TestAmplitudeToDb:
    """Tests for amplitude_to_db function."""

    def test_amplitude_to_db_matches_librosa(self, random_signal):
        """Test that amplitude_to_db matches librosa."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_amp = magnitude(S)

        expected = librosa.amplitude_to_db(np.array(S_amp))
        actual = amplitude_to_db(S_amp)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_amplitude_to_db_custom_ref(self, random_signal):
        """Test amplitude_to_db with custom reference."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_amp = magnitude(S)

        ref = 0.5
        expected = librosa.amplitude_to_db(np.array(S_amp), ref=ref)
        actual = amplitude_to_db(S_amp, ref=ref)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_amplitude_to_db_no_top_db(self, random_signal):
        """Test amplitude_to_db without top_db clipping."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_amp = magnitude(S)

        expected = librosa.amplitude_to_db(np.array(S_amp), top_db=None)
        actual = amplitude_to_db(S_amp, top_db=None)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)


class TestDbToAmplitude:
    """Tests for db_to_amplitude function."""

    def test_db_to_amplitude_matches_librosa(self, random_signal):
        """Test that db_to_amplitude matches librosa."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_amp = magnitude(S)
        S_db = amplitude_to_db(S_amp, top_db=None)

        expected = librosa.db_to_amplitude(np.array(S_db))
        actual = db_to_amplitude(S_db)
        actual_np = np.array(actual)

        np.testing.assert_allclose(actual_np, expected, rtol=1e-5, atol=1e-5)

    def test_amplitude_round_trip(self, random_signal):
        """Test amplitude_to_db -> db_to_amplitude round trip."""
        y_mlx = mx.array(random_signal)
        S = stft(y_mlx)
        S_amp = magnitude(S)

        # Round trip (without top_db clipping for exact round trip)
        S_db = amplitude_to_db(S_amp, top_db=None)
        S_recovered = db_to_amplitude(S_db)

        np.testing.assert_allclose(
            np.array(S_recovered), np.array(S_amp), rtol=1e-5, atol=1e-5
        )
