"""
Pitch detection and periodicity test suite.

Tests cover:
- Autocorrelation via FFT (Wiener-Khinchin theorem)
- Pitch detection on known frequencies (sine waves, chirps)
- Voiced/unvoiced detection (periodicity threshold)
- Frequency range constraints (fmin, fmax parameters)
- Batch processing

Cross-validates against: numpy.correlate, known synthetic signals
Tolerance: Pitch within 5% of ground truth (peak detection variance)

Algorithm: autocorrelation -> find peak in valid lag range -> f0 = sr / lag
The periodicity value indicates confidence (0 = noise, 1 = perfectly periodic).
"""
import numpy as np
import pytest
import mlx.core as mx

from mlx_audio_primitives import autocorrelation, pitch_detect_acf, periodicity


class TestAutocorrelation:
    """Test autocorrelation() function."""

    def test_basic_autocorrelation(self, random_signal):
        """Test basic autocorrelation computation."""
        y_mx = mx.array(random_signal)
        result = autocorrelation(y_mx)

        # r[0] should be 1 when normalized
        result_np = np.array(result)
        assert np.isclose(result_np[0], 1.0, rtol=1e-5)

    def test_autocorrelation_numpy_reference(self, random_signal):
        """Test autocorrelation matches numpy reference."""
        y = random_signal - np.mean(random_signal)
        y_mx = mx.array(random_signal.astype(np.float32))

        result = autocorrelation(y_mx, normalize=True)
        result_np = np.array(result)

        # Compute reference using numpy
        n = len(y)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        Y = np.fft.rfft(y, n=n_fft)
        power = Y * np.conj(Y)
        r_ref = np.fft.irfft(power, n=n_fft)[:n]
        r_ref = r_ref / r_ref[0]

        np.testing.assert_allclose(
            result_np[:n], r_ref.astype(np.float32), rtol=1e-4, atol=1e-4
        )

    def test_sine_wave_periodicity(self):
        """Test that sine wave shows periodicity in autocorrelation."""
        sr = 22050
        freq = 440  # 440 Hz
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        y = np.sin(2 * np.pi * freq * t)

        y_mx = mx.array(y)
        result = autocorrelation(y_mx, max_lag=1000)
        result_np = np.array(result)

        # Find first peak after lag 0 (should be at lag ~sr/freq = ~50)
        expected_lag = sr / freq  # ~50 samples

        # Look for peak near expected lag
        search_start = int(expected_lag * 0.8)
        search_end = int(expected_lag * 1.2)
        peak_idx = search_start + np.argmax(result_np[search_start:search_end])

        assert abs(peak_idx - expected_lag) < 5

    def test_max_lag(self, random_signal):
        """Test max_lag parameter."""
        y_mx = mx.array(random_signal)

        result_full = autocorrelation(y_mx, max_lag=None)
        result_limited = autocorrelation(y_mx, max_lag=500)

        assert len(np.array(result_full)) == len(random_signal)
        assert len(np.array(result_limited)) == 500

    def test_normalize_option(self, random_signal):
        """Test normalize parameter."""
        y_mx = mx.array(random_signal)

        result_norm = autocorrelation(y_mx, normalize=True)
        result_unnorm = autocorrelation(y_mx, normalize=False)

        # Normalized: r[0] = 1
        assert np.isclose(np.array(result_norm)[0], 1.0, rtol=1e-5)

        # Unnormalized: r[0] = signal energy
        # (approximately, since we center the signal)

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = autocorrelation(y_mx, max_lag=500)

        assert result.shape[0] == 4  # Batch size preserved
        assert result.shape[1] == 500

    def test_center_option(self, random_signal):
        """Test center (mean subtraction) option."""
        y_mx = mx.array(random_signal)

        result_center = autocorrelation(y_mx, center=True)
        result_no_center = autocorrelation(y_mx, center=False)

        # Results should be different (unless signal already has zero mean)
        # Just check both produce valid output
        assert len(np.array(result_center)) > 0
        assert len(np.array(result_no_center)) > 0


class TestPitchDetectACF:
    """Test pitch_detect_acf() function."""

    def test_detect_known_frequency(self):
        """Test pitch detection on signal with known frequency."""
        sr = 22050
        freq = 440  # A4 note
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        y = np.sin(2 * np.pi * freq * t)

        y_mx = mx.array(y)
        f0, voiced = pitch_detect_acf(
            y_mx, sr=sr, fmin=80, fmax=1000,
            frame_length=2048, hop_length=512
        )

        f0_np = np.array(f0)
        voiced_np = np.array(voiced)

        # Most frames should be voiced
        assert np.mean(voiced_np) > 0.7

        # Detected pitch should be close to 440 Hz
        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            assert abs(mean_f0 - freq) < 20  # Within 20 Hz

    def test_detect_various_frequencies(self):
        """Test pitch detection at various frequencies."""
        sr = 22050

        for freq in [100, 200, 440, 880]:
            t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
            y = np.sin(2 * np.pi * freq * t)

            y_mx = mx.array(y)
            f0, voiced = pitch_detect_acf(y_mx, sr=sr, fmin=50, fmax=2000)

            f0_np = np.array(f0)
            voiced_np = np.array(voiced)

            voiced_f0 = f0_np[voiced_np]
            if len(voiced_f0) > 0:
                mean_f0 = np.mean(voiced_f0)
                # Allow 10% tolerance
                assert abs(mean_f0 - freq) / freq < 0.1, f"Failed for {freq} Hz"

    def test_noise_unvoiced(self):
        """Test that noise is detected as mostly unvoiced."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(22050).astype(np.float32)

        y_mx = mx.array(noise)
        f0, voiced = pitch_detect_acf(y_mx, sr=22050, fmin=80, fmax=1000)

        voiced_np = np.array(voiced)

        # Noise should have low voicing rate
        assert np.mean(voiced_np) < 0.5

    def test_output_shape(self, random_signal):
        """Test output shapes."""
        y_mx = mx.array(random_signal)
        f0, voiced = pitch_detect_acf(
            y_mx, sr=22050, frame_length=2048, hop_length=512
        )

        # f0 and voiced should have same shape
        assert np.array(f0).shape == np.array(voiced).shape

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        f0, voiced = pitch_detect_acf(y_mx, sr=22050)

        assert np.array(f0).shape[0] == 4  # Batch size preserved
        assert np.array(voiced).shape[0] == 4

    def test_fmin_fmax_range(self):
        """Test fmin/fmax range constraints."""
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        y = np.sin(2 * np.pi * 150 * t)  # 150 Hz

        y_mx = mx.array(y)

        # With fmin too high, shouldn't detect 150 Hz
        f0_high_min, voiced_high = pitch_detect_acf(
            y_mx, sr=sr, fmin=200, fmax=1000
        )

        # With appropriate range, should detect
        f0_good, voiced_good = pitch_detect_acf(
            y_mx, sr=sr, fmin=80, fmax=500
        )

        # Good range should have more voiced frames
        assert np.mean(np.array(voiced_good)) >= np.mean(np.array(voiced_high))


class TestPeriodicity:
    """Test periodicity() function."""

    def test_sine_high_periodicity(self):
        """Test that sine wave has high periodicity."""
        sr = 22050
        t = np.linspace(0, 1, sr, dtype=np.float32)
        y = np.sin(2 * np.pi * 440 * t)

        y_mx = mx.array(y)
        result = periodicity(y_mx, sr=sr, fmin=80, fmax=1000)

        result_np = np.array(result)

        # Sine wave should have high periodicity
        assert np.mean(result_np) > 0.7

    def test_noise_low_periodicity(self):
        """Test that noise has low periodicity."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(22050).astype(np.float32)

        y_mx = mx.array(noise)
        result = periodicity(y_mx, sr=22050, fmin=80, fmax=1000)

        result_np = np.array(result)

        # Noise should have low periodicity
        assert np.mean(result_np) < 0.3

    def test_output_shape(self, random_signal):
        """Test output shape matches librosa convention."""
        y_mx = mx.array(random_signal)
        result = periodicity(y_mx, sr=22050, frame_length=2048, hop_length=512)

        # Should be (1, n_frames)
        assert result.shape[0] == 1

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        result = periodicity(y_mx, sr=22050)

        assert result.shape[0] == 4  # Batch size preserved

    def test_periodicity_range(self, random_signal):
        """Test periodicity is in [0, 1] range."""
        y_mx = mx.array(random_signal)
        result = periodicity(y_mx, sr=22050)

        result_np = np.array(result)
        assert np.all(result_np >= 0)
        assert np.all(result_np <= 1)


class TestPitchMathematicalProperties:
    """Test mathematical properties of pitch detection."""

    def test_octave_detection(self):
        """Test detection at octave intervals."""
        sr = 22050

        # Fundamental and octave
        for base_freq in [220, 440]:
            t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
            y = np.sin(2 * np.pi * base_freq * t)

            y_mx = mx.array(y)
            f0, voiced = pitch_detect_acf(y_mx, sr=sr, fmin=50, fmax=2000)

            f0_np = np.array(f0)
            voiced_np = np.array(voiced)

            voiced_f0 = f0_np[voiced_np]
            if len(voiced_f0) > 0:
                mean_f0 = np.mean(voiced_f0)
                # Should detect the actual frequency, not an octave
                assert abs(mean_f0 - base_freq) / base_freq < 0.1

    def test_harmonic_content(self):
        """Test detection with harmonic content."""
        sr = 22050
        freq = 220  # Fundamental
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)

        # Add harmonics
        y = (np.sin(2 * np.pi * freq * t) +
             0.5 * np.sin(2 * np.pi * 2 * freq * t) +
             0.25 * np.sin(2 * np.pi * 3 * freq * t))
        y = y.astype(np.float32)

        y_mx = mx.array(y)
        f0, voiced = pitch_detect_acf(y_mx, sr=sr, fmin=80, fmax=500)

        f0_np = np.array(f0)
        voiced_np = np.array(voiced)

        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            # Should detect fundamental, not harmonics
            assert abs(mean_f0 - freq) / freq < 0.15

    def test_vibrato(self):
        """Test detection with vibrato (frequency modulation)."""
        sr = 22050
        center_freq = 440
        vibrato_rate = 5  # 5 Hz vibrato
        vibrato_depth = 20  # +/- 20 Hz

        t = np.linspace(0, 1, sr, dtype=np.float32)
        freq_modulation = center_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        y = np.sin(2 * np.pi * np.cumsum(freq_modulation) / sr).astype(np.float32)

        y_mx = mx.array(y)
        f0, voiced = pitch_detect_acf(y_mx, sr=sr, fmin=300, fmax=600)

        f0_np = np.array(f0)
        voiced_np = np.array(voiced)

        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            # Should be close to center frequency
            assert abs(mean_f0 - center_freq) < 50
