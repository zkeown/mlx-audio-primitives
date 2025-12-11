"""
Mathematical property validation tests for mlx-audio-primitives.

These tests verify mathematical invariants and properties that should hold
for correct DSP implementations, providing confidence beyond reference comparison.
"""
import numpy as np
import pytest
import librosa
import scipy.signal
import mlx.core as mx

from mlx_audio_primitives import (
    stft,
    istft,
    magnitude,
    phase,
    mel_filterbank,
    melspectrogram,
    hz_to_mel,
    mel_to_hz,
    power_to_db,
    db_to_power,
    amplitude_to_db,
    db_to_amplitude,
    get_window,
)

# Test tolerance constants
# Standard tolerance for most floating-point comparisons
RTOL_STANDARD = 1e-4
# Strict tolerance for high-precision tests
RTOL_STRICT = 1e-5
ATOL_STRICT = 1e-5
# Relaxed tolerance for tests involving accumulated numerical error
RTOL_RELAXED = 1e-3
ATOL_RELAXED = 1e-6


class TestParsevalsTheorem:
    """
    Parseval's theorem states that the total energy in the time domain
    equals the total energy in the frequency domain.

    For DFT: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)

    For STFT with proper windowing and overlap, energy should be approximately
    preserved (within window normalization factors).
    """

    def test_parseval_rfft_basic(self):
        """Test Parseval's theorem for basic rfft."""
        np.random.seed(42)
        n = 1024
        x = np.random.randn(n).astype(np.float32)

        # Time domain energy
        time_energy = np.sum(x ** 2)

        # Frequency domain energy (rfft)
        x_mlx = mx.array(x)
        X = mx.fft.rfft(x_mlx)
        X_np = np.array(X)

        # For rfft, we need to account for the one-sided spectrum
        # DC and Nyquist are not doubled, middle frequencies are
        freq_energy = np.abs(X_np[0]) ** 2  # DC
        freq_energy += 2 * np.sum(np.abs(X_np[1:-1]) ** 2)  # Middle frequencies (doubled)
        freq_energy += np.abs(X_np[-1]) ** 2  # Nyquist
        freq_energy /= n

        np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-5)

    def test_parseval_stft_energy_conservation(self):
        """
        Test that STFT energy relates predictably to time-domain energy.

        For windowed STFT, the energy relationship depends on the window
        and overlap. We verify using librosa as reference.
        """
        np.random.seed(42)
        n = 8192
        x = np.random.randn(n).astype(np.float32)

        n_fft = 1024
        hop_length = 256  # 75% overlap for Hann window

        # Compare our STFT energy to librosa's
        x_mlx = mx.array(x)
        S_ours = stft(x_mlx, n_fft=n_fft, hop_length=hop_length, center=False)
        S_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, center=False)

        our_energy = np.sum(np.abs(np.array(S_ours)) ** 2)
        librosa_energy = np.sum(np.abs(S_librosa) ** 2)

        # Our energy should match librosa's energy closely
        np.testing.assert_allclose(
            our_energy, librosa_energy, rtol=1e-4,
            err_msg="STFT energy differs from librosa"
        )

    def test_parseval_round_trip_energy(self):
        """Test that STFT->ISTFT preserves signal energy."""
        np.random.seed(42)
        n = 4096
        x = np.random.randn(n).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        x_reconstructed = istft(S, hop_length=hop_length, length=n)
        x_reconstructed_np = np.array(x_reconstructed)

        original_energy = np.sum(x ** 2)
        reconstructed_energy = np.sum(x_reconstructed_np ** 2)

        np.testing.assert_allclose(
            original_energy, reconstructed_energy, rtol=1e-4,
            err_msg="STFT round-trip should preserve energy"
        )


class TestSTFTLinearity:
    """
    Test linearity property: STFT(ax + by) = a*STFT(x) + b*STFT(y)
    """

    def test_stft_linearity_addition(self):
        """Test STFT(x + y) = STFT(x) + STFT(y)."""
        np.random.seed(42)
        n = 4096
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        y_mlx = mx.array(y)
        xy_mlx = mx.array(x + y)

        S_x = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        S_y = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        S_xy = stft(xy_mlx, n_fft=n_fft, hop_length=hop_length)

        S_sum = S_x + S_y

        np.testing.assert_allclose(
            np.array(S_xy), np.array(S_sum), rtol=1e-5, atol=1e-5,
            err_msg="STFT should be linear: STFT(x+y) = STFT(x) + STFT(y)"
        )

    def test_stft_linearity_scaling(self):
        """Test STFT(ax) = a*STFT(x)."""
        np.random.seed(42)
        n = 4096
        x = np.random.randn(n).astype(np.float32)
        a = 2.5

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        ax_mlx = mx.array(a * x)

        S_x = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        S_ax = stft(ax_mlx, n_fft=n_fft, hop_length=hop_length)

        np.testing.assert_allclose(
            np.array(S_ax), a * np.array(S_x), rtol=1e-5, atol=1e-5,
            err_msg="STFT should be linear: STFT(ax) = a*STFT(x)"
        )

    def test_stft_linearity_full(self):
        """Test full linearity: STFT(ax + by) = a*STFT(x) + b*STFT(y)."""
        np.random.seed(42)
        n = 4096
        x = np.random.randn(n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        a, b = 1.5, -0.7

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        y_mlx = mx.array(y)
        combined_mlx = mx.array(a * x + b * y)

        S_x = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        S_y = stft(y_mlx, n_fft=n_fft, hop_length=hop_length)
        S_combined = stft(combined_mlx, n_fft=n_fft, hop_length=hop_length)

        S_expected = a * np.array(S_x) + b * np.array(S_y)

        np.testing.assert_allclose(
            np.array(S_combined), S_expected, rtol=1e-5, atol=1e-5,
            err_msg="STFT should satisfy full linearity"
        )


class TestPureToneAccuracy:
    """
    Test that pure tones are correctly localized in frequency.
    A sine wave at frequency f should produce a peak at bin f*n_fft/sr.
    """

    @pytest.mark.parametrize("frequency", [440, 1000, 2000, 5000])
    def test_pure_tone_peak_location(self, frequency):
        """Test that a pure tone produces a peak at the correct frequency bin."""
        sr = 22050
        duration = 1.0
        n_fft = 2048

        # Generate pure tone
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        x = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=512)
        mag = np.array(magnitude(S))

        # Average magnitude across time frames (excluding edges)
        avg_mag = mag[:, 2:-2].mean(axis=1)

        # Find peak bin
        peak_bin = np.argmax(avg_mag)

        # Expected bin
        freq_resolution = sr / n_fft
        expected_bin = int(round(frequency / freq_resolution))

        # Peak should be at expected bin (within 1 bin due to windowing)
        assert abs(peak_bin - expected_bin) <= 1, \
            f"Pure tone at {frequency}Hz: peak at bin {peak_bin}, expected {expected_bin}"

    def test_pure_tone_magnitude_concentration(self):
        """Test that pure tone energy is concentrated near the tone frequency."""
        sr = 22050
        frequency = 1000
        duration = 1.0
        n_fft = 2048

        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        x = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=512)
        mag = np.array(magnitude(S))

        # Average magnitude across time
        avg_mag = mag.mean(axis=1)

        # Find peak and its neighbors
        peak_bin = np.argmax(avg_mag)

        # Energy in peak region (peak +/- 3 bins)
        peak_region = slice(max(0, peak_bin - 3), min(len(avg_mag), peak_bin + 4))
        peak_energy = np.sum(avg_mag[peak_region] ** 2)
        total_energy = np.sum(avg_mag ** 2)

        # Most energy should be in peak region (>90%)
        concentration = peak_energy / total_energy
        assert concentration > 0.9, \
            f"Pure tone energy concentration {concentration:.2%} < 90%"


class TestDCOffsetHandling:
    """Test handling of DC offset (zero frequency component)."""

    def test_dc_offset_in_stft(self):
        """Test that DC offset appears only in bin 0."""
        np.random.seed(42)
        n = 4096
        dc_offset = 0.5
        x = np.random.randn(n).astype(np.float32) * 0.1 + dc_offset

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        mag = np.array(magnitude(S))

        # DC component should be significant in bin 0
        dc_magnitude = mag[0, :].mean()

        # Average magnitude of other low-frequency bins
        low_freq_magnitude = mag[1:10, :].mean()

        # DC should be larger (signal has DC offset)
        assert dc_magnitude > low_freq_magnitude, \
            "DC offset should produce larger magnitude in bin 0"

    def test_dc_offset_round_trip(self):
        """Test that DC offset is preserved through STFT->ISTFT."""
        n = 4096
        dc_offset = 0.5
        x = np.ones(n, dtype=np.float32) * dc_offset

        n_fft = 1024
        hop_length = 256

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        x_reconstructed = istft(S, hop_length=hop_length, length=n)
        x_reconstructed_np = np.array(x_reconstructed)

        # Mean should be preserved
        np.testing.assert_allclose(
            x_reconstructed_np.mean(), dc_offset, rtol=1e-3,
            err_msg="DC offset should be preserved through STFT round-trip"
        )


class TestExtremeEdgeCases:
    """Test behavior with extreme parameter values and edge cases."""

    def test_very_short_signal(self):
        """Test STFT with signal shorter than n_fft."""
        x = np.random.randn(512).astype(np.float32)
        n_fft = 1024

        x_mlx = mx.array(x)
        # With center=True, signal is padded
        S = stft(x_mlx, n_fft=n_fft, hop_length=256, center=True)

        # Should not raise, output should have valid shape
        assert S.shape[0] == n_fft // 2 + 1
        assert S.shape[1] >= 1

    def test_signal_exactly_n_fft(self):
        """Test STFT with signal exactly n_fft samples."""
        n_fft = 1024
        x = np.random.randn(n_fft).astype(np.float32)

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=256, center=True)

        assert S.shape[0] == n_fft // 2 + 1

    def test_minimum_hop_length(self):
        """Test STFT with hop_length=1 (maximum overlap)."""
        x = np.random.randn(256).astype(np.float32)
        n_fft = 64
        hop_length = 1

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length, center=False)

        # Should produce many frames
        expected_frames = 1 + (len(x) - n_fft) // hop_length
        assert S.shape[1] == expected_frames

    def test_hop_length_equals_n_fft(self):
        """Test STFT with hop_length=n_fft (no overlap)."""
        x = np.random.randn(4096).astype(np.float32)
        n_fft = 512
        hop_length = 512

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length, center=False)

        expected_frames = len(x) // n_fft
        assert S.shape[1] == expected_frames

    def test_small_n_fft(self):
        """Test STFT with very small n_fft."""
        x = np.random.randn(1024).astype(np.float32)
        n_fft = 32
        hop_length = 8

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)

        assert S.shape[0] == n_fft // 2 + 1

    def test_large_n_fft(self):
        """Test STFT with large n_fft."""
        x = np.random.randn(16384).astype(np.float32)
        n_fft = 8192
        hop_length = 2048

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)

        assert S.shape[0] == n_fft // 2 + 1

    @pytest.mark.parametrize("n_fft", [64, 128, 256, 512, 1024, 2048, 4096])
    def test_round_trip_various_n_fft(self, n_fft):
        """Test STFT round-trip with various n_fft values."""
        np.random.seed(42)
        x = np.random.randn(4096).astype(np.float32)
        hop_length = n_fft // 4

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=n_fft, hop_length=hop_length)
        x_reconstructed = istft(S, hop_length=hop_length, length=len(x))

        np.testing.assert_allclose(
            np.array(x_reconstructed), x, rtol=1e-4, atol=1e-4,
            err_msg=f"Round-trip failed for n_fft={n_fft}"
        )


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_near_zero_values(self):
        """Test handling of very small amplitude signals."""
        x = np.random.randn(4096).astype(np.float32) * 1e-7

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=1024, hop_length=256)

        # Should not produce NaN or Inf
        S_np = np.array(S)
        assert not np.any(np.isnan(S_np)), "STFT produced NaN for small values"
        assert not np.any(np.isinf(S_np)), "STFT produced Inf for small values"

    def test_large_values(self):
        """Test handling of large amplitude signals."""
        x = np.random.randn(4096).astype(np.float32) * 1e4

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=1024, hop_length=256)

        S_np = np.array(S)
        assert not np.any(np.isnan(S_np)), "STFT produced NaN for large values"
        assert not np.any(np.isinf(S_np)), "STFT produced Inf for large values"

    def test_db_conversion_near_zero(self):
        """Test dB conversion doesn't produce -Inf for small values."""
        x = np.array([1e-15, 1e-10, 1e-5, 1.0], dtype=np.float32)
        x_mlx = mx.array(x)

        # Power to dB with amin protection
        db = power_to_db(x_mlx, amin=1e-10)
        db_np = np.array(db)

        assert not np.any(np.isnan(db_np)), "power_to_db produced NaN"
        assert not np.any(np.isinf(db_np)), "power_to_db produced Inf"

    def test_db_conversion_round_trip_precision(self):
        """Test dB conversion round-trip maintains precision."""
        x = np.logspace(-5, 2, 100).astype(np.float32)
        x_mlx = mx.array(x)

        # Power round-trip
        db = power_to_db(x_mlx, top_db=None, amin=1e-10)
        recovered = db_to_power(db)

        # Clip to valid range before comparison
        x_clipped = np.maximum(x, 1e-10)

        np.testing.assert_allclose(
            np.array(recovered), x_clipped, rtol=1e-5,
            err_msg="dB round-trip lost precision"
        )

    def test_mel_filterbank_sum_to_one(self):
        """Test that mel filterbank filters have reasonable properties."""
        sr = 22050
        n_fft = 2048
        n_mels = 128

        fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, norm=None)
        fb_np = np.array(fb)

        # Each frequency bin should be covered by at most a few filters
        coverage = (fb_np > 0).sum(axis=0)
        assert coverage.max() <= 3, "Too many overlapping filters"

        # Most middle frequencies should be covered
        middle_coverage = coverage[10:-10]
        assert (middle_coverage > 0).mean() > 0.9, "Poor frequency coverage"


class TestCrossValidationWithLibrosa:
    """Additional cross-validation tests against librosa."""

    def test_stft_phase_continuity(self):
        """Test that phase is continuous for slowly varying signals."""
        sr = 22050
        duration = 0.5
        frequency = 440

        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        x = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        x_mlx = mx.array(x)
        S = stft(x_mlx, n_fft=2048, hop_length=512)
        ph = np.array(phase(S))

        # Find the bin closest to 440 Hz
        freq_bin = int(round(frequency * 2048 / sr))

        # Phase should change smoothly between frames
        phase_diff = np.diff(np.unwrap(ph[freq_bin, :]))

        # Phase difference should be relatively constant for a pure tone
        phase_std = np.std(phase_diff)
        assert phase_std < 0.1, f"Phase not continuous: std={phase_std}"

    def test_melspectrogram_energy_distribution(self):
        """Test that melspectrogram distributes energy reasonably."""
        np.random.seed(42)
        x = np.random.randn(22050).astype(np.float32)  # 1 second of noise

        x_mlx = mx.array(x)
        mel = melspectrogram(x_mlx, sr=22050, n_mels=128, power=2.0)
        mel_np = np.array(mel)

        # White noise should have relatively even energy across mel bands
        # (with some variation due to mel scale compression at high frequencies)
        band_energies = mel_np.mean(axis=1)

        # Energy should be present in all bands
        assert np.all(band_energies > 0), "Some mel bands have zero energy"

        # Ratio of max to min shouldn't be extreme for white noise
        energy_ratio = band_energies.max() / band_energies.min()
        assert energy_ratio < 100, f"Energy distribution too uneven: ratio={energy_ratio}"

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman"])
    def test_stft_window_comparison(self, window_type):
        """Test STFT with different windows against librosa."""
        np.random.seed(42)
        x = np.random.randn(4096).astype(np.float32)

        n_fft = 1024
        hop_length = 256

        # librosa
        expected = librosa.stft(
            x, n_fft=n_fft, hop_length=hop_length, window=window_type
        )

        # Our implementation
        x_mlx = mx.array(x)
        actual = stft(x_mlx, n_fft=n_fft, hop_length=hop_length, window=window_type)

        np.testing.assert_allclose(
            np.array(actual), expected, rtol=1e-4, atol=1e-4,
            err_msg=f"STFT mismatch for window={window_type}"
        )

    def test_mel_scale_monotonicity(self):
        """Test that mel scale is monotonically increasing."""
        freqs = np.linspace(0, 8000, 1000)

        # Slaney
        mels_slaney = hz_to_mel(freqs, htk=False)
        assert np.all(np.diff(mels_slaney) >= 0), "Slaney mel scale not monotonic"

        # HTK
        mels_htk = hz_to_mel(freqs, htk=True)
        assert np.all(np.diff(mels_htk) >= 0), "HTK mel scale not monotonic"

    def test_mel_filterbank_triangular_shape(self):
        """Test that mel filters have proper triangular shape."""
        fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=40, norm=None)
        fb_np = np.array(fb)

        for i in range(fb_np.shape[0]):
            filt = fb_np[i, :]
            nonzero_idx = np.where(filt > 0)[0]

            if len(nonzero_idx) > 2:
                # Find peak
                peak_idx = nonzero_idx[np.argmax(filt[nonzero_idx])]

                # Values before peak should increase
                before_peak = filt[nonzero_idx[nonzero_idx < peak_idx]]
                if len(before_peak) > 1:
                    assert np.all(np.diff(before_peak) >= -1e-6), \
                        f"Filter {i} not monotonically increasing before peak"

                # Values after peak should decrease
                after_peak = filt[nonzero_idx[nonzero_idx > peak_idx]]
                if len(after_peak) > 1:
                    assert np.all(np.diff(after_peak) <= 1e-6), \
                        f"Filter {i} not monotonically decreasing after peak"


class TestWindowProperties:
    """Test mathematical properties of window functions."""

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman", "bartlett"])
    def test_window_symmetry(self, window_type):
        """Test that symmetric windows are actually symmetric."""
        n = 1024
        window = get_window(window_type, n, fftbins=False)
        window_np = np.array(window)

        # Symmetric window should be perfectly symmetric
        # (computed in float64, cast to float32)
        np.testing.assert_allclose(
            window_np, window_np[::-1], rtol=1e-7, atol=0,
            err_msg=f"{window_type} window not symmetric"
        )

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman", "bartlett"])
    def test_window_endpoints(self, window_type):
        """Test window endpoint values."""
        n = 1024
        window = get_window(window_type, n, fftbins=False)
        window_np = np.array(window)

        if window_type in ["hann", "blackman"]:
            # These windows should be zero at endpoints
            assert window_np[0] < 0.01, f"{window_type} window start not near zero"
            assert window_np[-1] < 0.01, f"{window_type} window end not near zero"

        # All windows should peak near center
        peak_idx = np.argmax(window_np)
        center = n // 2
        assert abs(peak_idx - center) <= 1, f"{window_type} window peak not at center"

    @pytest.mark.parametrize("window_type", ["hann", "hamming", "blackman", "bartlett"])
    def test_window_non_negative(self, window_type):
        """Test that windows are non-negative."""
        window = get_window(window_type, 1024, fftbins=True)
        window_np = np.array(window)

        # Windows should be non-negative (computed in float64)
        min_val = window_np.min()
        assert min_val >= 0, \
            f"{window_type} window has negative values: {min_val}"

    def test_periodic_window_differs_from_symmetric(self):
        """Test that periodic and symmetric windows differ correctly."""
        n = 1024

        periodic = get_window("hann", n, fftbins=True)
        symmetric = get_window("hann", n, fftbins=False)

        periodic_np = np.array(periodic)
        symmetric_np = np.array(symmetric)

        # They should be different
        assert not np.allclose(periodic_np, symmetric_np), \
            "Periodic and symmetric windows should differ"

        # Periodic window's last element should match symmetric's second-to-last
        # (because periodic is computed as symmetric[:-1] from n+1 points)
        symmetric_n1 = get_window("hann", n + 1, fftbins=False)
        symmetric_n1_np = np.array(symmetric_n1)

        np.testing.assert_allclose(
            periodic_np, symmetric_n1_np[:-1], rtol=1e-6,
            err_msg="Periodic window doesn't match symmetric[:-1] from n+1"
        )
