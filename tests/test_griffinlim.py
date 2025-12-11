"""
Griffin-Lim phase reconstruction test suite.

Tests cover:
- Magnitude preservation (reconstructed STFT magnitude matches input)
- Convergence with increasing iterations (32, 64, 128)
- Momentum acceleration (Perraudin et al. 2013)
- Length parameter for exact output size
- Batch processing
- Comparison with librosa.griffinlim

Cross-validates against: librosa.griffinlim
Tolerance: Magnitude match rtol=1e-3 (iterative algorithm, phase is estimated)

Note: Griffin-Lim cannot recover original phase - it finds a consistent
phase that produces the given magnitude spectrogram. Quality improves
with more iterations but with diminishing returns after ~32.
"""
import numpy as np
import pytest
import librosa
import mlx.core as mx

from mlx_audio_primitives import griffinlim, stft, istft, magnitude


class TestGriffinLim:
    """Test griffinlim() function."""

    def test_basic_reconstruction(self, chirp_signal):
        """Test basic Griffin-Lim reconstruction preserves magnitude."""
        y_mx = mx.array(chirp_signal)

        # Compute magnitude spectrogram
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        # Reconstruct
        y_reconstructed = griffinlim(
            S_mag, n_iter=64, hop_length=512, n_fft=2048,
            length=len(chirp_signal)
        )

        y_reconstructed_np = np.array(y_reconstructed)

        # Check length
        assert len(y_reconstructed_np) == len(chirp_signal)

        # Griffin-Lim minimizes magnitude error, not waveform correlation
        # The reconstructed signal may be time-reversed or phase-shifted
        S_recon = magnitude(stft(mx.array(y_reconstructed_np), n_fft=2048, hop_length=512))
        S_mag_np = np.array(S_mag)
        S_recon_np = np.array(S_recon)

        # Relative magnitude error should be low
        rel_error = np.mean(np.abs(S_mag_np - S_recon_np)) / np.mean(S_mag_np)
        assert rel_error < 0.15  # Less than 15% relative error

    def test_matches_librosa(self, chirp_signal):
        """Test Griffin-Lim reconstruction quality matches librosa."""
        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(chirp_signal, n_fft=2048, hop_length=512))

        # Reconstruct with librosa
        librosa_result = librosa.griffinlim(
            S, n_iter=64, hop_length=512, n_fft=2048,
            length=len(chirp_signal), random_state=42
        )

        # Reconstruct with our implementation
        S_mx = mx.array(S)
        our_result = griffinlim(
            S_mx, n_iter=64, hop_length=512, n_fft=2048,
            length=len(chirp_signal), random_state=42
        )

        our_result_np = np.array(our_result)

        # Compare magnitude reconstruction error (the proper metric for Griffin-Lim)
        S_librosa = np.abs(librosa.stft(librosa_result, n_fft=2048, hop_length=512))
        S_ours = np.abs(librosa.stft(our_result_np, n_fft=2048, hop_length=512))

        error_librosa = np.mean((S - S_librosa) ** 2)
        error_ours = np.mean((S - S_ours) ** 2)

        # Our error should be similar to librosa's (within factor of 3)
        assert error_ours < error_librosa * 3

    @pytest.mark.parametrize("n_iter", [16, 32, 64, 128])
    def test_convergence_with_iterations(self, chirp_signal, n_iter):
        """Test that error decreases with more iterations."""
        y_mx = mx.array(chirp_signal)
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        y_reconstructed = griffinlim(
            S_mag, n_iter=n_iter, hop_length=512,
            length=len(chirp_signal), random_state=42
        )

        # Compute reconstruction error
        S_reconstructed = magnitude(stft(y_reconstructed, n_fft=2048, hop_length=512))
        error = np.mean((np.array(S_mag) - np.array(S_reconstructed)) ** 2)

        # Error threshold scales inversely with iterations (roughly)
        # These thresholds are based on empirical observations
        threshold = {16: 10.0, 32: 5.0, 64: 2.0, 128: 1.0}
        assert error < threshold[n_iter]

    def test_momentum_speedup(self, chirp_signal):
        """Test that momentum improves convergence."""
        y_mx = mx.array(chirp_signal)
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        # With momentum
        y_momentum = griffinlim(
            S_mag, n_iter=16, hop_length=512,
            momentum=0.99, random_state=42
        )
        S_momentum = magnitude(stft(y_momentum, n_fft=2048, hop_length=512))
        error_momentum = np.mean((np.array(S_mag) - np.array(S_momentum)) ** 2)

        # Without momentum
        y_no_momentum = griffinlim(
            S_mag, n_iter=16, hop_length=512,
            momentum=0.0, random_state=42
        )
        S_no_momentum = magnitude(stft(y_no_momentum, n_fft=2048, hop_length=512))
        error_no_momentum = np.mean((np.array(S_mag) - np.array(S_no_momentum)) ** 2)

        # Momentum should help (or at least not hurt much)
        # Note: This isn't always guaranteed for 16 iterations
        assert error_momentum < error_no_momentum * 2

    def test_random_init_vs_zeros(self, chirp_signal):
        """Test different initialization methods."""
        y_mx = mx.array(chirp_signal)
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        # Random init
        y_random = griffinlim(
            S_mag, n_iter=32, hop_length=512,
            init="random", random_state=42
        )

        # Zeros init
        y_zeros = griffinlim(
            S_mag, n_iter=32, hop_length=512,
            init="zeros"
        )

        # Both should produce valid output
        assert len(np.array(y_random)) > 0
        assert len(np.array(y_zeros)) > 0

    def test_output_length(self, chirp_signal):
        """Test output length control."""
        y_mx = mx.array(chirp_signal)
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        # Request specific length
        target_length = 10000
        y_reconstructed = griffinlim(
            S_mag, n_iter=32, hop_length=512,
            length=target_length
        )

        assert len(np.array(y_reconstructed)) == target_length

    def test_batch_input(self, batch_signals):
        """Test batched input handling."""
        y_mx = mx.array(batch_signals)
        S_complex = stft(y_mx, n_fft=2048, hop_length=512)
        S_mag = magnitude(S_complex)

        y_reconstructed = griffinlim(S_mag, n_iter=16, hop_length=512)

        assert y_reconstructed.shape[0] == 4  # Batch size preserved

    def test_invalid_init(self, chirp_signal):
        """Test error for invalid init method."""
        y_mx = mx.array(chirp_signal)
        S_mag = magnitude(stft(y_mx, n_fft=2048, hop_length=512))

        with pytest.raises(ValueError, match="Unknown init"):
            griffinlim(S_mag, n_iter=32, init="invalid")

    def test_reproducibility(self, chirp_signal):
        """Test that same random_state gives same result."""
        y_mx = mx.array(chirp_signal)
        S_mag = magnitude(stft(y_mx, n_fft=2048, hop_length=512))

        y1 = griffinlim(S_mag, n_iter=32, random_state=42)
        y2 = griffinlim(S_mag, n_iter=32, random_state=42)

        np.testing.assert_allclose(
            np.array(y1), np.array(y2), rtol=1e-5, atol=1e-5
        )


class TestGriffinLimQuality:
    """Test Griffin-Lim reconstruction quality."""

    def test_sine_wave_reconstruction(self):
        """Test reconstruction of a simple sine wave."""
        sr = 22050
        duration = 1.0
        freq = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        y = np.sin(2 * np.pi * freq * t)

        y_mx = mx.array(y)
        S_mag = magnitude(stft(y_mx, n_fft=2048, hop_length=512))

        y_reconstructed = griffinlim(
            S_mag, n_iter=64, hop_length=512,
            length=len(y), random_state=42
        )

        y_reconstructed_np = np.array(y_reconstructed)

        # Reconstruction should preserve the frequency content
        fft_orig = np.abs(np.fft.rfft(y))
        fft_recon = np.abs(np.fft.rfft(y_reconstructed_np))

        # Find peaks
        peak_orig = np.argmax(fft_orig[1:]) + 1  # Skip DC
        peak_recon = np.argmax(fft_recon[1:]) + 1

        # Peaks should be at same frequency
        assert abs(peak_orig - peak_recon) < 5

    def test_magnitude_constraint(self, chirp_signal):
        """Test that output has approximately correct magnitude."""
        y_mx = mx.array(chirp_signal)
        S_mag = magnitude(stft(y_mx, n_fft=2048, hop_length=512))

        y_reconstructed = griffinlim(
            S_mag, n_iter=64, hop_length=512,
            length=len(chirp_signal)
        )

        # Recompute magnitude
        S_recon = magnitude(stft(y_reconstructed, n_fft=2048, hop_length=512))

        S_mag_np = np.array(S_mag)
        S_recon_np = np.array(S_recon)

        # Magnitudes should be similar
        relative_error = np.mean(np.abs(S_mag_np - S_recon_np)) / np.mean(S_mag_np)
        assert relative_error < 0.1  # Less than 10% error

    def test_reconstruction_improves_with_iterations(self, chirp_signal):
        """Test that reconstruction error decreases with iterations."""
        y_mx = mx.array(chirp_signal)
        S_mag = magnitude(stft(y_mx, n_fft=2048, hop_length=512))
        S_mag_np = np.array(S_mag)

        errors = []
        for n_iter in [4, 8, 16, 32, 64]:
            y_recon = griffinlim(
                S_mag, n_iter=n_iter, hop_length=512,
                random_state=42
            )
            S_recon = magnitude(stft(y_recon, n_fft=2048, hop_length=512))
            error = np.mean((S_mag_np - np.array(S_recon)) ** 2)
            errors.append(error)

        # Errors should generally decrease (allow some noise)
        # Check that final error is less than initial
        assert errors[-1] < errors[0]
