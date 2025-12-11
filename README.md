# mlx-audio-primitives

> *Sound engineering for Apple Silicon*

Foundational audio DSP primitives for [Apple MLX](https://github.com/ml-explore/mlx), built on Metal. Drop-in replacements for librosa's core functions, optimized for the M-series chips in your Mac.

*Built for Metal, because your audio deserves to be heavy.*

## Why mlx-audio-primitives?

- **Native Apple Silicon** - Runs on Metal GPU, not CPU-bound NumPy
- **librosa-compatible API** - Same function signatures, same results (within float32 precision)
- **Perfect reconstruction** - ISTFT inverts STFT exactly, we won't drop a single beat
- **Aggressive caching** - Window functions and filterbanks computed once, reused forever
- **Battle-tested** - 245 tests validating against librosa, scipy, and torchaudio

## Installation

```bash
pip install mlx-audio-primitives
```

Or install from source:

```bash
git clone https://github.com/yourusername/mlx-audio-primitives.git
cd mlx-audio-primitives
pip install -e .
```

## Quick Start (No Amp Required)

```python
import mlx.core as mx
from mlx_audio_primitives import stft, istft, melspectrogram, power_to_db

# Load your audio (shape: samples or batch x samples)
audio = mx.array(your_audio_data)

# STFT - transform at the frequency of thought
spectrogram = stft(audio, n_fft=2048, hop_length=512)

# Perfect reconstruction - your signal comes back intact
reconstructed = istft(spectrogram, hop_length=512, length=audio.shape[-1])

# Mel spectrogram - the bread and butter of speech/music ML
mel = melspectrogram(audio, sr=22050, n_mels=128)
mel_db = power_to_db(mel)  # Log scale for visualization/training
```

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STFT Pipeline                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Signal ──▶ Pad ──▶ Frame ──▶ Window ──▶ FFT ──▶ Complex Spectrogram       │
│     │         │        │         │        │                                  │
│     │         │        │         │        └── mx.fft.rfft (Metal)           │
│     │         │        │         └── Cached window * frames                  │
│     │         │        └── as_strided or gather-based                        │
│     │         └── reflect/constant/edge padding                              │
│     └── (batch, samples) or (samples,)                                       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           ISTFT Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Spectrogram ──▶ IFFT ──▶ Overlap-Add ──▶ Normalize ──▶ Trim ──▶ Signal   │
│                     │           │              │                             │
│                     │           │              └── Divide by squared window  │
│                     │           └── Custom Metal kernel (atomic scatter-add) │
│                     └── mx.fft.irfft                                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Mel Spectrogram Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Signal ──▶ STFT ──▶ |Magnitude|^power ──▶ Mel Filterbank @ ──▶ Mel Spec  │
│                                                    │                         │
│                                                    └── Cached (n_mels, bins) │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Performance

Benchmarked on Apple Silicon (results vary by chip):

| Operation | vs librosa | vs torchaudio | Notes |
|-----------|------------|---------------|-------|
| STFT (n_fft=2048) | ~1x | ~0.5x | Competitive with CPU libs |
| Mel Spectrogram | **2x faster** | ~0.8x | Real-world pipeline |
| Window Functions | **25-90x faster** | - | Cached after first call |
| Mel Filterbank | **380-1200x faster** | - | Cached after first call |
| ISTFT Round-trip | - | - | Max error: 1.2e-6 |

*The massive speedups for windows/filterbanks come from our two-tier caching strategy. Once computed, they're stored as MLX arrays on the GPU - no CPU-to-GPU transfer on subsequent calls.*

## Features

### STFT/ISTFT

```python
from mlx_audio_primitives import stft, istft, magnitude, phase

# Forward transform
S = stft(y, n_fft=2048, hop_length=512, window='hann', center=True)
# S.shape: (1025, n_frames) for 1D, (batch, 1025, n_frames) for batched

# Get magnitude and phase
mag = magnitude(S)   # |S|
phi = phase(S)       # angle(S) in radians

# Perfect reconstruction
y_hat = istft(S, hop_length=512, length=len(y))
# max(|y - y_hat|) < 1e-5 for COLA-compliant windows
```

### Mel Spectrogram

```python
from mlx_audio_primitives import melspectrogram, mel_filterbank, power_to_db

# End-to-end mel spectrogram
mel = melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)

# Or build it yourself
S = stft(y, n_fft=2048, hop_length=512)
mel_fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=128)  # Cached!
mel = mel_fb @ magnitude(S) ** 2

# Log scale (standard for ML)
mel_db = power_to_db(mel, ref=mx.max, top_db=80.0)
```

### Window Functions

```python
from mlx_audio_primitives import get_window

# Supported: hann, hamming, blackman, bartlett, rectangular
# Aliases: hanning=hann, triangular=bartlett, boxcar/ones=rectangular
window = get_window('hann', 2048, fftbins=True)

# Or pass your own
custom_window = mx.kaiser(2048, beta=14.0)
S = stft(y, window=custom_window)
```

### dB Conversions

```python
from mlx_audio_primitives import (
    power_to_db, db_to_power,
    amplitude_to_db, db_to_amplitude
)

# Power spectrogram to dB (10 * log10)
S_power = magnitude(stft(y)) ** 2
S_db = power_to_db(S_power, ref=1.0, top_db=80.0)

# Amplitude to dB (20 * log10)
S_amp = magnitude(stft(y))
S_db = amplitude_to_db(S_amp, ref=mx.max)  # Normalize to peak

# Round-trip
S_power_back = db_to_power(S_db)
```

### Spectral Features

*Measure the spectrum of possibilities.*

```python
from mlx_audio_primitives import (
    spectral_centroid, spectral_bandwidth, spectral_rolloff,
    spectral_flatness, spectral_contrast, zero_crossing_rate
)

# Spectral centroid - "brightness" of the sound
centroid = spectral_centroid(y, sr=22050)  # Shape: (1, n_frames)

# Spectral bandwidth - spread around the centroid
bandwidth = spectral_bandwidth(y, sr=22050)

# Spectral rolloff - frequency below which 85% of energy lies
rolloff = spectral_rolloff(y, sr=22050, roll_percent=0.85)

# Spectral flatness - how noise-like vs tonal (0=tonal, 1=noise)
flatness = spectral_flatness(y)

# Spectral contrast - peak-to-valley ratio per octave band
contrast = spectral_contrast(y, sr=22050, n_bands=6)  # Shape: (7, n_frames)

# Zero crossing rate - useful for speech/music discrimination
zcr = zero_crossing_rate(y, frame_length=2048, hop_length=512)
```

### MFCC & Cepstral Features

*MFCCs - because cepstral coefficients deserve love too.*

```python
from mlx_audio_primitives import mfcc, delta

# Mel-Frequency Cepstral Coefficients - standard for speech/speaker recognition
mfccs = mfcc(y, sr=22050, n_mfcc=13, n_mels=128)  # Shape: (13, n_frames)

# Delta features - capture temporal dynamics
delta1 = delta(mfccs, width=9, order=1)   # First derivative
delta2 = delta(mfccs, width=9, order=2)   # Second derivative (acceleration)

# Stack for ML input (39-dimensional feature vector)
features = mx.concatenate([mfccs, delta1, delta2], axis=0)
```

### Additional Filterbanks

*The Bark scale - no dogs were harmed in this filterbank.*

```python
from mlx_audio_primitives import (
    linear_filterbank, bark_filterbank,
    hz_to_bark, bark_to_hz
)

# Linear filterbank - equal-width frequency bands
linear_fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)

# Bark filterbank - psychoacoustic scale matching human hearing
bark_fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)

# Frequency conversions
bark_vals = hz_to_bark(freqs, formula='zwicker')  # or 'traunmuller'
hz_vals = bark_to_hz(bark_vals, formula='zwicker')
```

### Time-Domain Primitives

```python
from mlx_audio_primitives import frame, rms, preemphasis, deemphasis

# Frame signal into overlapping windows
frames = frame(y, frame_length=2048, hop_length=512)  # (n_frames, 2048)

# RMS energy per frame
energy = rms(y, frame_length=2048, hop_length=512)  # Shape: (1, n_frames)

# Pre-emphasis - boost high frequencies (common for speech)
y_emph = preemphasis(y, coef=0.97)

# De-emphasis - inverse operation
y_recovered = deemphasis(y_emph, coef=0.97)
```

### Resampling

*We'll resample your audio at any rate - we're flexible like that.*

```python
from mlx_audio_primitives import resample, resample_poly

# FFT-based resampling (high quality, bandlimited)
y_16k = resample(y_44k, orig_sr=44100, target_sr=16000)

# Polyphase resampling (efficient for integer ratios)
y_22k = resample_poly(y_44k, up=1, down=2)  # 44100 -> 22050
```

### Phase Reconstruction

*Phase reconstruction: we iterate until we get it right.*

```python
from mlx_audio_primitives import griffinlim

# Reconstruct audio from magnitude spectrogram
S_mag = magnitude(stft(y))
y_reconstructed = griffinlim(
    S_mag,
    n_iter=32,          # More iterations = better quality
    momentum=0.99,      # Accelerates convergence
    hop_length=512
)
```

### Pitch & Periodicity

*Finding your fundamental frequency - it's a peak experience.*

```python
from mlx_audio_primitives import autocorrelation, pitch_detect_acf, periodicity

# Autocorrelation via FFT (Wiener-Khinchin theorem)
r = autocorrelation(y, max_lag=2000, normalize=True)

# Pitch detection using autocorrelation
f0, voiced = pitch_detect_acf(y, sr=22050, fmin=80, fmax=500)
# f0: fundamental frequency per frame, voiced: boolean mask

# Periodicity strength (0=noise, 1=perfectly periodic)
p = periodicity(y, sr=22050)
```

## Coming from librosa?

The API is designed to be a drop-in replacement:

```python
# librosa
import librosa
S = librosa.stft(y, n_fft=2048, hop_length=512)
mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128)

# mlx-audio-primitives (same API!)
from mlx_audio_primitives import stft, melspectrogram
S = stft(mx.array(y), n_fft=2048, hop_length=512)
mel = melspectrogram(mx.array(y), sr=22050, n_mels=128)
```

Key differences:

- Input must be `mx.array`, not numpy (use `mx.array(y)`)
- Output is `mx.array` (use `np.array(result)` if needed)
- All computation is float32 (matches MLX optimization)
- Numerical differences up to ~5e-5 due to FFT implementation

See [NUMERICAL_ACCURACY.md](NUMERICAL_ACCURACY.md) for detailed precision guarantees.

## API Reference

### STFT Functions

| Function | Description |
|----------|-------------|
| `stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')` | Short-Time Fourier Transform |
| `istft(stft_matrix, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, length=None)` | Inverse STFT with perfect reconstruction |
| `magnitude(stft_matrix)` | Compute magnitude spectrogram |
| `phase(stft_matrix)` | Compute phase spectrogram (radians) |
| `check_nola(window, hop_length, n_fft, tol=1e-10)` | Verify NOLA constraint for invertibility |

### Mel Functions

| Function | Description |
|----------|-------------|
| `mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')` | Create mel filterbank matrix |
| `melspectrogram(y, sr=22050, n_fft=2048, hop_length=None, ...)` | Compute mel spectrogram end-to-end |
| `hz_to_mel(frequencies, htk=False)` | Convert Hz to mel scale |
| `mel_to_hz(mels, htk=False)` | Convert mel scale to Hz |

### dB Conversion Functions

| Function | Description |
|----------|-------------|
| `power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)` | Power spectrogram to decibels |
| `db_to_power(S_db, ref=1.0)` | Decibels back to power |
| `amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0)` | Amplitude spectrogram to decibels |
| `db_to_amplitude(S_db, ref=1.0)` | Decibels back to amplitude |

### Window Function

| Function | Description |
|----------|-------------|
| `get_window(window, n_fft, fftbins=True)` | Get window by name or pass through custom array |

### Spectral Feature Functions

| Function | Description |
|----------|-------------|
| `spectral_centroid(y, sr=22050, S=None, ...)` | Center of mass of spectrum (brightness) |
| `spectral_bandwidth(y, sr=22050, S=None, p=2.0, ...)` | Weighted spread around centroid |
| `spectral_rolloff(y, sr=22050, roll_percent=0.85, ...)` | Frequency containing X% of energy |
| `spectral_flatness(y, S=None, ...)` | Wiener entropy (noise-likeness, 0-1) |
| `spectral_contrast(y, sr=22050, n_bands=6, ...)` | Peak-to-valley contrast per octave |
| `zero_crossing_rate(y, frame_length=2048, ...)` | Rate of sign changes |

### MFCC Functions

| Function | Description |
|----------|-------------|
| `mfcc(y, sr=22050, n_mfcc=20, n_mels=128, ...)` | Mel-frequency cepstral coefficients |
| `delta(data, width=9, order=1, ...)` | Delta (derivative) features |
| `dct(x, type=2, n=None, axis=-1, norm='ortho')` | Discrete Cosine Transform (Type II) |

### Filterbank Functions

| Function | Description |
|----------|-------------|
| `linear_filterbank(sr, n_fft, n_bands=64, ...)` | Linear-scale filterbank matrix |
| `bark_filterbank(sr, n_fft, n_bands=24, ...)` | Bark-scale filterbank matrix |
| `hz_to_bark(frequencies, formula='zwicker')` | Convert Hz to Bark scale |
| `bark_to_hz(bark, formula='zwicker')` | Convert Bark scale to Hz |

### Time-Domain Functions

| Function | Description |
|----------|-------------|
| `frame(y, frame_length, hop_length)` | Frame signal into overlapping windows |
| `rms(y, frame_length=2048, hop_length=512, ...)` | Root-mean-square energy per frame |
| `preemphasis(y, coef=0.97, ...)` | Pre-emphasis filter (boost high freq) |
| `deemphasis(y, coef=0.97, ...)` | De-emphasis filter (inverse) |

### Resampling Functions

| Function | Description |
|----------|-------------|
| `resample(y, orig_sr, target_sr, res_type='fft', ...)` | Resample to different sample rate |
| `resample_poly(y, up, down, ...)` | Polyphase resampling for integer ratios |

### Phase Reconstruction Function

| Function | Description |
|----------|-------------|
| `griffinlim(S, n_iter=32, momentum=0.99, ...)` | Reconstruct audio from magnitude spectrogram |

### Pitch & Periodicity Functions

| Function | Description |
|----------|-------------|
| `autocorrelation(y, max_lag=None, normalize=True, ...)` | Autocorrelation via FFT |
| `pitch_detect_acf(y, sr=22050, fmin=50, fmax=2000, ...)` | Pitch detection via autocorrelation |
| `periodicity(y, sr=22050, ...)` | Periodicity strength per frame (0-1) |

## Numerical Accuracy

All computations use float32 precision (Apple Silicon optimized). Key tolerances:

| Function | Max Error | Notes |
|----------|-----------|-------|
| STFT | ~5e-5 | FFT implementation differences |
| ISTFT round-trip | <4e-6 | Perfect reconstruction |
| Windows/Filterbanks | 0.00 | Exact match to librosa |
| Spectral Features | ~1e-4 | Matches librosa.feature |
| MFCC | ~1e-4 | DCT + mel spectrogram chain |
| Resampling | ~1e-4 | Uses scipy.signal.resample |
| Griffin-Lim | varies | Iterative convergence |

See [NUMERICAL_ACCURACY.md](NUMERICAL_ACCURACY.md) for detailed precision guarantees and mathematical property validation.

## Requirements

- Python >= 3.10
- MLX >= 0.30.0
- NumPy

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
pip install -e ".[bench]"
mlx-audio-bench --verbose

# Lint
ruff check mlx_audio_primitives/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Documentation

- [NUMERICAL_ACCURACY.md](NUMERICAL_ACCURACY.md) - Precision guarantees and tolerances
- [ARCHITECTURE.md](ARCHITECTURE.md) - Internal design and data flow
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup and guidelines
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Tutorial: WAV to Mel Spectrogram

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [librosa](https://librosa.org/) - The gold standard for audio analysis in Python
- [Apple MLX](https://github.com/ml-explore/mlx) - The framework that makes this possible
- [scipy](https://scipy.org/) - Reference implementations for window functions
- [torchaudio](https://pytorch.org/audio/) - Cross-validation reference

---

*We're precise down to the last bit - no rounding errors in our commitment to quality.*
