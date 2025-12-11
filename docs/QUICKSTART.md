# Quickstart: From Audio to Mel Spectrogram in 5 Minutes

> *"The fastest way to transform your audio - and we're not just winging it."*

This guide walks you through loading audio, computing spectrograms, and visualizing results using mlx-audio-primitives.

## Prerequisites

```bash
pip install mlx-audio-primitives numpy matplotlib soundfile
```

## Step 1: Load Audio

```python
import numpy as np
import mlx.core as mx
import soundfile as sf

# Load a WAV file
audio_np, sample_rate = sf.read("your_audio.wav")

# Convert to MLX array
# If stereo, take first channel
if audio_np.ndim > 1:
    audio_np = audio_np[:, 0]

audio = mx.array(audio_np.astype(np.float32))
print(f"Loaded {len(audio)} samples at {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
```

**No audio file?** Generate a test signal:

```python
# Generate a 2-second chirp (100 Hz → 1000 Hz)
sample_rate = 22050
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 100 + 900 * t / duration  # Linear sweep
audio_np = np.sin(2 * np.pi * frequency * t).astype(np.float32)
audio = mx.array(audio_np)
```

## Step 2: Compute STFT

```python
from mlx_audio_primitives import stft, magnitude

# Compute the Short-Time Fourier Transform
S = stft(audio, n_fft=2048, hop_length=512)
print(f"STFT shape: {S.shape}")
# Output: (1025, n_frames) - 1025 frequency bins, variable time frames

# Get magnitude spectrogram
S_mag = magnitude(S)
```

### Understanding the Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fft` | 2048 | FFT size. Larger = better frequency resolution, worse time resolution |
| `hop_length` | n_fft // 4 | Samples between frames. Smaller = more frames, more overlap |
| `window` | 'hann' | Window function. Hann is standard for audio analysis |
| `center` | True | Pad signal so frames are centered |

## Step 3: Compute Mel Spectrogram

For speech and music ML, mel spectrograms are the standard input:

```python
from mlx_audio_primitives import melspectrogram, power_to_db

# Compute mel spectrogram (power=2.0 for power spectrogram)
mel = melspectrogram(
    audio,
    sr=sample_rate,
    n_fft=2048,
    hop_length=512,
    n_mels=128,  # 128 mel bands is common
)
print(f"Mel shape: {mel.shape}")
# Output: (128, n_frames)

# Convert to log scale (dB) - standard for ML and visualization
mel_db = power_to_db(mel, ref=mx.max)  # Normalize to peak
```

### Choosing n_mels

| Use Case | n_mels | Notes |
|----------|--------|-------|
| Speech recognition | 40-80 | Lower frequency range matters most |
| Music analysis | 128 | Capture full harmonic content |
| General purpose | 80 | Good balance |

## Step 4: Visualize

```python
import matplotlib.pyplot as plt

# Convert to NumPy for plotting
mel_db_np = np.array(mel_db)

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Waveform
time = np.arange(len(audio)) / sample_rate
axes[0].plot(time, np.array(audio), linewidth=0.5)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Waveform")

# Mel spectrogram
img = axes[1].imshow(
    mel_db_np,
    aspect="auto",
    origin="lower",
    extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("Mel Spectrogram (dB)")
plt.colorbar(img, ax=axes[1], format="%+2.0f dB")

plt.tight_layout()
plt.savefig("spectrogram.png", dpi=150)
plt.show()
```

## Step 5: Perfect Reconstruction (Optional)

Want to verify the STFT is invertible?

```python
from mlx_audio_primitives import istft

# Reconstruct from STFT
S = stft(audio, n_fft=2048, hop_length=512)
reconstructed = istft(S, hop_length=512, length=len(audio))

# Check reconstruction quality
error = mx.abs(audio - reconstructed)
print(f"Max reconstruction error: {float(mx.max(error)):.2e}")
print(f"Mean reconstruction error: {float(mx.mean(error)):.2e}")
# Expect: Max ~1e-6, Mean ~1e-7 (essentially perfect!)
```

## Complete Example

Here's everything in one script:

```python
"""
Complete mel spectrogram pipeline example.

Usage:
    python mel_example.py input.wav output.png
"""
import sys
import numpy as np
import mlx.core as mx
import soundfile as sf
import matplotlib.pyplot as plt

from mlx_audio_primitives import melspectrogram, power_to_db

def main(audio_path: str, output_path: str):
    # Load audio
    audio_np, sr = sf.read(audio_path)
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # Mono
    audio = mx.array(audio_np.astype(np.float32))

    # Compute mel spectrogram
    mel = melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = power_to_db(mel, ref=mx.max)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(
        np.array(mel_db),
        aspect="auto",
        origin="lower",
        extent=[0, len(audio) / sr, 0, sr / 2],
    )
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Mel Spectrogram: {audio_path}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mel_example.py input.wav output.png")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

---

## Extended Features

> *"Now that you've got the basics down, let's crank up the features."*

### Computing MFCCs for Speech Recognition

MFCCs (Mel-Frequency Cepstral Coefficients) are the gold standard for speech recognition:

```python
from mlx_audio_primitives import mfcc, delta

# Compute 13 MFCCs (standard for ASR)
mfccs = mfcc(
    audio,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
)
print(f"MFCC shape: {mfccs.shape}")
# Output: (13, n_frames)

# Add delta (velocity) and delta-delta (acceleration) features
mfcc_delta = delta(mfccs, width=9)
mfcc_delta2 = delta(mfccs, width=9, order=2)

# Stack for a complete feature set
features = mx.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
print(f"Full feature shape: {features.shape}")
# Output: (39, n_frames) - 13 static + 13 delta + 13 delta-delta
```

### Extracting Spectral Features for Music Analysis

Spectral features characterize the "shape" of the frequency spectrum:

```python
from mlx_audio_primitives import (
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    zero_crossing_rate,
)

# Spectral centroid: "brightness" of the sound
centroid = spectral_centroid(audio, sr=sample_rate, n_fft=2048, hop_length=512)
print(f"Centroid shape: {centroid.shape}")  # (1, n_frames)

# Spectral bandwidth: spread around the centroid
bandwidth = spectral_bandwidth(audio, sr=sample_rate, n_fft=2048, hop_length=512)

# Spectral rolloff: frequency below which X% of energy lies
rolloff = spectral_rolloff(audio, sr=sample_rate, n_fft=2048, hop_length=512)

# Spectral flatness: how "noisy" vs "tonal" (0=tonal, 1=noise)
flatness = spectral_flatness(audio, n_fft=2048, hop_length=512)

# Zero crossing rate: rough estimate of dominant frequency
zcr = zero_crossing_rate(audio, frame_length=2048, hop_length=512)

# Stack all features
spectral_features = mx.concatenate([
    centroid, bandwidth, rolloff, flatness, zcr
], axis=0)
print(f"Spectral features shape: {spectral_features.shape}")
# Output: (5, n_frames)
```

### Resampling Audio

Change sample rate while preserving audio content:

```python
from mlx_audio_primitives import resample

# Downsample from 44.1kHz to 16kHz (common for speech models)
audio_44k = audio  # Assuming 44.1kHz source
audio_16k = resample(audio_44k, orig_sr=44100, target_sr=16000)
print(f"Downsampled: {audio_44k.shape} → {audio_16k.shape}")

# Upsample from 16kHz to 22.05kHz
audio_22k = resample(audio_16k, orig_sr=16000, target_sr=22050)
```

**When to use each method:**

- `resample()` - FFT-based, best quality, good for general use
- `resample_poly()` - Polyphase filter, more efficient for integer ratios

### Griffin-Lim Phase Reconstruction

Reconstruct audio from magnitude-only spectrograms (useful for vocoder-free synthesis):

```python
from mlx_audio_primitives import stft, magnitude, griffinlim

# Get magnitude spectrogram (lose phase information)
S = stft(audio, n_fft=2048, hop_length=512)
S_mag = magnitude(S)

# Reconstruct with Griffin-Lim (iterative algorithm)
reconstructed = griffinlim(
    S_mag,
    n_fft=2048,
    hop_length=512,
    n_iter=32,          # More iterations = better quality
    momentum=0.99,      # Accelerates convergence
    length=len(audio),  # Match original length
)

# The reconstruction won't be perfect (phase is lost), but it's surprisingly good
print(f"Reconstructed shape: {reconstructed.shape}")
```

### Pitch Detection

Estimate fundamental frequency (f0) using autocorrelation:

```python
from mlx_audio_primitives import pitch_detect_acf, periodicity

# Detect pitch in Hz and voicing decision
f0, voiced = pitch_detect_acf(
    audio,
    sr=sample_rate,
    fmin=50.0,   # Minimum expected pitch (Hz)
    fmax=400.0,  # Maximum expected pitch (Hz)
)
print(f"Pitch shape: {f0.shape}")  # (n_frames,) or (batch, n_frames)
print(f"Voiced frames: {float(mx.sum(voiced))} / {voiced.shape[-1]}")

# Get periodicity (confidence of pitch estimate)
period = periodicity(audio, sr=sample_rate, fmin=50.0, fmax=400.0)
# Values near 1.0 = periodic (voiced), near 0.0 = aperiodic (unvoiced/noise)
```

---

## FAQ

### Q: My STFT output shape doesn't match librosa?

**A:** Check your input shape. mlx-audio-primitives expects:
- 1D: `(samples,)` → output: `(freq_bins, n_frames)`
- 2D: `(batch, samples)` → output: `(batch, freq_bins, n_frames)`

Librosa only accepts 1D.

### Q: Why is my first call slow?

**A:** MLX compiles operations on first use. Subsequent calls are fast:

```python
# First call: ~10ms (includes compilation)
S1 = stft(audio, n_fft=2048)

# Second call: ~0.3ms (compiled, cached)
S2 = stft(audio, n_fft=2048)
```

### Q: How do I process batches?

**A:** Stack signals along the first dimension:

```python
# Process 4 audio clips at once
batch = mx.stack([audio1, audio2, audio3, audio4])  # (4, samples)
S_batch = stft(batch, n_fft=2048)  # (4, 1025, n_frames)
```

### Q: Which window function should I use?

**A:** Hann (default) is almost always right:
- **Hann**: General purpose, good frequency resolution
- **Hamming**: Similar to Hann, slightly less sidelobe suppression
- **Blackman**: Best sidelobe suppression, wider main lobe
- **Rectangular**: No windowing (rarely used)

### Q: How do I get the same output as librosa?

**A:** Use the same parameters and convert types:

```python
import librosa
import numpy as np
import mlx.core as mx
from mlx_audio_primitives import stft

# librosa
S_librosa = librosa.stft(audio_np, n_fft=2048, hop_length=512)

# mlx-audio-primitives
S_mlx = stft(mx.array(audio_np), n_fft=2048, hop_length=512)

# Compare (expect differences < 5e-5)
np.testing.assert_allclose(np.array(S_mlx), S_librosa, rtol=1e-4, atol=1e-4)
```

## Next Steps

- [NUMERICAL_ACCURACY.md](../NUMERICAL_ACCURACY.md) - Understanding precision
- [ARCHITECTURE.md](../ARCHITECTURE.md) - How it works under the hood
- [README.md](../README.md) - Full API reference

---

*Now you're ready to make some noise - just don't tell the neighbors it was us.*
