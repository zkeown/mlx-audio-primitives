# mlx-audio-primitives

Foundational audio DSP primitives for Apple MLX. Provides librosa-compatible STFT, ISTFT, mel filterbank, and dB conversion functions optimized for Apple Silicon.

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

## Features

- **STFT/ISTFT**: Short-Time Fourier Transform with perfect reconstruction
- **Window functions**: Hann, Hamming, Blackman, Bartlett, rectangular
- **Mel filterbank**: Slaney and HTK formulas
- **Mel spectrogram**: End-to-end mel spectrogram computation
- **dB conversions**: Power/amplitude to/from decibels

All functions are validated against librosa for numerical accuracy.

## Usage

### STFT and ISTFT

```python
import mlx.core as mx
from mlx_audio_primitives import stft, istft

# Load or create audio signal
y = mx.array(audio_data)  # shape: (samples,) or (batch, samples)

# Compute STFT
S = stft(y, n_fft=2048, hop_length=512)
# S.shape: (1025, n_frames) or (batch, 1025, n_frames)

# Reconstruct signal
y_reconstructed = istft(S, hop_length=512, length=y.shape[-1])
```

### Mel Spectrogram

```python
from mlx_audio_primitives import melspectrogram, power_to_db

# Compute mel spectrogram
mel = melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)

# Convert to decibels
mel_db = power_to_db(mel)
```

### Mel Filterbank

```python
from mlx_audio_primitives import mel_filterbank

# Create mel filterbank matrix
mel_fb = mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
# mel_fb.shape: (128, 1025)
```

### Window Functions

```python
from mlx_audio_primitives import get_window

# Get window function
window = get_window("hann", 2048, fftbins=True)
```

### dB Conversions

```python
from mlx_audio_primitives import (
    power_to_db, db_to_power,
    amplitude_to_db, db_to_amplitude
)

# Power spectrogram to decibels
S_db = power_to_db(S_power, ref=1.0, top_db=80.0)

# Back to power
S_power = db_to_power(S_db)
```

## API Reference

### STFT Functions

- `stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='constant')` - Short-Time Fourier Transform
- `istft(stft_matrix, hop_length=None, win_length=None, n_fft=None, window='hann', center=True, length=None)` - Inverse STFT
- `magnitude(stft_matrix)` - Compute magnitude spectrogram
- `phase(stft_matrix)` - Compute phase spectrogram
- `check_nola(window, hop_length, n_fft, tol=1e-10)` - Check NOLA constraint

### Mel Functions

- `mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney')` - Create mel filterbank
- `melspectrogram(y, sr=22050, n_fft=2048, hop_length=None, ...)` - Compute mel spectrogram
- `hz_to_mel(frequencies, htk=False)` - Convert Hz to mel scale
- `mel_to_hz(mels, htk=False)` - Convert mel scale to Hz

### dB Conversion Functions

- `power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)` - Power to decibels
- `db_to_power(S_db, ref=1.0)` - Decibels to power
- `amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0)` - Amplitude to decibels
- `db_to_amplitude(S_db, ref=1.0)` - Decibels to amplitude

### Window Functions

- `get_window(window, n_fft, fftbins=True)` - Get window function

## Requirements

- MLX >= 0.30.0
- NumPy

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
