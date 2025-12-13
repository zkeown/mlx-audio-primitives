# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-12

### Added

#### Core STFT Operations
- `stft` - Short-Time Fourier Transform with librosa-compatible output shape
- `istft` - Inverse STFT with custom Metal kernel for overlap-add
- `magnitude` - Compute magnitude from complex STFT
- `phase` - Compute phase from complex STFT
- `check_nola` - Verify NOLA constraint for perfect reconstruction

#### Mel-Scale Operations
- `mel_filterbank` - Mel-scale filterbank matrix (Slaney normalization)
- `melspectrogram` - Compute mel spectrogram from waveform
- `hz_to_mel` - Convert Hz to mel scale (htk/slaney)
- `mel_to_hz` - Convert mel scale to Hz

#### Spectral Features
- `spectral_centroid` - Spectral center of mass
- `spectral_bandwidth` - Spectral spread around centroid
- `spectral_rolloff` - Frequency below which percentage of energy is contained
- `spectral_flatness` - Geometric/arithmetic mean ratio (tonality measure)
- `spectral_contrast` - Peak-valley contrast per frequency band
- `zero_crossing_rate` - Rate of sign changes in signal

#### MFCC
- `mfcc` - Mel-frequency cepstral coefficients
- `delta` - Delta (derivative) features via Savitzky-Golay filter
- `dct` - Discrete Cosine Transform Type II (orthonormal)

#### Filterbanks
- `linear_filterbank` - Linear-scale triangular filterbank
- `bark_filterbank` - Bark-scale filterbank (Zwicker/Traunmuller formulas)
- `hz_to_bark` - Convert Hz to Bark scale
- `bark_to_hz` - Convert Bark scale to Hz

#### Time-Domain Primitives
- `frame` - Frame signal into overlapping windows
- `rms` - Root-mean-square energy per frame
- `preemphasis` - Pre-emphasis filter
- `deemphasis` - De-emphasis filter (inverse)

#### Resampling
- `resample` - Resample audio via polyphase/FFT methods
- `resample_poly` - Polyphase resampling for integer ratios

#### Phase Reconstruction
- `griffinlim` - Griffin-Lim algorithm with momentum

#### Pitch and Periodicity
- `autocorrelation` - Autocorrelation via FFT (Wiener-Khinchin)
- `pitch_detect_acf` - Pitch detection via autocorrelation
- `periodicity` - Periodicity strength measure

#### Decibel Conversions
- `power_to_db` - Power spectrogram to decibels
- `db_to_power` - Decibels to power spectrogram
- `amplitude_to_db` - Amplitude spectrogram to decibels
- `db_to_amplitude` - Decibels to amplitude spectrogram

#### Window Functions
- `get_window` - Window function generator (hann, hamming, blackman, bartlett)

#### Performance
- Custom Metal kernel for overlap-add scatter operations
- Optional C++ extension for `pad_signal`, `frame_signal`, `overlap_add`
- LRU caching for window functions and mel filterbanks
- Batch processing support for all operations

### Technical Details
- Python 3.10+ required
- MLX 0.30.0+ required
- Validated against librosa, scipy, and torchaudio
- Maximum numerical error <4e-6 for STFT round-trip
- All operations in float32 (Apple Silicon optimized)
