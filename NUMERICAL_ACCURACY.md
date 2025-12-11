# Numerical Accuracy Documentation

This document details the numerical precision guarantees for mlx-audio-primitives and explains the testing methodology used to validate mathematical correctness.

## Summary

All functions in this library are validated against reference implementations (librosa, scipy, torchaudio) with the following typical error bounds:

| Function | Max Error | Tolerance Used | Reference |
|----------|-----------|----------------|-----------|
| STFT | ~5e-5 | rtol=1e-4, atol=1e-4 | librosa, torchaudio |
| ISTFT (round-trip) | ~3e-6 | rtol=1e-4, atol=1e-4 | Perfect reconstruction |
| Window functions | 0.00 | rtol=1e-5, atol=1e-5 | scipy.signal |
| Mel filterbank | 0.00 | rtol=1e-5, atol=1e-5 | librosa.filters.mel |
| Mel scale (Slaney/HTK) | 0.00 | exact | librosa |
| dB conversions | ~4e-6 | rtol=1e-4, atol=1e-4 | librosa |

## Precision Considerations

### Float32 Computation

All computations use float32 precision, which provides approximately 7 significant decimal digits. This is standard for GPU-accelerated audio processing and provides sufficient precision for all practical applications.

**Why float32?**
- MLX is optimized for float32 on Apple Silicon
- Audio processing rarely requires float64 precision
- Matches the precision of most audio hardware (24-bit = ~7 digits)

### Window Functions

Window functions are computed in float64 internally and cast to float32 to ensure:
- Perfect symmetry (verified: asymmetry = 0)
- Exact match with scipy.signal
- Non-negative values for Blackman window (theoretical endpoints are 0, but float64 may produce tiny negatives ~1e-17, which are clamped)

### FFT Implementation Differences

The MLX FFT implementation differs slightly from SciPy's FFT by up to ~5e-5 for n_fft=2048. This is due to:
- Different underlying FFT algorithms (Apple Accelerate vs FFTPACK)
- Floating-point accumulation order differences
- Both are mathematically correct within float32 precision

### Overlap-Add Normalization

The ISTFT overlap-add normalization uses a minimum threshold of 1e-8 to avoid division by zero:

```python
window_sum = np.maximum(window_sum, 1e-8)
```

This threshold is:
- Far below any audible signal level
- Orders of magnitude smaller than the NOLA tolerance (1e-10)
- Ensures numerical stability without affecting reconstruction quality

## Mathematical Properties Verified

### 1. Parseval's Theorem (Energy Conservation)

For the basic FFT:
```
sum(|x|^2) = (|X[0]|^2 + 2*sum(|X[1:-1]|^2) + |X[-1]|^2) / N
```
Verified to machine precision (ratio = 1.000000).

### 2. STFT Linearity

```
STFT(a*x + b*y) = a*STFT(x) + b*STFT(y)
```
Verified with max difference < 1e-5.

### 3. Perfect Reconstruction

For COLA-compliant windows (Hann, Hamming, etc.):
```
ISTFT(STFT(x)) ≈ x
```
Verified with max error < 4e-6.

### 4. Pure Tone Localization

A pure tone at frequency f appears in the correct FFT bin:
```
bin = round(f * n_fft / sr)
```
Verified to within ±1 bin (due to spectral leakage from windowing).

## Testing Methodology

### Reference Implementations

1. **librosa** (primary): Industry-standard audio analysis library
2. **scipy.signal**: Reference for window functions
3. **torchaudio**: Independent cross-validation

### Test Types

1. **Direct comparison**: Output matches reference within tolerance
2. **Round-trip tests**: ISTFT(STFT(x)) ≈ x
3. **Mathematical property tests**: Parseval, linearity, etc.
4. **Edge case tests**: Short signals, extreme values, boundary conditions
5. **Numerical stability tests**: Near-zero, large values, NaN/Inf checks

### Tolerance Rationale

- **rtol=1e-4, atol=1e-4**: Accounts for FFT implementation differences while ensuring functional equivalence
- **rtol=1e-5, atol=1e-5**: Used for exact matches (windows, mel filterbanks)
- **rtol=1e-6**: Used for simple operations (magnitude, phase)

## Known Limitations

1. **Float32 ceiling**: Maximum precision is ~7 significant digits
2. **FFT differences**: MLX vs SciPy FFT differ by ~5e-5
3. **Mel spectrogram chain**: Errors can accumulate through STFT → magnitude → mel filterbank

## Validation Commands

Run the full test suite:
```bash
pytest tests/ -v
```

Run mathematical property tests only:
```bash
pytest tests/test_mathematical_properties.py -v
```

Run cross-validation tests:
```bash
pytest tests/test_torchaudio_crossval.py -v
```
