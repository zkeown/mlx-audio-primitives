# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-audio-primitives provides librosa-compatible audio DSP primitives for Apple's MLX framework. Core functionality includes STFT/ISTFT, mel spectrograms, and dB conversions optimized for Apple Silicon.

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Install with test dependencies (includes torch/torchaudio for cross-validation)
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_stft.py -v

# Run tests matching a pattern
pytest tests/ -k "test_stft_basic"

# Run mathematical property tests
pytest tests/test_mathematical_properties.py -v

# Run cross-validation tests against torchaudio
pytest tests/test_torchaudio_crossval.py -v

# Lint with ruff
ruff check mlx_audio_primitives/

# Run benchmarks (requires bench dependencies)
pip install -e ".[bench]"
mlx-audio-bench                    # All benchmarks
mlx-audio-bench --suite stft       # STFT only
mlx-audio-bench --verbose          # Include accuracy metrics
```

## Architecture

### Module Structure

- `stft.py` - STFT/ISTFT with custom Metal kernel for overlap-add scatter operations
- `mel.py` - Mel filterbank (NumPy-computed for precision, cached via lru_cache) and melspectrogram
- `windows.py` - Window functions (hann, hamming, blackman, bartlett)
- `convert.py` - dB conversion functions (power_to_db, amplitude_to_db, etc.)
- `_extension.py` - C++ extension loader with graceful fallback (HAS_CPP_EXT flag)
- `_validation.py` - Input validation helpers

### Key Implementation Details

**STFT Pipeline:**
1. Signal framing via `_frame_signal()` - uses vectorized indexing
2. FFT via `mx.fft.rfft()`
3. Output shape convention matches librosa: `(freq_bins, n_frames)` for 1D, `(batch, freq_bins, n_frames)` for batched

**ISTFT Pipeline:**
1. Inverse FFT via `mx.fft.irfft()`
2. Overlap-add via `_overlap_add()` - uses custom Metal kernel (`_SCATTER_ADD_SOURCE`) with atomic operations
3. Squared-window normalization with epsilon threshold (1e-8)

**C++ Extension Pattern:**
Functions check `HAS_CPP_EXT` and `_ext` before using C++ implementations, falling back to Python/Metal otherwise. The extension provides `pad_signal`, `frame_signal`, and `overlap_add`.

**Mel Filterbank Caching:**
`_compute_mel_filterbank_np()` is cached via `@lru_cache`. Returns bytes for hashability, converted to mx.array on retrieval. NumPy is used for filterbank computation (not C++ extension) to match librosa precision exactly.

## Numerical Accuracy

All functions validated against librosa/scipy/torchaudio. See NUMERICAL_ACCURACY.md for detailed tolerances.

Key tolerances:
- STFT: rtol=1e-4, atol=1e-4 (FFT implementation differences ~5e-5)
- Windows/Mel filterbank: rtol=1e-5, atol=1e-5 (exact match)
- Round-trip STFT->ISTFT: max error <4e-6

All computation is float32 (MLX optimized for this on Apple Silicon).

## Test Fixtures

Shared fixtures in `tests/conftest.py`:
- `random_signal` - 22050 samples, seeded RNG
- `chirp_signal` - Swept sine 100-1000 Hz
- `short_signal` - 1024 samples for edge cases
- `batch_signals` - Shape (4, 22050) for batch testing
