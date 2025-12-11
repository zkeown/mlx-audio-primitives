# Architecture

> *"Understanding the signal flow - because knowing is half the battle, and the other half is FFT."*

This document describes the internal architecture of mlx-audio-primitives, including module dependencies, data flow pipelines, and key implementation decisions.

## Module Dependency Graph

```text
                              ┌──────────────────┐
                              │   __init__.py    │
                              │  (public API)    │
                              └────────┬─────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│     stft.py      │       │      mel.py      │       │   convert.py     │
│  STFT / ISTFT    │       │  Mel filterbank  │       │  dB conversions  │
└────────┬─────────┘       └────────┬─────────┘       └──────────────────┘
         │                          │
         │                          │
         ▼                          │
┌──────────────────┐                │
│   windows.py     │◄───────────────┘
│ Window functions │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐       ┌──────────────────┐
│  _extension.py   │◄──────│  _validation.py  │
│  C++ extension   │       │  Input checks    │
│    loader        │       └──────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   csrc/_ext      │
│  C++ / Metal     │
│   primitives     │
└──────────────────┘
```

## Data Flow Pipelines

### STFT Pipeline

The Short-Time Fourier Transform converts time-domain signals to time-frequency representations.

```text
Input Signal                     Output Spectrogram
(batch, samples)                 (batch, freq_bins, n_frames)
     │                                    ▲
     │                                    │
     ▼                                    │
┌─────────────┐                  ┌────────┴────────┐
│  1. Pad     │                  │  5. Transpose   │
│  (center)   │                  │  to librosa     │
└──────┬──────┘                  │  convention     │
       │                         └────────┬────────┘
       ▼                                  │
┌─────────────┐                  ┌────────┴────────┐
│  2. Frame   │                  │  4. rfft        │
│  signal     │                  │  (Metal GPU)    │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
       │    ┌─────────────┐              │
       └───▶│ 3. Window   │──────────────┘
            │ multiply    │
            └─────────────┘
```

#### Step Details

1. **Padding** (`_pad_signal`)
   - Adds `n_fft // 2` samples to each side when `center=True`
   - Modes: `constant` (zeros), `reflect`, `edge`
   - C++ extension: `_ext.pad_signal()` or Python fallback

2. **Framing** (`_frame_signal`)
   - Slices signal into overlapping windows
   - Output shape: `(batch, n_frames, n_fft)`
   - Uses `mx.as_strided` for zero-copy views when available
   - Falls back to gather-based indexing on older MLX

3. **Windowing**
   - Element-wise multiply: `frames * window`
   - Window is cached to avoid recomputation
   - Computed in float64 for precision, cast to float32

4. **FFT**
   - `mx.fft.rfft(windowed_frames, axis=-1)`
   - Real-to-complex FFT, outputs `n_fft // 2 + 1` bins
   - Runs on Metal GPU

5. **Transpose**
   - Reshape from `(batch, n_frames, freq_bins)` to `(batch, freq_bins, n_frames)`
   - Matches librosa output convention

### ISTFT Pipeline

The inverse transform reconstructs time-domain signals from spectrograms.

```text
Input Spectrogram                Output Signal
(batch, freq_bins, n_frames)     (batch, samples)
     │                                  ▲
     │                                  │
     ▼                                  │
┌─────────────┐                 ┌───────┴───────┐
│ 1. Transpose│                 │ 5. Trim       │
│ to frames   │                 │ (center pad)  │
└──────┬──────┘                 └───────┬───────┘
       │                                │
       ▼                                │
┌─────────────┐                 ┌───────┴───────┐
│ 2. irfft    │                 │ 4. Normalize  │
│ (Metal GPU) │                 │ by win²       │
└──────┬──────┘                 └───────┬───────┘
       │                                │
       │    ┌─────────────┐            │
       └───▶│ 3. Overlap- │────────────┘
            │    Add      │
            └─────────────┘
```

#### Step Details

1. **Transpose**
   - From `(batch, freq_bins, n_frames)` to `(batch, n_frames, freq_bins)`

2. **Inverse FFT**
   - `mx.fft.irfft(spectrogram, n=n_fft, axis=-1)`
   - Complex-to-real FFT

3. **Overlap-Add** (`_overlap_add`)
   - Accumulates overlapping frames into output signal
   - **Critical for perfect reconstruction**
   - Uses custom Metal kernel with atomic operations
   - C++ extension: `_ext.overlap_add()` or Metal kernel fallback

4. **Normalization**
   - Divide by sum of squared windows: `output / window_sum`
   - Uses epsilon threshold (1e-8) to avoid division by zero
   - This is what enables perfect reconstruction

5. **Trimming**
   - Remove padding added during STFT
   - Crop to original signal length if specified

### Mel Spectrogram Pipeline

```text
Input Signal                    Output Mel Spectrogram
(batch, samples)                (batch, n_mels, n_frames)
     │                                   ▲
     │                                   │
     ▼                                   │
┌─────────────┐                 ┌────────┴────────┐
│   STFT      │                 │  3. Matmul      │
│   (above)   │                 │  mel @ mag      │
└──────┬──────┘                 └────────┬────────┘
       │                                 │
       ▼                                 │
┌─────────────┐    ┌──────────┐         │
│ 1. Magnitude│    │ 2. Get   │         │
│ |S|         │    │ mel FB   │─────────┘
└──────┬──────┘    │ (cached) │
       │           └──────────┘
       │                ▲
       │                │
       ▼                │
┌─────────────┐         │
│ |S|^power   │─────────┘
└─────────────┘
```

### MFCC Pipeline

*MFCCs - because cepstral coefficients deserve love too.*

Mel-Frequency Cepstral Coefficients transform mel spectrograms into a compact representation suitable for speech/speaker recognition.

```text
Input Signal                    Output MFCCs
(batch, samples)                (batch, n_mfcc, n_frames)
     │                                   ▲
     │                                   │
     ▼                                   │
┌─────────────┐                 ┌────────┴────────┐
│    Mel      │                 │  4. Lifter      │
│ Spectrogram │                 │  (optional)     │
│   (above)   │                 └────────┬────────┘
└──────┬──────┘                          │
       │                         ┌───────┴───────┐
       ▼                         │  3. DCT-II    │
┌─────────────┐                  │  (cached)     │
│ 1. power_   │                  └───────┬───────┘
│    to_db    │                          │
└──────┬──────┘                          │
       │                                 │
       ▼                                 │
┌─────────────┐                         │
│ 2. Log-mel  │─────────────────────────┘
│ spectrogram │
└─────────────┘
```

#### Step Details

1. **Log scale** - Convert power mel spectrogram to decibels
2. **Transpose** - Move mel axis for DCT application
3. **DCT-II** - Discrete Cosine Transform decorrelates mel bands
   - Cached basis matrix: `cos(π * k * (2n + 1) / (2N))`
   - Orthonormal scaling for energy preservation
4. **Liftering** (optional) - Cepstral smoothing: `M[n] *= 1 + (L/2) * sin(π(n+1)/L)`

### Spectral Features Pipeline

```text
Input Signal                    Output Features
(batch, samples)                (batch, 1, n_frames) each
     │
     │         ┌──────────────────────────────────────────┐
     ▼         │                                          │
┌─────────────┐│   ┌─────────────┐  ┌─────────────┐      │
│   STFT      ││   │  Centroid   │  │  Bandwidth  │      │
└──────┬──────┘│   │  Σ(f·S)/ΣS  │  │  Σ(|f-c|S)  │      │
       │       │   └─────────────┘  └─────────────┘      │
       ▼       │                                          │
┌─────────────┐│   ┌─────────────┐  ┌─────────────┐      │
│  Magnitude  │├──▶│  Rolloff    │  │  Flatness   │      │
│     |S|     ││   │  cumsum     │  │  gmean/amean│      │
└──────┬──────┘│   └─────────────┘  └─────────────┘      │
       │       │                                          │
       │       │   ┌─────────────┐  ┌─────────────┐      │
       │       │   │  Contrast   │  │     ZCR     │◄─────┘
       │       │   │  per octave │  │  sign diffs │  (from signal)
       │       │   └─────────────┘  └─────────────┘
       │       │
       └───────┘
```

### Griffin-Lim Pipeline

*Phase reconstruction: we iterate until we get it right.*

```text
┌───────────────────────────────────────────────────────────────┐
│                    Griffin-Lim Algorithm                       │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Magnitude S ──┐     ┌──────────────────────────────────────┐ │
│                │     │                                      │ │
│                ▼     │    ┌─────────────┐                   │ │
│         ┌───────────┐│    │  2. ISTFT   │                   │ │
│         │ 1. Init   ││    │  (above)    │                   │ │
│         │ phase     │├───▶└──────┬──────┘                   │ │
│         │ (random)  ││           │                          │ │
│         └───────────┘│           ▼                          │ │
│                      │    ┌─────────────┐                   │ │
│         S·e^(jφ)  ◄──┘    │  3. STFT    │                   │ │
│              │            │  (above)    │                   │ │
│              │            └──────┬──────┘                   │ │
│              │                   │                          │ │
│              │                   ▼                          │ │
│              │            ┌─────────────┐                   │ │
│              └────────────│ 4. Extract  │◄── Repeat n_iter  │ │
│                           │ phase(S')   │    with momentum  │ │
│                           └─────────────┘                   │ │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

**Momentum acceleration** (Perraudin et al. 2013):

```python
rebuilt = rebuilt_new + momentum * (rebuilt_new - prev)
```
Default `momentum=0.99` significantly speeds convergence.

### Pitch Detection Pipeline (ACF)

```text
Input Signal                    Output (f0, voiced)
(batch, samples)                (batch, n_frames) each
     │
     │
     ▼
┌─────────────┐
│  1. Frame   │
│  signal     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. Center   │    For each frame:
│ (subtract   │    ┌─────────────────────────────────┐
│  mean)      │    │                                 │
└──────┬──────┘    │  ┌─────────────┐               │
       │           │  │ 3. FFT      │               │
       └──────────▶│  │ |FFT(x)|²   │               │
                   │  └──────┬──────┘               │
                   │         │                      │
                   │         ▼                      │
                   │  ┌─────────────┐               │
                   │  │ 4. IFFT     │               │
                   │  │ = autocorr  │               │
                   │  └──────┬──────┘               │
                   │         │                      │
                   │         ▼                      │
                   │  ┌─────────────┐               │
                   │  │ 5. Find     │               │
                   │  │ peak in     │───▶ f0 = sr/lag
                   │  │ [min,max]   │               │
                   │  │ lag range   │───▶ voiced = peak > threshold
                   │  └─────────────┘               │
                   │                                 │
                   └─────────────────────────────────┘
```

**Lag-to-frequency**: `f0 = sample_rate / peak_lag`

## Caching Strategy

We use a **two-tier caching strategy** to maximize performance:

### Tier 1: Bytes Cache (LRU)

```python
@lru_cache(maxsize=128)
def _get_window_cached(name: str, n_fft: int, fftbins: bool) -> tuple[bytes, int]:
    """Returns window as bytes (hashable for lru_cache)."""
    ...
```

**Why bytes?**
- `lru_cache` requires hashable arguments and return values
- `mx.array` is not hashable
- Bytes are compact and hashable

### Tier 2: MLX Array Cache (Dict)

```python
_mlx_window_cache: dict[tuple, mx.array] = {}
```

**Why a second cache?**
- Avoids `bytes → np.frombuffer → mx.array` conversion on every access
- Avoids CPU-to-GPU transfer on repeated calls
- The MLX array stays on the GPU

### Cache Flow

```text
get_window("hann", 2048, True)
         │
         ▼
┌─────────────────────────────────┐
│ Check MLX cache (Tier 2)        │ ──▶ HIT: Return mx.array directly
└─────────────┬───────────────────┘
              │ MISS
              ▼
┌─────────────────────────────────┐
│ Check bytes cache (Tier 1)      │ ──▶ HIT: Convert bytes → mx.array
└─────────────┬───────────────────┘            Store in Tier 2, return
              │ MISS
              ▼
┌─────────────────────────────────┐
│ Compute window (NumPy float64)  │
│ Cast to float32, convert bytes  │
│ Store in Tier 1                 │
│ Convert to mx.array             │
│ Store in Tier 2, return         │
└─────────────────────────────────┘
```

## C++ Extension Integration

The optional C++ extension provides optimized implementations for:

- `pad_signal` - Signal padding with various modes
- `frame_signal` - Efficient signal framing
- `overlap_add` - Atomic scatter-add for reconstruction
- `generate_window` - Window function generation
- `mel_filterbank` - Mel filterbank matrix construction
- `hz_to_mel` / `mel_to_hz` - Frequency conversions

### Fallback Pattern

```python
# mlx_audio_primitives/_extension.py
try:
    from . import _ext
    HAS_CPP_EXT = True
except ImportError:
    _ext = None
    HAS_CPP_EXT = False

# In stft.py, mel.py, etc.
def _some_operation(...):
    if HAS_CPP_EXT and _ext is not None:
        return _ext.some_operation(...)

    # Pure Python/Metal fallback
    ...
```

### Why C++?

| Operation | C++ Advantage | Fallback |
|-----------|---------------|----------|
| `overlap_add` | Fused kernel, no intermediate buffers | Custom Metal kernel |
| `frame_signal` | Optimized gather | `mx.as_strided` or gather |
| `pad_signal` | All modes in one kernel | Python conditionals |

## Metal Kernel: Scatter-Add

The ISTFT overlap-add requires accumulating values from multiple frames into overlapping output positions. This is a **scatter operation** with potential race conditions.

### The Problem

```text
Frame 0:  [a b c d]
Frame 1:      [e f g h]
Frame 2:          [i j k l]
Output:   [a b c+e d+f g+i h+j k l]
             ▲     ▲     ▲
             └─────┴─────┴── Race conditions!
```

### Our Solution: Atomic Operations

```metal
// csrc/metal/overlap_add.metal (simplified)
atomic_fetch_add_explicit(
    (device atomic_float*)&output[out_pos],
    val,
    memory_order_relaxed
);
```

**Trade-offs:**
- Atomics have overhead (~10% slower than non-atomic)
- But avoids multiple kernel launches for reduction
- Profiling shows atomics win for typical audio (n_frames < 1000)

### Alternative Considered: Reduction

```text
1. Write all contributions to separate buffer
2. Reduction kernel to sum contributions
```

This was rejected because:
- Requires 2x memory for intermediate buffer
- Multiple kernel launches (dispatch overhead)
- For small n_frames, atomics are faster

## Numerical Precision

### Why Float32?

1. **MLX optimization** - Apple Silicon is optimized for float32
2. **Audio precision** - 24-bit audio ≈ 7 significant digits (float32 provides ~7)
3. **Memory bandwidth** - Half the data = 2x throughput

### Float64 for Intermediate Calculations

Windows and filterbanks are computed in float64, then cast to float32:

```python
# windows.py
k = np.arange(n, dtype=np.float64)  # High precision
window = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
return mx.array(window.astype(np.float32))  # Cast for GPU
```

**Why?**
- Ensures perfect symmetry in window functions
- Matches scipy/librosa reference implementations
- Small arrays, so float64 computation cost is negligible

## File Organization

```text
mlx_audio_primitives/
├── __init__.py          # Public API exports
├── stft.py              # STFT, ISTFT, magnitude, phase
├── mel.py               # Mel filterbank, melspectrogram
├── windows.py           # Window functions
├── convert.py           # dB conversions
├── _extension.py        # C++ extension loader
├── _validation.py       # Input validation helpers
└── _ext.cpython-*.so    # Compiled C++ extension (built)

csrc/
├── CMakeLists.txt       # Build configuration
├── bindings.cpp         # Python bindings (nanobind)
├── bindings_wrappers.h  # Wrapper functions
├── overlap_add.{h,cpp}  # Overlap-add primitive
├── frame_signal.{h,cpp} # Signal framing primitive
├── pad_signal.{h,cpp}   # Signal padding primitive
├── windows.{h,cpp}      # Window generation
├── mel_filterbank.{h,cpp} # Mel filterbank
└── metal/               # Metal shader sources
    ├── overlap_add.metal
    ├── frame_signal.metal
    ├── pad_signal.metal
    ├── windows.metal
    └── mel_filterbank.metal

tests/
├── conftest.py          # Shared fixtures
├── test_stft.py         # STFT/ISTFT tests
├── test_mel.py          # Mel tests
├── test_windows.py      # Window tests
├── test_convert.py      # dB conversion tests
├── test_mathematical_properties.py  # DSP invariants
├── test_torchaudio_crossval.py      # Cross-validation
└── test_cpp_extension.py            # C++ extension tests

benchmarks/
├── __init__.py
├── run.py               # CLI entry point
├── utils.py             # Benchmark utilities
├── bench_stft.py        # STFT benchmarks
├── bench_mel.py         # Mel benchmarks
└── bench_windows.py     # Window benchmarks
```

## Future Considerations

### Potential Optimizations

1. **Fused STFT kernel** - Combine framing + windowing + FFT
2. **Streaming support** - Process audio in chunks
3. **Mixed precision** - Float16 for memory-bound operations
4. **Batch parallelism** - Process multiple signals concurrently

### Potential Features

1. **CQT** - Constant-Q Transform
2. **Chroma** - Pitch class profiles
3. **Onset detection** - Event detection primitives
4. **Pitch tracking** - F0 estimation

---

*Architecture is like audio engineering: it's all about the signal flow.*
