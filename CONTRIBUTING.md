# Contributing to mlx-audio-primitives

> *"Pull requests welcome - let's make some noise together!"*

Thanks for your interest in contributing! This guide will help you get set up and understand our development workflow.

## Development Setup

### Prerequisites

- Python >= 3.10
- macOS with Apple Silicon (M1/M2/M3) for Metal acceleration
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-audio-primitives.git
cd mlx-audio-primitives

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install with all development dependencies
pip install -e ".[dev,test,bench]"
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Run benchmarks
mlx-audio-bench --verbose
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Guidelines

1. **Type hints everywhere** - All public functions must have complete type annotations
2. **NumPy-style docstrings** - Parameters, Returns, Raises, Examples sections
3. **Float32 by default** - MLX is optimized for float32 on Apple Silicon
4. **No unnecessary dependencies** - Only mlx and numpy are required at runtime

### Running the Linter

```bash
# Check for issues
ruff check mlx_audio_primitives/

# Auto-fix where possible
ruff check --fix mlx_audio_primitives/

# Format code
ruff format mlx_audio_primitives/
```

### Docstring Format

```python
def your_function(
    input_array: mx.array,
    param: int = 42,
) -> mx.array:
    """
    Brief one-line description.

    Longer description if needed, explaining the algorithm or
    any important details.

    Parameters
    ----------
    input_array : mx.array
        Description of the input. Shape: (batch, samples).
    param : int, default=42
        Description of the parameter.

    Returns
    -------
    mx.array
        Description of the output. Shape: (batch, features).

    Raises
    ------
    ValueError
        When invalid parameters are provided.

    Examples
    --------
    >>> result = your_function(mx.array([1, 2, 3]))
    >>> result.shape
    (3,)
    """
```

## Testing Requirements

**Every new feature must include tests.** We validate against reference implementations.

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_stft.py             # STFT/ISTFT tests
├── test_mel.py              # Mel scale tests
├── test_windows.py          # Window function tests
├── test_convert.py          # dB conversion tests
├── test_mathematical_properties.py  # DSP invariants
├── test_torchaudio_crossval.py      # Cross-validation
└── test_cpp_extension.py    # C++ extension tests
```

### Writing Tests

1. **Cross-validate against librosa** - Our primary reference
2. **Test mathematical properties** - Not just outputs, but invariants
3. **Include edge cases** - Short signals, extreme parameters
4. **Test batched inputs** - Both 1D and 2D (batch, samples)

```python
import numpy as np
import mlx.core as mx
import pytest
import librosa

from mlx_audio_primitives import your_function


class TestYourFunction:
    """Tests for your_function."""

    def test_matches_librosa(self, random_signal):
        """Verify output matches librosa within tolerance."""
        # Get librosa result
        np_signal = np.array(random_signal)
        expected = librosa.your_function(np_signal)

        # Get our result
        result = your_function(random_signal)

        # Compare with appropriate tolerance
        np.testing.assert_allclose(
            np.array(result),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_batch_processing(self, batch_signals):
        """Verify batched input produces correct output shape."""
        result = your_function(batch_signals)
        assert result.ndim == batch_signals.ndim
```

### Tolerance Guidelines

| Function Type | rtol | atol | Why |
|---------------|------|------|-----|
| STFT/ISTFT | 1e-4 | 1e-4 | FFT implementation differences |
| Windows | 1e-5 | 1e-5 | Should match exactly |
| Mel filterbank | 1e-5 | 1e-5 | Numerical operations |
| dB conversions | 1e-4 | 1e-4 | Log operations |

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_stft.py -v

# Run tests matching a pattern
pytest tests/ -k "test_stft_basic"

# Run with coverage
pytest tests/ --cov=mlx_audio_primitives --cov-report=html
```

## Adding New Features

### Adding a New Window Function

1. **Add the implementation** in `mlx_audio_primitives/windows.py`:

```python
def _your_window(n: int) -> mx.array:
    """
    Your window function: w[k] = formula here.

    Computed in float64 for precision, then cast to float32.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    # Implementation here (use NumPy for precision)
    k = np.arange(n, dtype=np.float64)
    window = ...  # Your formula
    return mx.array(window.astype(np.float32))
```

2. **Add to the dispatch table**:

```python
_WINDOW_FUNCTIONS: dict[str, Callable[[int], mx.array]] = {
    # ... existing windows ...
    "your_window": _your_window,
    "alias": _your_window,  # If you want aliases
}
```

3. **Add tests** in `tests/test_windows.py`:

```python
def test_your_window_matches_scipy(self):
    """Verify your_window matches scipy.signal.windows.your_window."""
    from scipy.signal import windows

    for n in [64, 128, 512, 2048]:
        expected = windows.your_window(n)
        result = get_window("your_window", n, fftbins=False)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)
```

4. **Update documentation** in `README.md`

### Adding a New C++ Primitive

See [csrc/README.md](csrc/README.md) for the C++ extension development guide.

High-level steps:

1. Create header file: `csrc/your_primitive.h`
2. Create implementation: `csrc/your_primitive.cpp`
3. Create Metal kernel (optional): `csrc/metal/your_primitive.metal`
4. Add to CMakeLists.txt
5. Add Python bindings in `csrc/bindings.cpp`
6. Add fallback in Python module
7. Add tests in `tests/test_cpp_extension.py`

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest tests/`)
- [ ] Linter passes (`ruff check mlx_audio_primitives/`)
- [ ] New features have tests
- [ ] Docstrings are complete (NumPy style)
- [ ] Type hints are present
- [ ] `NUMERICAL_ACCURACY.md` updated if tolerances change
- [ ] `README.md` updated if API changes
- [ ] Benchmarks run if performance-related

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information about:

- Module dependencies
- Pipeline data flow
- Caching strategies
- Metal kernel design

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

---

*Thanks for helping make audio processing on Apple Silicon even better!*
