"""
Pytest configuration and shared fixtures for mlx-audio-primitives tests.
"""
import numpy as np
import pytest


# Use np.random.Generator for better test isolation instead of global seed
_TEST_SEED = 42


@pytest.fixture
def random_signal():
    """Generate a random audio signal for testing."""
    rng = np.random.default_rng(_TEST_SEED)
    return rng.standard_normal(22050).astype(np.float32)


@pytest.fixture
def chirp_signal():
    """Generate a chirp signal (swept sine) for testing."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Chirp from 100 Hz to 1000 Hz
    signal = np.sin(2 * np.pi * (100 + 900 * t / 2) * t).astype(np.float32)
    return signal


@pytest.fixture
def short_signal():
    """Generate a short signal for edge case testing."""
    rng = np.random.default_rng(_TEST_SEED)
    return rng.standard_normal(1024).astype(np.float32)


@pytest.fixture
def batch_signals():
    """Generate a batch of random signals for testing."""
    rng = np.random.default_rng(_TEST_SEED)
    return rng.standard_normal((4, 22050)).astype(np.float32)


@pytest.fixture
def sine_signal():
    """Generate a pure sine wave for testing."""
    sr = 22050
    duration = 1.0
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture
def spectrogram(random_signal):
    """Pre-computed magnitude spectrogram."""
    import mlx.core as mx
    from mlx_audio_primitives import stft, magnitude

    y_mx = mx.array(random_signal)
    S = stft(y_mx, n_fft=2048, hop_length=512)
    return np.array(magnitude(S))


@pytest.fixture
def mel_spec(random_signal):
    """Pre-computed mel spectrogram."""
    import mlx.core as mx
    from mlx_audio_primitives import melspectrogram

    y_mx = mx.array(random_signal)
    return np.array(melspectrogram(y_mx, sr=22050, n_mels=128))
