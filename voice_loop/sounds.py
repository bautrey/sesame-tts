"""Chime and feedback sounds for voice loop state transitions."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def _generate_tone(
    frequency: float,
    duration_s: float,
    sample_rate: int = 24000,
    amplitude: float = 0.3,
    fade_ms: float = 20,
) -> np.ndarray:
    """Generate a sine wave tone with fade-in/fade-out."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # Apply fade
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and fade_samples < len(tone) // 2:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

    return tone


def play_listening_chime(device: int | str | None = None) -> None:
    """Play a short ascending two-tone chime (C5 + E5) to indicate listening started."""
    sr = 24000
    tone1 = _generate_tone(523.25, 0.08, sr)  # C5
    tone2 = _generate_tone(659.25, 0.12, sr)  # E5
    silence = np.zeros(int(sr * 0.02), dtype=np.float32)
    chime = np.concatenate([tone1, silence, tone2])
    try:
        sd.play(chime, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play listening chime: %s", e)


def play_error_chime(device: int | str | None = None) -> None:
    """Play a short descending two-tone chime (E4 + C4) to indicate an error."""
    sr = 24000
    tone1 = _generate_tone(329.63, 0.12, sr)  # E4
    tone2 = _generate_tone(261.63, 0.15, sr)  # C4
    silence = np.zeros(int(sr * 0.03), dtype=np.float32)
    chime = np.concatenate([tone1, silence, tone2])
    try:
        sd.play(chime, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play error chime: %s", e)


def play_complete_chime(device: int | str | None = None) -> None:
    """Play a single short tone to indicate response complete."""
    sr = 24000
    tone = _generate_tone(880.0, 0.06, sr, amplitude=0.15)  # A5, quiet
    try:
        sd.play(tone, samplerate=sr, device=device, blocking=False)
    except Exception as e:
        logger.warning("Failed to play complete chime: %s", e)
