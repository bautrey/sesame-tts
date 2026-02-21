"""Tests for chime/feedback sounds."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


class TestGenerateTone:
    """Verify tone generation helper."""

    def test_generate_tone_shape(self):
        """Generated tone has correct length and dtype."""
        from voice_loop.sounds import _generate_tone

        tone = _generate_tone(440.0, 0.1, sample_rate=24000)
        expected_samples = int(24000 * 0.1)
        assert len(tone) == expected_samples
        assert tone.dtype == np.float32

    def test_generate_tone_amplitude(self):
        """Generated tone respects amplitude parameter."""
        from voice_loop.sounds import _generate_tone

        tone = _generate_tone(440.0, 0.1, sample_rate=24000, amplitude=0.5)
        # Peak should not exceed amplitude (plus some tolerance for float precision)
        # After fade, interior samples should be close to amplitude
        assert np.max(np.abs(tone)) <= 0.5 + 1e-5

    def test_generate_tone_fade(self):
        """Generated tone has fade-in and fade-out."""
        from voice_loop.sounds import _generate_tone

        tone = _generate_tone(440.0, 0.5, sample_rate=24000, amplitude=1.0, fade_ms=50)
        fade_samples = int(24000 * 50 / 1000)
        # First sample should be near zero (fade-in)
        assert abs(tone[0]) < 0.01
        # Last sample should be near zero (fade-out)
        assert abs(tone[-1]) < 0.01


class TestPlayListeningChime:
    """Verify listening chime plays correctly."""

    @patch("voice_loop.sounds.sd")
    def test_play_listening_chime_calls_sd(self, mock_sd):
        """play_listening_chime calls sd.play with non-blocking."""
        from voice_loop.sounds import play_listening_chime

        play_listening_chime()

        mock_sd.play.assert_called_once()
        call_kwargs = mock_sd.play.call_args[1]
        assert call_kwargs["blocking"] is False
        assert call_kwargs["samplerate"] == 24000

    @patch("voice_loop.sounds.sd")
    def test_play_listening_chime_with_device(self, mock_sd):
        """play_listening_chime passes device parameter."""
        from voice_loop.sounds import play_listening_chime

        play_listening_chime(device=2)

        call_kwargs = mock_sd.play.call_args[1]
        assert call_kwargs["device"] == 2

    @patch("voice_loop.sounds.sd")
    def test_play_listening_chime_error_suppressed(self, mock_sd):
        """play_listening_chime suppresses playback errors."""
        from voice_loop.sounds import play_listening_chime

        mock_sd.play.side_effect = RuntimeError("no device")
        play_listening_chime()  # Should not raise


class TestPlayErrorChime:
    """Verify error chime plays correctly."""

    @patch("voice_loop.sounds.sd")
    def test_play_error_chime_calls_sd(self, mock_sd):
        """play_error_chime calls sd.play with non-blocking."""
        from voice_loop.sounds import play_error_chime

        play_error_chime()

        mock_sd.play.assert_called_once()
        call_kwargs = mock_sd.play.call_args[1]
        assert call_kwargs["blocking"] is False


class TestPlayCompleteChime:
    """Verify complete chime plays correctly."""

    @patch("voice_loop.sounds.sd")
    def test_play_complete_chime_calls_sd(self, mock_sd):
        """play_complete_chime calls sd.play with non-blocking."""
        from voice_loop.sounds import play_complete_chime

        play_complete_chime()

        mock_sd.play.assert_called_once()
        call_kwargs = mock_sd.play.call_args[1]
        assert call_kwargs["blocking"] is False
